"""
Poisson Regression of Section-Crossing Counts (scikit-learn backend)
====================================================================

This is the inferential layer: given the tidy count table from
:mod:`rotmd.analysis.composition`, fit a Poisson generalised linear model that
asks *does bilayer composition change the rate of orientational crossings (or
binding events)?*

Modelling choices and why they matter
-------------------------------------

* **Offset, not raw counts.** Crossing counts grow with how long you watched.
  scikit-learn's :class:`~sklearn.linear_model.PoissonRegressor` has no native
  offset, so we use the standard equivalence: fit the *rate* ``count /
  exposure`` with ``sample_weight = exposure``. Minimising the
  exposure-weighted Poisson deviance is identical to fitting counts with a
  ``log(exposure)`` offset, so the coefficients are log-rate-ratios.

* **Unpenalised by default.** ``PoissonRegressor`` applies L2 shrinkage with
  ``alpha=1`` out of the box, which would bias coefficients toward zero. For
  inference we set ``alpha=0`` to recover the maximum-likelihood estimates; a
  small positive ``alpha`` is exposed for the case of many collinear
  covariates where the unpenalised fit is unstable.

* **Cluster bootstrap for uncertainty.** scikit-learn reports no standard
  errors, so we resample *replicas* (the independent unit) with replacement,
  refit, and take percentile confidence intervals and two-sided p-values from
  the bootstrap distribution. Resampling whole replicas also makes the
  inference robust to **overdispersion** -- crossings cluster in time, so the
  per-replica variance exceeds the nominal Poisson variance, and the bootstrap
  captures that excess directly. This is what replaces a Negative-Binomial
  model: we do not assume ``variance == mean``, we estimate the variability
  empirically.

* **Permutation test for the composition effect.** In place of a
  likelihood-ratio test we shuffle the composition labels across replicas and
  recompute the Poisson deviance explained; the p-value is the fraction of
  shuffles that explain at least as much as the real labelling. This is exact
  under the null "composition does not matter" and needs no distributional
  assumption.

* **Effect sizes as rate ratios.** ``exp(coefficient)`` is the multiplicative
  change in crossing rate for that predictor. A rate ratio below 1 for the
  complex bilayer is the quantitative statement "this composition stabilises
  orientation".

``pandas`` and ``scikit-learn`` are imported lazily with a clear error so the
rest of ``rotmd`` works even when they are not installed.

Author: rotmd contributors
Date: 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def _require_stats() -> tuple[Any, Any]:
    """Import the modelling stack with an actionable error if missing."""
    try:
        import pandas as pd
        from sklearn.linear_model import PoissonRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Crossing-count regression needs 'pandas' and 'scikit-learn'. "
            "Install them with: poetry add pandas scikit-learn"
        ) from exc
    return pd, PoissonRegressor


# ==============================================================================
# Report container
# ==============================================================================


@dataclass
class GLMReport:
    """Outcome of fitting a Poisson rate model to one count subset."""

    response: str
    terms: list[str]
    n_obs: int
    rate_ratios: pd.DataFrame
    dispersion: float
    overdispersed: bool
    effect_stat: float
    effect_pvalue: float
    n_bootstrap: int
    n_permutations: int
    inference: str = "cluster bootstrap over replicas + label permutation test"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "terms": self.terms,
            "n_obs": self.n_obs,
            "dispersion": self.dispersion,
            "overdispersed": self.overdispersed,
            "effect_stat": self.effect_stat,
            "effect_pvalue": self.effect_pvalue,
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
            "inference": self.inference,
            "rate_ratios": self.rate_ratios.to_dict(orient="list"),
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Poisson rate model for response={self.response!r}",
            f"  terms      : {', '.join(self.terms) or '(intercept only)'}",
            f"  n_obs      : {self.n_obs}",
            f"  dispersion : {self.dispersion:.3f} "
            f"({'OVERDISPERSED -> rely on bootstrap CIs' if self.overdispersed else 'ok'})",
            f"  composition effect (permutation): "
            f"deviance-explained={self.effect_stat:.3f}, p={self.effect_pvalue:.4g}",
            f"  inference  : {self.inference} "
            f"({self.n_bootstrap} boot, {self.n_permutations} perm)",
            "  rate ratios (exp(coef), bootstrap 95% CI):",
        ]
        for _, row in self.rate_ratios.iterrows():
            lines.append(
                f"    {row['term']:<24} RR={row['rate_ratio']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  p={row['pvalue']:.4g}"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ==============================================================================
# Design matrix
# ==============================================================================
#
# We avoid a formula mini-language (patsy) and build the design matrix directly:
# the composition `label` becomes indicator columns against a reference level
# (`simple` by default, so that a rate ratio < 1 reads as "stabilised relative
# to the simple bilayer"), and numeric covariates enter as-is.


def _build_design(
    data: pd.DataFrame,
    predictors: list[str],
    reference: str = "simple",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(X, term_names, notes)`` for the requested predictors."""
    columns: dict[str, np.ndarray] = {}
    notes: list[str] = []

    for predictor in predictors:
        if predictor == "label":
            cats = sorted(data["label"].astype(str).unique())
            if len(cats) < 2:
                continue  # nothing to contrast; handled by the caller
            ref = reference if reference in cats else cats[0]
            for cat in cats:
                if cat == ref:
                    continue
                columns[f"label[{cat}]"] = (data["label"].astype(str) == cat).astype(float).to_numpy()
        else:
            values = data[predictor].to_numpy(dtype=float)
            if np.unique(values[~np.isnan(values)]).size <= 1:
                notes.append(f"dropped predictor {predictor!r}: constant across systems")
                continue
            columns[predictor] = values

    term_names = list(columns)
    if not term_names:
        return np.empty((len(data), 0)), term_names, notes
    matrix = np.column_stack([columns[name] for name in term_names])
    return matrix, term_names, notes


# ==============================================================================
# Poisson fitting + deviance helpers
# ==============================================================================


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance ``2 * sum(y log(y/mu) - (y - mu))`` with the y=0 limit."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    # y * log(y/mu) -> 0 as y -> 0, handled explicitly to avoid log(0).
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def _fit_rate(
    poisson_cls: Any,
    X: np.ndarray,
    counts: np.ndarray,
    exposure: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a Poisson rate model; return predicted *counts* ``mu``.

    The model is fit on the rate ``counts / exposure`` weighted by exposure,
    which is equivalent to a counts model with a ``log(exposure)`` offset.
    """
    rate = counts / exposure
    if X.shape[1] == 0:
        # Intercept-only closed form: the MLE rate is the exposure-weighted mean.
        pooled = counts.sum() / exposure.sum()
        return pooled * exposure
    model = poisson_cls(alpha=alpha, fit_intercept=True, max_iter=1000)
    model.fit(X, rate, sample_weight=exposure)
    return model.predict(X) * exposure


def _fit_coefficients(
    poisson_cls: Any,
    X: np.ndarray,
    counts: np.ndarray,
    exposure: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit and return the log-rate-ratio coefficients (no intercept)."""
    rate = counts / exposure
    model = poisson_cls(alpha=alpha, fit_intercept=True, max_iter=1000)
    model.fit(X, rate, sample_weight=exposure)
    return np.asarray(model.coef_, dtype=float)


def _deviance_explained(
    poisson_cls: Any,
    X: np.ndarray,
    counts: np.ndarray,
    exposure: np.ndarray,
    alpha: float,
) -> float:
    """Drop in Poisson deviance from intercept-only to the full model."""
    mu_null = _fit_rate(poisson_cls, np.empty((len(counts), 0)), counts, exposure, alpha)
    mu_full = _fit_rate(poisson_cls, X, counts, exposure, alpha)
    return _poisson_deviance(counts, mu_null) - _poisson_deviance(counts, mu_full)


# ==============================================================================
# Resampling-based inference
# ==============================================================================


def _bootstrap_rate_ratios(
    poisson_cls: Any,
    data: pd.DataFrame,
    predictors: list[str],
    term_names: list[str],
    point_coefs: np.ndarray,
    *,
    alpha: float,
    n_bootstrap: int,
    reference: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Percentile CIs and two-sided p-values from a cluster bootstrap.

    Replicas are resampled with replacement *within each composition* so that
    every label stays represented and the design never collapses. Each resample
    is refit; the spread of the coefficients gives the uncertainty that
    scikit-learn does not report.
    """
    import pandas as pd

    counts_all = data["count"].to_numpy(dtype=float)
    exposure_all = data["exposure"].to_numpy(dtype=float)
    labels = data["label"].astype(str).to_numpy()
    groups = {lab: np.where(labels == lab)[0] for lab in np.unique(labels)}

    boot = np.full((n_bootstrap, len(term_names)), np.nan)
    for b in range(n_bootstrap):
        idx = np.concatenate(
            [rng.choice(members, size=members.size, replace=True) for members in groups.values()]
        )
        resampled = data.iloc[idx]
        X_b, names_b, _ = _build_design(resampled, predictors, reference)
        # Skip degenerate resamples (e.g. a column lost all its variation).
        if names_b != term_names or X_b.shape[1] != len(term_names):
            continue
        try:
            boot[b] = _fit_coefficients(
                poisson_cls, X_b, counts_all[idx], exposure_all[idx], alpha
            )
        except Exception:  # noqa: BLE001 - solver may fail on a bad resample
            continue

    rows = []
    for j, name in enumerate(term_names):
        samples = boot[~np.isnan(boot[:, j]), j]
        if samples.size == 0:
            rows.append(
                {
                    "term": name,
                    "rate_ratio": float(np.exp(point_coefs[j])),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "pvalue": float("nan"),
                }
            )
            continue
        # Two-sided bootstrap p-value: how often the effect flips sign.
        frac_pos = float(np.mean(samples > 0))
        frac_neg = float(np.mean(samples < 0))
        pvalue = min(1.0, 2.0 * min(frac_pos, frac_neg))
        rows.append(
            {
                "term": name,
                "rate_ratio": float(np.exp(point_coefs[j])),
                "ci_low": float(np.exp(np.percentile(samples, 2.5))),
                "ci_high": float(np.exp(np.percentile(samples, 97.5))),
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def _permutation_pvalue(
    poisson_cls: Any,
    data: pd.DataFrame,
    predictors: list[str],
    observed: float,
    *,
    alpha: float,
    n_permutations: int,
    reference: str,
    rng: np.random.Generator,
) -> float:
    """Permutation p-value for the joint composition effect.

    The predictor block is shuffled relative to the counts, breaking any
    association while preserving the predictors' own correlation structure
    (so the confounded PIP2/CHOL columns move together). The p-value is the
    fraction of shuffles whose deviance-explained matches or beats the real
    labelling.
    """
    counts = data["count"].to_numpy(dtype=float)
    exposure = data["exposure"].to_numpy(dtype=float)
    n = len(data)

    ge = 1  # +1 for the observed labelling (standard permutation correction)
    valid = 1
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        shuffled = data.iloc[perm].reset_index(drop=True)
        X_p, names_p, _ = _build_design(shuffled, predictors, reference)
        if X_p.shape[1] == 0:
            continue
        try:
            stat = _deviance_explained(poisson_cls, X_p, counts, exposure, alpha)
        except Exception:  # noqa: BLE001
            continue
        valid += 1
        if stat >= observed - 1e-9:
            ge += 1
    return ge / valid


# ==============================================================================
# High-level: analyse one count subset
# ==============================================================================


def analyze_counts(
    table: pd.DataFrame,
    *,
    predictors: list[str] | None = None,
    reference: str = "simple",
    alpha: float = 0.0,
    dispersion_threshold: float = 1.5,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    random_state: int | None = 0,
    verbose: bool = True,
) -> GLMReport:
    """
    Fit a Poisson rate model to one count subset with resampling-based inference.

    The ``table`` passed here should already be filtered to a *single*
    comparable response/coordinate (e.g. tilt crossings only); mixing
    coordinates would conflate different physical quantities in one rate.

    Args:
        table: Tidy count table (already subset to one response/coordinate).
        predictors: Predictor columns; defaults to ``["label"]`` (composition).
        reference: Composition level used as the baseline for ``label``.
        alpha: L2 penalty for ``PoissonRegressor`` (0 = maximum likelihood;
            raise it only when many collinear covariates destabilise the fit).
        dispersion_threshold: Pearson dispersion above which the data are
            flagged overdispersed (the bootstrap CIs remain valid either way).
        n_bootstrap: Cluster-bootstrap resamples for CIs / p-values.
        n_permutations: Label permutations for the composition-effect test.
        random_state: Seed for reproducible resampling.
        verbose: Print the fitted summary.

    Returns:
        A :class:`GLMReport`.

    Raises:
        ValueError: If the subset has too few rows or no predictor varies.
    """
    pd, poisson_cls = _require_stats()
    rng = np.random.default_rng(random_state)

    predictors = predictors or ["label"]
    data = table.reset_index(drop=True).copy()
    notes: list[str] = []

    if len(data) < len(predictors) + 2:
        raise ValueError(
            f"too few observations ({len(data)}) to fit {len(predictors)} "
            "predictor(s); add more replicas"
        )

    X, term_names, design_notes = _build_design(data, predictors, reference)
    notes.extend(design_notes)
    if X.shape[1] == 0:
        raise ValueError(
            "no predictor varies across the supplied systems; with a single "
            "composition there is nothing to compare"
        )

    # Warn about perfect collinearity among numeric predictors (PIP2/CHOL
    # confound): the fit will then alias their effects onto each other.
    numeric = [p for p in predictors if p != "label" and p in data]
    if len(numeric) > 1:
        corr = data[numeric].corr().abs()
        for i, a in enumerate(numeric):
            for b in numeric[i + 1 :]:
                if corr.loc[a, b] > 0.999:
                    notes.append(
                        f"predictors {a!r} and {b!r} are perfectly collinear; "
                        "their effects cannot be separated with these systems"
                    )

    counts = data["count"].to_numpy(dtype=float)
    exposure = data["exposure"].to_numpy(dtype=float)

    # --- Point estimates + dispersion -----------------------------------------
    point_coefs = _fit_coefficients(poisson_cls, X, counts, exposure, alpha)
    mu = _fit_rate(poisson_cls, X, counts, exposure, alpha)
    df_resid = len(data) - (X.shape[1] + 1)
    if df_resid > 0:
        pearson_chi2 = float(np.sum((counts - mu) ** 2 / np.clip(mu, 1e-12, None)))
        dispersion = pearson_chi2 / df_resid
    else:
        dispersion = float("nan")
        notes.append("no residual df for a dispersion estimate (n <= params)")
    overdispersed = bool(dispersion > dispersion_threshold)

    # --- Resampling inference -------------------------------------------------
    rate_ratios = _bootstrap_rate_ratios(
        poisson_cls,
        data,
        predictors,
        term_names,
        point_coefs,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        reference=reference,
        rng=rng,
    )
    observed_effect = _deviance_explained(poisson_cls, X, counts, exposure, alpha)
    effect_pvalue = _permutation_pvalue(
        poisson_cls,
        data,
        predictors,
        observed_effect,
        alpha=alpha,
        n_permutations=n_permutations,
        reference=reference,
        rng=rng,
    )

    report = GLMReport(
        response=str(data["response"].iloc[0]) if "response" in data else "count",
        terms=term_names,
        n_obs=int(len(data)),
        rate_ratios=rate_ratios,
        dispersion=dispersion,
        overdispersed=overdispersed,
        effect_stat=float(observed_effect),
        effect_pvalue=float(effect_pvalue),
        n_bootstrap=int(n_bootstrap),
        n_permutations=int(n_permutations),
        notes=notes,
    )
    if verbose:
        print(report.summary())
    return report


def analyze_all(
    table: pd.DataFrame,
    *,
    predictors: list[str] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, GLMReport]:
    """
    Fit a separate model for every (response, coordinate) group in the table.

    Each physical quantity (tilt crossings, azimuth direction changes, binding
    events, ...) gets its own model so their rates are never pooled. Groups that
    are too small or degenerate are skipped with a recorded reason rather than
    aborting the whole run. Extra keyword arguments are forwarded to
    :func:`analyze_counts`.

    Returns:
        Mapping ``"{response}:{coordinate}" -> GLMReport`` for groups that fit.
    """
    reports: dict[str, GLMReport] = {}
    for (response, coordinate), group in table.groupby(["response", "coordinate"]):
        key = f"{response}:{coordinate}"
        if verbose:
            print(f"\n=== {key} ===")
        try:
            reports[key] = analyze_counts(
                group, predictors=predictors, verbose=verbose, **kwargs
            )
        except ValueError as exc:
            if verbose:
                print(f"  skipped: {exc}")
    if not reports:
        raise ValueError(
            "no group could be modelled; you likely have a single composition "
            "or a single replica -- add more independent replicas"
        )
    return reports
