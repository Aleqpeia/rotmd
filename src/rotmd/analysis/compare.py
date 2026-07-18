"""System comparison: ΔDCCM and effect sizes (T6).

Answers "what did the mutation change?" with uncertainty attached. The guiding
rule, from the plan: **no ΔDCCM cell and no scalar difference is reported as a
bare point estimate.** A difference map computed from finite sampling always
looks structured; without a confidence interval there is no way to tell that
structure from noise, and the distal (allosteric) signal the reviewer asked
about is exactly where the effect is small enough for that to matter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# atanh(±1) is infinite, so correlations are clipped just inside the boundary
# before transforming. 1 - 1e-7 keeps z finite (~8.4) without distorting any
# value a real DCCM produces.
_R_CLIP = 1.0 - 1e-7


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher r-to-z transform, safe at ``r = ±1`` (the DCCM diagonal)."""
    return np.arctanh(np.clip(np.asarray(r, dtype=np.float64), -_R_CLIP, _R_CLIP))


def inverse_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(z, dtype=np.float64))


def pool_matrices(matrices: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Average correlation matrices across replicas in Fisher-z space.

    Correlations are bounded and their sampling distribution is skewed, so a
    plain arithmetic mean is biased toward zero — increasingly so for strong
    correlations, which are precisely the ones the map is about. Averaging the
    variance-stabilised z values and transforming back removes that bias.
    """
    stack = np.asarray(matrices, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"expected a stack of 2-D matrices, got shape {stack.shape}")
    return inverse_fisher_z(fisher_z(stack).mean(axis=0))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for ``a`` vs ``b`` using the pooled standard deviation.

    Positive means ``a`` exceeds ``b``. Returns 0.0 when both groups are
    constant (no spread and no difference), which is the only sensible answer
    and avoids a 0/0.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        raise ValueError(f"need >= 2 observations per group, got {a.size} and {b.size}")

    na, nb = a.size, b.size
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over resamples of ``values``."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    n = len(values)
    draws = [statistic(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def block_bootstrap_delta(
    blocks_a: np.ndarray,
    blocks_b: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Confidence interval on ``pool(b) - pool(a)`` by resampling whole blocks.

    Each block is one DCCM computed over a contiguous stretch of frames at
    least ``g`` long, so blocks are approximately independent and resampling
    them with replacement propagates the *correlated* sampling error that a
    naive per-frame bootstrap would badly underestimate.

    A cell is called significant when its CI excludes zero.
    """
    rng = rng or np.random.default_rng(0)
    blocks_a = np.asarray(blocks_a, dtype=np.float64)
    blocks_b = np.asarray(blocks_b, dtype=np.float64)
    if blocks_a.shape[1:] != blocks_b.shape[1:]:
        raise ValueError(
            f"block matrices disagree in shape: {blocks_a.shape[1:]} vs {blocks_b.shape[1:]}"
        )

    za, zb = fisher_z(blocks_a), fisher_z(blocks_b)
    na, nb = len(za), len(zb)

    draws = np.empty((n_boot, *blocks_a.shape[1:]), dtype=np.float64)
    for i in range(n_boot):
        mean_a = za[rng.integers(0, na, na)].mean(axis=0)
        mean_b = zb[rng.integers(0, nb, nb)].mean(axis=0)
        draws[i] = inverse_fisher_z(mean_b) - inverse_fisher_z(mean_a)

    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    return {
        "ci_low": lo,
        "ci_high": hi,
        "significant": (lo > 0) | (hi < 0),
        "bootstrap_sd": draws.std(axis=0, ddof=1),
    }


def distal_mask(resids: np.ndarray, site: int, exclude_within: int) -> np.ndarray:
    """``(N, N)`` mask hiding pairs where either residue is near ``site``.

    Coupling changes next door to a mutation are expected and uninformative;
    masking them is what lets the long-range rewiring the reviewer asked about
    actually be seen instead of being drowned by the local signal.
    """
    resids = np.asarray(resids)
    near = np.abs(resids - site) < exclude_within
    return ~(near[:, None] | near[None, :])


def compare_dccm(
    dcc_a: list[np.ndarray],
    dcc_b: list[np.ndarray],
    resids: np.ndarray,
    blocks_a: list[np.ndarray] | None = None,
    blocks_b: list[np.ndarray] | None = None,
    site: int | None = None,
    exclude_within: int = 5,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Full T6 comparison: ``pooled(b) - pooled(a)`` with significance."""
    pooled_a = pool_matrices(dcc_a)
    pooled_b = pool_matrices(dcc_b)
    if pooled_a.shape != pooled_b.shape:
        raise ValueError(f"maps differ in size: {pooled_a.shape} vs {pooled_b.shape}")

    delta = pooled_b - pooled_a
    result: dict[str, Any] = {
        "dcc_a": pooled_a,
        "dcc_b": pooled_b,
        "ddccm": delta,
        "resids": np.asarray(resids, dtype=np.int32),
        "n_replicas_a": np.int64(len(dcc_a)),
        "n_replicas_b": np.int64(len(dcc_b)),
    }

    if blocks_a and blocks_b:
        boot = block_bootstrap_delta(
            np.concatenate(list(blocks_a)), np.concatenate(list(blocks_b)),
            n_boot=n_boot, alpha=alpha, rng=np.random.default_rng(seed),
        )
        result.update({
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "significant": boot["significant"],
            "bootstrap_sd": boot["bootstrap_sd"],
            "alpha": np.float64(alpha),
            "n_boot": np.int64(n_boot),
        })

    if site is not None:
        mask = distal_mask(result["resids"], site, exclude_within)
        result["distal_mask"] = mask
        result["site"] = np.int64(site)
        result["exclude_within"] = np.int64(exclude_within)
        # The headline number for reviewer #4: the strongest coupling change
        # that is both statistically supported and far from the mutation.
        scored = np.where(mask, np.abs(delta), 0.0)
        if "significant" in result:
            scored = np.where(result["significant"], scored, 0.0)
        result["max_distal_change"] = np.float64(scored.max())

    return result


def effect_size_table(
    values_a: dict[str, np.ndarray],
    values_b: dict[str, np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Cohen's d with a bootstrap CI for each scalar metric present in both.

    ``values_*`` map a metric name to per-replica observations, e.g.
    ``{"rmsf_mean": [...one value per replica...]}``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name in sorted(set(values_a) & set(values_b)):
        a = np.asarray(values_a[name], dtype=np.float64).ravel()
        b = np.asarray(values_b[name], dtype=np.float64).ravel()
        if a.size < 2 or b.size < 2:
            continue

        d = cohens_d(b, a)
        # Resample both groups jointly so the CI reflects error in the
        # *difference*, not in either group alone.
        draws = [
            cohens_d(b[rng.integers(0, b.size, b.size)], a[rng.integers(0, a.size, a.size)])
            for _ in range(n_boot)
        ]
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({
            "metric": name,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": float(b.mean() - a.mean()),
            "cohens_d": d,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0),
            "n_a": int(a.size),
            "n_b": int(b.size),
        })
    return rows
