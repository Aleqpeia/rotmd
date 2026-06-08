"""Tests for the sklearn-based Poisson crossing-count regression."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from rotmd.analysis.regression import analyze_counts  # noqa: E402

# Keep resampling light so the suite stays fast; the seeds make it deterministic.
_FAST = {"n_bootstrap": 300, "n_permutations": 299, "random_state": 0}


def _two_composition_table(rate_simple, rate_complex, n_rep=6, exposure=100.0, seed=0):
    """Synthetic per-replica counts with a known composition effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, rate in (("simple", rate_simple), ("complex", rate_complex)):
        for rep in range(n_rep):
            rows.append(
                {
                    "label": label,
                    "replica": rep,
                    "response": "crossings",
                    "coordinate": "tilt",
                    "count": int(rng.poisson(rate * exposure)),
                    "exposure": exposure,
                    "n_frames": int(exposure),
                    "has_complex": 0.0 if label == "simple" else 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_simple_is_the_reference_level():
    table = _two_composition_table(rate_simple=5.0, rate_complex=1.0)
    report = analyze_counts(table, predictors=["label"], verbose=False, **_FAST)
    assert report.terms == ["label[complex]"]


def test_recovers_stabilising_effect():
    # Complex bilayer has 1/5 the crossing rate -> rate ratio ~ 0.2, p tiny.
    table = _two_composition_table(rate_simple=5.0, rate_complex=1.0)
    report = analyze_counts(table, predictors=["label"], verbose=False, **_FAST)

    rr = report.rate_ratios
    complex_row = rr[rr["term"].str.contains("complex")].iloc[0]
    assert complex_row["rate_ratio"] < 0.4
    assert complex_row["ci_high"] < 1.0  # CI excludes "no effect"
    assert report.effect_pvalue < 0.05


def test_no_effect_gives_nonsignificant_permutation():
    table = _two_composition_table(rate_simple=3.0, rate_complex=3.0, seed=42)
    report = analyze_counts(table, predictors=["label"], verbose=False, **_FAST)
    assert report.effect_pvalue > 0.05


def test_single_composition_raises():
    table = _two_composition_table(rate_simple=3.0, rate_complex=3.0)
    only_simple = table[table["label"] == "simple"]
    with pytest.raises(ValueError, match="no predictor varies"):
        analyze_counts(only_simple, predictors=["label"], verbose=False, **_FAST)


def test_constant_covariate_is_dropped_with_note():
    table = _two_composition_table(rate_simple=4.0, rate_complex=1.0)
    table["always_one"] = 1.0
    report = analyze_counts(
        table, predictors=["label", "always_one"], verbose=False, **_FAST
    )
    assert any("always_one" in note for note in report.notes)


def test_overdispersion_flagged():
    # Inject heavy overdispersion by mixing two very different rates per label.
    rng = np.random.default_rng(7)
    rows = []
    for label, lo, hi in (("simple", 2.0, 20.0), ("complex", 0.5, 5.0)):
        for rep in range(8):
            rate = lo if rep % 2 == 0 else hi
            rows.append(
                {
                    "label": label,
                    "replica": rep,
                    "response": "crossings",
                    "coordinate": "tilt",
                    "count": int(rng.poisson(rate * 100.0)),
                    "exposure": 100.0,
                    "n_frames": 100,
                }
            )
    table = pd.DataFrame(rows)
    report = analyze_counts(table, predictors=["label"], verbose=False, **_FAST)
    assert report.dispersion > 1.5
    assert report.overdispersed
    # Bootstrap CIs are still produced and remain valid under overdispersion.
    assert report.rate_ratios["ci_low"].notna().all()


def test_alpha_zero_matches_unweighted_rate_ratio():
    # With a clean, non-overdispersed design the fitted RR should track the
    # empirical rate ratio (mean complex rate / mean simple rate).
    table = _two_composition_table(rate_simple=8.0, rate_complex=2.0, n_rep=10)
    report = analyze_counts(table, predictors=["label"], verbose=False, **_FAST)
    rate = table.groupby("label").apply(
        lambda g: g["count"].sum() / g["exposure"].sum(), include_groups=False
    )
    empirical_rr = rate["complex"] / rate["simple"]
    fitted_rr = report.rate_ratios.iloc[0]["rate_ratio"]
    assert fitted_rr == pytest.approx(empirical_rr, rel=0.05)
