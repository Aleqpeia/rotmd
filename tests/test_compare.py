"""T6: ΔDCCM, pooling and effect sizes.

The statistical claims are tested against ground truth rather than against
themselves: Fisher-z pooling is checked for the bias it exists to remove,
Cohen's d against its textbook definition, and the bootstrap for calibration
(no significance where there is no difference, significance where there is).
"""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.analysis.compare import (
    block_bootstrap_delta,
    bootstrap_ci,
    cohens_d,
    compare_dccm,
    distal_mask,
    effect_size_table,
    fisher_z,
    inverse_fisher_z,
    pool_matrices,
)

# ---------------------------------------------------------------------------
# Fisher-z
# ---------------------------------------------------------------------------

def test_fisher_z_roundtrips():
    r = np.linspace(-0.99, 0.99, 50)
    np.testing.assert_allclose(inverse_fisher_z(fisher_z(r)), r, atol=1e-12)


def test_fisher_z_is_finite_at_plus_minus_one():
    """The DCCM diagonal is exactly 1.0, so atanh must not blow up."""
    z = fisher_z(np.array([-1.0, 1.0]))
    assert np.isfinite(z).all()
    np.testing.assert_allclose(inverse_fisher_z(z), [-1.0, 1.0], atol=1e-6)


def test_pooling_preserves_a_unit_diagonal():
    mats = [np.eye(4) for _ in range(3)]
    np.testing.assert_allclose(np.diag(pool_matrices(mats)), 1.0, atol=1e-6)


def test_fisher_pooling_is_less_biased_than_the_arithmetic_mean():
    """The reason Fisher-z pooling is used at all.

    Sample correlations around a strong true value are skewed low, so a plain
    mean is biased toward zero; averaging in z space corrects it.
    """
    rho = 0.9
    rng = np.random.default_rng(0)
    n_obs, n_reps = 12, 200

    sampled = []
    for _ in range(n_reps):
        x = rng.normal(size=n_obs)
        y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n_obs)
        sampled.append(np.corrcoef(x, y)[0, 1])
    mats = [np.array([[1.0, r], [r, 1.0]]) for r in sampled]

    fisher = pool_matrices(mats)[0, 1]
    arithmetic = np.mean(sampled)
    assert abs(fisher - rho) < abs(arithmetic - rho)


def test_pool_matrices_rejects_non_stacks():
    with pytest.raises(ValueError, match="stack of 2-D matrices"):
        pool_matrices(np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def test_cohens_d_matches_the_textbook_definition():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([3.0, 4.0, 5.0, 6.0])
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)  # equal n
    assert cohens_d(b, a) == pytest.approx((b.mean() - a.mean()) / pooled)


def test_cohens_d_sign_and_scale_invariance():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 200)
    b = rng.normal(1, 1, 200)
    assert cohens_d(b, a) > 0
    assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))
    # Scaling both groups leaves a standardised effect size unchanged.
    assert cohens_d(3 * b, 3 * a) == pytest.approx(cohens_d(b, a))


def test_cohens_d_recovers_a_known_separation():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 20000)
    b = rng.normal(0.8, 1.0, 20000)
    assert cohens_d(b, a) == pytest.approx(0.8, abs=0.05)


def test_cohens_d_on_constant_groups_is_zero_not_nan():
    assert cohens_d(np.full(5, 2.0), np.full(5, 2.0)) == 0.0


def test_cohens_d_needs_two_observations():
    with pytest.raises(ValueError, match=">= 2 observations"):
        cohens_d(np.array([1.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_true_mean():
    rng = np.random.default_rng(3)
    values = rng.normal(5.0, 1.0, 500)
    lo, hi = bootstrap_ci(values, np.mean, n_boot=500, rng=rng)
    assert lo < 5.0 < hi
    assert lo < values.mean() < hi


def test_block_bootstrap_finds_no_significance_when_systems_are_identical():
    """Calibration: identical inputs must not manufacture a difference."""
    rng = np.random.default_rng(4)
    blocks = rng.normal(0.0, 0.05, size=(40, 6, 6))
    blocks = 0.5 * (blocks + np.swapaxes(blocks, 1, 2))
    out = block_bootstrap_delta(blocks, blocks.copy(), n_boot=300, rng=rng)
    # Same data both sides: essentially nothing should be flagged.
    assert out["significant"].mean() < 0.05


def test_block_bootstrap_detects_a_real_offset():
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 0.03, size=(40, 5, 5))
    b = a + 0.4                                  # a large, uniform shift
    out = block_bootstrap_delta(a, b, n_boot=300, rng=rng)
    assert out["significant"].mean() > 0.9
    assert (out["ci_low"] > 0).mean() > 0.9      # and in the right direction


def test_block_bootstrap_ci_ordering_and_shape():
    rng = np.random.default_rng(6)
    a = rng.normal(0, 0.1, size=(20, 4, 4))
    b = rng.normal(0, 0.1, size=(25, 4, 4))
    out = block_bootstrap_delta(a, b, n_boot=200, rng=rng)
    assert out["ci_low"].shape == (4, 4)
    assert np.all(out["ci_low"] <= out["ci_high"])


def test_block_bootstrap_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="disagree in shape"):
        block_bootstrap_delta(np.zeros((5, 4, 4)), np.zeros((5, 3, 3)))


# ---------------------------------------------------------------------------
# Distal mask
# ---------------------------------------------------------------------------

def test_distal_mask_hides_the_neighbourhood_of_the_site():
    resids = np.arange(70, 81)
    mask = distal_mask(resids, site=75, exclude_within=3)
    i = list(resids).index(75)
    assert not mask[i].any()          # every pair involving 75 is hidden
    j = list(resids).index(70)
    assert mask[j, list(resids).index(80)]   # far-far pairs survive
    assert not mask[j, i]


def test_distal_mask_is_symmetric():
    mask = distal_mask(np.arange(1, 30), site=15, exclude_within=4)
    np.testing.assert_array_equal(mask, mask.T)


# ---------------------------------------------------------------------------
# compare_dccm
# ---------------------------------------------------------------------------

def _system(n_reps=3, n_res=8, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    maps, blocks = [], []
    for _ in range(n_reps):
        base = rng.normal(0, 0.1, size=(10, n_res, n_res))
        base = 0.5 * (base + np.swapaxes(base, 1, 2)) + offset
        for m in base:
            np.fill_diagonal(m, 1.0)
        blocks.append(base.astype(np.float32))
        maps.append(pool_matrices(base))
    return maps, blocks


def test_compare_dccm_difference_and_provenance():
    maps_a, blocks_a = _system(seed=0)
    maps_b, blocks_b = _system(seed=1, offset=0.3)
    resids = np.arange(1, 9)

    result = compare_dccm(maps_a, maps_b, resids, blocks_a, blocks_b, n_boot=200)

    np.testing.assert_allclose(result["ddccm"], result["dcc_b"] - result["dcc_a"], atol=1e-12)
    np.testing.assert_array_equal(result["resids"], resids)
    assert int(result["n_replicas_a"]) == 3
    assert result["significant"].shape == (8, 8)
    assert np.all(result["ci_low"] <= result["ci_high"])


def test_compare_dccm_without_blocks_omits_significance():
    maps_a, _ = _system(seed=0)
    maps_b, _ = _system(seed=1)
    result = compare_dccm(maps_a, maps_b, np.arange(1, 9))
    assert "ddccm" in result
    assert "significant" not in result       # no CI => no claim


def test_compare_dccm_distal_summary():
    maps_a, blocks_a = _system(n_res=20, seed=0)
    maps_b, blocks_b = _system(n_res=20, seed=1, offset=0.25)
    resids = np.arange(65, 85)

    result = compare_dccm(
        maps_a, maps_b, resids, blocks_a, blocks_b,
        site=75, exclude_within=4, n_boot=200,
    )
    assert result["distal_mask"].shape == (20, 20)
    assert int(result["site"]) == 75
    assert float(result["max_distal_change"]) >= 0.0


def test_compare_dccm_rejects_mismatched_sizes():
    maps_a, _ = _system(n_res=8, seed=0)
    maps_b, _ = _system(n_res=6, seed=1)
    with pytest.raises(ValueError, match="differ in size"):
        compare_dccm(maps_a, maps_b, np.arange(8))


def test_compare_dccm_is_reproducible_for_a_fixed_seed():
    maps_a, blocks_a = _system(seed=0)
    maps_b, blocks_b = _system(seed=1, offset=0.2)
    kwargs = dict(blocks_a=blocks_a, blocks_b=blocks_b, n_boot=100, seed=7)
    first = compare_dccm(maps_a, maps_b, np.arange(1, 9), **kwargs)
    second = compare_dccm(maps_a, maps_b, np.arange(1, 9), **kwargs)
    np.testing.assert_array_equal(first["ci_low"], second["ci_low"])
    np.testing.assert_array_equal(first["significant"], second["significant"])


# ---------------------------------------------------------------------------
# Effect-size table
# ---------------------------------------------------------------------------

def test_effect_size_table_reports_d_with_a_ci():
    rng = np.random.default_rng(8)
    a = {"rmsf": rng.normal(1.0, 0.1, 30), "sasa": rng.normal(50.0, 5.0, 30)}
    b = {"rmsf": rng.normal(1.6, 0.1, 30), "sasa": rng.normal(50.2, 5.0, 30)}

    rows = {r["metric"]: r for r in effect_size_table(a, b, n_boot=300)}
    assert rows["rmsf"]["cohens_d"] > 2.0      # a large, real shift
    assert rows["rmsf"]["significant"] is True
    assert rows["rmsf"]["ci_low"] <= rows["rmsf"]["cohens_d"] <= rows["rmsf"]["ci_high"]
    # sasa barely moved, so its CI should straddle zero.
    assert abs(rows["sasa"]["cohens_d"]) < 0.5
    assert rows["sasa"]["significant"] is False


def test_effect_size_table_only_covers_shared_metrics():
    rng = np.random.default_rng(9)
    a = {"shared": rng.normal(size=10), "only_a": rng.normal(size=10)}
    b = {"shared": rng.normal(size=10), "only_b": rng.normal(size=10)}
    assert [r["metric"] for r in effect_size_table(a, b, n_boot=50)] == ["shared"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_dccm(tmp_path, name, offset=0.0, seed=0, n_res=10, blocks=True):
    from rotmd.io.output import save_npz

    maps, blk = _system(n_reps=1, n_res=n_res, offset=offset, seed=seed)
    payload = {"dcc": maps[0], "resids": np.arange(70, 70 + n_res, dtype=np.int32)}
    if blocks:
        payload["dcc_blocks"] = blk[0]
    return save_npz(tmp_path / name, payload)


def test_cli_compare_end_to_end(tmp_path):
    from rotmd.cli import main
    from rotmd.io.output import load_npz

    a = [_write_dccm(tmp_path, f"wt{i}.npz", seed=i) for i in range(3)]
    b = [_write_dccm(tmp_path, f"mut{i}.npz", offset=0.3, seed=10 + i) for i in range(3)]
    out = tmp_path / "compare.npz"
    fig = tmp_path / "ddccm.png"

    rc = main([
        "compare", "--a", *map(str, a), "--b", *map(str, b),
        "-o", str(out), "--label-a", "WT", "--label-b", "N75K",
        "--site", "75", "--n-boot", "100", "--figure", str(fig),
    ])
    assert rc == 0

    result = load_npz(out)
    assert result["ddccm"].shape == (10, 10)
    assert "significant" in result
    assert "distal_mask" in result
    assert fig.exists() and fig.stat().st_size > 5000


def test_cli_compare_without_blocks_still_runs(tmp_path):
    from rotmd.cli import main
    from rotmd.io.output import load_npz

    a = [_write_dccm(tmp_path, f"a{i}.npz", seed=i, blocks=False) for i in range(2)]
    b = [_write_dccm(tmp_path, f"b{i}.npz", seed=5 + i, blocks=False) for i in range(2)]
    out = tmp_path / "c.npz"
    assert main(["compare", "--a", *map(str, a), "--b", *map(str, b), "-o", str(out)]) == 0
    assert "significant" not in load_npz(out)


def test_cli_compare_rejects_mismatched_residue_numbering(tmp_path):
    from rotmd.cli import main

    a = [_write_dccm(tmp_path, "a.npz", n_res=10, seed=0)]
    b = [_write_dccm(tmp_path, "b.npz", n_res=8, seed=1)]
    with pytest.raises(SystemExit, match="different residue numbering"):
        main(["compare", "--a", str(a[0]), "--b", str(b[0]), "-o", str(tmp_path / "x.npz")])
