"""T3: equilibration detection and the window.json contract.

The statistical inefficiency is checked against an AR(1) process, whose
autocorrelation is exactly ``phi**t`` and whose ``g`` is therefore known
analytically as ``(1 + phi) / (1 - phi)`` — a real correctness check rather
than a self-consistency one.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.analysis.equilibration import (
    apply_window,
    block_average_t0,
    build_window,
    detect_equilibration,
    pool_windows,
    statistical_inefficiency,
)


def ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    """Stationary AR(1): x_t = phi x_{t-1} + eps, so C(t) = phi**t."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1.0 - phi**2)  # start on the stationary distribution
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


# ---------------------------------------------------------------------------
# statistical_inefficiency
# ---------------------------------------------------------------------------

def test_white_noise_has_g_near_one():
    g = statistical_inefficiency(np.random.default_rng(0).normal(size=20000))
    assert 0.9 <= g <= 1.6


def test_constant_series_returns_one():
    assert statistical_inefficiency(np.full(500, 3.14)) == 1.0


def test_short_series_returns_one():
    assert statistical_inefficiency(np.array([1.0])) == 1.0


@pytest.mark.parametrize("phi", [0.5, 0.8, 0.9])
def test_g_matches_the_ar1_analytic_value(phi):
    expected = (1.0 + phi) / (1.0 - phi)
    g = statistical_inefficiency(ar1(60000, phi, seed=1))
    assert g == pytest.approx(expected, rel=0.25)


def test_g_is_never_below_one():
    # Anticorrelated series would give g < 1 without the floor.
    alternating = np.tile([1.0, -1.0], 500)
    assert statistical_inefficiency(alternating) >= 1.0


# ---------------------------------------------------------------------------
# detect_equilibration
# ---------------------------------------------------------------------------

def test_stationary_series_keeps_almost_everything():
    result = detect_equilibration(ar1(5000, 0.5, seed=2), method="native")
    assert result["t0_index"] < 500  # nothing much to discard
    assert result["n_eff"] > 1000


def test_transient_is_detected_and_discarded():
    n = 6000
    t = np.arange(n)
    # A decaying transient (relaxation time 200 frames) on top of noise.
    series = 5.0 * np.exp(-t / 200.0) + 1.0 + 0.1 * ar1(n, 0.5, seed=3)
    result = detect_equilibration(series, method="native")
    # The transient is essentially gone after ~5 relaxation times.
    assert 100 < result["t0_index"] < 2500


def test_discarding_the_transient_raises_n_eff():
    n = 6000
    t = np.arange(n)
    series = 5.0 * np.exp(-t / 200.0) + 1.0 + 0.1 * ar1(n, 0.5, seed=3)
    result = detect_equilibration(series, method="native")
    # Keeping the whole series would leave far fewer independent samples,
    # which is precisely why the maximum-N_eff rule discards the transient.
    g_all = statistical_inefficiency(series)
    assert result["n_eff"] > n / g_all


def test_detection_is_deterministic():
    series = ar1(3000, 0.7, seed=4)
    a = detect_equilibration(series, method="native")
    b = detect_equilibration(series, method="native")
    assert a == b


def test_auto_method_falls_back_to_native_without_pymbar():
    result = detect_equilibration(ar1(1000, 0.5, seed=5), method="auto")
    assert result["method"] in {"native", "pymbar"}


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="must be 1-D"):
        detect_equilibration(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="too short"):
        detect_equilibration(np.zeros(2))


# ---------------------------------------------------------------------------
# block-average plateau cross-check
# ---------------------------------------------------------------------------

def test_block_average_finds_the_transient():
    n = 4000
    t = np.arange(n)
    with_transient = 5.0 * np.exp(-t / 200.0) + 1.0 + 0.05 * ar1(n, 0.3, seed=6)
    # True transient is spent after ~5 relaxation times (~1000 frames).
    assert 500 < block_average_t0(with_transient) < 2000


@pytest.mark.parametrize("seed", range(7, 17))
def test_block_average_does_not_fire_on_stationary_noise(seed):
    """The cross-check must not manufacture a transient that isn't there.

    Swept over seeds because a single-seed check would hide exactly the
    false-positive behaviour this estimator was chosen to avoid.
    """
    n = 4000
    assert block_average_t0(0.05 * ar1(n, 0.3, seed=seed)) < n // 20


def test_block_average_handles_constant_and_short_series():
    assert block_average_t0(np.full(500, 2.0)) == 0
    assert block_average_t0(np.zeros(10)) == 0


# ---------------------------------------------------------------------------
# build_window / pool_windows / apply_window
# ---------------------------------------------------------------------------

def _transient_series(n=4000, dt=10.0, seed=8):
    t = np.arange(n)
    series = 5.0 * np.exp(-t / 200.0) + 1.0 + 0.1 * ar1(n, 0.5, seed=seed)
    return np.arange(n) * dt, series


def test_build_window_maps_index_to_time_and_counts_frames():
    time_ps, series = _transient_series()
    w = build_window(time_ps, series, column="rmsd", method="native")

    assert w["t0_ps"] == time_ps[w["t0_index"]]
    assert w["n_frames_used"] == w["n_frames"] - w["t0_index"]
    assert w["frac_discarded"] == pytest.approx(w["t0_index"] / w["n_frames"])
    assert w["column"] == "rmsd"
    assert w["pooled"] is False
    assert set(w["cross_check"]) >= {"method", "t0_ps", "rel_diff", "agree"}


def test_build_window_agrees_with_cross_check_on_a_clean_transient():
    time_ps, series = _transient_series()
    w = build_window(time_ps, series, column="rmsd", method="native")
    assert w["cross_check"]["agree"] is True


def test_build_window_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        build_window(np.arange(10.0), np.zeros(9), column="rmsd")


def test_pool_takes_the_latest_t0():
    time_ps, series = _transient_series()
    a = build_window(time_ps, series, column="rmsd", method="native")
    b = dict(a, t0_ps=a["t0_ps"] + 5000.0, g=a["g"] * 2, n_eff=a["n_eff"] / 2)

    pooled = pool_windows([a, b])
    assert pooled["t0_ps"] == b["t0_ps"]  # conservative: the later start wins
    assert pooled["g"] == max(a["g"], b["g"])
    assert pooled["n_eff"] == min(a["n_eff"], b["n_eff"])
    assert pooled["pooled"] is True
    assert pooled["n_replicas"] == 2


def test_pool_requires_input():
    with pytest.raises(ValueError, match="at least one window"):
        pool_windows([])


def test_apply_window_selects_only_post_t0_frames():
    time_ps = np.arange(100) * 10.0
    mask = apply_window(time_ps, {"t0_ps": 250.0})
    assert mask.sum() == 75
    assert time_ps[mask].min() >= 250.0


def test_apply_window_raises_when_nothing_survives():
    with pytest.raises(ValueError, match="excludes every frame"):
        apply_window(np.arange(10.0), {"t0_ps": 1e9})


# ---------------------------------------------------------------------------
# `rotmd equilibrate` CLI
# ---------------------------------------------------------------------------

def _write_chunk(tmp_path, name="merged.npz", n=4000, dt=10.0, seed=8, with_meta=True):
    from rotmd.io.meta import meta_path, write_json
    from rotmd.io.output import save_npz

    t = np.arange(n)
    rmsd = 5.0 * np.exp(-t / 200.0) + 1.0 + 0.1 * ar1(n, 0.5, seed=seed)
    path = save_npz(tmp_path / name, {"time_ps": t * dt, "rmsd": rmsd, "rg": rmsd * 2})
    if with_meta:
        write_json(meta_path(path), {"system": "wt", "replica": 1, "window": None})
    return path


def test_cli_equilibrate_writes_window_and_updates_meta(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import meta_path, read_json

    npz = _write_chunk(tmp_path)
    out = tmp_path / "window.json"
    assert main(["equilibrate", str(npz), "-o", str(out), "--method", "native"]) == 0

    window = read_json(out)
    assert window["column"] == "rmsd"
    assert window["t0_ps"] > 0
    assert window["n_frames_used"] == window["n_frames"] - window["t0_index"]
    assert window["source_npz"] == str(npz)

    # The window travels with the data, not only in the standalone file.
    assert read_json(meta_path(npz))["window"]["t0_ps"] == window["t0_ps"]


def test_cli_equilibrate_respects_column_choice(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import read_json

    npz = _write_chunk(tmp_path)
    out = tmp_path / "window.json"
    main(["equilibrate", str(npz), "-o", str(out), "--column", "rg", "--method", "native"])
    assert read_json(out)["column"] == "rg"


def test_cli_equilibrate_reports_unknown_column(tmp_path):
    from rotmd.cli import main

    npz = _write_chunk(tmp_path)
    with pytest.raises(SystemExit, match="no column 'nope'"):
        main(["equilibrate", str(npz), "-o", str(tmp_path / "w.json"), "--column", "nope"])


def test_cli_equilibrate_rejects_multiple_npz_without_pool(tmp_path):
    from rotmd.cli import main

    a = _write_chunk(tmp_path, "a.npz")
    b = _write_chunk(tmp_path, "b.npz")
    with pytest.raises(SystemExit, match="exactly one merged .npz"):
        main(["equilibrate", str(a), str(b), "-o", str(tmp_path / "w.json")])


def test_cli_equilibrate_pools_replicas(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import read_json, write_json

    windows = []
    for i, t0 in enumerate([1000.0, 9000.0, 3000.0]):
        p = tmp_path / f"w{i}.json"
        write_json(p, {
            "t0_ps": t0, "g": 2.0 + i, "n_eff": 100.0 * (i + 1), "column": "rmsd",
            "method": "native", "cross_check": {"agree": True},
        })
        windows.append(str(p))

    out = tmp_path / "pooled.json"
    assert main(["equilibrate", *windows, "-o", str(out), "--pool"]) == 0

    pooled = read_json(out)
    assert pooled["pooled"] is True
    assert pooled["t0_ps"] == 9000.0  # conservative: latest across replicas
    assert pooled["n_replicas"] == 3


def test_cli_equilibrate_skips_existing_output_without_force(tmp_path):
    from rotmd.cli import main

    npz = _write_chunk(tmp_path)
    out = tmp_path / "window.json"
    out.write_text("sentinel")
    main(["equilibrate", str(npz), "-o", str(out), "--method", "native"])
    assert out.read_text() == "sentinel"

    main(["equilibrate", str(npz), "-o", str(out), "--method", "native", "--force"])
    assert out.read_text() != "sentinel"
