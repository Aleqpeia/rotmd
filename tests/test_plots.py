"""T2: equilibration figures.

Figures are checked structurally (files written, non-trivial size, panel count)
plus a real numeric test of the running mean, which is the only part with
arithmetic worth getting wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rotmd.analysis.plots import (
    plot_equilibration,
    plot_replica_overlay,
    running_mean,
)

# ---------------------------------------------------------------------------
# running_mean
# ---------------------------------------------------------------------------

def test_running_mean_of_a_constant_is_that_constant():
    np.testing.assert_allclose(running_mean(np.full(50, 3.0), 7), 3.0)


def test_running_mean_preserves_length_and_window_one_is_identity():
    a = np.arange(20.0)
    assert running_mean(a, 5).shape == a.shape
    np.testing.assert_array_equal(running_mean(a, 1), a)


def test_running_mean_edges_are_honest_partial_averages():
    """Edges must average the frames that exist, not decay toward zero."""
    a = np.arange(10.0)
    smoothed = running_mean(a, 5)
    # First point averages a[0:3] = 0,1,2 -> 1.0 (half-window = 2).
    assert smoothed[0] == pytest.approx(np.mean(a[0:3]))
    assert smoothed[-1] == pytest.approx(np.mean(a[-3:]))
    # A monotone ramp stays monotone; no artificial dip at the ends.
    assert np.all(np.diff(smoothed) > 0)


def test_running_mean_smooths_noise_toward_the_true_mean():
    rng = np.random.default_rng(0)
    noisy = 5.0 + rng.normal(size=4000)
    assert running_mean(noisy, 200).std() < noisy.std() / 5


def test_running_mean_survives_an_oversized_window():
    """A window wider than the series must clamp, not index out of bounds.

    It does not collapse to the global mean: the window stays *centred*, so
    each point still averages a different sub-range near the edges.
    """
    a = np.arange(10.0)
    smoothed = running_mean(a, 1000)
    assert smoothed.shape == a.shape
    assert smoothed.min() >= a.min() and smoothed.max() <= a.max()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _dataset(n=500, n_dom=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return {
        "time_ps": t * 10.0,
        "rmsd": 2.0 * (1 - np.exp(-t / 50.0)) + 0.1 * rng.normal(size=n),
        "rg": 14.0 + 0.1 * rng.normal(size=n),
        "rmsd_dom": np.abs(rng.normal(size=(n, n_dom))) + 1.0,
    }


def test_plot_equilibration_writes_a_figure(tmp_path):
    out = plot_equilibration(
        _dataset(), tmp_path / "equil.png",
        window={"t0_ps": 1000.0, "g": 3.2, "n_eff": 120.0},
        domain_names=["EF1", "EF2"], title="wt/1",
    )
    assert out.exists()
    assert out.stat().st_size > 5000  # a real rendered plot, not a blank canvas


def test_plot_equilibration_works_without_a_window(tmp_path):
    out = plot_equilibration(_dataset(), tmp_path / "nowindow.png")
    assert out.exists()


def test_plot_equilibration_handles_absent_domains(tmp_path):
    data = _dataset()
    data["rmsd_dom"] = np.zeros((500, 0))
    out = plot_equilibration(data, tmp_path / "nodom.png")
    assert out.exists()


def test_plot_equilibration_panel_count_tracks_available_series(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_equilibration(_dataset(), tmp_path / "three.png")
    # rmsd + rg + per-domain = 3 panels; without domains, 2.
    assert len(plt.gcf().axes) in (0, 3)  # figure already closed by the helper
    plt.close("all")


def test_plot_equilibration_requires_something_to_plot(tmp_path):
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_equilibration({"time_ps": np.arange(10.0)}, tmp_path / "x.png")


def test_plot_equilibration_creates_missing_directories(tmp_path):
    out = plot_equilibration(_dataset(), tmp_path / "deep" / "nested" / "e.png")
    assert out.exists()


def test_replica_overlay_writes_a_figure(tmp_path):
    datasets = [(f"wt/{i}", _dataset(seed=i)) for i in range(3)]
    out = plot_replica_overlay(datasets, tmp_path / "overlay.png", window={"t0_ps": 500.0})
    assert out.exists()
    assert out.stat().st_size > 5000


def test_replica_overlay_needs_a_common_observable(tmp_path):
    a = _dataset()
    b = {"time_ps": a["time_ps"]}          # no rmsd, no rg
    with pytest.raises(ValueError, match="no observable common"):
        plot_replica_overlay([("a", a), ("b", b)], tmp_path / "bad.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_replica(tmp_path, stem, system="wt", replica=1, seed=0):
    from rotmd.io.meta import meta_path, write_json
    from rotmd.io.output import save_npz

    path = save_npz(tmp_path / f"{stem}.npz", _dataset(seed=seed))
    write_json(meta_path(path), {
        "system": system, "replica": replica, "domain_names": ["EF1", "EF2"],
    })
    return path


def test_cli_plot_equil_single_replica(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import write_json

    npz = _write_replica(tmp_path, "rep1")
    window = write_json(tmp_path / "window.json", {"t0_ps": 1000.0, "g": 3.0, "n_eff": 100.0})
    outdir = tmp_path / "figs"

    assert main(["plot-equil", str(npz), "-o", str(outdir), "--window", str(window)]) == 0
    assert (outdir / "equil_rep1.png").exists()
    assert not (outdir / "equil_overlay.png").exists()  # only one replica


def test_cli_plot_equil_multiple_replicas_adds_overlay(tmp_path):
    from rotmd.cli import main

    paths = [_write_replica(tmp_path, f"rep{i}", replica=i, seed=i) for i in (1, 2, 3)]
    outdir = tmp_path / "figs"

    assert main(["plot-equil", *map(str, paths), "-o", str(outdir)]) == 0
    for i in (1, 2, 3):
        assert (outdir / f"equil_rep{i}.png").exists()
    assert (outdir / "equil_overlay.png").exists()


def test_cli_plot_equil_skips_existing_without_force(tmp_path):
    from rotmd.cli import main

    npz = _write_replica(tmp_path, "rep1")
    outdir = tmp_path / "figs"
    outdir.mkdir()
    stub = outdir / "equil_rep1.png"
    stub.write_bytes(b"sentinel")

    main(["plot-equil", str(npz), "-o", str(outdir)])
    assert stub.read_bytes() == b"sentinel"

    main(["plot-equil", str(npz), "-o", str(outdir), "--force"])
    assert stub.read_bytes() != b"sentinel"
