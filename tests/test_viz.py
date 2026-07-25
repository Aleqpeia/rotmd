"""The plotting API and the numeric parts of the restored plots.

Figures are checked structurally — a file exists, is not trivially small, has
the panel count asked for. Everything with arithmetic in it is checked against
a case with a known answer, because those are the parts that were wrong in the
legacy visualization package and the parts a rendered PNG cannot vouch for.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from rotmd.viz import (
    angle_grid_degrees,
    available,
    figure,
    finite_percentile,
    local_minima,
    poincare_crossings,
    power_spectrum,
    render,
    spectral_density,
    time_coloured_path,
)


# ---------------------------------------------------------------------------
# The @figure decorator
# ---------------------------------------------------------------------------

@figure(figsize=(4, 3))
def _single(ax, y):
    ax.plot(y)
    return "drawn"


@figure(nrows=2, ncols=2, figsize=(6, 5))
def _quad(axes, y):
    for ax in axes.reshape(-1):
        ax.plot(y)
    return axes.shape


@figure(nrows=1, ncols=3, figsize=(8, 3))
def _strip(axes, y):
    for ax in axes:
        ax.plot(y)
    return axes.shape


def test_output_mode_writes_the_file_and_returns_its_path(tmp_path):
    out = _single(np.arange(10.0), output=tmp_path / "nested" / "fig.png")
    assert out.exists() and out.stat().st_size > 1000


def test_figure_mode_returns_an_open_figure(tmp_path):
    fig = _single(np.arange(10.0))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_ax_mode_draws_into_the_caller_figure_and_returns_the_inner_value():
    """The mode that makes plots composable: no figure is created or closed."""
    fig, ax = plt.subplots()
    assert _single(np.arange(10.0), ax=ax) == "drawn"
    assert len(ax.lines) == 1
    assert plt.fignum_exists(fig.number), "ax mode must not close the caller's figure"
    plt.close(fig)


def test_ax_and_output_together_are_refused():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="not both"):
        _single(np.arange(10.0), ax=ax, output="x.png")
    plt.close(fig)


@pytest.mark.parametrize(
    ("nrows", "ncols", "expected"),
    [(1, 1, ()), (1, 3, (3,)), (3, 1, (3,)), (2, 2, (2, 2))],
)
def test_axes_are_shaped_by_the_grid_not_by_matplotlib_squeeze(nrows, ncols, expected):
    """One panel arrives bare, any strip flat, a true grid two-dimensional.

    Matplotlib's own squeeze collapses 1xN and Nx1 differently from a grid, so
    `axes[i]` would break on a transpose. The shape is pinned here instead.
    """
    seen = []

    @figure(nrows=nrows, ncols=ncols, figsize=(4, 3))
    def _probe(axes):
        seen.append(getattr(axes, "shape", ()))

    plt.close(_probe())
    assert seen == [expected]


def test_a_failed_draw_closes_its_figure(tmp_path):
    @figure()
    def _boom(ax):
        raise RuntimeError("boom")

    before = set(plt.get_fignums())
    with pytest.raises(RuntimeError, match="boom"):
        _boom(output=tmp_path / "never.png")
    assert set(plt.get_fignums()) == before, "figure leaked when the draw raised"


def test_title_overrides_and_per_call_figsize(tmp_path):
    fig = _single(np.arange(10.0), title="custom", figsize=(3, 2))
    assert fig.axes[0].get_title() == "custom"
    assert tuple(fig.get_size_inches()) == (3.0, 2.0)
    plt.close(fig)


def test_registry_reaches_plots_by_name(tmp_path):
    assert "pmf-heatmap" in available()
    out = render(
        "acf",
        np.linspace(0, 10, 60),
        np.exp(-np.linspace(0, 10, 60) / 3),
        output=tmp_path / "by-name.png",
    )
    assert out.exists()


def test_unknown_plot_name_lists_the_alternatives():
    with pytest.raises(KeyError, match="unknown plot"):
        render("no-such-plot")


# ---------------------------------------------------------------------------
# Unit handling -- the most common legacy defect
# ---------------------------------------------------------------------------

def test_angle_grid_returns_degrees_in_x_y_order():
    theta = np.array([0.0, np.pi / 2, np.pi])
    psi = np.array([0.0, np.pi, 2 * np.pi])
    psi_grid, theta_grid = angle_grid_degrees(theta, psi)

    assert psi_grid.shape == theta_grid.shape == (3, 3)
    np.testing.assert_allclose(theta_grid[:, 0], [0.0, 90.0, 180.0])
    np.testing.assert_allclose(psi_grid[0, :], [0.0, 180.0, 360.0])


def test_pmf_heatmap_minima_land_inside_the_drawn_mesh(tmp_path):
    """The legacy bug: mesh in radians, markers in degrees, 57x apart."""
    theta = np.linspace(0, np.pi, 30)
    psi = np.linspace(0, 2 * np.pi, 40)
    tt, pp = np.meshgrid(theta, psi, indexing="ij")
    pmf = 5.0 * (1 - np.exp(-(((tt - np.pi / 2) ** 2) / 0.1 + ((pp - np.pi) ** 2) / 0.1)))

    from rotmd.viz import plot_pmf_heatmap

    fig, ax = plt.subplots()
    plot_pmf_heatmap(pmf, theta, psi, ax=ax)

    markers = [line for line in ax.lines if line.get_marker() == "*"]
    assert markers, "expected at least one basin minimum to be marked"
    for marker in markers:
        x, y = marker.get_xdata()[0], marker.get_ydata()[0]
        assert 0 <= x <= 360 and 0 <= y <= 180, (
            f"minimum marker at ({x:.1f}, {y:.1f}) is outside the angle ranges — "
            f"radian/degree mix"
        )
    plt.close(fig)


def test_finite_percentile_ignores_infinities_and_survives_an_empty_grid():
    grid = np.array([1.0, 2.0, 3.0, np.inf, np.nan])
    assert finite_percentile(grid, 50) == pytest.approx(2.0)
    assert finite_percentile(np.full(5, np.inf), 95) == 1.0


# ---------------------------------------------------------------------------
# local_minima
# ---------------------------------------------------------------------------

def test_local_minima_finds_each_basin_once():
    grid = np.ones((21, 21)) * 10.0
    grid[5, 5] = 0.0
    grid[15, 15] = 0.5
    found = local_minima(grid, threshold=2.0)
    assert sorted(found) == [(5, 5), (15, 15)]


def test_a_flat_basin_is_reported_once_not_once_per_cell():
    """The legacy `values == minimum_filter(values)` test fired on every
    plateau cell, spraying 25 markers across this one basin."""
    grid = np.ones((20, 20)) * 5.0
    grid[8:13, 8:13] = 1.0  # a 5x5 flat floor: 25 tied cells, one basin
    found = local_minima(grid)
    assert len(found) == 1
    assert found[0] == (10, 10), "the marker belongs in the middle of the plateau"


def test_a_shelf_that_drains_downhill_is_not_a_basin():
    """A flat region with a lower neighbour is on the way down, not a minimum."""
    grid = np.tile(np.linspace(10.0, 0.0, 20), (20, 1))
    grid[:, 8:12] = grid[:, 8][:, None]  # flatten a stripe partway down the slope
    assert all(j >= 18 for _i, j in local_minima(grid, threshold=1.0)), (
        "only the true bottom of the slope should be reported"
    )


def test_symmetric_ties_at_a_basin_floor_still_report_a_minimum():
    """An even-sized grid puts the basin centre between bins, so the two
    deepest cells are exactly equal; a strict test would report nothing."""
    axis = np.linspace(-1, 1, 20)  # no sample at 0
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    found = local_minima(xx**2 + yy**2, threshold=0.05)
    assert len(found) == 1


def test_local_minima_ignores_unvisited_bins():
    grid = np.full((10, 10), np.inf)
    grid[4:8, 4:8] = 3.0
    grid[6, 6] = 1.0
    assert local_minima(grid) == [(6, 6)]


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def test_spectral_density_of_an_exponential_acf_matches_the_lorentzian():
    """C(t) = exp(-t/tau)  ->  S(f) = 2 tau / (1 + (2 pi f tau)^2).

    The legacy version omitted the factor of 2 from folding the negative lags,
    so every value it reported was half the truth. S(0) = 2*tau pins that down.
    """
    tau = 5.0
    t = np.arange(0, 400, 0.05)
    freqs, density = spectral_density(t, np.exp(-t / tau))

    assert density[0] == pytest.approx(2 * tau, rel=0.02)
    expected = 2 * tau / (1 + (2 * np.pi * freqs[:200] * tau) ** 2)
    np.testing.assert_allclose(density[:200], expected, rtol=0.05, atol=0.02)


def test_power_spectrum_peaks_at_the_frequency_that_is_there():
    dt, f0 = 0.05, 0.7
    t = np.arange(0, 200, dt)
    freqs, power = power_spectrum(t, 3.0 * np.sin(2 * np.pi * f0 * t) + 10.0)
    assert freqs[np.argmax(power)] == pytest.approx(f0, abs=0.01)


def test_power_spectrum_drops_the_dc_bin():
    t = np.arange(0, 50, 0.1)
    freqs, _ = power_spectrum(t, np.full_like(t, 7.0))
    assert (freqs > 0).all()


# ---------------------------------------------------------------------------
# Poincare crossings
# ---------------------------------------------------------------------------

def test_crossings_are_counted_once_per_transit_with_direction_honoured():
    # phi sweeps 0 -> 6 pi: three increasing crossings of each plane.
    phi = np.linspace(0, 6 * np.pi, 4001) % (2 * np.pi)
    up, _ = poincare_crossings(phi, np.pi, direction=1)
    down, _ = poincare_crossings(phi, np.pi, direction=-1)
    both, _ = poincare_crossings(phi, np.pi, direction=0)

    assert up.size == 3
    assert down.size == 0, "a monotonically increasing phi has no decreasing crossings"
    assert both.size == 3


def test_slow_approach_is_one_crossing_not_many():
    """A proximity test -- the legacy `|phi - target| < tol` -- reports every
    frame spent near the plane. A crossing test reports the transit."""
    rng = np.random.default_rng(0)
    phi = np.concatenate([
        # Hundreds of frames loitering just short of the plane, never reaching it.
        np.pi - 0.02 - 0.005 * np.abs(rng.standard_normal(300)),
        np.linspace(np.pi - 0.02, np.pi + 0.5, 20),  # then one transit
    ])
    index, _ = poincare_crossings(phi, np.pi, direction=1)
    assert index.size == 1

    proximity = int(np.count_nonzero(np.abs(phi - np.pi) < 0.05))
    assert proximity > 50, "fixture must actually dwell near the plane"


def test_wraparound_is_not_mistaken_for_a_crossing_of_pi():
    """phi jumping 2pi -> 0 passes through the branch cut, not through pi."""
    phi = np.array([0.1, 0.2, 6.2, 6.25, 0.05, 0.1])
    index, _ = poincare_crossings(phi, np.pi, direction=0)
    assert index.size == 0


def test_crossing_of_zero_is_found_from_either_side_of_the_wrap():
    phi = np.array([6.20, 6.25, 0.05, 0.10])
    index, _ = poincare_crossings(phi, 0.0, direction=0)
    assert index.size == 1


def test_crossing_position_is_interpolated_between_frames():
    # Linear ramp crossing pi/2 exactly midway between two samples.
    phi = np.array([np.pi / 2 - 0.1, np.pi / 2 + 0.1])
    index, frac = poincare_crossings(phi, np.pi / 2, direction=1)
    assert index.tolist() == [0]
    assert frac[0] == pytest.approx(0.5, abs=1e-6)


def test_poincare_sections_reject_mismatched_frame_counts():
    """The legacy version sliced its two inputs differently and indexed one
    with a mask built from the other."""
    from rotmd.viz import plot_poincare_sections

    with pytest.raises(ValueError, match="same frames"):
        plot_poincare_sections(np.zeros((100, 3)), np.zeros((98, 3)))


# ---------------------------------------------------------------------------
# Trajectory rendering
# ---------------------------------------------------------------------------

def test_time_coloured_path_is_one_artist_not_one_per_frame():
    """25 000 Line2D objects is what the legacy per-frame loop produced."""
    n = 5000
    x = np.linspace(0, 1, n)
    fig, ax = plt.subplots()
    collection = time_coloured_path(ax, x, x**2, x)

    assert len(ax.lines) == 0
    assert len(ax.collections) == 1
    assert len(collection.get_segments()) == n - 1
    plt.close(fig)


def test_segment_colours_are_centred_on_the_segment():
    fig, ax = plt.subplots()
    collection = time_coloured_path(ax, np.arange(4.0), np.arange(4.0), np.array([0.0, 2.0, 4.0, 6.0]))
    np.testing.assert_allclose(collection.get_array(), [1.0, 3.0, 5.0])
    plt.close(fig)


def test_energy_phase_space_rejects_a_misaligned_colour_series():
    """The legacy version passed `energy[1:]` against full-length angles."""
    from rotmd.viz import plot_energy_phase_space

    theta = np.linspace(0, np.pi, 50)
    with pytest.raises(ValueError, match="frames"):
        plot_energy_phase_space(theta, theta, energy=np.zeros(49))


# ---------------------------------------------------------------------------
# Every registered plot renders
# ---------------------------------------------------------------------------

def test_every_registered_plot_closes_its_figure(tmp_path):
    """Guards the whole registry against the leak the legacy `plt.close()`
    (bare, closing whatever figure happened to be current) allowed."""
    from rotmd.viz import plot_autocorrelation

    before = set(plt.get_fignums())
    t = np.linspace(0, 30, 200)
    plot_autocorrelation(t, np.exp(-t / 4), output=tmp_path / "a.png")
    assert set(plt.get_fignums()) == before
