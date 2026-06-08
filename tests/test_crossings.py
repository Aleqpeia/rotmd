"""Known-answer and property tests for orientation section-crossing counts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rotmd.analysis.crossings import (
    body_axis_vectors,
    count_direction_changes,
    count_grid_crossings,
    count_level_crossings,
    count_orientation_crossings,
    count_plane_crossings,
    geodesic_angle_series,
    make_grid_levels,
)


# ------------------------------------------------------------------ level crossings
def test_level_crossings_basic_up_down():
    # -1, +1, -1, +1 crosses zero three times: up, down, up.
    total, up, down = count_level_crossings([-1.0, 1.0, -1.0, 1.0], 0.0)
    assert (total, up, down) == (3, 2, 1)


def test_level_crossings_grazing_is_not_a_crossing():
    # Touch the level and return to the same side -> no crossing.
    total, up, down = count_level_crossings([-1.0, 0.0, -1.0], 0.0)
    assert (total, up, down) == (0, 0, 0)


def test_level_crossings_through_zero_sample_counts_once():
    # Pass cleanly from below to above, landing on zero mid-way.
    total, up, down = count_level_crossings([-1.0, 0.0, 2.0], 0.0)
    assert (total, up, down) == (1, 1, 0)


# ------------------------------------------------------------------- grid crossings
def test_grid_crossings_linear_ramp():
    # 0 -> 100 with 15 deg spacing crosses gridlines 15,30,45,60,75,90 -> 6.
    ramp = np.linspace(0.0, 100.0, 50)
    total, up, down = count_grid_crossings(ramp, 15.0)
    assert (total, up, down) == (6, 6, 0)


def test_grid_crossings_counts_multiple_lines_per_step():
    # A single jump of 50 deg sweeps gridlines 15,30,45 -> 3 crossings.
    total, _, _ = count_grid_crossings([0.0, 50.0], 15.0)
    assert total == 3


def test_grid_crossings_periodic_single_wrap_is_small():
    # 179 -> -179 deg is a 2 deg move, not a near-full rotation.
    total, _, _ = count_grid_crossings([179.0, -179.0], 15.0, periodic=True)
    assert total <= 1


def test_grid_crossings_periodic_full_rotation_counts_all_lines():
    # A genuine full turn must cross every one of the 360/15 = 24 gridlines.
    angle = np.linspace(0.0, 359.0, 360)
    wrapped = (angle + 180.0) % 360.0 - 180.0
    total, _, _ = count_grid_crossings(wrapped, 15.0, periodic=True)
    assert total == 23  # 24 lines, last one at 360==0 not yet reached at 359


def test_make_grid_levels_interior_only():
    levels = make_grid_levels(0.0, 45.0, 15.0)
    assert np.allclose(levels, [15.0, 30.0])


# --------------------------------------------------------------- direction changes
def test_direction_changes_triangle():
    # up, up, down, down, up -> two reversals.
    assert count_direction_changes([0.0, 1.0, 2.0, 1.0, 0.0, 1.0]) == 2


def test_direction_changes_deadband_suppresses_jitter():
    signal = [0.0, 0.1, 0.0, 0.1, 0.0, 5.0]
    assert count_direction_changes(signal, min_step=1.0) == 0


# ------------------------------------------------------------------ plane crossings
def test_plane_crossings_of_rotating_axis():
    # A unit vector sweeping the xy-plane crosses the x=0 plane twice per turn.
    angle = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    u = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
    total, _, _ = count_plane_crossings(u, plane_normal=(1.0, 0.0, 0.0))
    assert total == 2


# ------------------------------------------------------- SO(3) high-level wrappers
def test_body_axis_vectors_identity():
    R = np.tile(np.eye(3), (5, 1, 1))
    u = body_axis_vectors(R, body_axis=(0.0, 0.0, 1.0))
    assert np.allclose(u, np.array([0.0, 0.0, 1.0]))


def test_geodesic_angle_zero_for_constant_orientation():
    R = np.tile(np.eye(3), (4, 1, 1))
    assert np.allclose(geodesic_angle_series(R), 0.0)


def test_count_orientation_crossings_tilt_monotone():
    # Build rotations that tilt the body z-axis steadily from 0 to 80 deg about x.
    # (80 deg is chosen off the 15 deg grid to avoid an on-gridline endpoint.)
    tilts = np.linspace(0.0, np.radians(80.0), 50)
    R = np.array(
        [
            [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
            for t in tilts
        ]
    )
    times = np.arange(50, dtype=float)
    counts = count_orientation_crossings(
        R, times, coordinate="tilt", spacing_deg=15.0
    )
    # Tilt sweeps 0..80 deg -> crosses 15,30,45,60,75 -> 5 gridlines.
    assert counts.n_crossings == 5
    assert counts.exposure == 49.0


# ------------------------------------------------------------------- property tests
_series = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=200,
)
_spacing = st.floats(min_value=0.5, max_value=90.0)


@given(_series, _spacing)
def test_grid_crossings_time_reversal_symmetry(values, spacing):
    # Running the trajectory backwards crosses the same gridlines the same number
    # of times; only the up/down direction labels swap. This is convention-free.
    series = np.asarray(values)
    total, up, down = count_grid_crossings(series, spacing)
    rtotal, rup, rdown = count_grid_crossings(series[::-1], spacing)
    assert total == rtotal
    assert up == rdown
    assert down == rup
