"""Tests for core.orientation: extract-scope orientation trajectory + tilt.

Post-processing helpers (Euler/quaternion conversions, angular displacement,
unwrapping, derivatives, autocorrelation) moved to rotmd.analysis and are tested
in test_analysis.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import ellipsoid_positions
from rotmd.core.orientation import (
    extract_orientation_trajectory,
    fold_tilt_and_psi,
    fold_tilt_and_psi_continuous,
    membrane_tilt_angle,
)


def _v_c_direction(theta, psi):
    """Reconstruct v_c's lab-frame direction from (theta, psi), per the
    R[2,0] = -sin(theta)cos(psi), R[2,1] = sin(theta)sin(psi), R[2,2] = cos(theta)
    identities in rotation_matrix_to_euler_zyz's derivation comment.
    """
    return np.stack(
        [-np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)],
        axis=-1,
    )


def _is_rotation(R):
    return (
        np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        and np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    )


def test_extract_orientation_trajectory_shapes_and_validity():
    rng = np.random.default_rng(0)
    base = ellipsoid_positions(50, axes=(1.0, 2.0, 4.0), seed=1)
    n_frames = 6
    # Small random rotations frame-to-frame.
    positions = np.stack([base + 0.01 * rng.normal(size=base.shape) for _ in range(n_frames)])
    masses = np.ones(50)
    euler, R = extract_orientation_trajectory(positions, masses)
    assert euler.shape == (n_frames, 3)
    assert R.shape == (n_frames, 3, 3)
    # Every frame's matrix must be a proper rotation (regression for the
    # det = -1 reflection bug from odd-count eigenvector sign flips).
    for f in range(n_frames):
        assert _is_rotation(R[f])
    # theta is constrained to [0, pi].
    assert np.all(euler[:, 1] >= -1e-9)
    assert np.all(euler[:, 1] <= np.pi + 1e-9)


def test_membrane_tilt_angle_anchor_points():
    # 0° when the principal axis is along the normal (theta = 0 or π),
    # 90° when it lies in the membrane plane (theta = π/2).
    theta = np.array([0.0, np.pi / 2, np.pi])
    tilt = membrane_tilt_angle(theta)
    assert np.degrees(tilt) == pytest.approx([0.0, 90.0, 0.0], abs=1e-9)


def test_membrane_tilt_angle_is_folded_and_in_range():
    # theta and (π − theta) are the same undirected line → identical tilt,
    # and every value stays within [0, π/2].
    theta = np.linspace(0.0, np.pi, 50)
    tilt = membrane_tilt_angle(theta)
    assert np.all(tilt >= -1e-12) and np.all(tilt <= np.pi / 2 + 1e-12)
    assert tilt == pytest.approx(membrane_tilt_angle(np.pi - theta), abs=1e-12)
    # tilt IS the folded (acute) nutation angle: tilt = min(theta, π − theta).
    folded = np.minimum(theta, np.pi - theta)
    assert tilt == pytest.approx(folded, abs=1e-9)


def test_fold_tilt_and_psi_preserves_the_headless_axis_direction():
    """(theta, psi) and (tilt, psi_folded) must describe the same undirected
    v_c line: reconstructing v_c's lab-frame direction from each should give
    the same vector up to an overall sign flip.
    """
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, np.pi, 200)
    psi = rng.uniform(0.0, 2 * np.pi, 200)

    tilt, psi_folded = fold_tilt_and_psi(theta, psi)

    v_before = _v_c_direction(theta, psi)
    v_after = _v_c_direction(tilt, psi_folded)

    same = np.all(np.isclose(v_before, v_after), axis=-1)
    opposite = np.all(np.isclose(v_before, -v_after), axis=-1)
    assert np.all(same | opposite)


def test_fold_tilt_and_psi_merges_a_bimodal_psi_split():
    """The scenario that motivated this: raw theta/psi split into two
    branches ~180deg apart in psi because some frames have the opposite
    v_c sign convention from others. Folding should collapse them.
    """
    rng = np.random.default_rng(1)
    n = 2000
    # One consistent physical orientation, expressed with a coin-flip choice
    # of which v_c sign convention each "frame" happened to use.
    true_tilt = np.full(n, np.radians(20.0))
    true_psi = np.full(n, np.radians(10.0))
    flip = rng.random(n) < 0.5
    theta = np.where(flip, np.pi - true_tilt, true_tilt)
    psi = np.where(flip, (true_psi + np.pi) % (2 * np.pi), true_psi)

    assert np.degrees(np.std(psi)) > 50  # the raw, unfolded split (regression guard)

    tilt, psi_folded = fold_tilt_and_psi(theta, psi)
    assert tilt == pytest.approx(true_tilt, abs=1e-9)
    assert psi_folded == pytest.approx(true_psi, abs=1e-9)


def test_fold_tilt_and_psi_continuous_matches_independent_fold_away_from_the_pivot():
    """Far from the pi/2 pivot (two clean, separated branches, as in the
    bimodal-split test above), the continuity-aware fold must agree with the
    simple per-frame one — continuity tracking shouldn't change anything when
    the independent decision was already unambiguous.
    """
    rng = np.random.default_rng(2)
    n = 500
    true_tilt = np.full(n, np.radians(20.0))
    true_psi = np.full(n, np.radians(10.0))
    flip = rng.random(n) < 0.5
    theta = np.where(flip, np.pi - true_tilt, true_tilt)
    psi = np.where(flip, (true_psi + np.pi) % (2 * np.pi), true_psi)

    tilt, psi_folded = fold_tilt_and_psi_continuous(theta, psi)
    assert tilt == pytest.approx(true_tilt, abs=1e-9)
    assert psi_folded == pytest.approx(true_psi, abs=1e-9)


def test_a_clean_sign_flip_is_always_exactly_reversible_by_the_independent_threshold():
    """A full v_c sign flip always crosses theta=pi/2 exactly, however close
    the true orientation sits to the pivot — so for *this* kind of
    corruption the independent per-frame fold is already exact, and
    continuity tracking cannot do better (there is nothing left to fix).

    This documents why fold_tilt_and_psi_continuous doesn't rescue a real
    system like N75K sitting at tilt~90deg: checked against the real
    trajectory, its raw theta changes smoothly frame-to-frame (no jumps
    bigger than a few degrees) — meaning it isn't corrupted by a discrete
    per-frame sign artifact at all. It's the same undirected-axis fold
    genuinely, continuously crossing the pivot as real thermal motion, and
    folding that always leaves a seam at each true crossing — an inherent
    property of the fold, not a bug either function can fix.
    """
    rng = np.random.default_rng(3)
    n = 1000
    true_theta = np.full(n, np.radians(89.9))
    true_psi = np.full(n, np.radians(15.0))
    v_true = np.stack(
        [
            -np.sin(true_theta) * np.cos(true_psi),
            np.sin(true_theta) * np.sin(true_psi),
            np.cos(true_theta),
        ],
        axis=-1,
    )

    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    v_noisy = v_true * sign[:, None]
    theta_noisy = np.arccos(np.clip(v_noisy[:, 2], -1.0, 1.0))
    psi_noisy = np.arctan2(v_noisy[:, 1], -v_noisy[:, 0]) % (2 * np.pi)

    tilt_i, psi_i = fold_tilt_and_psi(theta_noisy, psi_noisy)
    tilt_c, psi_c = fold_tilt_and_psi_continuous(theta_noisy, psi_noisy)
    assert psi_i == pytest.approx(true_psi, abs=1e-9)
    assert tilt_i == pytest.approx(true_theta, abs=1e-9)
    # Continuity tracking agrees exactly — there was no ambiguity for it to
    # resolve differently.
    assert psi_c == pytest.approx(psi_i, abs=1e-9)
    assert tilt_c == pytest.approx(tilt_i, abs=1e-9)

    # Continuity tracking recovers the (nearly) constant orientation.
    tilt_c, psi_c = fold_tilt_and_psi_continuous(theta_noisy, psi_noisy)
    assert np.degrees(np.std(psi_c)) < 5.0
    assert np.degrees(np.std(tilt_c)) < 5.0
