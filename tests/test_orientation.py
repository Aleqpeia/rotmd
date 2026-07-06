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
    membrane_tilt_angle,
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
    # 90° when the principal axis is along the normal (theta = 0 or π),
    # 0° when it lies in the membrane plane (theta = π/2).
    theta = np.array([0.0, np.pi / 2, np.pi])
    tilt = membrane_tilt_angle(theta)
    assert np.degrees(tilt) == pytest.approx([90.0, 0.0, 90.0], abs=1e-9)


def test_membrane_tilt_angle_is_folded_and_in_range():
    # theta and (π − theta) are the same undirected line → identical tilt,
    # and every value stays within [0, π/2].
    theta = np.linspace(0.0, np.pi, 50)
    tilt = membrane_tilt_angle(theta)
    assert np.all(tilt >= -1e-12) and np.all(tilt <= np.pi / 2 + 1e-12)
    assert tilt == pytest.approx(membrane_tilt_angle(np.pi - theta), abs=1e-12)
    # Complementary to the folded (acute) nutation angle: tilt = π/2 − foldedθ.
    folded = np.minimum(theta, np.pi - theta)
    assert tilt == pytest.approx(np.pi / 2 - folded, abs=1e-9)
