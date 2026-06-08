"""Tests for core.orientation: Euler/quaternion conversions and trajectories."""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.core.orientation import (
    compute_angular_displacement,
    compute_orientation_time_derivative,
    euler_zyz_to_rotation_matrix,
    extract_orientation,
    extract_orientation_trajectory,
    membrane_tilt_angle,
    orientation_autocorrelation,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler_zyz,
    rotation_matrix_to_quaternion,
    unwrap_euler_angles,
)
from _helpers import ellipsoid_positions


def _is_rotation(R):
    return (
        np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        and np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    )


@pytest.mark.parametrize(
    "phi,theta,psi",
    [
        (0.3, 0.8, 1.2),
        (2.5, 1.5, 0.4),
        (1.0, 0.1, 5.0),
    ],
)
def test_euler_roundtrip(phi, theta, psi):
    R = euler_zyz_to_rotation_matrix(phi, theta, psi)
    assert _is_rotation(R)
    phi2, theta2, psi2 = rotation_matrix_to_euler_zyz(R)
    R2 = euler_zyz_to_rotation_matrix(phi2, theta2, psi2)
    # Euler angles are not unique, but the matrix they encode must match.
    assert R2 == pytest.approx(R, abs=1e-8)


def test_euler_to_matrix_is_rotation_for_identity():
    assert euler_zyz_to_rotation_matrix(0, 0, 0) == pytest.approx(np.eye(3))


def test_gimbal_lock_theta_zero():
    # theta ~ 0 is the degenerate case; conversion must stay finite + valid.
    R = euler_zyz_to_rotation_matrix(0.7, 0.0, 0.0)
    phi, theta, psi = rotation_matrix_to_euler_zyz(R)
    assert np.isfinite([phi, theta, psi]).all()
    assert theta == pytest.approx(0.0, abs=1e-6)


def test_quaternion_roundtrip():
    R = euler_zyz_to_rotation_matrix(0.5, 1.0, 2.0)
    q = rotation_matrix_to_quaternion(R)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    R_back = quaternion_to_rotation_matrix(q)
    assert R_back == pytest.approx(R, abs=1e-8)


def test_angular_displacement_zero_and_known():
    e = np.array([0.3, 0.8, 1.2])
    assert compute_angular_displacement(e, e) == pytest.approx(0.0, abs=1e-7)
    # Pure 90-degree spin about z (ZYZ with theta=0): displacement = pi/2.
    e1 = np.array([0.0, 0.0, 0.0])
    e2 = np.array([0.0, 0.0, np.pi / 2])
    assert compute_angular_displacement(e1, e2) == pytest.approx(np.pi / 2, abs=1e-6)


def test_unwrap_removes_discontinuity():
    phi = np.array([0.0, 6.0, 0.1, 6.1])  # wraps across 2*pi
    euler = np.stack([phi, np.zeros(4), np.zeros(4)], axis=1)
    unwrapped = unwrap_euler_angles(euler)
    # After unwrapping, consecutive jumps should never exceed pi.
    assert np.all(np.abs(np.diff(unwrapped[:, 0])) <= np.pi + 1e-9)


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
    for f in range(n_frames):
        assert _is_rotation(R[f])
    # theta is constrained to [0, pi].
    assert np.all(euler[:, 1] >= -1e-9)
    assert np.all(euler[:, 1] <= np.pi + 1e-9)


def test_extract_orientation_single_is_rotation():
    pos = ellipsoid_positions(40, axes=(1.0, 2.0, 4.0), seed=2)
    masses = np.ones(40)
    R = extract_orientation(pos, masses)
    assert R.shape == (3, 3)
    assert _is_rotation(R)
    # Relative to itself the orientation is the identity.
    R_rel = extract_orientation(pos, masses, reference_frame=R)
    assert R_rel == pytest.approx(np.eye(3), abs=1e-8)


def test_orientation_time_derivative_linear_drift():
    times = np.linspace(0, 9, 10)
    # phi increases linearly at 0.2 rad/ps; theta, psi constant.
    euler = np.zeros((10, 3))
    euler[:, 0] = 0.2 * times
    deriv = compute_orientation_time_derivative(euler, times, smooth_window=5)
    assert deriv.shape == (10, 3)
    # Interior derivative of phi recovers the slope.
    assert deriv[5, 0] == pytest.approx(0.2, abs=1e-6)


def test_orientation_autocorrelation_starts_at_one():
    rng = np.random.default_rng(0)
    euler = rng.normal(size=(20, 3)) * 0.1
    acf = orientation_autocorrelation(euler, max_lag=5)
    assert acf.shape == (5,)
    # Zero-lag autocorrelation is exactly 1 (cos of zero displacement).
    assert acf[0] == pytest.approx(1.0, abs=1e-9)


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
