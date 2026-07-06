"""Tests for the rotmd.analysis package (analyze-worktree helpers).

These cover the CLI-unused helpers that were moved out of the extract scope:
orientation post-processing, inertia-derived quantities, vector-observable
serialization/stats, and the streaming trajectory readers.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import ellipsoid_positions, write_synthetic_gromacs
from rotmd.analysis.inertia import asymmetry_parameter, parallel_axis_theorem
from rotmd.analysis.observables import (
    compute_spin_nutation_ratio,
    observable_mean,
    observable_std,
    observable_to_dict,
    observable_to_xarray,
)
from rotmd.analysis.orientation import (
    compute_angular_displacement,
    compute_orientation_time_derivative,
    euler_zyz_to_rotation_matrix,
    extract_orientation,
    orientation_autocorrelation,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    unwrap_euler_angles,
)
from rotmd.core.orientation import rotation_matrix_to_euler_zyz
from rotmd.core.vector_observables import create_vector_observable


def _is_rotation(R):
    return (
            np.allclose(R.T @ R, np.eye(3), atol=1e-9)
            and np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    )


# ---------------------------------------------------------------------------
# Orientation post-processing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inertia-derived quantities
# ---------------------------------------------------------------------------

def test_parallel_axis_theorem_zero_displacement_is_identity():
    I = np.diag([10.0, 20.0, 30.0])
    assert parallel_axis_theorem(I, 5.0, np.zeros(3)) == pytest.approx(I)


def test_parallel_axis_theorem_shift():
    # Point mass M at origin, shift origin by d along x.
    I_com = np.zeros((3, 3))
    M, d = 2.0, np.array([3.0, 0.0, 0.0])
    shifted = parallel_axis_theorem(I_com, M, d)
    # Steiner term: M(|d|² I - d⊗d) -> I_xx stays 0, I_yy = I_zz = M|d|².
    assert shifted == pytest.approx(np.diag([0.0, M * 9, M * 9]))


def test_asymmetry_parameter_limits():
    # κ = (2I_b - I_a - I_c) / (I_c - I_a).
    # Prolate symmetric top (I_a < I_b = I_c) -> kappa = +1.
    assert asymmetry_parameter(np.array([1.0, 3.0, 3.0])) == pytest.approx(1.0)
    # Oblate symmetric top (I_a = I_b < I_c) -> kappa = -1.
    assert asymmetry_parameter(np.array([1.0, 1.0, 3.0])) == pytest.approx(-1.0)
    # Spherical top is degenerate; defined to return 0.
    assert asymmetry_parameter(np.array([2.0, 2.0, 2.0])) == 0.0


# ---------------------------------------------------------------------------
# Vector-observable serialization / statistics
# ---------------------------------------------------------------------------

def test_spin_nutation_ratio():
    obs = create_vector_observable(np.array([[3.0, 4.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
    # parallel mag = 3 (x), perp mag = 4 (y) -> ratio 0.75
    ratio = compute_spin_nutation_ratio(obs)
    assert ratio == pytest.approx([0.75], rel=1e-6)


def test_observable_to_xarray_with_and_without_times():
    xr = pytest.importorskip("xarray")
    vec = np.random.default_rng(0).normal(size=(8, 3))
    obs = create_vector_observable(vec, np.array([1.0, 0, 0]), times=np.arange(8.0), name="L")
    ds = observable_to_xarray(obs)
    assert isinstance(ds, xr.Dataset)
    assert "L" in ds and "L_mag" in ds
    assert "time" in ds.coords
    # Without times, frames are used as the coordinate instead.
    obs2 = create_vector_observable(vec, np.array([1.0, 0, 0]), name="L")
    ds2 = observable_to_xarray(obs2)
    assert "frame" in ds2.coords


def test_observable_to_dict_and_stats():
    vec = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    obs = create_vector_observable(vec, np.array([1.0, 0, 0]), name="tau")
    d = observable_to_dict(obs)
    assert set(d) >= {"tau", "tau_parallel", "tau_perp", "tau_mag"}
    assert observable_mean(obs) == pytest.approx(2.5)  # mean of [5, 0]
    assert observable_std(obs) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Streaming trajectory readers (require MDAnalysis)
# ---------------------------------------------------------------------------

def test_load_gromacs_trajectory_chunked_yields_all_frames(tmp_path):
    pytest.importorskip("MDAnalysis")
    from rotmd.analysis.io import load_gromacs_trajectory_chunked

    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=7)
    chunks = list(
        load_gromacs_trajectory_chunked(top, trj, selection="all", chunk_size=3, verbose=False)
    )
    total = sum(c["positions"].shape[0] for c in chunks)
    assert total == 7
    assert all(c["n_atoms"] == info["n_atoms"] for c in chunks)


def test_chunked_trajectory_reader_generator(tmp_path):
    pytest.importorskip("MDAnalysis")
    from rotmd.analysis.io import chunked_trajectory_reader

    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=6)
    chunks = list(
        chunked_trajectory_reader(top, trj, chunk_size=4, selection="all")
    )
    total = sum(c["n_frames"] for c in chunks)
    assert total == 6
