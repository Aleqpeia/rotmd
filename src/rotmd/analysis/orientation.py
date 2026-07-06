"""Orientation analysis helpers (analyze worktree).

These operate on the orientation time series produced by the extract pipeline
(Euler angles, rotation matrices). They are deliberately kept out of
``rotmd.core.orientation`` — the extract CLI never calls them, so they live here
where the downstream *analyze* stage can build on them without pulling extra
dependencies into the extract hot path.

The single primitive shared with extract is
:func:`rotmd.core.orientation.rotation_matrix_to_euler_zyz`, imported below.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def euler_zyz_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (ZYZ convention) to rotation matrix.

    Args:
        phi: First rotation about Z-axis (radians)
        theta: Rotation about Y-axis (radians)
        psi: Second rotation about Z-axis (radians)

    Returns:
        R: (3, 3) rotation matrix
    """
    # Individual rotation matrices
    Rz_phi = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    Ry_theta = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    Rz_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

    # R = Rz(φ) Ry(θ) Rz(ψ)
    return Rz_phi @ Ry_theta @ Rz_psi


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to unit quaternion.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        q: (4,) quaternion [w, x, y, z] with ||q|| = 1

    Notes:
        Uses scipy.spatial.transform.Rotation for numerical stability
    """
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # Returns [x, y, z, w] in scipy
    return np.array([q[3], q[0], q[1], q[2]])  # Convert to [w, x, y, z]


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion to rotation matrix.

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        R: (3, 3) rotation matrix
    """
    # Convert from [w, x, y, z] to scipy format [x, y, z, w]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    rot = Rotation.from_quat(q_scipy)
    return rot.as_matrix()


def extract_orientation(
        positions: np.ndarray,
        masses: np.ndarray,
        reference_frame: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract the orientation rotation matrix from atomic positions.

    Single-frame counterpart of
    :func:`rotmd.core.orientation.extract_orientation_trajectory`:

    1. Compute inertia tensor from positions
    2. Find principal axes (eigenvectors)
    3. Construct rotation matrix from principal axes

    Args:
        positions: (n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses
        reference_frame: Optional (3, 3) reference orientation.
                        If provided, rotations are relative to this frame.
                        If None, uses lab frame.

    Returns:
        R: (3, 3) rotation matrix

    Notes:
        - Positions should be centered at origin (center=True on load).
        - Principal axes are ordered by ascending eigenvalue.
    """
    from rotmd.core.inertia import inertia_tensor, principal_axes

    # Compute inertia tensor and principal axes
    I = inertia_tensor(positions, masses)
    moments, axes = principal_axes(I)

    # Rotation matrix from lab frame to body frame
    R = axes.T  # Principal axes as columns → rotation matrix
    # If reference frame provided, compute relative rotation
    if reference_frame is not None:
        R = R @ reference_frame.T

    return R


def compute_angular_displacement(euler1: np.ndarray, euler2: np.ndarray) -> float:
    """
    Compute angular displacement between two orientations.

    Uses quaternion representation for geodesic distance on SO(3).

    Args:
        euler1: (3,) Euler angles [phi, theta, psi] for first orientation
        euler2: (3,) Euler angles [phi, theta, psi] for second orientation

    Returns:
        angle: Angular displacement in radians (0 to π)

    Notes:
        - This is the geodesic distance on SO(3)
        - Avoids gimbal lock issues with Euler angles
    """
    # Convert both to rotation matrices
    R1 = euler_zyz_to_rotation_matrix(*euler1)
    R2 = euler_zyz_to_rotation_matrix(*euler2)

    # Relative rotation
    R_rel = R2 @ R1.T

    # Convert to quaternion and extract angle
    q = rotation_matrix_to_quaternion(R_rel)

    # Angle from quaternion: θ = 2 * arccos(w)
    return 2 * np.arccos(np.clip(q[0], -1.0, 1.0))


def unwrap_euler_angles(euler_angles: np.ndarray) -> np.ndarray:
    """
    Unwrap Euler angles to remove 2π discontinuities.

    Args:
        euler_angles: (n_frames, 3) array of [phi, theta, psi]

    Returns:
        unwrapped: (n_frames, 3) unwrapped angles

    Notes:
        - Only unwraps phi and psi (which are periodic)
        - theta naturally lives in [0, π]
    """
    unwrapped = euler_angles.copy()

    # Unwrap phi and psi (columns 0 and 2)
    unwrapped[:, 0] = np.unwrap(euler_angles[:, 0])
    unwrapped[:, 2] = np.unwrap(euler_angles[:, 2])

    return unwrapped


def compute_orientation_time_derivative(
        euler_angles: np.ndarray, times: np.ndarray, smooth_window: int = 5
) -> np.ndarray:
    """
    Compute time derivatives of Euler angles.

    Args:
        euler_angles: (n_frames, 3) array of [phi, theta, psi]
        times: (n_frames,) timestamps in ps
        smooth_window: Window size for Savitzky-Golay smoothing

    Returns:
        derivatives: (n_frames, 3) array of [dphi/dt, dtheta/dt, dpsi/dt] in rad/ps

    Notes:
        - Uses central differences with smoothing
        - Edges use forward/backward differences
    """
    from scipy.signal import savgol_filter

    # Unwrap angles first
    unwrapped = unwrap_euler_angles(euler_angles)

    n_frames = len(euler_angles)
    derivatives = np.zeros((n_frames, 3))

    for i in range(3):
        # Smooth if requested
        if smooth_window > 1:
            smoothed = savgol_filter(
                unwrapped[:, i], window_length=min(smooth_window, n_frames), polyorder=2
            )
        else:
            smoothed = unwrapped[:, i]

        # Compute derivative using central differences
        dt = np.diff(times)
        derivatives[1:-1, i] = (smoothed[2:] - smoothed[:-2]) / (times[2:] - times[:-2])

        # Edge cases
        derivatives[0, i] = (smoothed[1] - smoothed[0]) / dt[0]
        derivatives[-1, i] = (smoothed[-1] - smoothed[-2]) / dt[-1]

    return derivatives


def orientation_autocorrelation(euler_angles: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Compute orientational autocorrelation function.

    C(t) = <cos(Δθ(t))> where Δθ(t) is angular displacement after time t.

    Args:
        euler_angles: (n_frames, 3) array of Euler angles
        max_lag: Maximum lag time in frames (None = n_frames // 2)

    Returns:
        acf: (max_lag,) autocorrelation function

    Notes:
        - Uses quaternion-based angular displacement
        - Decays from 1.0 to ~0 as orientation decorrelates
    """
    n_frames = len(euler_angles)
    if max_lag is None:
        max_lag = n_frames // 2

    acf = np.zeros(max_lag)

    for lag in range(max_lag):
        cos_angles = []
        for i in range(n_frames - lag):
            angle = compute_angular_displacement(euler_angles[i], euler_angles[i + lag])
            cos_angles.append(np.cos(angle))
        acf[lag] = np.mean(cos_angles)

    return acf
