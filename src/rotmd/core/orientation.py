"""
Protein Orientation Analysis

This module provides utilities for extracting and analyzing protein orientations
in 3D space using Euler angles, quaternions, and rotation matrices.

Key Features:
- Euler angle extraction (ZYZ convention)
- Quaternion representations
- SO(3) rotation matrix operations
- Integration with inertia tensor principal axes
- Time series analysis of orientational dynamics

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


def rotation_matrix_to_euler_zyz(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles using ZYZ convention.

    The ZYZ convention is commonly used in molecular dynamics:
    - First rotation φ about lab Z-axis
    - Second rotation θ about new Y-axis (nutation angle)
    - Third rotation ψ about final Z-axis (spin angle)

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (phi, theta, psi): Euler angles in radians
        - phi ∈ [0, 2π]: First rotation about Z
        - theta ∈ [0, π]: Nutation angle
        - psi ∈ [0, 2π]: Spin angle

    Notes:
        - Handles gimbal lock at θ = 0 and θ = π
        - Uses atan2 for proper quadrant handling
    """
    # Extract elements for ZYZ convention
    # R = Rz(φ) Ry(θ) Rz(ψ)
    theta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))

    # Handle gimbal lock
    if np.abs(np.sin(theta)) < 1e-10:
        # θ ≈ 0 or θ ≈ π
        if theta < np.pi / 2:
            # θ ≈ 0: φ + ψ is determined, set ψ = 0
            phi = np.arctan2(R[1, 0], R[0, 0])
            psi = 0.0
        else:
            # θ ≈ π: φ - ψ is determined, set ψ = 0
            phi = np.arctan2(-R[1, 0], R[0, 0])
            psi = 0.0
    else:
        phi = np.arctan2(R[2, 1], R[2, 0])
        psi = np.arctan2(R[1, 2], -R[0, 2])

    # Normalize to [0, 2π] for phi and psi, [0, π] for theta
    phi = phi % (2 * np.pi)
    psi = psi % (2 * np.pi)

    return phi, theta, psi


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
    Rz_phi = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )

    Ry_theta = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    Rz_psi = np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
    )

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
    reference_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract Euler angles trajectory from atomic positions.

    This function computes the protein's orientation at each frame by:
    1. Computing inertia tensor from positions
    2. Finding principal axes (eigenvectors)
    3. Constructing rotation matrix from principal axes

    Args:
        positions: (n_frames, n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses
        reference_frame: Optional (3, 3) reference orientation.
                        If provided, rotations are relative to this frame.
                        If None, uses lab frame.

    Returns:
        Rotation matrix

    Notes:
        - Positions should be centered at origin (use trajectory.load_trajectory with center=True)
        - Principal axes are ordered by eigenvalues (I1 >= I2 >= I3)
        - Handles sign ambiguity in eigenvectors for smooth trajectories
    """
    from .inertia import inertia_tensor, principal_axes

    prev_axes = None

    # Compute inertia tensor and principal axes
    I = inertia_tensor(positions, masses)
    moments, axes = principal_axes(I)

    # Rotation matrix from lab frame to body frame
    R = axes.T  # Principal axes as columns → rotation matrix
    # If reference frame provided, compute relative rotation
    if reference_frame is not None:
        R = R @ reference_frame.T

    return R


def extract_orientation_trajectory(
    positions: np.ndarray,
    masses: np.ndarray,
    reference_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract Euler angles trajectory from atomic positions.

    This function computes the protein's orientation at each frame by:
    1. Computing inertia tensor from positions
    2. Finding principal axes (eigenvectors)
    3. Constructing rotation matrix from principal axes
    4. Converting to Euler angles

    Args:
        positions: (n_frames, n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses
        reference_frame: Optional (3, 3) reference orientation.
                        If provided, rotations are relative to this frame.
                        If None, uses lab frame.

    Returns:
        euler_angles: (n_frames, 3) array of [phi, theta, psi] in radians
        rotation_matrix: (n_frames, 3, 3) rotation matrices

    Notes:
        - Positions should be centered at origin (use trajectory.load_trajectory with center=True)
        - Principal axes are ordered by eigenvalues (I1 >= I2 >= I3)
        - Handles sign ambiguity in eigenvectors for smooth trajectories

    Examples:
        >>> import numpy as np
        >>> positions = np.random.rand(1000, 100, 3)  # 1000 frames, 100 atoms
        >>> masses = np.ones(100)
        >>> euler, R = extract_orientation_trajectory(positions, masses)
        >>> print(euler.shape, R.shape)
        (1000, 3) (1000, 3, 3)
    """

    # Original NumPy path (for debugging/validation)
    from .inertia import inertia_tensor, principal_axes

    n_frames = len(positions)
    euler_angles = np.zeros((n_frames, 3))
    rotation_matrix = np.zeros((n_frames, 3, 3))
    prev_axes = None

    for i in range(n_frames):
        pos = positions[i]

        # Compute inertia tensor and principal axes
        I = inertia_tensor(pos, masses)
        moments, axes = principal_axes(I)

        # Ensure consistent sign convention across frames
        if prev_axes is not None:
            # Flip axes if they point in opposite direction from previous frame
            for j in range(3):
                if np.dot(axes[:, j], prev_axes[:, j]) < 0:
                    axes[:, j] *= -1

        prev_axes = axes.copy()

        # Rotation matrix from lab frame to body frame
        R = axes.T  # Principal axes as columns → rotation matrix

        # If reference frame provided, compute relative rotation
        if reference_frame is not None:
            R = R @ reference_frame.T

        # Convert to Euler angles
        phi, theta, psi = rotation_matrix_to_euler_zyz(R)
        euler_angles[i] = [phi, theta, psi]
        rotation_matrix[i] = R

    return euler_angles, rotation_matrix


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
    angle = 2 * np.arccos(np.clip(q[0], -1.0, 1.0))

    return angle


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


def orientation_autocorrelation(
    euler_angles: np.ndarray, max_lag: Optional[int] = None
) -> np.ndarray:
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


if __name__ == "__main__":
    # Example usage
    print("Protein Orientation Module")
    print("==========================")
    print()
    print("Example usage:")
    print()
    print(
        "from protein_orientation.core.orientation import extract_orientation_trajectory"
    )
    print("from protein_orientation.core.trajectory import load_trajectory")
    print()
    print("# Load trajectory")
    print("traj = load_trajectory('system.gro', 'traj.trr')")
    print()
    print("# Extract Euler angles")
    print("euler = extract_orientation_trajectory(traj.positions, traj.masses)")
    print()
    print("# Analyze orientation dynamics")
    print("phi, theta, psi = euler.T")
    print("print(f'Mean nutation angle: {np.mean(theta):.2f} rad')")
