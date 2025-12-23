#!/usr/bin/env python
"""
Angular Velocity Calculation from Rotation Matrices

This module computes angular velocity ω from time series of rotation matrices
or quaternions. The angular velocity is a fundamental quantity in SO(3) dynamics.

Mathematical Background:
========================

For a rotation matrix R(t) ∈ SO(3), the angular velocity ω is defined via:

    dR/dt = [ω]× · R

where [ω]× ∈ so(3) is the skew-symmetric matrix:

    [ω]× = ⎡  0    -ω_z   ω_y ⎤
           ⎢  ω_z    0   -ω_x ⎥
           ⎣ -ω_y   ω_x    0  ⎦

Solving for ω:
-------------
Given R(t) and R(t+dt), we can compute:

    Ω = (dR/dt) · R^T = [ω]×

Then extract ω from the skew-symmetric matrix:
    ω_x = Ω[2,1] = -Ω[1,2]
    ω_y = Ω[0,2] = -Ω[2,0]
    ω_z = Ω[1,0] = -Ω[0,1]

Finite Difference Approximation:
--------------------------------
For discrete time series R(t_i):

    dR/dt ≈ [R(t+Δt) - R(t)] / Δt           (forward difference)
    dR/dt ≈ [R(t) - R(t-Δt)] / Δt           (backward difference)
    dR/dt ≈ [R(t+Δt) - R(t-Δt)] / (2Δt)     (central difference, recommended)

Quaternion Method:
------------------
For quaternions q = [w, x, y, z] (unit norm), angular velocity is:

    ω = 2 · q^{-1} · dq/dt

where q^{-1} = [w, -x, -y, -z] for unit quaternions.

Decomposition into Components:
-------------------------------
Given ω and principal axes p (longest axis) and n (membrane normal):

    ω_∥ = (ω · p) p         (spin around protein axis)
    ω_⊥ = ω - ω_∥           (nutation/wobbling)
    ω_z = (ω · n) n         (rotation around membrane normal)

Physical Interpretation:
------------------------
- ω_∥: Spin angular velocity (rotation about protein's own axis)
- ω_⊥: Nutation angular velocity (precession, wobbling)
- |ω|: Total angular speed
- For well-bound protein: small |ω|, constrained rotation
- For poorly-bound protein: large |ω|, free rotation

References:
-----------
- Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
  Chapter 2: Rigid Body Motion.
- Landau & Lifshitz (1976). Mechanics (3rd ed.), §32-35.
"""

from MDAnalysis.lib.transformations import rotation_from_matrix
import numpy as np
from typing import Tuple, Optional, List
import warnings
from protein_orientation import extract_orientation_trajectory
from protein_orientation.core.orientation import extract_orientation
def skew_symmetric_to_vector(Omega: np.ndarray) -> np.ndarray:
    """
    Extract angular velocity vector ω from skew-symmetric matrix [ω]×.

    Args:
        Omega: Skew-symmetric matrix [ω]×, shape (3, 3)

    Returns:
        Angular velocity vector ω, shape (3,)

    Mathematical Formula:
        ω = [Ω[2,1], Ω[0,2], Ω[1,0]]^T

    Example:
        >>> Omega = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        >>> omega = skew_symmetric_to_vector(Omega)
        >>> print(omega)
        [1. 2. 3.]
    """
    if Omega.shape != (3, 3):
        raise ValueError(f"Matrix must be 3×3, got {Omega.shape}")

    # Check skew-symmetry (Ω^T = -Ω)
    if not np.allclose(Omega, -Omega.T, atol=1e-10):
        warnings.warn(
            "Matrix is not skew-symmetric. "
            "This may indicate numerical errors in rotation matrix differentiation."
        )

    # Extract components
    omega = np.array([
        Omega[2, 1],  # ω_x = Ω[2,1]
        Omega[0, 2],  # ω_y = Ω[0,2]
        Omega[1, 0]   # ω_z = Ω[1,0]
    ])

    return omega


def vector_to_skew_symmetric(omega: np.ndarray) -> np.ndarray:
    """
    Convert angular velocity vector ω to skew-symmetric matrix [ω]×.

    Args:
        omega: Angular velocity vector, shape (3,)

    Returns:
        Skew-symmetric matrix [ω]×, shape (3, 3)

    Mathematical Formula:
        [ω]× = ⎡  0    -ω_z   ω_y ⎤
               ⎢  ω_z    0   -ω_x ⎥
               ⎣ -ω_y   ω_x    0  ⎦

    Example:
        >>> omega = np.array([1, 2, 3])
        >>> Omega = vector_to_skew_symmetric(omega)
        >>> print(Omega)
        [[ 0. -3.  2.]
         [ 3.  0. -1.]
         [-2.  1.  0.]]
    """
    if len(omega) != 3:
        raise ValueError(f"Vector must have 3 components, got {len(omega)}")

    omega_x, omega_y, omega_z = omega

    Omega = np.array([
        [0,       -omega_z,  omega_y],
        [omega_z,  0,       -omega_x],
        [-omega_y,  omega_x,  0      ]
    ])

    return Omega


def angular_velocity_from_rotation_matrices(
    R_prev: np.ndarray,
    R_curr: np.ndarray,
    R_next: np.ndarray,
    dt: float,
    method: str = 'central'
) -> np.ndarray:
    """
    Compute angular velocity ω from rotation matrix time series.

    Args:
        R_prev: Rotation matrix at t-Δt, shape (3, 3)
        R_curr: Rotation matrix at t, shape (3, 3)
        R_next: Rotation matrix at t+Δt, shape (3, 3)
        dt: Time step Δt in ps
        method: Finite difference method:
            - 'forward': (R_next - R_curr) / dt
            - 'backward': (R_curr - R_prev) / dt
            - 'central': (R_next - R_prev) / (2·dt)  [default, most accurate]

    Returns:
        Angular velocity ω in space frame, shape (3,) in rad/ps

    Mathematical Formula:
        dR/dt = [ω]× · R

        Therefore:
        [ω]× = (dR/dt) · R^T

    Notes:
        - Central difference is O(dt²) accurate vs O(dt) for forward/backward
        - Returns ω in space frame (lab frame), not body frame
        - For body frame: ω_body = R^T · ω_space

    Example:
        >>> # Small rotation around z-axis
        >>> R_prev = np.eye(3)
        >>> theta = 0.1  # radians
        >>> R_next = np.array([[np.cos(theta), -np.sin(theta), 0],
        ...                     [np.sin(theta),  np.cos(theta), 0],
        ...                     [0, 0, 1]])
        >>> dt = 0.001  # ps
        >>> omega = angular_velocity_from_rotation_matrices(
        ...     R_prev, (R_prev + R_next)/2, R_next, dt, method='central'
        ... )
        >>> # Expect ω_z ≈ theta/(2*dt) = 50 rad/ps
    """
    # Validate inputs
    for R, name in [(R_prev, 'R_prev'), (R_curr, 'R_curr'), (R_next, 'R_next')]:
        if R.shape != (3, 3):
            raise ValueError(f"{name} must be 3×3, got {R.shape}")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
            warnings.warn(f"{name} is not orthogonal (R·R^T ≠ I)")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
            warnings.warn(f"{name} has det ≠ 1 (not a proper rotation)")

    # Compute finite difference
    if method == 'forward':
        dR_dt = (R_next - R_curr) / dt
        R_ref = R_curr
    elif method == 'backward':
        dR_dt = (R_curr - R_prev) / dt
        R_ref = R_curr
    elif method == 'central':
        dR_dt = (R_next - R_prev) / (2 * dt)
        R_ref = R_curr
    else:
        raise ValueError(f"Unknown method: {method}. Use 'forward', 'backward', or 'central'")

    # Compute [ω]× = (dR/dt) · R^T
    Omega = dR_dt @ R_ref.T
    
    # Extract ω from skew-symmetric matrix
    omega_lab = skew_symmetric_to_vector(Omega)
    omega_body = R_ref.T @ omega_lab
    omega_spin_body = np.array([0, 0, omega_body[2]])
    omega_tilt_body = np.array([omega_body[0], omega_body[1], 0])
    body_frame_components = {'ω': omega_body, 'parallel': omega_spin_body, 'perpendicular': omega_tilt_body}
    return omega_lab, body_frame_components





def decompose_angular_velocity(
    omega: np.ndarray,
    principal_axis: np.ndarray,
    membrane_normal: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Decompose angular velocity into spin and nutation components.

    Args:
        omega: Angular velocity vector ω, shape (3,)
        principal_axis: Protein principal axis p (longest inertial axis), shape (3,)
        membrane_normal: Membrane normal n, shape (3,). If None, no z-component returned.

    Returns:
        Tuple of:
            - omega_parallel: Spin component ω_∥ = (ω · p) p, shape (3,)
            - omega_perp: Nutation component ω_⊥ = ω - ω_∥, shape (3,)
            - omega_z: Membrane-normal component ω_z = (ω · n) n, shape (3,) or None

    Physical Interpretation:
        - ω_∥: Spin angular velocity (rotation about protein axis)
          * Well-bound: small, protein doesn't spin freely
          * Poorly-bound: large, protein spins around its axis

        - ω_⊥: Nutation angular velocity (wobbling, precession)
          * Well-bound: very small, protein is constrained
          * Poorly-bound: large, protein wobbles and nutates

        - ω_z: Membrane-normal component
          * Measures rotation in membrane plane

    Example:
        >>> omega = np.array([1, 2, 3])  # rad/ps
        >>> p = np.array([0, 0, 1])  # Protein axis along z
        >>> n = np.array([1, 0, 0])  # Membrane normal along x
        >>> omega_par, omega_perp, omega_z = decompose_angular_velocity(
        ...     omega, p, n
        ... )
        >>> print(f"|ω_∥| = {np.linalg.norm(omega_par):.2f} rad/ps")
        >>> print(f"|ω_⊥| = {np.linalg.norm(omega_perp):.2f} rad/ps")
    """
    # Validate inputs
    if len(omega) != 3:
        raise ValueError(f"omega must have 3 components, got {len(omega)}")
    if len(principal_axis) != 3:
        raise ValueError(f"principal_axis must have 3 components")

    # Normalize principal axis
    p = np.array(principal_axis) / np.linalg.norm(principal_axis)

    # Parallel component: ω_∥ = (ω · p) p
    omega_parallel = np.dot(omega, p) * p

    # Perpendicular component: ω_⊥ = ω - ω_∥
    omega_perp = omega - omega_parallel

    # Membrane-normal component (if provided)
    if membrane_normal is not None:
        if len(membrane_normal) != 3:
            raise ValueError(f"membrane_normal must have 3 components")
        n = np.array(membrane_normal) / np.linalg.norm(membrane_normal)
        omega_z = np.dot(omega, n) * n
    else:
        omega_z = None

    return omega_parallel, omega_perp, omega_z



def compute_angular_velocity_from_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    interval: int | None=1,
    verbose: bool = True
) -> dict:
    """
    Compute angular velocity from entire trajectory using velocities.

    This is the recommended method when velocities are available in the trajectory.
    More accurate than differentiating Euler angles or rotation matrices.

    Args:
        positions: (n_frames, n_atoms, 3) positions
        velocities: (n_frames, n_atoms, 3) velocities
        masses: (n_atoms,) masses
        verbose: Print progress

    Returns:
        results: Dictionary with:
            - 'omega': (n_frames, 3) angular velocity in rad/ps
            - 'omega_spin': (n_frames, 3) spin component
            - 'omega_nutation': (n_frames, 3) nutation component
            - 'omega_magnitude': (n_frames,) total angular speed
            - 'spin_magnitude': (n_frames,) spin speed
            - 'nutation_magnitude': (n_frames,) nutation speed

    Example:
        >>> results = compute_angular_velocity_from_trajectory(
        ...     traj['positions'],
        ...     traj['velocities'],
        ...     traj['masses']
        ... )
        >>> omega = results['omega']
        >>> print(f"Mean |ω| = {np.mean(results['omega_magnitude']):.3f} rad/ps")
    """
    n_frames = len(positions)

    if verbose:
        print(f"Computing angular velocity for {n_frames} frames...")

    omega_lab_list = []
    omega_body_list = []
    omega_body_spin_list = []
    omega_body_nutation_list = []

    for i in range(1, n_frames - 1):
        if verbose and i % max(1, n_frames // 10) == 0:
            print(f"  Frame {i}/{n_frames}")
        r = positions
        v = velocities
        m = masses
        R = extract_orientation
        result, body_frame = angular_velocity_from_rotation_matrices(
            R_prev=R(r[i-1], m), R_curr=R(r[i], m), R_next=R(r[i+1],m), dt=100*interval
        )
        omega_lab_list.append(result)
        omega_body_list.append(body_frame['ω'])
        omega_body_spin_list.append(body_frame['parallel'])
        omega_body_nutation_list.append(body_frame['perpendicular'])

    omega = np.array(omega_lab_list)
    omega_body = np.array(omega_body_list)
    omega_body_spin = np.array(omega_body_spin_list)
    omega_body_nutation = np.array(omega_body_nutation_list)

    lab = {
        'omega_lab': omega,

        'omega_magnitude': np.linalg.norm(omega, axis=1)

    }
    body = {
        'omega_body': omega_body,
        'omega_body_spin': omega_body_spin,
        'omega_body_nutation': omega_body_nutation,
        'spin_magnitude': np.linalg.norm(omega_body_spin, axis=1),
        'nutation_magnitude': np.linalg.norm(omega_body_nutation, axis=1)
    }
    if verbose:
        print(f"  ✓ Complete")
        print(f"\nSummary:")
        print(f"  |ω| = {np.mean(lab['omega_magnitude']):.4f} ± {np.std(lab['omega_magnitude']):.4f} rad/ps")
        print(f"  |ω_spin| = {np.mean(body['spin_magnitude']):.4f} ± {np.std(body['spin_magnitude']):.4f} rad/ps")
        print(f"  |ω_nut| = {np.mean(body['nutation_magnitude']):.4f} ± {np.std(body['nutation_magnitude']):.4f} rad/ps")

    return lab, body 

