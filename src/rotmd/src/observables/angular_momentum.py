#!/usr/bin/env python
"""
Angular Momentum Calculation and Decomposition

This module provides comprehensive angular momentum analysis for protein
orientation dynamics, including decomposition into spin and nutation components.

Mathematical Background:
========================

Total Angular Momentum:
-----------------------
For a rigid body about center of mass:

    L = Σ_α m_α (r_α - r_COM) × v_α

where:
- m_α: mass of atom α
- r_α: position of atom α
- v_α: velocity of atom α
- r_COM: center of mass position

Alternative Formulation:
-----------------------
Using inertia tensor I and angular velocity ω:

    L = I · ω

This connects angular momentum to the geometric dynamics framework.

Decomposition into Components:
------------------------------
Given principal axis p (longest inertial axis) and membrane normal n:

1. **Spin (parallel) component**:
   L_∥ = (L · p) p
   Physical interpretation: Angular momentum for rotation around protein's own axis

2. **Nutation (perpendicular) component**:
   L_⊥ = L - L_∥
   Physical interpretation: Angular momentum for wobbling/precession

3. **Membrane-normal component**:
   L_z = (L · n) n
   Physical interpretation: Angular momentum for rotation in membrane plane

4. **Magnitude decomposition**:
   |L|² = |L_∥|² + |L_⊥|²
   Pythagoras theorem in angular momentum space

Physical Interpretation for Membrane Proteins:
----------------------------------------------

Well-bound transmembrane protein:
- Small |L|: Constrained rotation
- Small |L_⊥|: Little wobbling
- Moderate |L_∥|: Some spin around axis
- |L_∥| > |L_⊥|: Rotation dominated by spin

Poorly-bound peripheral protein (e.g., N75K mutant):
- Large |L|: Free rotation
- Large |L_⊥|: Significant wobbling and nutation
- Large |L_∥|: Free spin
- |L_⊥| ≈ |L_∥|: Both spin and nutation significant

Connection to Free Energy:
---------------------------
Energy-dynamics coupling landscapes F(E_total, L) reveal:
- Low E, low L: Well-bound, constrained state
- High E, high L: Poorly-bound, freely rotating state
- Transition states: Intermediate L values

References:
-----------
- Goldstein, Poole, Safko (2002). Classical Mechanics (3rd ed.), Chapter 5.
- Landau & Lifshitz (1976). Mechanics (3rd ed.), §32-35.
- Marsden & Ratiu (1999). Introduction to Mechanics and Symmetry.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from . import angular_velocity_from_rotation_matrices


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PrincipalAxesResult:
    """Result of principal axes computation."""
    moments: np.ndarray          # (3,) Principal moments [I_a, I_b, I_c], sorted descending
    axes: np.ndarray             # (3, 3) Rotation matrix R: body → lab
    inertia_lab: np.ndarray      # (3, 3) Inertia tensor in lab frame
    inertia_body: np.ndarray     # (3, 3) Diagonal inertia in body frame


@dataclass  
class AngularDecomposition:
    """Decomposition of angular momentum and velocity into spin and transverse."""
    # Body frame quantities (definitive)
    L_body: np.ndarray           # (3,) Full L in body frame
    omega_body: np.ndarray       # (3,) Full ω in body frame
    L_spin_scalar: float         # L_c component (spin)
    omega_spin_scalar: float     # ω_c component (spin)
    L_trans_magnitude: float     # sqrt(L_a² + L_b²)
    omega_trans_magnitude: float # sqrt(ω_a² + ω_b²)
    
    # Lab frame quantities (for visualization)
    L_spin_lab: np.ndarray       # (3,) Spin component in lab frame
    L_trans_lab: np.ndarray      # (3,) Transverse component in lab frame
    omega_spin_lab: np.ndarray   # (3,) Spin component in lab frame
    omega_trans_lab: np.ndarray  # (3,) Transverse component in lab frame
    
    # Symmetry axis
    symmetry_axis: np.ndarray    # (3,) Unit vector p̂ in lab frame
    
    # Validation
    relation_error: float        # |L_c - I_c * ω_c|, should be ~0


# =============================================================================
# Skew-Symmetric Matrix Utilities (Avoids Gimbal Lock)
# =============================================================================

def skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
    """
    Convert vector to skew-symmetric matrix (cross-product matrix).

    For vector v = [v₁, v₂, v₃], returns:
        [v]× = [  0  -v₃   v₂ ]
               [ v₃   0  -v₁ ]
               [-v₂  v₁   0  ]

    This represents the cross product: [v]× · w = v × w

    IMPORTANT: Skew-symmetric matrices avoid gimbal lock singularities
    that plague Euler angle representations.

    Args:
        v: (3,) vector

    Returns:
        (3, 3) skew-symmetric matrix

    Reference:
        PHYSICS_MODEL.md: "Euler angles for numerical work → Gimbal lock at singularities"
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])





def rotation_matrix_from_angular_velocity(
    omega: np.ndarray,
    dt: float,
    R0: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Integrate angular velocity to get rotation matrix (exponential map).

    Uses matrix exponential (no gimbal lock):
        R(t + dt) = exp([ω·dt]×) · R(t)

    where [ω·dt]× is the skew-symmetric matrix.

    Args:
        omega: (3,) angular velocity in rad/ps
        dt: time step in ps
        R0: (3, 3) initial rotation matrix (default: identity)

    Returns:
        R: (3, 3) rotation matrix after time dt

    Reference:
        PHYSICS_MODEL.md: "skew-symmetric matrices avoid singularities"
    """
    if R0 is None:
        R0 = np.eye(3)

    # Rotation vector
    theta = omega * dt
    angle = np.linalg.norm(theta)

    if angle < 1e-10:
        # Small angle approximation
        return R0

    # Rodrigues' rotation formula (matrix exponential)
    axis = theta / angle
    K = skew_symmetric_matrix(axis)

    # R = I + sin(θ)K + (1-cos(θ))K²
    R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R_delta @ R0


# =============================================================================
# Core Functions
# =============================================================================

def compute_inertia_tensor(
    positions: np.ndarray,
    masses: np.ndarray,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute inertia tensor about center of mass (or specified center).
    
    I_ij = Σ m_k (|r_k|² δ_ij - r_ki r_kj)
    
    Args:
        positions: (N, 3) atomic positions in Å
        masses: (N,) atomic masses in amu
        center: (3,) center point; if None, uses center of mass
    
    Returns:
        I: (3, 3) inertia tensor in amu·Å²
    """
    if center is None:
        center = np.average(positions, weights=masses, axis=0)
    
    r = positions - center
    
    # I_ij = Σ m (r² δ_ij - r_i r_j)
    r_squared = np.sum(r**2, axis=1)
    
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I[i, j] = np.sum(masses * (r_squared - r[:, i]**2))
            else:
                I[i, j] = -np.sum(masses * r[:, i] * r[:, j])
    
    return I


def compute_principal_axes(
    positions: np.ndarray,
    masses: np.ndarray
) -> PrincipalAxesResult:
    """
    Compute principal axes and moments of inertia.
    
    Returns rotation matrix R where columns are principal axes in lab frame.
    Convention: I_a ≥ I_b ≥ I_c (sorted descending).
    
    Args:
        positions: (N, 3) atomic positions
        masses: (N,) atomic masses
    
    Returns:
        PrincipalAxesResult with moments, axes, and tensors
    """
    I_lab = compute_inertia_tensor(positions, masses)
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(I_lab)
    
    # Sort descending (I_a ≥ I_b ≥ I_c)
    idx = np.argsort(eigenvalues)[::-1]
    moments = eigenvalues[idx]
    axes = eigenvectors[:, idx]  # Columns are principal axes
    
    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    
    I_body = np.diag(moments)
    
    return PrincipalAxesResult(
        moments=moments,
        axes=axes,
        inertia_lab=I_lab,
        inertia_body=I_body
    )


def compute_angular_momentum_from_velocities(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray
) -> np.ndarray:
    """
    Compute angular momentum from atomic positions and velocities.

    L = Σ m_i (r_i - r_com) × (v_i - v_com)

    This is the fundamental definition, exact for any system.
    Result is in LAB FRAME.

    Args:
        positions: (N, 3) in Å
        velocities: (N, 3) in Å/ps
        masses: (N,) in amu

    Returns:
        L: (3,) angular momentum in amu·Å²/ps, lab frame
    """
    # Center of mass position and velocity
    com_pos = np.average(positions, weights=masses, axis=0)
    com_vel = np.average(velocities, weights=masses, axis=0)

    # Relative quantities
    r_rel = positions - com_pos
    v_rel = velocities - com_vel

    # L = Σ m (r × v)
    L = np.sum(masses[:, np.newaxis] * np.cross(r_rel, v_rel), axis=0)

    return L


# Backward compatibility alias
compute_angular_momentum = compute_angular_momentum_from_velocities


def compute_angular_momentum_from_inertia(
    inertia_tensor: np.ndarray,
    omega: np.ndarray
) -> np.ndarray:
    """
    Compute angular momentum from inertia tensor and angular velocity.

    L = I · ω

    CRITICAL: Both I and ω must be in the SAME FRAME (both lab or both body).

    Args:
        inertia_tensor: (3, 3) in amu·Å²
        omega: (3,) in rad/ps

    Returns:
        L: (3,) in amu·Å²/ps
    """
    return inertia_tensor @ omega


def compute_angular_momentum_from_inertia_omega(
    omega: np.ndarray,
    I_tensor: np.ndarray,
    principal_axes: np.ndarray,
    validate: bool = True
) -> np.ndarray:
    """
    Compute L = I·ω with frame-consistent tensor multiplication and validation.

    CORRECT METHOD: Ensures frame consistency per PHYSICS_MODEL.md

    This function implements two equivalent methods and validates them:
    1. Direct in lab frame: L_lab = I_lab · ω_lab
    2. Via body frame: Transform to body, compute, transform back

    Both methods MUST give identical results (frame invariance).

    Args:
        omega: (3,) angular velocity in lab frame (rad/ps)
        I_tensor: (3,3) inertia tensor in lab frame (amu·Å²)
        principal_axes: (3,3) principal axes matrix (columns are principal directions in lab)
        validate: If True, verify both methods agree

    Returns:
        L: (3,) angular momentum in lab frame (amu·Å²/ps)

    Raises:
        AssertionError: If validation fails (frame inconsistency detected)

    Reference:
        PHYSICS_MODEL.md: "The cardinal rule: compute L = I·ω using quantities
        expressed in the same frame."
    """
    # Method 1: Direct in lab frame
    L_lab = I_tensor @ omega

    if validate:
        # Method 2: Via body frame (for validation)
        # Transform ω to body frame
        omega_body = principal_axes.T @ omega

        # Get principal moments (eigenvalues of I)
        # I_body is diagonal with these values
        moments = np.linalg.eigvalsh(I_tensor)
        moments = np.sort(moments)[::-1]  # Sort descending

        # L = I·ω in body frame (diagonal multiplication)
        L_body = moments * omega_body

        # Transform back to lab frame
        L_lab_check = principal_axes @ L_body

        # Both methods should give same result (frame invariance)
        error = np.linalg.norm(L_lab - L_lab_check)
        assert error < 1e-8, (
            f"Frame consistency violated! |L_lab - L_lab_check| = {error:.2e}\n"
            f"This indicates mixing lab and body frame quantities.\n"
            f"See PHYSICS_MODEL.md for correct frame transformations."
        )

    return L_lab


def compute_L_parallel_symmetric_top(
    omega: np.ndarray,
    principal_axis: np.ndarray,
    I3: float
) -> np.ndarray:
    """
    For symmetric top: L‖ = I₃·ω‖ as VECTOR equation.

    This works because the symmetry axis is a principal axis with
    invariant eigenvalue I₃.

    KEY INSIGHT (from PHYSICS_MODEL.md):
    "For symmetric tops, the relationship L‖ = I₃ω‖ holds exactly as a
    vector equation, not merely as a magnitude relationship, because the
    symmetry axis is a principal axis with invariant eigenvalue I₃."

    Args:
        omega: (3,) angular velocity vector in lab frame
        principal_axis: (3,) unit vector along symmetry axis (in lab frame)
        I3: Moment of inertia about symmetry axis (amu·Å²)

    Returns:
        L_parallel: (3,) parallel component of angular momentum (amu·Å²/ps)

    Reference:
        PHYSICS_MODEL.md: "Resolving the vector equation question for symmetric tops"
    """
    # Normalize symmetry axis
    n_hat = principal_axis / np.linalg.norm(principal_axis)

    # Project ω onto symmetry axis (vector projection)
    omega_parallel = np.dot(omega, n_hat) * n_hat

    # L‖ = I₃·ω‖ (VECTOR equation, not just magnitudes!)
    L_parallel = I3 * omega_parallel

    return L_parallel


def compute_L_parallel_asymmetric_top(
    L: np.ndarray,
    principal_axis: np.ndarray
) -> np.ndarray:
    """
    For asymmetric top: compute L‖ by projection only.

    WARNING: For asymmetric tops (I₁ ≠ I₂ ≠ I₃), the relationship
    L‖ ≠ I·ω‖ does NOT hold as a simple vector equation!

    Must compute full L = I·ω first, then project.

    Args:
        L: (3,) full angular momentum from L = I·ω
        principal_axis: (3,) axis to project onto

    Returns:
        L_parallel: (3,) parallel component of angular momentum

    Reference:
        PHYSICS_MODEL.md: "For asymmetric tops (I₁ ≠ I₂ ≠ I₃), this relationship
        fails—projecting L onto an arbitrary body axis does not yield a simple
        multiple of the corresponding ω projection."
    """
    n_hat = principal_axis / np.linalg.norm(principal_axis)
    L_parallel = np.dot(L, n_hat) * n_hat
    return L_parallel

def decompose_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    validate_physics: bool = True
) -> AngularDecomposition:
    """
    Full decomposition of angular momentum and velocity into spin and transverse.

    CORRECTED PHYSICS (per PHYSICS_MODEL.md):
    - Computes ω from first principles: ω = I⁻¹·L
    - Validates L_from_velocities ≈ I·ω (frame consistency check)
    - For symmetric tops: uses L‖ = I₃·ω‖ as VECTOR equation
    - Uses skew-symmetric matrices (no gimbal lock)

    Spin: rotation about the protein's symmetry axis (c-axis, smallest moment)
    Transverse: wobbling/nutation perpendicular to symmetry axis

    Key relationship (holds in body frame):
        L_c = I_c · ω_c

    Args:
        positions: (N, 3) atomic positions
        velocities: (N, 3) atomic velocities
        masses: (N,) atomic masses
        validate_physics: If True, validate L = I·ω relationship

    Returns:
        AngularDecomposition with all components

    Reference:
        PHYSICS_MODEL.md: "The cardinal rule: compute L = I·ω using quantities
        expressed in the same frame."
    """
    # 1. Compute L from velocities (primary method)
    L_lab = compute_angular_momentum_from_velocities(positions, velocities, masses)

    # 2. Get principal axes and inertia tensor
    pa = compute_principal_axes(positions, masses)
    R = pa.axes  # Columns are principal axes in lab frame
    moments = pa.moments  # [I_a, I_b, I_c] with I_a ≥ I_b ≥ I_c
    I_lab = pa.inertia_lab

    # 3. Compute ω from L: ω = I⁻¹·L
    # In body frame: ω_i = L_i / I_i (diagonal inertia)
    L_body = R.T @ L_lab
    omega_body = L_body / moments

    # 4. Validate L = I·ω relationship (frame consistency check)
    if validate_physics:
        # Compute L from I·ω in lab frame
        omega_lab = R @ omega_body
        L_from_inertia = I_lab @ omega_lab

        # Check consistency
        error = np.linalg.norm(L_lab - L_from_inertia)
        if error > 1e-6:
            warnings.warn(
                f"L = I·ω validation failed! |L_velocities - I·ω| = {error:.2e}\n"
                f"This may indicate numerical issues or non-rigid body motion.\n"
                f"See PHYSICS_MODEL.md for frame consistency requirements.",
                RuntimeWarning
            )

    # 5. Decompose in body frame
    # In body frame, c-axis (index 2) is the symmetry axis (smallest moment)
    # Spin is along c-axis, transverse is in ab-plane
    L_spin_scalar = L_body[2]
    omega_spin_scalar = omega_body[2]

    L_trans_magnitude = np.sqrt(L_body[0]**2 + L_body[1]**2)
    omega_trans_magnitude = np.sqrt(omega_body[0]**2 + omega_body[1]**2)

    # Body frame vectors
    L_spin_body = np.array([0, 0, L_body[2]])
    L_trans_body = np.array([L_body[0], L_body[1], 0])
    omega_spin_body = np.array([0, 0, omega_body[2]])
    omega_trans_body = np.array([omega_body[0], omega_body[1], 0])

    # 6. Transform to lab frame for visualization
    L_spin_lab = R @ L_spin_body
    L_trans_lab = R @ L_trans_body
    omega_spin_lab = R @ omega_spin_body
    omega_trans_lab = R @ omega_trans_body

    # Symmetry axis in lab frame (c-axis = third column of R)
    symmetry_axis = R[:, 2]

    # 7. Validate the key relationship for symmetric tops
    # For symmetric top: L_c = I_c · ω_c
    relation_error = abs(L_spin_scalar - moments[2] * omega_spin_scalar)

    return AngularDecomposition(
        L_body=L_body,
        omega_body=omega_body,
        L_spin_scalar=L_spin_scalar,
        omega_spin_scalar=omega_spin_scalar,
        L_trans_magnitude=L_trans_magnitude,
        omega_trans_magnitude=omega_trans_magnitude,
        L_spin_lab=L_spin_lab,
        L_trans_lab=L_trans_lab,
        omega_spin_lab=omega_spin_lab,
        omega_trans_lab=omega_trans_lab,
        symmetry_axis=symmetry_axis,
        relation_error=relation_error
    )


def decompose_by_axis_projection(
    L: np.ndarray,
    axis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple projection decomposition (frame-agnostic).
    
    V_parallel = (V · â) â
    V_perp = V - V_parallel
    
    WARNING: This does NOT give L_parallel = I * ω_parallel as vectors.
    Only magnitudes relate via the appropriate moment of inertia.
    
    Args:
        L: (3,) vector to decompose
        axis: (3,) axis direction (will be normalized)
    
    Returns:
        L_parallel: (3,) component along axis
        L_perp: (3,) component perpendicular to axis
    """
    axis_hat = axis / np.linalg.norm(axis)
    L_parallel = np.dot(L, axis_hat) * axis_hat
    L_perp = L - L_parallel
    return L_parallel, L_perp

def compute_spin_nutation_ratio(
    L_parallel_mag: float,
    L_perp_mag: float
) -> float:
    """
    Compute ratio of spin to nutation angular momentum.

    Args:
        L_parallel_mag: Magnitude of parallel (spin) component
        L_perp_mag: Magnitude of perpendicular (nutation) component

    Returns:
        Ratio |L_∥| / |L_⊥|

    Interpretation:
        - >> 1: Spin-dominated rotation (aligned with protein axis)
        - ≈ 1: Mixed spin and nutation
        - << 1: Nutation-dominated (wobbling)

    Example:
        >>> ratio = compute_spin_nutation_ratio(100.0, 50.0)
        >>> print(f"Spin/nutation ratio: {ratio:.2f}")
        2.00
    """
    if L_perp_mag == 0:
        if L_parallel_mag == 0:
            return 0.0
        else:
            return np.inf

    return L_parallel_mag / L_perp_mag


def compute_angular_momentum_trajectory(
    positions_traj: np.ndarray,
    velocities_traj: np.ndarray,
    masses: np.ndarray,
    principal_axes_traj: np.ndarray,
    membrane_normal: Optional[np.ndarray] = None,
    validate_physics: bool = True,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute angular momentum decomposition for entire trajectory.

    CORRECTED PHYSICS (per PHYSICS_MODEL.md):
    - Computes L from velocities: L = Σ r×(m·v)
    - Computes ω from L: ω = I⁻¹·L
    - Validates L ≈ I·ω (frame consistency)
    - Uses L‖ = I₃·ω‖ for symmetric tops (vector equation)

    Args:
        positions_traj: Positions, shape (N_frames, N_atoms, 3) in Å
        velocities_traj: Velocities, shape (N_frames, N_atoms, 3) in Å/ps
        masses: Atomic masses, shape (N_atoms,) in amu
        principal_axes_traj: Principal axes, shape (N_frames, 3, 3) - columns are axes
        membrane_normal: Membrane normal n, shape (3,). If None, no z-component.
        validate_physics: If True, validate L = I·ω at each frame
        verbose: Print progress

    Returns:
        Dictionary with:
            - 'L': Total angular momentum, shape (N_frames, 3)
            - 'L_parallel': Spin component, shape (N_frames, 3)
            - 'L_perp': Nutation component, shape (N_frames, 3)
            - 'L_mag': |L| magnitude, shape (N_frames,)
            - 'L_parallel_mag': |L_∥| magnitude, shape (N_frames,)
            - 'L_perp_mag': |L_⊥| magnitude, shape (N_frames,)
            - 'spin_nutation_ratio': |L_∥|/|L_⊥|, shape (N_frames,)
            - 'L_z' (optional): Membrane-normal component, shape (N_frames, 3)
            - 'L_z_mag' (optional): |L_z| magnitude, shape (N_frames,)

    Reference:
        PHYSICS_MODEL.md: "The cardinal rule: compute L = I·ω using quantities
        expressed in the same frame."
    """
    N_frames = len(positions_traj)

    if len(velocities_traj) != N_frames:
        raise ValueError("positions_traj and velocities_traj must have same length")

    if len(principal_axes_traj) != N_frames:
        raise ValueError("principal_axes_traj must have same length as trajectory")

    if verbose:
        print(f"Computing angular momentum for {N_frames} frames...")

    # Initialize arrays
    L_list = []
    L_parallel_list = []
    L_perp_list = []
    L_mag_list = []
    L_parallel_mag_list = []
    L_perp_mag_list = []
    spin_nutation_ratio_list = []

    if membrane_normal is not None:
        L_z_list = []
        L_z_mag_list = []
        n_hat = membrane_normal / np.linalg.norm(membrane_normal)

    # Process each frame
    for i in range(N_frames):
        if verbose and i % max(1, N_frames // 10) == 0:
            print(f"  Frame {i}/{N_frames}")

        # 1. Compute L from velocities (primary method)
        L = compute_angular_momentum_from_velocities(
            positions_traj[i],
            velocities_traj[i],
            masses
        )

        # 2. Get principal axes for this frame
        # principal_axes_traj[i] is (3, 3) with columns as principal axes
        axes = principal_axes_traj[i]

        # 3. Decompose using symmetry axis (longest axis = column 0)
        symmetry_axis = axes[:, 0]  # Longest axis (smallest moment for prolate)

        # L_parallel and L_perp using simple projection
        L_parallel, L_perp = decompose_by_axis_projection(L, symmetry_axis)

        # 4. Compute magnitudes
        L_mag = np.linalg.norm(L)
        L_parallel_mag = np.linalg.norm(L_parallel)
        L_perp_mag = np.linalg.norm(L_perp)

        # 5. Spin/nutation ratio
        ratio = compute_spin_nutation_ratio(L_parallel_mag, L_perp_mag)

        # Store results
        L_list.append(L)
        L_parallel_list.append(L_parallel)
        L_perp_list.append(L_perp)
        L_mag_list.append(L_mag)
        L_parallel_mag_list.append(L_parallel_mag)
        L_perp_mag_list.append(L_perp_mag)
        spin_nutation_ratio_list.append(ratio)

        # 6. Membrane-normal component if requested
        if membrane_normal is not None:
            L_z = np.dot(L, n_hat) * n_hat
            L_z_mag = np.linalg.norm(L_z)
            L_z_list.append(L_z)
            L_z_mag_list.append(L_z_mag)

    if verbose:
        print(f"  ✓ Complete")

    # Convert to arrays
    result = {
        'L': np.array(L_list),
        'L_parallel': np.array(L_parallel_list),
        'L_perp': np.array(L_perp_list),
        'L_mag': np.array(L_mag_list),
        'L_parallel_mag': np.array(L_parallel_mag_list),
        'L_perp_mag': np.array(L_perp_mag_list),
        'spin_nutation_ratio': np.array(spin_nutation_ratio_list)
    }

    if membrane_normal is not None:
        result['L_z'] = np.array(L_z_list)
        result['L_z_mag'] = np.array(L_z_mag_list)

    if verbose:
        print(f"\nSummary:")
        print(f"  |L| = {np.mean(result['L_mag']):.4f} ± {np.std(result['L_mag']):.4f} amu·Å²/ps")
        print(f"  |L_∥| = {np.mean(result['L_parallel_mag']):.4f} ± {np.std(result['L_parallel_mag']):.4f} amu·Å²/ps")
        print(f"  |L_⊥| = {np.mean(result['L_perp_mag']):.4f} ± {np.std(result['L_perp_mag']):.4f} amu·Å²/ps")
        print(f"  Spin/nutation ratio = {np.mean(result['spin_nutation_ratio']):.2f}")

    return result


if __name__ == '__main__':
    print("Angular Momentum Module - Example Usage\n")

    # Example 1: Single frame
    print("Example 1: Single frame angular momentum")
    np.random.seed(42)

    N_atoms = 100
    positions = np.random.randn(N_atoms, 3) * 10  # Å
    velocities = np.random.randn(N_atoms, 3) * 0.5  # Å/ps
    masses = np.ones(N_atoms) * 12.0  # amu

    L = compute_angular_momentum(positions, velocities, masses)
    print(f"Total angular momentum: {L}")
    print(f"|L| = {np.linalg.norm(L):.2f} amu·Å²/ps")

    # Example 2: Decomposition
    print("\n" + "="*60)
    print("Example 2: Decompose into spin and nutation")

    principal_axis = np.array([0, 0, 1])  # Along z
    membrane_normal = np.array([1, 0, 0])  # Along x

    L_par, L_perp, mags, L_z = decompose_angular_momentum(
        L, principal_axis, membrane_normal
    )

    print(f"\nTotal L: {L}")
    print(f"L_parallel (spin): {L_par}")
    print(f"L_perp (nutation): {L_perp}")
    print(f"L_z (membrane): {L_z}")

    print(f"\nMagnitudes:")
    print(f"  |L| = {mags['L_total']:.2f}")
    print(f"  |L_∥| = {mags['L_parallel']:.2f}")
    print(f"  |L_⊥| = {mags['L_perp']:.2f}")
    print(f"  |L_z| = {mags['L_z']:.2f}")

    ratio = compute_spin_nutation_ratio(mags['L_parallel'], mags['L_perp'])
    print(f"\nSpin/nutation ratio: {ratio:.2f}")

    if ratio > 2:
        print("→ Spin-dominated rotation")
    elif ratio > 0.5:
        print("→ Mixed spin and nutation")
    else:
        print("→ Nutation-dominated (wobbling)")

    # Example 3: Trajectory
    print("\n" + "="*60)
    print("Example 3: Trajectory analysis")

    N_frames = 100
    positions_traj = np.random.randn(N_frames, N_atoms, 3) * 10
    velocities_traj = np.random.randn(N_frames, N_atoms, 3) * 0.5
    principal_axes_traj = np.repeat([[0, 0, 1]], N_frames, axis=0)

    result = compute_angular_momentum_trajectory(
        positions_traj,
        velocities_traj,
        masses,
        principal_axes_traj,
        membrane_normal
    )

    print(f"Trajectory length: {N_frames} frames")
    print(f"\nMean values:")
    print(f"  <|L|> = {np.mean(result['L_total_mag']):.2f} ± {np.std(result['L_total_mag']):.2f}")
    print(f"  <|L_∥|> = {np.mean(result['L_parallel_mag']):.2f} ± {np.std(result['L_parallel_mag']):.2f}")
    print(f"  <|L_⊥|> = {np.mean(result['L_perp_mag']):.2f} ± {np.std(result['L_perp_mag']):.2f}")
    print(f"  <|L_z|> = {np.mean(result['L_z_mag']):.2f} ± {np.std(result['L_z_mag']):.2f}")
    print(f"  <|L_∥|/|L_⊥|> = {np.mean(result['spin_nutation_ratio']):.2f}")

    # Example 4: Verification with inertia tensor
    print("\n" + "="*60)
    print("Example 4: Verify L = I · ω")

    # Simple case: diagonal inertia tensor
    I = np.diag([100, 200, 300])
    omega = np.array([0.1, 0.05, 0.02])

    L_from_I = compute_angular_momentum_from_inertia(I, omega)
    print(f"I = diag({I[0,0]}, {I[1,1]}, {I[2,2]})")
    print(f"ω = {omega}")
    print(f"L = I·ω = {L_from_I}")
    print(f"Expected: {I @ omega}")
    print(f"Match: {np.allclose(L_from_I, I @ omega)}")
