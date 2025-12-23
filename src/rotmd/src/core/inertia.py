#!/usr/bin/env python
"""
Inertia Tensor Calculations for Rigid Body Dynamics

This module provides utilities for computing inertia tensors, principal axes,
and principal moments of inertia for protein systems. These are fundamental
quantities for analyzing rotational dynamics on SO(3).

Mathematical Background:
========================

The inertia tensor I ∈ ℝ³ˣ³ is a symmetric positive-definite matrix that
characterizes the mass distribution of a rigid body:

    I_ij = Σ_α m_α [(r_α · r_α) δ_ij - r_α,i r_α,j]

where:
- m_α: mass of atom α
- r_α: position of atom α relative to center of mass
- δ_ij: Kronecker delta

Principal Axes Decomposition:
-----------------------------
Since I is real and symmetric, it can be diagonalized:

    I = R · diag(I_a, I_b, I_c) · R^T

where:
- R ∈ SO(3): rotation matrix (principal axes as columns)
- I_a ≤ I_b ≤ I_c: principal moments of inertia

Physical Interpretation:
------------------------
The principal axes define the body frame where:
- I is diagonal (no products of inertia)
- Angular momentum L = I · ω simplifies to component-wise multiplication
- Rotational kinetic energy T = (1/2) ω^T · I · ω simplifies

Protein Classification by Shape:
---------------------------------
Based on principal moments (I_a, I_b, I_c):

1. Spherical top: I_a = I_b = I_c (e.g., spherical proteins)
2. Symmetric top: I_a = I_b ≠ I_c or I_a ≠ I_b = I_c (e.g., α-helical bundles)
3. Asymmetric top: I_a ≠ I_b ≠ I_c (most proteins)

References:
-----------
- Goldstein, Poole, Safko (2002). Classical Mechanics (3rd ed.), Chapter 5.
- Landau & Lifshitz (1976). Mechanics (3rd ed.), §32.
"""

import numpy as np
from typing import Tuple, Optional


def inertia_tensor(
    positions: np.ndarray,
    masses: np.ndarray,
    center_of_mass: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the 3×3 inertia tensor for a collection of point masses.

    Args:
        positions: Atomic positions, shape (N, 3) in Å
        masses: Atomic masses, shape (N,) in amu
        center_of_mass: Center of mass position (3,). If None, will be computed.

    Returns:
        Inertia tensor I, shape (3, 3) in amu·Å²

    Mathematical Formula:
        I_ij = Σ_α m_α [(r_α · r_α) δ_ij - r_α,i r_α,j]

    Notes:
        - Positions are shifted to center of mass frame
        - Result is symmetric: I^T = I
        - Result is positive semi-definite: x^T I x ≥ 0 for all x
        - Diagonal elements: I_xx = Σ m(y² + z²), etc.
        - Off-diagonal elements: I_xy = -Σ m·x·y, etc.

    Example:
        >>> positions = np.random.randn(100, 3)  # 100 atoms
        >>> masses = np.ones(100)  # Equal masses
        >>> I = inertia_tensor(positions, masses)
        >>> print(I.shape)
        (3, 3)
        >>> assert np.allclose(I, I.T)  # Symmetric
    """
    if positions.shape[0] != len(masses):
        raise ValueError(
            f"Number of positions ({positions.shape[0]}) must match "
            f"number of masses ({len(masses)})"
        )

    if positions.shape[1] != 3:
        raise ValueError(f"Positions must have shape (N, 3), got {positions.shape}")

    # Compute center of mass if not provided
    if center_of_mass is None:
        center_of_mass = np.average(positions, weights=masses, axis=0)

    # Shift to COM frame
    r = positions - center_of_mass  # Shape (N, 3)

    # Compute r² = x² + y² + z² for each atom
    r_squared = np.sum(r**2, axis=1)  # Shape (N,)

    # Initialize inertia tensor
    I = np.zeros((3, 3))

    # Diagonal elements: I_ii = Σ m_α (r_α² - r_α,i²)
    for i in range(3):
        I[i, i] = np.sum(masses * (r_squared - r[:, i]**2))

    # Off-diagonal elements: I_ij = -Σ m_α r_α,i r_α,j
    I[0, 1] = I[1, 0] = -np.sum(masses * r[:, 0] * r[:, 1])
    I[0, 2] = I[2, 0] = -np.sum(masses * r[:, 0] * r[:, 2])
    I[1, 2] = I[2, 1] = -np.sum(masses * r[:, 1] * r[:, 2])

    return I


def principal_axes(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize inertia tensor to find principal axes and moments.

    Args:
        I: Inertia tensor, shape (3, 3) in amu·Å²

    Returns:
        Tuple of:
            - moments: Principal moments (I_a, I_b, I_c) in amu·Å², sorted ascending
            - axes: Rotation matrix R ∈ SO(3) with principal axes as columns

    Mathematical Details:
        Solves the eigenvalue problem:
            I · v_i = I_i · v_i

        Returns:
            - moments: eigenvalues sorted I_a ≤ I_b ≤ I_c
            - axes: R = [v_a | v_b | v_c] where columns are eigenvectors

    Physical Interpretation:
        - v_a: axis of minimum moment (easiest rotation)
        - v_c: axis of maximum moment (hardest rotation)
        - For proteins: typically v_c is longest axis

    Properties:
        - det(R) = +1 (proper rotation, no reflections)
        - R^T R = I (orthonormal)
        - I_body = R^T · I · R = diag(I_a, I_b, I_c)

    Example:
        >>> I = np.diag([100, 200, 300])  # Already diagonal
        >>> moments, axes = principal_axes(I)
        >>> print(moments)
        [100. 200. 300.]
        >>> print(axes)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    if I.shape != (3, 3):
        raise ValueError(f"Inertia tensor must be 3×3, got {I.shape}")

    if not np.allclose(I, I.T):
        raise ValueError("Inertia tensor must be symmetric")

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Sort by eigenvalues (ascending: I_a ≤ I_b ≤ I_c)
    sort_indices = np.argsort(eigenvalues)
    moments = eigenvalues[sort_indices]
    axes = eigenvectors[:, sort_indices]

    # Ensure proper rotation (det = +1, not -1)
    if np.linalg.det(axes) < 0:
        # Flip one axis to convert reflection → rotation
        axes[:, 0] *= -1

    return moments, axes


def principal_moments(
    positions: np.ndarray,
    masses: np.ndarray,
    center_of_mass: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal moments and axes directly from positions and masses.

    This is a convenience function combining inertia_tensor() and principal_axes().

    Args:
        positions: Atomic positions, shape (N, 3) in Å
        masses: Atomic masses, shape (N,) in amu
        center_of_mass: Center of mass position (3,). If None, computed automatically.

    Returns:
        Tuple of:
            - moments: Principal moments (I_a, I_b, I_c) in amu·Å², sorted ascending
            - axes: Rotation matrix R ∈ SO(3) with principal axes as columns

    Example:
        >>> # Elongated protein along z-axis
        >>> positions = np.random.randn(100, 3)
        >>> positions[:, 2] *= 3  # Stretch along z
        >>> masses = np.ones(100)
        >>> moments, axes = principal_moments(positions, masses)
        >>> print(f"I_a = {moments[0]:.1f}")
        >>> print(f"I_b = {moments[1]:.1f}")
        >>> print(f"I_c = {moments[2]:.1f}")
        >>> # Expect I_c (longest axis) to be largest
    """
    I = inertia_tensor(positions, masses, center_of_mass)
    return principal_axes(I)


def parallel_axis_theorem(
    I_com: np.ndarray,
    total_mass: float,
    displacement: np.ndarray
) -> np.ndarray:
    """
    Apply parallel axis theorem to shift inertia tensor to a new origin.

    The parallel axis theorem states:
        I_new = I_com + M [(d · d) I_3 - d ⊗ d]

    where:
        - I_com: inertia tensor about center of mass
        - M: total mass
        - d: displacement vector from COM to new origin
        - I_3: 3×3 identity matrix

    Args:
        I_com: Inertia tensor about COM, shape (3, 3) in amu·Å²
        total_mass: Total mass in amu
        displacement: Displacement vector d, shape (3,) in Å

    Returns:
        Inertia tensor about new origin, shape (3, 3) in amu·Å²

    Example:
        >>> I_com = np.diag([100, 200, 300])
        >>> M = 1000.0
        >>> d = np.array([1.0, 0.0, 0.0])  # Shift 1 Å along x
        >>> I_new = parallel_axis_theorem(I_com, M, d)
        >>> # I_xx stays same (parallel to shift)
        >>> # I_yy, I_zz increase by M·d²
    """
    d = np.asarray(displacement)
    d_squared = np.dot(d, d)

    # Steiner term: M [(d · d) I_3 - d ⊗ d]
    steiner = total_mass * (d_squared * np.eye(3) - np.outer(d, d))

    return I_com + steiner


def is_spherical_top(
    moments: np.ndarray,
    rtol: float = 1e-2
) -> bool:
    """
    Check if protein is a spherical top (I_a = I_b = I_c).

    Args:
        moments: Principal moments (I_a, I_b, I_c)
        rtol: Relative tolerance for equality check

    Returns:
        True if all moments are equal within tolerance

    Example:
        >>> moments = np.array([100, 100, 100])
        >>> assert is_spherical_top(moments)
        >>> moments = np.array([100, 200, 300])
        >>> assert not is_spherical_top(moments)
    """
    I_mean = np.mean(moments)
    return np.allclose(moments, I_mean, rtol=rtol)


def is_symmetric_top(
    moments: np.ndarray,
    rtol: float = 1e-2
) -> bool:
    """
    Check if protein is a symmetric top (I_a = I_b ≠ I_c or I_a ≠ I_b = I_c).

    Args:
        moments: Principal moments (I_a, I_b, I_c), sorted ascending
        rtol: Relative tolerance for equality check

    Returns:
        True if two moments are equal (but not all three)

    Physical Examples:
        - Prolate (cigar-shaped): I_a = I_b < I_c (α-helix)
        - Oblate (disk-shaped): I_a < I_b = I_c (β-barrel)

    Example:
        >>> moments = np.array([100, 100, 300])  # Prolate
        >>> assert is_symmetric_top(moments)
        >>> moments = np.array([100, 200, 200])  # Oblate
        >>> assert is_symmetric_top(moments)
    """
    I_a, I_b, I_c = moments

    # Check if any two are equal (but not all three)
    ab_equal = np.isclose(I_a, I_b, rtol=rtol)
    bc_equal = np.isclose(I_b, I_c, rtol=rtol)
    ac_equal = np.isclose(I_a, I_c, rtol=rtol)

    # Symmetric if exactly one pair is equal
    return (ab_equal and not bc_equal) or (bc_equal and not ab_equal)


def is_asymmetric_top(
    moments: np.ndarray,
    rtol: float = 1e-2
) -> bool:
    """
    Check if protein is an asymmetric top (I_a ≠ I_b ≠ I_c).

    Args:
        moments: Principal moments (I_a, I_b, I_c)
        rtol: Relative tolerance for equality check

    Returns:
        True if all moments are distinct

    Note:
        Most proteins are asymmetric tops. This is the generic case.

    Example:
        >>> moments = np.array([100, 200, 300])
        >>> assert is_asymmetric_top(moments)
    """
    return not (is_spherical_top(moments, rtol) or is_symmetric_top(moments, rtol))


def asymmetry_parameter(moments: np.ndarray) -> float:
    """
    Compute Ray's asymmetry parameter κ.

    The asymmetry parameter quantifies deviation from a symmetric top:
        κ = (2I_b - I_a - I_c) / (I_c - I_a)

    Range:
        - κ = -1: prolate symmetric top (I_a = I_b < I_c)
        - κ = 0: maximally asymmetric
        - κ = +1: oblate symmetric top (I_a < I_b = I_c)

    Args:
        moments: Principal moments (I_a, I_b, I_c), sorted ascending

    Returns:
        Asymmetry parameter κ ∈ [-1, +1]

    References:
        - Ray, B. (1932). Z. Phys. 78, 74.

    Example:
        >>> moments = np.array([100, 100, 300])  # Prolate
        >>> kappa = asymmetry_parameter(moments)
        >>> assert np.isclose(kappa, -1.0)
        >>> moments = np.array([100, 200, 200])  # Oblate
        >>> kappa = asymmetry_parameter(moments)
        >>> assert np.isclose(kappa, 1.0)
    """
    I_a, I_b, I_c = moments

    if np.isclose(I_c, I_a):
        # Spherical top: κ undefined, return 0
        return 0.0

    kappa = (2*I_b - I_a - I_c) / (I_c - I_a)
    return kappa


if __name__ == '__main__':
    print("Inertia Tensor Module - Example Usage\n")

    # Example 1: Random protein-like structure
    print("Example 1: Random protein (100 atoms)")
    np.random.seed(42)
    positions = np.random.randn(100, 3) * 10  # 10 Å spread
    masses = np.ones(100) * 12.0  # Carbon-like masses

    I = inertia_tensor(positions, masses)
    print(f"Inertia tensor:\n{I}\n")

    moments, axes = principal_axes(I)
    print(f"Principal moments: I_a = {moments[0]:.1f}, I_b = {moments[1]:.1f}, I_c = {moments[2]:.1f}")
    print(f"Principal axes (columns):\n{axes}\n")

    kappa = asymmetry_parameter(moments)
    print(f"Asymmetry parameter κ = {kappa:.3f}")

    if is_spherical_top(moments):
        print("Shape: Spherical top")
    elif is_symmetric_top(moments):
        print("Shape: Symmetric top")
    else:
        print("Shape: Asymmetric top (generic)")

    # Example 2: Elongated structure (cigar-shaped)
    print("\n" + "="*60)
    print("Example 2: Elongated structure (prolate symmetric top)")
    positions_cigar = np.random.randn(100, 3)
    positions_cigar[:, 2] *= 5  # Stretch along z
    masses_cigar = np.ones(100)

    moments_cigar, _ = principal_moments(positions_cigar, masses_cigar)
    print(f"Principal moments: I_a = {moments_cigar[0]:.1f}, I_b = {moments_cigar[1]:.1f}, I_c = {moments_cigar[2]:.1f}")

    kappa_cigar = asymmetry_parameter(moments_cigar)
    print(f"Asymmetry parameter κ = {kappa_cigar:.3f} (expect ≈ -1 for prolate)")

    # Example 3: Parallel axis theorem
    print("\n" + "="*60)
    print("Example 3: Parallel axis theorem")
    I_com = np.diag([100, 200, 300])
    M = 1000.0
    d = np.array([2.0, 0.0, 0.0])

    I_shifted = parallel_axis_theorem(I_com, M, d)
    print(f"I_com diagonal: {np.diag(I_com)}")
    print(f"I_shifted diagonal: {np.diag(I_shifted)}")
    print(f"Change in I_yy and I_zz: {M * np.dot(d, d):.1f} (expected from Steiner term)")
