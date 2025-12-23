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


References:
-----------
- Goldstein, Poole, Safko (2002). Classical Mechanics (3rd ed.), Chapter 5.
- Landau & Lifshitz (1976). Mechanics (3rd ed.), §32.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional


def inertia_tensor(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    center_of_mass: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
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
        >>> positions = jnp.random.randn(100, 3)  # 100 atoms
        >>> masses = jnp.ones(100)  # Equal masses
        >>> I = inertia_tensor(positions, masses)
        >>> print(I.shape)
        (3, 3)
        >>> assert jnp.allclose(I, I.T)  # Symmetric
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
        center_of_mass = jnp.average(positions, weights=masses, axis=0)

    # Shift to COM frame
    r = positions - center_of_mass  # Shape (N, 3)

    # Compute r² = x² + y² + z² for each atom
    r_squared = jnp.sum(r**2, axis=1)  # Shape (N,)

    # Initialize inertia tensor
    I = jnp.zeros((3, 3))

    # Diagonal elements: I_ii = Σ m_α (r_α² - r_α,i²)
    for i in range(3):
        I[i, i] = jnp.sum(masses * (r_squared - r[:, i]**2))

    # Off-diagonal elements: I_ij = -Σ m_α r_α,i r_α,j
    I[0, 1] = I[1, 0] = -jnp.sum(masses * r[:, 0] * r[:, 1])
    I[0, 2] = I[2, 0] = -jnp.sum(masses * r[:, 0] * r[:, 2])
    I[1, 2] = I[2, 1] = -jnp.sum(masses * r[:, 1] * r[:, 2])

    return I


def principal_axes(I: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        >>> I = jnp.diag([100, 200, 300])  # Already diagonal
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

    if not jnp.allclose(I, I.T):
        raise ValueError("Inertia tensor must be symmetric")

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = jnp.linalg.eigh(I)

    # Sort by eigenvalues (ascending: I_a ≤ I_b ≤ I_c)
    sort_indices = jnp.argsort(eigenvalues)
    moments = eigenvalues[sort_indices]
    axes = eigenvectors[:, sort_indices]

    # Ensure proper rotation (det = +1, not -1)
    if jnp.linalg.det(axes) < 0:
        # Flip one axis to convert reflection → rotation
        axes[:, 0] *= -1

    return moments, axes


def principal_moments(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    center_of_mass: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, np.ndarray]:
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
        >>> positions = jnp.random.randn(100, 3)
        >>> positions[:, 2] *= 3  # Stretch along z
        >>> masses = jnp.ones(100)
        >>> moments, axes = principal_moments(positions, masses)
        >>> print(f"I_a = {moments[0]:.1f}")
        >>> print(f"I_b = {moments[1]:.1f}")
        >>> print(f"I_c = {moments[2]:.1f}")
        >>> # Expect I_c (longest axis) to be largest
    """
    I = inertia_tensor(positions, masses, center_of_mass)
    return principal_axes(I)


def parallel_axis_theorem(
    I_com: jnp.ndarray,
    total_mass: float,
    displacement: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply parallel axis theorem to shift inertia tensor to a new origin.

    The parallel axis theorem states:
        I_new = I_com + M [(d · d) I_3 - d ⊗ d]
    """
    d = jnp.asarray(displacement)
    d_squared = jnp.dot(d, d)
    # Steiner term: M [(d · d) I_3 - d ⊗ d]
    steiner = total_mass * (d_squared * jnp.eye(3) - np.outer(d, d))

    return I_com + steiner


def asymmetry_parameter(moments: jnp.ndarray) -> float:
    """
    Compute Ray's asymmetry parameter κ.

    The asymmetry parameter quantifies deviation from a symmetric top:
        κ = (2I_b - I_a - I_c) / (I_c - I_a)
    """
    I_a, I_b, I_c = moments

    if jnp.isclose(I_c, I_a):
        # Spherical top: κ undefined, return 0
        return 0.0

    kappa = (2*I_b - I_a - I_c) / (I_c - I_a)
    return kappa

