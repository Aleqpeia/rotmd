"""
Unified Vector Observable Framework (Runtime-Accelerated)

This module provides a common abstraction for all 3D vector observables:
- Angular momentum (L)
- Angular velocity (ω)
- Torque (τ)
- Forces (F)

Now supports multiple computational backends (Numba/PyTorch/JAX) for flexible
performance optimization on different hardware.

Key Abstractions:
-----------------
1. VectorField: 3D vector at each frame (n_frames, 3)
2. Decomposition: parallel, perpendicular, z-component
3. Magnitudes: |v|, |v_∥|, |v_⊥|, etc.

Backend Selection:
- Default: Numba (CPU-optimized)
- Optional: PyTorch (GPU), JAX (TPU/GPU)
- Automatically uses best available backend

Author: Mykyta Bobylyow
Date: 2025
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# Import runtime-agnostic kernels
from . import kernels as K


# =============================================================================
# Runtime-Accelerated Core Functions (Numba/PyTorch/JAX)
# =============================================================================


def decompose_vector_parallel(
    vectors: np.ndarray, reference_axis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose vector into parallel and perpendicular components.

    Uses active computational backend (Numba/PyTorch/JAX).

    Args:
        vectors: (n_frames, 3) vector field
        reference_axis: (3,) or (n_frames, 3) reference direction

    Returns:
        v_parallel: (n_frames, 3) parallel component
        v_perp: (n_frames, 3) perpendicular component

    Examples:
        >>> import numpy as np
        >>> vectors = np.random.rand(1000, 3)
        >>> ref = np.array([0, 0, 1])  # z-axis
        >>> v_par, v_perp = decompose_vector_parallel(vectors, ref)
        >>> # v_perp should be perpendicular to z
        >>> print(np.allclose(v_perp[:, 2], 0))
        True
    """
    # Choose optimized function based on reference type
    if reference_axis.ndim == 1:
        # Static reference: use optimized version
        v_parallel, v_perp = K.decompose_vector_batch_static_ref(
            vectors, reference_axis
        )
    else:
        # Time-varying reference: use standard batch version
        v_parallel, v_perp = K.decompose_vector_batch(vectors, reference_axis)

    return v_parallel, v_perp


def compute_magnitudes(vectors: np.ndarray) -> np.ndarray:
    """
    Compute vector magnitudes.

    Uses active computational backend for acceleration.

    Args:
        vectors: (n_frames, 3) or (n_frames, n_atoms, 3) vector field

    Returns:
        magnitudes: (n_frames,) or (n_frames, n_atoms) |v|

    Examples:
        >>> vectors = np.array([[3, 4, 0], [5, 12, 0]])
        >>> mags = compute_magnitudes(vectors)
        >>> print(mags)
        [5. 13.]
    """
    return K.compute_magnitude_batch(vectors)


def compute_cross_product_trajectory(
    positions: np.ndarray, vectors: np.ndarray, masses: np.ndarray, com: np.ndarray
) -> np.ndarray:
    """
    Compute Σ_i m_i (r_i - COM) × v_i for entire trajectory.

    Uses active computational backend for acceleration.

    Used for angular momentum and torque calculations.

    Args:
        positions: (n_frames, n_atoms, 3)
        vectors: (n_frames, n_atoms, 3) velocities or forces
        masses: (n_atoms,)
        com: (n_frames, 3) center of mass

    Returns:
        result: (n_frames, 3) summed cross products

    Examples:
        >>> n_frames, n_atoms = 100, 1000
        >>> pos = np.random.rand(n_frames, n_atoms, 3)
        >>> vel = np.random.rand(n_frames, n_atoms, 3)
        >>> masses = np.ones(n_atoms)
        >>> com = np.mean(pos, axis=1)
        >>> L = compute_cross_product_trajectory(pos, vel, masses, com)
        >>> print(L.shape)
        (100, 3)
    """
    # Use batched cross product (vmap over frames)
    result = K.cross_product_trajectory(positions, vectors, masses, com)

    return result


# =============================================================================
# High-Level Data Structure
# =============================================================================


@dataclass
class VectorObservable:
    """
    Container for vector observable with automatic decomposition.

    All decompositions are computed using the active backend (Numba/PyTorch).

    Attributes:
        vector: (n_frames, 3) full 3D vector
        parallel: (n_frames, 3) component parallel to reference
        perp: (n_frames, 3) component perpendicular to reference
        z_component: (n_frames, 3) component along z-axis (membrane normal)
        magnitude: (n_frames,) |vector|
        parallel_mag: (n_frames,) |parallel|
        perp_mag: (n_frames,) |perp|
        z_mag: (n_frames,) |z_component|
        times: (n_frames,) time values
        name: Observable name (L, omega, tau, etc.)

    Examples:
        >>> import numpy as np
        >>> vectors = np.random.rand(100, 3)
        >>> ref = np.array([1, 0, 0])
        >>> obs = create_vector_observable(vectors, ref, name="L")
        >>> print(obs.magnitude.mean())
        1.732...
    """

    vector: np.ndarray
    parallel: np.ndarray
    perp: np.ndarray
    z_component: np.ndarray
    magnitude: np.ndarray
    parallel_mag: np.ndarray
    perp_mag: np.ndarray
    z_mag: np.ndarray
    times: Optional[np.ndarray] = None
    name: str = "vector"


# =============================================================================
# Factory Functions
# =============================================================================


def create_vector_observable(
    vector: np.ndarray,
    reference_axis: np.ndarray,
    membrane_normal: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    name: str = "vector",
) -> VectorObservable:
    """
    Create VectorObservable with automatic decomposition.

    All decompositions computed via accelerated backend kernels.

    Args:
        vector: (n_frames, 3) vector field
        reference_axis: (3,) or (n_frames, 3) - spin axis (usually a1)
        membrane_normal: (3,) or (n_frames, 3) - z-axis
        times: (n_frames,) time values
        name: Observable name

    Returns:
        VectorObservable: With all decompositions computed

    Examples:
        >>> import numpy as np
        >>> L = np.random.rand(1000, 3)
        >>> axis = np.array([1, 0, 0])
        >>> obs = create_vector_observable(L, axis, name="L")
        >>> print(f"Mean |L|: {obs.magnitude.mean():.3f}")
        Mean |L|: 1.732
    """
    # Decompose into parallel/perpendicular relative to reference axis
    v_parallel, v_perp = decompose_vector_parallel(vector, reference_axis)

    # Decompose along membrane normal (z-axis) if provided
    if membrane_normal is None:
        membrane_normal = np.array([0.0, 0.0, 1.0])

    v_z, _ = decompose_vector_parallel(vector, membrane_normal)

    # Compute magnitudes
    magnitude = compute_magnitudes(vector)
    parallel_mag = compute_magnitudes(v_parallel)
    perp_mag = compute_magnitudes(v_perp)
    z_mag = compute_magnitudes(v_z)

    return VectorObservable(
        vector=vector,
        parallel=v_parallel,
        perp=v_perp,
        z_component=v_z,
        magnitude=magnitude,
        parallel_mag=parallel_mag,
        perp_mag=perp_mag,
        z_mag=z_mag,
        times=times,
        name=name,
    )


__all__ = [
    "decompose_vector_parallel",
    "compute_magnitudes",
    "compute_cross_product_trajectory",
    "VectorObservable",
    "create_vector_observable",
]
