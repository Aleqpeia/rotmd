#!/usr/bin/env python
"""
Core geometric and physical primitives for the rotmd extract pipeline.

Rigid-body quantities computed per frame during extraction:
- Inertia tensor, principal axes and moments, center of mass
- Body-frame orientation (rotation matrices, ZYZ Euler angles, membrane tilt)
- Vector observables (L/ω/τ) with parallel/perpendicular/z decomposition

Analysis-stage extensions of these (quaternions, asymmetry parameter,
autocorrelations, xarray export) live in :mod:`rotmd.analysis`.
"""

from .inertia import (
    compute_center_of_mass,
    inertia_tensor,
    principal_axes,
    principal_moments,
)
from .orientation import (
    rotation_matrix_to_euler_zyz,
    membrane_tilt_angle,
    extract_orientation_trajectory,
)
from .vector_observables import (
    VectorObservable,
    create_vector_observable,
    decompose_vector_parallel,
    compute_magnitudes,
    compute_cross_product_trajectory,
)

__all__ = [
    # inertia
    "compute_center_of_mass",
    "inertia_tensor",
    "principal_axes",
    "principal_moments",
    # orientation
    "rotation_matrix_to_euler_zyz",
    "membrane_tilt_angle",
    "extract_orientation_trajectory",
    # vector observables
    "VectorObservable",
    "create_vector_observable",
    "decompose_vector_parallel",
    "compute_magnitudes",
    "compute_cross_product_trajectory",
]
