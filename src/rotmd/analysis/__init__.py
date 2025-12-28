#!/usr/bin/env python
"""
Analysis modules for protein orientation dynamics.

This module provides high-level analysis tools:
- Autocorrelation functions and time correlation analysis
- Friction coefficient extraction from dynamics
- Potential of mean force (PMF) calculations
- Membrane analysis (curvature, packing, Voronoi, etc.)
"""

from .correlations import (
    angular_momentum_acf,
    angular_velocity_acf,
    autocorrelation_function,
    cross_correlation_function,
    extract_correlation_time,
    fit_exponential_decay,
)
from .friction import (
    anisotropic_friction_tensor,
    extract_friction_from_acf,
    orientation_dependent_friction,
)
from .pmf import (
    compute_pmf_1d,
    compute_pmf_2d,
    compute_pmf_6d_projection,
    free_energy_difference,
    jacobian_euler_angles,
)

# Membrane submodule
from . import membrane

__all__ = [
    # Correlations
    "autocorrelation_function",
    "cross_correlation_function",
    "fit_exponential_decay",
    "extract_correlation_time",
    "angular_velocity_acf",
    "angular_momentum_acf",
    # Friction
    "extract_friction_from_acf",
    "orientation_dependent_friction",
    "anisotropic_friction_tensor",
    # PMF
    "jacobian_euler_angles",
    "compute_pmf_1d",
    "compute_pmf_2d",
    "compute_pmf_6d_projection",
    "free_energy_difference",
    # Membrane submodule
    "membrane",
]
