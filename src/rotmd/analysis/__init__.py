#!/usr/bin/env python
"""
Analysis modules for protein orientation dynamics.

This module provides high-level analysis tools:
- Autocorrelation functions and time correlation analysis
- Friction coefficient extraction from dynamics
- Potential of mean force (PMF) calculations
"""

from .correlations import (
    autocorrelation_function,
    cross_correlation_function,
    fit_exponential_decay,
    extract_correlation_time,
    angular_velocity_acf,
    angular_momentum_acf
)

from .friction import (
    extract_friction_from_acf,
    orientation_dependent_friction,
    anisotropic_friction_tensor
)

from .pmf import (
    jacobian_euler_angles,
    compute_pmf_1d,
    compute_pmf_2d,
    compute_pmf_6d_projection,
    free_energy_difference
)

__all__ = [
    'autocorrelation_function',
    'cross_correlation_function',
    'fit_exponential_decay',
    'extract_correlation_time',
    'angular_velocity_acf',
    'angular_momentum_acf',
    'extract_friction_from_acf',
    'orientation_dependent_friction',
    'anisotropic_friction_tensor',
    'jacobian_euler_angles',
    'compute_pmf_1d',
    'compute_pmf_2d',
    'compute_pmf_6d_projection',
    'free_energy_difference'
]
