#!/usr/bin/env python
"""
Analysis modules for protein orientation dynamics.

This module provides high-level analysis tools:
- Autocorrelation functions and time correlation analysis
- Friction coefficient extraction from dynamics
- Potential of mean force (PMF) calculations
- Hopf bifurcation detection and stability analysis
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

from .hopf_bifurcation import (
    HopfBifurcationResult,
    LimitCycleClassification,
    detect_hopf_bifurcation,
    analyze_fixed_point_stability,
    detect_limit_cycle,
    classify_limit_cycle,
    plot_hopf_bifurcation_diagram
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
    'free_energy_difference',
    'HopfBifurcationResult',
    'LimitCycleClassification',
    'detect_hopf_bifurcation',
    'analyze_fixed_point_stability',
    'detect_limit_cycle',
    'classify_limit_cycle',
    'plot_hopf_bifurcation_diagram'
]
