#!/usr/bin/env python
"""
Observable quantities for protein orientation dynamics.

This module provides calculations for dynamical observables:
- Angular velocity from rotation matrices
- Angular momentum (total, spin, nutation components)
- Torque from dL/dt and force fields
- Structural parameters (RMSD, Rg, shape)
- Energetic parameters (kinetic energy, temperature)
"""

from .angular_velocity import (
    angular_velocity_from_rotation_matrices,
    decompose_angular_velocity,
    compute_angular_velocity_from_trajectory,
)

from .angular_momentum import (
    compute_angular_momentum,
    compute_angular_momentum_from_inertia,
    compute_angular_momentum_from_inertia_omega,
    compute_L_parallel_symmetric_top,
    compute_L_parallel_asymmetric_top,
    decompose_angular_momentum,
    compute_spin_nutation_ratio,
    compute_angular_momentum_trajectory,
    skew_symmetric_matrix,
    rotation_matrix_from_angular_velocity
)

from .torque import (
    compute_torque,
    decompose_torque,
    compute_dL_dt,
    validate_euler_equation,
    compute_torque_trajectory,
    torque_field,
    torque_field_from_pmf
)

from .structural import (
    compute_rmsd,
    radius_of_gyration,
    radius_of_gyration_components,
    asphericity,
    acylindricity,
    end_to_end_distance,
    compute_structural_trajectory
)

from .energetics import (
    kinetic_energy_translational,
    kinetic_energy_rotational,
    kinetic_energy_total,
    instantaneous_temperature,
    potential_energy_from_forces,
    virial_tensor,
    compute_energetics,
    compute_energetics_trajectory
)

from .diffusion import (
    analyze_diffusion,
    anisotropic_diffusion_tensor,
    diffusion_from_velocity_acf,
    extract_diffusion_coefficient,
    mean_squared_angular_displacement
)


__all__ = [
    # Angular dynamics
    'angular_velocity_from_rotation_matrices',
    'decompose_angular_velocity',
    'compute_angular_momentum',
    'compute_angular_momentum_from_inertia',
    'compute_angular_momentum_from_inertia_omega',
    'compute_L_parallel_symmetric_top',
    'compute_L_parallel_asymmetric_top',
    'decompose_angular_momentum',
    'compute_spin_nutation_ratio',
    'compute_angular_momentum_trajectory',
    'skew_symmetric_matrix',
    'angular_velocity_from_rotation_matrix_derivative',
    'rotation_matrix_from_angular_velocity',
    'compute_torque',
    'decompose_torque',
    'compute_dL_dt',
    'validate_euler_equation',
    'compute_torque_trajectory',
    # Diffusion
    'analyze_diffusion',
    'anisotropic_diffusion_tensor',
    'mean_squared_angular_displacement',
    # Structural parameters
    'compute_rmsd',
    'radius_of_gyration',
    'radius_of_gyration_components',
    'asphericity',
    'acylindricity',
    'end_to_end_distance',
    'compute_structural_trajectory',
    # Energetics
    'kinetic_energy_translational',
    'kinetic_energy_rotational',
    'kinetic_energy_total',
    'instantaneous_temperature',
    'potential_energy_from_forces',
    'virial_tensor',
    'compute_energetics',
    'compute_energetics_trajectory',
]
