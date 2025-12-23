#!/usr/bin/env python
"""
Observable quantities for protein orientation dynamics.

This module provides calculations for dynamical observables:
- Angular momentum, torque, angular velocity (functional class-based)
- Structural parameters (RMSD, Rg, shape)
- Energetic parameters (kinetic energy, temperature)
- Diffusion analysis
"""

# Unified functional API (recommended)
from .unified import compute_all_observables

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
    # Unified functional API
    'compute_all_observables',
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
