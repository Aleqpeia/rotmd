#!/usr/bin/env python
"""
Observable quantities for protein orientation dynamics.

This module provides calculations for dynamical observables:
- Angular momentum, torque, angular velocity (functional class-based)
- Structural parameters (RMSD, Rg, shape)
- Energetic parameters (kinetic energy, temperature)
- Diffusion analysis
"""

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
from .structural import (
    compute_rmsd,
    radius_of_gyration,
    radius_of_gyration_components,
    asphericity,
    acylindricity,
    end_to_end_distance,
    compute_structural_trajectory
)
# Unified functional API (recommended)
from .unified import compute_all_observables

__all__ = [
    'compute_all_observables',
    'compute_rmsd',
    'radius_of_gyration',
    'radius_of_gyration_components',
    'asphericity',
    'acylindricity',
    'end_to_end_distance',
    'compute_structural_trajectory',
    'kinetic_energy_translational',
    'kinetic_energy_rotational',
    'kinetic_energy_total',
    'instantaneous_temperature',
    'potential_energy_from_forces',
    'virial_tensor',
    'compute_energetics',
    'compute_energetics_trajectory',
]
