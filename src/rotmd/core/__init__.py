#!/usr/bin/env python
"""
Core geometric and physical utilities for protein orientation dynamics.

This module provides fundamental calculations for rigid body mechanics:
- Inertia tensor computation
- Principal axes determination
- Moment of inertia calculations
"""

from .inertia import (
    inertia_tensor,
    principal_axes,
    principal_moments,
    parallel_axis_theorem,
    is_symmetric_top,
    is_spherical_top,
    is_asymmetric_top
)

__all__ = [
    'inertia_tensor',
    'principal_axes',
    'principal_moments',
    'parallel_axis_theorem',
    'is_symmetric_top',
    'is_spherical_top',
    'is_asymmetric_top'
]
