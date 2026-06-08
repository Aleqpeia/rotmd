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
)

__all__ = [
    'inertia_tensor',
    'principal_axes',
    'principal_moments',
]
