#!/usr/bin/python
"""
Protein Orientation Analysis Package

This package provides tools for analyzing peripheral membrane protein orientation
using a 3D pendulum model. It includes:

- Orientation extraction from MD trajectories
- Free energy landscape calculation
- Energy minima identification
- Membrane interface utilities
- Visualization tools
"""

__version__ = "0.1.0"

from . import membrane_interface
from . import leaflet_util

__all__ = [
    'membrane_interface',
    'leaflet_util'
]
