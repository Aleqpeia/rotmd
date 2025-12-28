"""
Membrane analysis submodule for rotmd.

This module provides tools for analyzing membrane properties:
- Surface and curvature analysis (MembraneCurvature)
- Lipid packing defects (PackingDefects)
- Voronoi-based 2D analysis (Voronoi2D)
- Cumulative 2D distributions (Cumulative2D)
- Membrane properties base class (MembProp)
- Protein-membrane interactions (BioPolymer2D)
- Order parameters analysis (OrderParameters)
"""

# Surface analysis
from .surface import normalized_grid, derive_surface, get_z_surface

# Curvature calculations
from .curvature import mean_curvature, gaussian_curvature

# Base analysis class (MDAnalysis-based)
from .base import MembraneCurvature

# Membrane properties base class
from .properties import MembProp

# Lipid packing defects
from .packing import PackingDefects

# Voronoi-based analysis
from .voro import Voronoi2D

# Cumulative 2D analysis
from .projection import Cumulative2D

# Protein-membrane interactions
from .protein import BioPolymer2D

# Order parameters
from .analysis import OrderParameters

__all__ = [
    # Surface
    "normalized_grid",
    "derive_surface",
    "get_z_surface",
    # Curvature
    "mean_curvature",
    "gaussian_curvature",
    # Base class
    "MembraneCurvature",
    # Properties
    "MembProp",
    # Packing
    "PackingDefects",
    # Voronoi
    "Voronoi2D",
    # Projection
    "Cumulative2D",
    # Protein
    "BioPolymer2D",
    # Analysis
    "OrderParameters",
]
