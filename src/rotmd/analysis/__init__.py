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
from .crossings import (
    CrossingCounts,
    count_direction_changes,
    count_grid_crossings,
    count_orientation_crossings,
    count_plane_crossings,
)
from .composition import Manifest, SystemRecord, build_count_table, load_manifest
from .regression import GLMReport, analyze_all, analyze_counts

# Membrane submodule (optional): it depends on the third-party `twodanalysis`
# package, which is not a hard requirement of rotmd. Import it lazily so that a
# missing membrane dependency does not break unrelated analysis (crossings,
# regression, PMF, ...). Access `rotmd.analysis.membrane` triggers the real
# ImportError with installation guidance only if someone actually uses it.
try:
    from . import membrane
except ImportError as _membrane_exc:  # pragma: no cover - optional dependency
    import warnings as _warnings

    _warnings.warn(
        "rotmd.analysis.membrane is unavailable "
        f"({_membrane_exc}). Install its optional dependency with "
        "'poetry add twodanalysis' (or 'pip install twodanalysis') to enable "
        "membrane analysis.",
        stacklevel=2,
    )
    membrane = None  # type: ignore[assignment]

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
    # Crossings / Poisson-NB regression
    "CrossingCounts",
    "count_grid_crossings",
    "count_direction_changes",
    "count_plane_crossings",
    "count_orientation_crossings",
    "Manifest",
    "SystemRecord",
    "load_manifest",
    "build_count_table",
    "GLMReport",
    "analyze_counts",
    "analyze_all",
    # Membrane submodule
    "membrane",
]
