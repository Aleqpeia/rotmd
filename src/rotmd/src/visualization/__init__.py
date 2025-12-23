"""
Visualization Module for Protein Orientation Analysis

Provides plotting utilities for phase space, PMF surfaces, and spectral analysis.

Architecture:
- phase_space: High-level plotting API (publication-quality)
- _phase_space_physics: Physics computations (flow fields, stability metrics)
- _plot_utils: Matplotlib helpers (reusable styling utilities)
"""

# Main plotting API
from .phase_space import (
    plot_L_phase_space,
    plot_phase_portrait_2d,
    plot_phase_portrait_with_vector_field,
    plot_energy_landscape_trajectory,
    plot_poincare_section
)

# Physics dataclasses (for type hints and advanced usage)
from ._phase_space_physics import (
    FlowFieldResult,
    StabilityMetrics,
    DensityField2D,
    compute_flow_field,
    compute_stability_metrics,
    compute_density_2d,
    validate_angular_momentum_frame
)

# Plotting utilities (for custom visualizations)
from ._plot_utils import (
    COLORBLIND_COLORMAPS,
    setup_publication_style,
    add_colorbar_with_label,
    plot_contour_density,
    add_frame_annotation,
    format_angle_axis,
    save_publication_figure
)

from .surfaces import (
    plot_pmf_heatmap,
    plot_pmf_contour,
    plot_pmf_3d_surface,
    plot_torque_vector_field,
    plot_free_energy_landscape
)

from .spectra import (
    plot_autocorrelation,
    plot_multiple_acfs,
    plot_power_spectrum,
    plot_spectral_density,
    plot_friction_extraction,
    plot_correlation_comparison
)

from .metrics import (
    plot_rmsd_timeseries,
    plot_rg_timeseries,
    plot_rmsd_rg_comparison,
    plot_metric_distribution,
    plot_multi_metric_panel
)

from .bifurcation import (
    compute_poincare_section,
    plot_poincare_bifurcation,
    plot_multi_section_bifurcation
)

__all__ = [
    # Phase space plotting API
    'plot_L_phase_space',
    'plot_phase_portrait_2d',
    'plot_phase_portrait_with_vector_field',
    'plot_energy_landscape_trajectory',
    'plot_poincare_section',

    # Physics dataclasses (for type hints)
    'FlowFieldResult',
    'StabilityMetrics',
    'DensityField2D',

    # Physics computation functions
    'compute_flow_field',
    'compute_stability_metrics',
    'compute_density_2d',
    'validate_angular_momentum_frame',

    # Plotting utilities
    'COLORBLIND_COLORMAPS',
    'setup_publication_style',
    'add_colorbar_with_label',
    'plot_contour_density',
    'add_frame_annotation',
    'format_angle_axis',
    'save_publication_figure',

    # Surfaces
    'plot_pmf_heatmap',
    'plot_pmf_contour',
    'plot_pmf_3d_surface',
    'plot_torque_vector_field',
    'plot_free_energy_landscape',

    # Spectra
    'plot_autocorrelation',
    'plot_multiple_acfs',
    'plot_power_spectrum',
    'plot_spectral_density',
    'plot_friction_extraction',
    'plot_correlation_comparison',

    # Metrics (simple visualizations)
    'plot_rmsd_timeseries',
    'plot_rg_timeseries',
    'plot_rmsd_rg_comparison',
    'plot_metric_distribution',
    'plot_multi_metric_panel',

    # Bifurcation analysis
    'compute_poincare_section',
    'plot_poincare_bifurcation',
    'plot_multi_section_bifurcation',
]
