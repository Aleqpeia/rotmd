"""
Visualization Module for Protein Orientation Analysis

Provides plotting utilities for phase space, PMF surfaces, and spectral analysis.
"""

from .phase_space import (
    plot_L_phase_space,
    plot_phase_portrait_2d,
    plot_phase_portrait_with_vector_field,
    plot_energy_phase_space,
    plot_multi_panel_summary,
    plot_poincare_section_improved
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

__all__ = [
    # Phase space
    'plot_L_phase_space',
    'plot_phase_portrait_2d',
    'plot_phase_portrait_with_vector_field',
    'plot_energy_phase_space',
    'plot_multi_panel_summary',
    'plot_poincare_section_improved',
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
]
