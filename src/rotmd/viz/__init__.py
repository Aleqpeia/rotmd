"""Plotting for rotmd.

:mod:`rotmd.viz.core` is the API — a ``@figure`` decorator that gives any
drawing function a save mode, a return-the-figure mode and a draw-into-my-axes
mode, plus the shared unit and style helpers. The other modules are plots built
on it.

Nothing here is imported by the analysis code, keeping the dependency direction
one-way: extract <- visualize <- analyze. Callers that only sometimes draw
should import lazily, as :mod:`rotmd.cli` does, so a headless run never pays
for matplotlib.

Plots are registered by name, so a string from a CLI flag or a config file is
enough to reach one::

    from rotmd.viz import available, render
    render("pmf-heatmap", pmf, theta_bins, psi_bins, output="pmf.png")
"""

from __future__ import annotations

from .core import (
    RC,
    angle_grid_degrees,
    available,
    figure,
    finite_percentile,
    label_orientation_axes,
    pyplot,
    render,
    save,
    style,
    time_coloured_path,
)
from .phase_space import (
    plot_energy_phase_space,
    plot_L_phase_space,
    plot_multi_panel_summary,
    plot_phase_portrait_2d,
    plot_phase_portrait_with_vector_field,
    plot_poincare_sections,
    poincare_crossings,
)
from .spectra import (
    plot_autocorrelation,
    plot_correlation_comparison,
    plot_friction_extraction,
    plot_multiple_acfs,
    plot_power_spectrum,
    plot_spectral_density,
    power_spectrum,
    spectral_density,
)
from .surfaces import (
    local_minima,
    plot_free_energy_landscape,
    plot_friction_map,
    plot_pmf_3d_surface,
    plot_pmf_contour,
    plot_pmf_heatmap,
    plot_torque_vector_field,
)
from .transitions import plot_committor, plot_state_trajectory

__all__ = [
    # API
    "RC",
    "figure",
    "render",
    "available",
    "save",
    "style",
    "pyplot",
    "angle_grid_degrees",
    "label_orientation_axes",
    "time_coloured_path",
    "finite_percentile",
    # Phase space
    "plot_phase_portrait_2d",
    "plot_phase_portrait_with_vector_field",
    "plot_energy_phase_space",
    "plot_L_phase_space",
    "plot_poincare_sections",
    "plot_multi_panel_summary",
    "poincare_crossings",
    # Surfaces
    "plot_pmf_heatmap",
    "plot_pmf_contour",
    "plot_pmf_3d_surface",
    "plot_torque_vector_field",
    "plot_free_energy_landscape",
    "plot_friction_map",
    "local_minima",
    # Spectra
    "plot_autocorrelation",
    "plot_multiple_acfs",
    "plot_power_spectrum",
    "plot_spectral_density",
    "plot_friction_extraction",
    "plot_correlation_comparison",
    "power_spectrum",
    "spectral_density",
    # Transitions
    "plot_state_trajectory",
    "plot_committor",
]
