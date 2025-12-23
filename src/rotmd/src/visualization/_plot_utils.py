"""
Plotting Utilities for Publication-Quality Visualizations

Reusable matplotlib helpers to enforce consistent styling and reduce
code duplication across phase space plotting functions.

Key Features:
- Colorblind-safe color schemes
- Publication-ready styling (PDF/SVG with proper fonts)
- Reusable colorbar and annotation helpers
- Standardized density visualization

Compliance:
- Font embedding: Type 42 TrueType (editable in Illustrator)
- DPI: 300 for raster, vector for PDF/SVG
- Colorblind palettes: viridis, cividis, RdBu_r

Author: Mykyta Bobylyow
Date: 2025
"""

from typing import Optional, Tuple, Literal
import numpy as np
from numpy.typing import NDArray
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")


# Colorblind-safe color schemes (validated with colorbrewer2.org)
COLORBLIND_COLORMAPS = {
    'sequential': 'viridis',      # For time, single-variable gradients
    'diverging': 'RdBu_r',         # For signed quantities (torque, divergence)
    'density': 'cividis',          # For density/probability (best contrast)
    'energy': 'plasma',            # For energy landscapes (warm colors)
    'qualitative': 'tab10'         # For discrete categories
}


def setup_publication_style() -> None:
    """
    Configure matplotlib for publication-quality output.

    Sets fonts, line widths, and defaults per matplotlib best practices.
    Supports vector output (PDF, SVG) with proper font embedding.

    Changes:
    - Font: Arial/DejaVu Sans (sans-serif, professional)
    - Font sizes: 11pt (body), 12pt (labels), 14pt (titles)
    - Line widths: 1.2-1.5pt (visible but not thick)
    - DPI: 100 (screen), 300 (save)
    - PDF font type 42: TrueType (editable in Illustrator)

    Notes
    -----
    This function modifies global plt.rcParams. Call it once at module
    import or per-function for isolation.

    For publication:
    - Saves as PDF with embedded fonts (editable)
    - SVG for web/presentations
    - PNG at 300 DPI for raster fallback

    Examples
    --------
    >>> setup_publication_style()
    >>> fig, ax = plt.subplots()
    >>> # ... plotting code ...
    >>> fig.savefig('output.pdf')  # Uses publication settings
    """
    if not HAS_MATPLOTLIB:
        return

    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Line widths
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.5,

        # Figure settings
        'figure.dpi': 100,           # Screen display
        'savefig.dpi': 300,          # Publication quality
        'savefig.format': 'pdf',     # Default format
        'pdf.fonttype': 42,          # TrueType fonts (editable)
        'ps.fonttype': 42,           # PostScript TrueType

        # Grid and axes
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,      # Grid behind plot elements

        # Tight layout
        'figure.autolayout': False,  # Use tight_layout() manually
    })


def add_colorbar_with_label(
    mappable,
    ax: Axes,
    label: str,
    orientation: Literal['vertical', 'horizontal'] = 'vertical',
    location: Optional[str] = None,
    fontsize: int = 11
) -> Colorbar:
    """
    Add publication-quality colorbar with proper formatting.

    Extracts repeated colorbar creation code (~50 lines across old module).
    Provides consistent styling and positioning.

    Parameters
    ----------
    mappable : ScalarMappable
        The plot object to create colorbar for (contour, scatter, etc.)
    ax : Axes
        Axes to attach colorbar to
    label : str
        Colorbar label with units (e.g., "Time (ps)", "Energy (kcal/mol)")
    orientation : {'vertical', 'horizontal'}
        Colorbar orientation
    location : str, optional
        Colorbar location ('right', 'top', 'bottom', 'left')
        If provided with horizontal orientation, uses axes_grid1 for placement
    fontsize : int
        Label font size (default: 11)

    Returns
    -------
    Colorbar
        Configured colorbar instance

    Notes
    -----
    For top/bottom placement with horizontal orientation, this uses
    mpl_toolkits.axes_grid1 to create an inset axes. This ensures
    proper spacing and doesn't overlap the plot.

    Examples
    --------
    >>> scatter = ax.scatter(x, y, c=times, cmap='viridis')
    >>> cbar = add_colorbar_with_label(scatter, ax, 'Time (ps)')

    >>> # Top horizontal colorbar
    >>> cbar = add_colorbar_with_label(
    ...     scatter, ax, 'Energy (kcal/mol)',
    ...     orientation='horizontal', location='top'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    if location and orientation == 'horizontal':
        # Use axes_grid1 for top/bottom placement
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(location, size="4%", pad=0.3)
        cbar = plt.colorbar(mappable, cax=cax, orientation=orientation)

        # Position label based on location
        if location == 'top':
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.xaxis.set_ticks_position('top')
        # 'bottom' uses default positioning

    else:
        # Standard colorbar (vertical on right, horizontal on bottom)
        cbar = plt.colorbar(mappable, ax=ax, orientation=orientation)

    # Format colorbar
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)

    return cbar


def plot_contour_density(
    ax: Axes,
    x_centers: NDArray,
    y_centers: NDArray,
    density: NDArray,
    n_levels: int = 7,
    cmap: str = 'cividis',
    label: str = 'Density',
    linewidths: float = 1.2,
    alpha: float = 0.8
) -> Tuple:
    """
    Plot density field as contour lines (not filled).

    Standardized density visualization for phase portraits.
    Uses contour lines (not hexbin or filled contours) for
    publication clarity.

    Parameters
    ----------
    ax : Axes
        Axes to plot on
    x_centers : ndarray, shape (nx,)
        X bin centers from DensityField2D
    y_centers : ndarray, shape (ny,)
        Y bin centers from DensityField2D
    density : ndarray, shape (nx, ny)
        Density values (normalized)
    n_levels : int
        Number of contour levels (default: 7)
    cmap : str
        Colormap name (should be colorblind-safe)
    label : str
        Label for colorbar
    linewidths : float
        Contour line width
    alpha : float
        Contour transparency

    Returns
    -------
    contour : QuadContourSet
        Contour object
    colorbar : Colorbar
        Colorbar object

    Notes
    -----
    Design decision: Contour lines chosen over:
    - Hexbin: Less professional, harder to read contours
    - Filled contours (contourf): Can obscure trajectory overlay

    Contour lines provide:
    - Clear density levels
    - Professional appearance
    - Easy overlay with trajectories

    Examples
    --------
    >>> from ._phase_space_physics import compute_density_2d
    >>> density = compute_density_2d(theta, omega, bins=60)
    >>> contour, cbar = plot_contour_density(
    ...     ax, density.x_centers, density.y_centers, density.counts,
    ...     cmap='cividis', label='Density (normalized)'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    # Plot contour lines (NOT filled)
    contour = ax.contour(
        x_centers, y_centers, density.T,  # Transpose for correct orientation
        levels=n_levels,
        linewidths=linewidths,
        cmap=cmap,
        alpha=alpha
    )

    # Add colorbar
    cbar = add_colorbar_with_label(contour, ax, label)

    return contour, cbar


def add_frame_annotation(
    ax: Axes,
    frame: Literal['body', 'lab'],
    location: Tuple[float, float] = (0.02, 0.98),
    fontsize: int = 9
) -> None:
    """
    Add frame annotation to plot (REQUIRED by .cursorrules).

    Per .cursorrules line 10: "Frame consistency is non-negotiable."
    All angular momentum plots must clearly indicate reference frame.

    Parameters
    ----------
    ax : Axes
        Axes to annotate
    frame : {'body', 'lab'}
        Reference frame to annotate
    location : tuple (x, y)
        Text position in axes coordinates (0-1)
        Default: (0.02, 0.98) = top-left corner
    fontsize : int
        Annotation font size

    Notes
    -----
    Frame meanings:
    - 'body': Principal axis frame (inertia tensor diagonal)
    - 'lab': Laboratory frame (fixed reference)

    The annotation appears as a small box in the plot corner:
    "Frame: body" or "Frame: lab"

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> # ... plot L_parallel vs L_perp ...
    >>> add_frame_annotation(ax, 'body')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    frame_text = f"Frame: {frame}"

    ax.text(
        location[0], location[1], frame_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='gray')
    )


def format_angle_axis(
    ax: Axes,
    axis: Literal['x', 'y'],
    angle_data: NDArray,
    angle_label: str = 'θ'
) -> None:
    """
    Format axis for angular quantities with π labels if appropriate.

    For angles spanning more than π radians, uses π-based tick labels
    for clarity. Otherwise uses standard decimal labels.

    Parameters
    ----------
    ax : Axes
        Axes to format
    axis : {'x', 'y'}
        Which axis to format
    angle_data : ndarray
        Angle values in radians to determine range
    angle_label : str
        Angle symbol for axis label (e.g., 'θ', 'ψ', 'φ')

    Notes
    -----
    Formatting rules:
    - If range > π: Use π ticks (0, π/2, π, 3π/2, 2π)
    - If range ≤ π: Use decimal radians
    - Label always includes "(rad)" for clarity

    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(theta, np.sin(theta))
    >>> format_angle_axis(ax, 'x', theta, 'θ')
    >>> # Result: x-axis labeled "θ (rad)" with ticks [0, π/2, π, 3π/2, 2π]
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    angle_range = angle_data.max() - angle_data.min()

    if angle_range > np.pi:
        # Use π ticks for large ranges
        ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        tick_labels = ['0', 'π/2', 'π', '3π/2', '2π']

        if axis == 'x':
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(f'{angle_label} (rad)', fontsize=12)
        else:  # axis == 'y'
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel(f'{angle_label} (rad)', fontsize=12)
    else:
        # Use decimal labels for small ranges
        if axis == 'x':
            ax.set_xlabel(f'{angle_label} (rad)', fontsize=12)
        else:
            ax.set_ylabel(f'{angle_label} (rad)', fontsize=12)


def save_publication_figure(
    fig: Figure,
    save_path: str,
    formats: Tuple[str, ...] = ('pdf',),
    dpi: int = 300,
    close: bool = True
) -> None:
    """
    Save figure in multiple publication-ready formats.

    Saves to PDF (vector, editable), SVG (web), and/or PNG (raster)
    with proper settings for each format.

    Parameters
    ----------
    fig : Figure
        Figure to save
    save_path : str
        Base path without extension
        Example: 'figures/theta_phase' → saves theta_phase.pdf, etc.
    formats : tuple of str
        File formats to save: 'pdf', 'svg', 'png'
        Default: ('pdf',) for publication
    dpi : int
        DPI for raster formats (PNG)
        Ignored for vector formats (PDF, SVG)
    close : bool
        If True, close figure after saving (frees memory)

    Notes
    -----
    Format specifics:
    - PDF: Vector, Type 42 fonts (editable in Illustrator)
    - SVG: Vector, web-friendly, good for presentations
    - PNG: Raster, fallback for journals that don't accept vector

    All formats use tight_layout and bbox_inches='tight' to
    remove unnecessary whitespace.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> save_publication_figure(
    ...     fig, 'figures/test',
    ...     formats=('pdf', 'svg', 'png')
    ... )
    # Saves: test.pdf, test.svg, test.png
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    from pathlib import Path

    base_path = Path(save_path).with_suffix('')  # Remove any extension

    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with format-specific settings
        fig.savefig(
            output_path,
            dpi=dpi if fmt == 'png' else None,  # DPI only for raster
            bbox_inches='tight',                 # Remove whitespace
            format=fmt,
            facecolor='white',                   # Ensure white background
            edgecolor='none'
        )

        print(f"✓ Saved: {output_path}")

    if close:
        plt.close(fig)


if __name__ == '__main__':
    print("Plotting Utilities Module")
    print("=" * 60)
    print("\nColorblind-safe colormaps:")
    for key, cmap in COLORBLIND_COLORMAPS.items():
        print(f"  {key:15s} : {cmap}")
    print("\nUtility functions:")
    print("  - setup_publication_style()")
    print("  - add_colorbar_with_label()")
    print("  - plot_contour_density()")
    print("  - add_frame_annotation()")
    print("  - format_angle_axis()")
    print("  - save_publication_figure()")
    print("\nPublication settings:")
    print("  ✓ Font: Arial/DejaVu Sans, 11pt")
    print("  ✓ DPI: 300 (PNG), vector (PDF/SVG)")
    print("  ✓ PDF Type 42: Editable fonts")
    print("=" * 60)
