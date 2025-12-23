"""
Bifurcation Diagrams and Poincaré Sections

Specialized visualization for bifurcation analysis of protein dynamics,
particularly Poincaré sections parameterized by system energy.

Key Features:
- Poincaré sections at specific angle crossings
- Bifurcation diagrams showing transitions vs energy
- Stability analysis from fixed point detection
- Multi-panel bifurcation visualization

Physics:
Based on BIFURCATION_ANALYSIS.md and PHASE_PORTRAIT.md:
- Saddle-node bifurcations: Fixed points appear/disappear
- Pitchfork bifurcations: Symmetry breaking
- Transcritical bifurcations: Stability exchange
- Hopf bifurcations: Periodic orbits emerge

Author: Mykyta Bobylyow
Date: 2025
"""

from typing import Optional, Tuple, Literal, List
import numpy as np
from numpy.typing import NDArray
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

from ._plot_utils import setup_publication_style, save_publication_figure


def compute_poincare_section(
    angle: NDArray,
    angular_velocity: NDArray,
    energy: NDArray,
    section_angle: float = 0.0,
    tolerance: float = 0.05
) -> Tuple[NDArray, NDArray]:
    """
    Compute Poincaré section crossings.

    A Poincaré section samples the trajectory each time it crosses
    a specific angle with positive angular velocity.

    Parameters
    ----------
    angle : ndarray, shape (n,)
        Angular coordinate (radians)
    angular_velocity : ndarray, shape (n,)
        Angular velocity (rad/ps)
    energy : ndarray, shape (n,)
        Total energy (kcal/mol)
    section_angle : float
        Angle at which to sample (radians)
    tolerance : float
        Crossing detection tolerance (radians)

    Returns
    -------
    omega_crossings : ndarray
        Angular velocities at crossings
    energy_crossings : ndarray
        Energies at crossings

    Notes
    -----
    Classical Poincaré section for 2D phase space (φ, ω).
    Crossings detected when φ passes through section_angle with ω > 0.

    Example
    -------
    >>> omega_cross, E_cross = compute_poincare_section(
    ...     theta, omega, total_energy, section_angle=0.0
    ... )
    >>> plt.scatter(E_cross, omega_cross, s=1, alpha=0.5)
    >>> plt.xlabel('Total Energy (kcal/mol)')
    >>> plt.ylabel('Angular Velocity at θ=0 (rad/ps)')
    """
    # Normalize angle to [0, π/6)
    angle_normalized = np.mod(angle, np.pi/4)
    section_normalized = np.mod(section_angle, np.pi/4)

    # Find crossings: where angle passes through section with positive velocity
    crossings = []
    energies = []

    for i in range(len(angle) - 1):
        # Check if we cross the section angle
        a1 = angle_normalized[i]
        a2 = angle_normalized[i+1]

        # Handle wrapping around π/6
        if abs(a1 - section_normalized) < tolerance and angular_velocity[i] > 0:
            crossings.append(angular_velocity[i])
            energies.append(energy[i])

        # Also check for actual crossing (sign change)
        delta = a2 - a1
        if abs(delta) < np.pi:  # Not wrapping
            if (a1 - section_normalized) * (a2 - section_normalized) < 0:
                if angular_velocity[i] > 0:
                    # Linear interpolation for better accuracy
                    frac = (section_normalized - a1) / (a2 - a1)
                    omega_interp = angular_velocity[i] + frac * (angular_velocity[i+1] - angular_velocity[i])
                    energy_interp = energy[i] + frac * (energy[i+1] - energy[i])
                    crossings.append(omega_interp)
                    energies.append(energy_interp)

    return np.array(crossings), np.array(energies)


def plot_poincare_bifurcation(
    energy: NDArray,
    omega_crossings: NDArray,
    xlabel: str = 'Total Energy (kcal/mol)',
    ylabel: str = 'Angular Velocity at Section (rad/ps)',
    title: str = 'Poincaré Bifurcation Diagram',
    energy_bins: int = 50,
    figsize: Tuple[float, float] = (10, 7),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot Poincaré section as bifurcation diagram vs energy.

    Shows how fixed points and periodic orbits evolve as energy changes.
    This reveals bifurcations where system behavior transitions.

    Parameters
    ----------
    energy : ndarray, shape (n_crossings,)
        Energy values at Poincaré crossings
    omega_crossings : ndarray, shape (n_crossings,)
        Angular velocity values at crossings
    xlabel, ylabel : str
        Axis labels
    title : str
        Plot title
    energy_bins : int
        Number of bins for binned statistics overlay
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        File formats to save

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Notes
    -----
    Bifurcation signatures:
    - Single branch: Single stable state
    - Fork split: Pitchfork bifurcation (symmetry breaking)
    - Branch appears/disappears: Saddle-node bifurcation
    - Loops/circles: Periodic orbits (Hopf bifurcation)

    Example
    -------
    >>> omega_cross, E_cross = compute_poincare_section(theta, omega, energy)
    >>> fig = plot_poincare_bifurcation(
    ...     E_cross, omega_cross,
    ...     title='Tilt Angle Bifurcation vs Total Energy',
    ...     save_path='figures/bifurcation'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot of all crossings
    scatter = ax.scatter(energy, omega_crossings,
                        s=1, alpha=0.5, c='steelblue',
                        rasterized=True)

    # Binned statistics to show fixed point evolution
    energy_range = energy.max() - energy.min()
    bin_edges = np.linspace(energy.min(), energy.max(), energy_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    means = []
    stds = []

    for i in range(1, len(bin_edges)):
        mask = (energy >= bin_edges[i-1]) & (energy < bin_edges[i])
        if np.sum(mask) > 0:
            means.append(np.mean(omega_crossings[mask]))
            stds.append(np.std(omega_crossings[mask]))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    means = np.array(means)
    stds = np.array(stds)
    print(means.shape, stds.shape)
    # Plot mean trajectory (fixed point evolution)
    valid = ~np.isnan(means)
    ax.plot(bin_centers[valid], means[valid],
            color='red', linewidth=2, label='Mean', zorder=10)

    # Show standard deviation bands (spread indicates stability)
    ax.fill_between(bin_centers[valid],
                    means[valid] - stds[valid],
                    means[valid] + stds[valid],
                    color='red', alpha=0.2, label='±1σ', zorder=5)

    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.text(0.02, 0.98, f'Crossings: {len(energy)}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


def plot_multi_section_bifurcation(
    angle: NDArray,
    angular_velocity: NDArray,
    energy: NDArray,
    section_angles: List[float] = [0.0, np.pi/2, np.pi],
    angle_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (15, 5),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot Poincaré sections at multiple angles.

    Shows how bifurcation structure depends on section angle choice.

    Parameters
    ----------
    angle : ndarray, shape (n,)
        Angular coordinate
    angular_velocity : ndarray, shape (n,)
        Angular velocity
    energy : ndarray, shape (n,)
        Total energy
    section_angles : list
        Angles at which to compute sections (radians)
    angle_labels : list, optional
        Labels for each section angle
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        File formats to save

    Returns
    -------
    fig : Figure
        Matplotlib figure with multiple panels

    Example
    -------
    >>> fig = plot_multi_section_bifurcation(
    ...     theta, omega, total_energy,
    ...     section_angles=[0, np.pi/2, np.pi],
    ...     angle_labels=['θ=0', 'θ=π/2', 'θ=π'],
    ...     save_path='figures/multi_poincare'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    n_sections = len(section_angles)
    fig, axes = plt.subplots(1, n_sections, figsize=figsize, sharey=True)

    if n_sections == 1:
        axes = [axes]

    if angle_labels is None:
        angle_labels = [f'θ = {a:.2f}' for a in section_angles]

    for i, (section_angle, label) in enumerate(zip(section_angles, angle_labels)):
        ax = axes[i]

        # Compute Poincaré section
        omega_cross, energy_cross = compute_poincare_section(
            angle, angular_velocity, energy,
            section_angle=section_angle
        )

        # Plot bifurcation diagram
        if len(omega_cross) > 0:
            ax.scatter(energy_cross, omega_cross,
                      s=1, alpha=0.3, c='steelblue',
                      rasterized=True)

            # Binned mean
            energy_bins = 40
            bin_edges = np.linspace(energy_cross.min(), energy_cross.max(), energy_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            means = []
            for j in range(len(bin_edges) - 1):
                mask = (energy_cross >= bin_edges[j]) & (energy_cross < bin_edges[j+1])
                if np.sum(mask) > 0:
                    means.append(np.mean(omega_cross[mask]))
                else:
                    means.append(np.nan)

            means = np.array(means)
            valid = ~np.isnan(means)
            ax.plot(bin_centers[valid], means[valid],
                   color='red', linewidth=2, zorder=10)

        # Styling
        ax.set_xlabel('Total Energy (kcal/mol)', fontsize=11)
        if i == 0:
            ax.set_ylabel('Angular Velocity (rad/ps)', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Annotation
        n_crossings = len(omega_cross)
        ax.text(0.02, 0.98, f'N = {n_crossings}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.suptitle('Poincaré Bifurcation Diagrams', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


if __name__ == '__main__':
    print("Bifurcation Diagram Visualization Module")
    print("=" * 60)
    print("\nFunctions:")
    print("  - compute_poincare_section()")
    print("  - plot_poincare_bifurcation()")
    print("  - plot_multi_section_bifurcation()")
    print("\nBifurcation types detected:")
    print("  - Saddle-node: Fixed points appear/disappear")
    print("  - Pitchfork: Symmetry breaking (single→double branch)")
    print("  - Transcritical: Stability exchange")
    print("  - Hopf: Periodic orbits (loops in section)")
    print("=" * 60)
