"""
Surface Visualization

This module provides utilities for visualizing PMF surfaces, torque vector fields,
and other 2D/3D scalar/vector fields in orientation space.

Key Features:
- 2D PMF heatmaps (θ vs ψ, θ vs φ)
- 3D surface plots
- Torque vector fields
- Contour plots with minima/saddle points
- Landscape analysis

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")


def plot_pmf_heatmap(pmf_values: np.ndarray,
                    theta_bins: np.ndarray,
                    psi_bins: np.ndarray,
                    vmax: Optional[float] = None,
                    mark_minima: bool = True,
                    figsize: Tuple[float, float] = (10, 8),
                    save_path: Optional[str] = None) -> None:
    """
    Plot 2D PMF heatmap.

    Args:
        pmf_values: (n_theta, n_psi) PMF values in kcal/mol
        theta_bins: (n_theta,) θ bin centers
        psi_bins: (n_psi,) ψ bin centers
        vmax: Maximum value for colorbar (clips high energies)
        mark_minima: Mark local minima with points
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> plot_pmf_heatmap(pmf, theta_bins, psi_bins, vmax=10)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # Clip high energies for better visualization
    if vmax is None:
        vmax = np.percentile(pmf_values[np.isfinite(pmf_values)], 95)

    pmf_plot = np.clip(pmf_values, 0, vmax)

    # Create meshgrid for pcolormesh
    theta_grid, psi_grid = np.meshgrid(theta_bins, psi_bins, indexing='ij')

    # Plot heatmap
    im = ax.pcolormesh(psi_grid, theta_grid, pmf_plot,
                       cmap='seismic', shading='auto', vmin=0, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PMF (kcal/mol)', fontsize=12)

    # Mark minima
    if mark_minima:
        minima = _find_local_minima(pmf_values)
        for i, j in minima:
            ax.plot(np.degrees(psi_bins[j]), np.degrees(theta_bins[i]),
                   'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)

    ax.set_xlabel('ψ (degrees)', fontsize=12)
    ax.set_ylabel('θ (degrees)', fontsize=12)
    ax.set_title('Potential of Mean Force', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()


def plot_pmf_contour(pmf_values: np.ndarray,
                    theta_bins: np.ndarray,
                    psi_bins: np.ndarray,
                    n_levels: int = 15,
                    vmax: Optional[float] = None,
                    figsize: Tuple[float, float] = (10, 8),
                    save_path: Optional[str] = None) -> None:
    """
    Plot PMF as contour plot.

    Args:
        pmf_values: (n_theta, n_psi) PMF values
        theta_bins: (n_theta,) θ bin centers
        psi_bins: (n_psi,) ψ bin centers
        n_levels: Number of contour levels
        vmax: Maximum value for contours
        figsize: Figure size
        save_path: Optional save path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    if vmax is None:
        vmax = np.percentile(pmf_values[np.isfinite(pmf_values)], 95)

    # Create contour levels
    levels = np.linspace(0, vmax, n_levels)

    theta_grid, psi_grid = np.meshgrid(theta_bins, psi_bins, indexing='ij')

    # Filled contours
    contourf = ax.contourf(np.rad2deg(psi_grid), np.rad2deg(theta_grid), pmf_values,
                          levels=levels, cmap='coolwarm', extend='max')

    # Contour lines
    contour = ax.contour(np.degrees(psi_grid), np.degrees(theta_grid), pmf_values,
                        levels=levels, colors='black', linewidths=0.5, alpha=0.3)

    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('PMF (kcal/mol)', fontsize=12)

    ax.set_xlabel('ψ (degrees)', fontsize=12)
    ax.set_ylabel('θ (degrees)', fontsize=12)
    ax.set_title('PMF Contour Plot', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()


def plot_pmf_3d_surface(pmf_values: np.ndarray,
                        theta_bins: np.ndarray,
                        psi_bins: np.ndarray,
                        vmax: Optional[float] = None,
                        figsize: Tuple[float, float] = (12, 10),
                        save_path: Optional[str] = None) -> None:
    """
    Plot 3D surface of PMF.

    Args:
        pmf_values: (n_theta, n_psi) PMF values
        theta_bins: (n_theta,) θ bins
        psi_bins: (n_psi,) ψ bins
        vmax: Maximum z value (clips high barriers)
        figsize: Figure size
        save_path: Optional save path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if vmax is None:
        vmax = np.percentile(pmf_values[np.isfinite(pmf_values)], 95)

    pmf_plot = np.clip(pmf_values, 0, vmax)

    theta_grid, psi_grid = np.meshgrid(theta_bins, psi_bins, indexing='ij')

    # Surface plot
    surf = ax.plot_surface(np.degrees(psi_grid), np.degrees(theta_grid), pmf_plot,
                           cmap='viridis', edgecolor='none', alpha=0.9)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('PMF (kcal/mol)', fontsize=11)

    ax.set_xlabel('ψ (degrees)', fontsize=10)
    ax.set_ylabel('θ (degrees)', fontsize=10)
    ax.set_zlabel('PMF (kcal/mol)', fontsize=10)
    ax.set_title('3D PMF Surface', fontsize=12, fontweight='bold')

    ax.view_init(elev=30, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()


def plot_torque_vector_field(torque_field: Callable,
                             theta_range: Tuple[float, float] = (0, np.pi),
                             psi_range: Tuple[float, float] = (0, 2*np.pi),
                             n_grid: int = 10,
                             figsize: Tuple[float, float] = (10, 8),
                             save_path: Optional[str] = None) -> None:
    """
    Plot torque vector field on (θ, ψ) plane.

    Args:
        torque_field: Function torque_field(theta, psi) -> (τ_θ, τ_ψ)
        theta_range: (min, max) for θ
        psi_range: (min, max) for ψ
        n_grid: Grid resolution
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> def tau_field(theta, psi):
        ...     tau_theta = -2.0 * np.sin(2*theta)
        ...     tau_psi = 0.0
        ...     return tau_theta, tau_psi
        >>> plot_torque_vector_field(tau_field)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # Create grid
    theta = np.linspace(theta_range[0], theta_range[1], n_grid)
    psi = np.linspace(psi_range[0], psi_range[1], n_grid)
    THETA, PSI = np.meshgrid(theta, psi, indexing='ij')

    # Compute torque at each point
    TAU_THETA = np.zeros_like(THETA)
    TAU_PSI = np.zeros_like(PSI)

    for i in range(n_grid):
        for j in range(n_grid):
            tau_theta, tau_psi = torque_field(THETA[i, j], PSI[i, j])
            TAU_THETA[i, j] = tau_theta
            TAU_PSI[i, j] = tau_psi

    # Plot vector field
    ax.quiver(PSI, THETA,
             TAU_PSI, TAU_THETA,
             scale=100, scale_units='xy', alpha=0.7)

    ax.set_xlabel('ψ (degrees)', fontsize=12)
    ax.set_ylabel('θ (degrees)', fontsize=12)
    ax.set_title('Torque ', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()


def plot_free_energy_landscape(pmf_values: np.ndarray,
                               theta_bins: np.ndarray,
                               psi_bins: np.ndarray,
                               trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                               figsize: Tuple[float, float] = (12, 10),
                               save_path: Optional[str] = None) -> None:
    """
    Combined visualization: PMF + trajectory overlay.

    Args:
        pmf_values: (n_theta, n_psi) PMF
        theta_bins: θ bins
        psi_bins: ψ bins
        trajectory: Optional (theta_traj, psi_traj) to overlay
        figsize: Figure size
        save_path: Optional save path
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # PMF heatmap
    vmax = np.percentile(pmf_values[np.isfinite(pmf_values)], 95)
    pmf_plot = np.clip(pmf_values, 0, vmax)

    theta_grid, psi_grid = np.meshgrid(theta_bins, psi_bins, indexing='ij')

    im = ax.pcolormesh(np.degrees(psi_grid), np.degrees(theta_grid), pmf_plot,
                       cmap='viridis', shading='auto', alpha=0.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PMF (kcal/mol)', fontsize=12)

    # Overlay trajectory
    if trajectory is not None:
        theta_traj, psi_traj = trajectory
        ax.plot(np.degrees(psi_traj), np.degrees(theta_traj),
               'r-', linewidth=0.5, alpha=0.3, label='Trajectory')
        ax.plot(np.degrees(psi_traj[0]), np.degrees(theta_traj[0]),
               'go', markersize=8, label='Start', markeredgecolor='white')
        ax.plot(np.degrees(psi_traj[-1]), np.degrees(theta_traj[-1]),
               'rs', markersize=8, label='End', markeredgecolor='white')
        ax.legend()

    # Mark minima
    minima = _find_local_minima(pmf_values)
    for i, j in minima:
        ax.plot(np.degrees(psi_bins[j]), np.degrees(theta_bins[i]),
               'w*', markersize=15, markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlabel('ψ (degrees)', fontsize=12)
    ax.set_ylabel('θ (degrees)', fontsize=12)
    ax.set_title('Free Energy Landscape', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()


def _find_local_minima(values: np.ndarray,
                      threshold: float = 2.0) -> List[Tuple[int, int]]:
    """
    Find local minima in 2D array.

    Args:
        values: 2D array
        threshold: Only report minima below global_min + threshold

    Returns:
        minima: List of (i, j) indices
    """
    from scipy.ndimage import minimum_filter

    # Find local minima using minimum filter
    minima_mask = (values == minimum_filter(values, size=3))

    # Only keep significant minima
    global_min = np.min(values[np.isfinite(values)])
    significant = values < (global_min + threshold)

    minima_mask = minima_mask & significant

    # Get indices
    minima = np.argwhere(minima_mask)

    return [(int(i), int(j)) for i, j in minima]


if __name__ == '__main__':
    print("Surface Visualization Module")
    print("============================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.visualization.surfaces import plot_pmf_heatmap")
    print("from protein_orientation.analysis.pmf import compute_pmf_2d")
    print()
    print("# Compute PMF")
    print("pmf, theta_bins, psi_bins = compute_pmf_2d(euler_trajectory)")
    print()
    print("# Visualize")
    print("plot_pmf_heatmap(pmf, theta_bins, psi_bins, vmax=10)")
