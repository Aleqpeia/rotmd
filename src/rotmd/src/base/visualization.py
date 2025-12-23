#!/usr/bin/python
"""
Visualization Tools

This module provides plotting functions for protein orientation analysis,
including free energy landscapes, time series, and distributions.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from .orientation_analyzer import OrientationData, FreeEnergyLandscape


def plot_free_energy_landscape(
    landscape: FreeEnergyLandscape,
    output_file: Optional[Path] = None,
    vmax: Optional[float] = None,
    show_minima: bool = True
) -> Figure:
    """
    Plot 2D free energy landscape.

    Supports both F(θ, φ) and F(θ, ב) landscapes.

    Args:
        landscape: FreeEnergyLandscape object
        output_file: Output file path (if None, don't save)
        vmax: Maximum value for color scale (kcal/mol)
        show_minima: Whether to mark energy minima

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create meshgrid
    theta_grid, second_var_grid = np.meshgrid(
        landscape.theta_bin_centers,
        landscape.second_var_bin_centers,
        indexing='ij'
    )

    # Plot contour
    if vmax is None:
        vmax = min(10.0, np.percentile(landscape.free_energy, 95))

    contourf = ax.contourf(
        second_var_grid, theta_grid, landscape.free_energy,
        levels=20,
        cmap='inferno',
        vmin=0,
        vmax=vmax
    )   

    # Mark minima if requested
    if show_minima:
        minima = landscape.find_minima()
        if minima:
            theta_min = [m['theta'] for m in minima]
            # Get second variable from minima dict
            second_var_key = landscape.second_var_name.lower()
            second_var_min = [m[second_var_key] for m in minima]

            ax.scatter(
                second_var_min, theta_min,
                c='red',
                marker='x',
                s=200,
                linewidths=3,
                label='Energy minima',
                zorder=5
            )

            # Annotate minima
            for i, minimum in enumerate(minima[:3]):  # Show top 3
                annotation_text = f"{i+1}: θ={minimum['theta']:.1f}°"
                if landscape.landscape_type == 'theta_sasa':
                    annotation_text += f"\nSASA={minimum[second_var_key]:.0f}"
                ax.annotate(
                    annotation_text,
                    xy=(minimum[second_var_key], minimum['theta']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )

    # Labels and formatting - adapt based on landscape type
    if landscape.landscape_type == 'theta_sasa':
        ax.set_xlabel(f'{landscape.second_var_name} ({landscape.second_var_unit})', fontsize=12)
        title = f'Free Energy Landscape F(θ, {landscape.second_var_name})'
    else:  # theta_phi
        ax.set_xlabel(f'{landscape.second_var_name} ({landscape.second_var_unit})', fontsize=12)
        title = 'Free Energy Landscape F(θ, φ)'

    ax.set_ylabel('Tilt angle θ (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Free energy (kcal/mol)', fontsize=12)

    if show_minima and minima:
        ax.legend(loc='upper right')

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved free energy landscape to {output_file}")

    return fig


def plot_marginal_distributions(
    landscape: FreeEnergyLandscape,
    output_file: Optional[Path] = None
) -> Figure:
    """
    Plot marginal distributions F(θ) and F(second_var).

    For theta_sasa: plots F(θ) and F(SASA)
    For theta_phi: plots F(θ) and F(φ)

    Args:
        landscape: FreeEnergyLandscape object
        output_file: Output file path (if None, don't save)

    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # F(θ) - Tilt angle distribution
    theta_centers, f_theta = landscape.get_marginal_theta()
    ax1.plot(theta_centers, f_theta, 'b-', linewidth=2)
    ax1.fill_between(theta_centers, 0, f_theta, alpha=0.3)
    ax1.set_xlabel('Tilt angle θ (degrees)', fontsize=12)
    ax1.set_ylabel('Free energy (kcal/mol)', fontsize=12)
    ax1.set_title('Marginal Distribution F(θ)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 180)

    # Mark minima
    minima = landscape.find_minima()
    if minima:
        theta_min = [m['theta'] for m in minima]
        f_min = [m['free_energy'] for m in minima]
        ax1.scatter(theta_min, f_min, c='red', marker='o', s=100, zorder=5, label='Minima')
        ax1.legend()

    # F(second_var) - Second variable distribution
    second_var_centers, f_second_var = landscape.get_marginal_second_var()
    ax2.plot(second_var_centers, f_second_var, 'g-', linewidth=2)
    ax2.fill_between(second_var_centers, 0, f_second_var, alpha=0.3)

    if landscape.landscape_type == 'theta_sasa':
        ax2.set_xlabel(f'{landscape.second_var_name} ({landscape.second_var_unit})', fontsize=12)
        ax2.set_title(f'Marginal Distribution F({landscape.second_var_name})', fontsize=14, fontweight='bold')
    else:  # theta_phi
        ax2.set_xlabel(f'{landscape.second_var_name} ({landscape.second_var_unit})', fontsize=12)
        ax2.set_title('Marginal Distribution F(φ)', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 360)

    ax2.set_ylabel('Free energy (kcal/mol)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved marginal distributions to {output_file}")

    return fig


def plot_time_series(
    orientations: List[OrientationData],
    output_file: Optional[Path] = None
) -> Figure:
    """
    Plot time series of orientation angles and z-position.

    Args:
        orientations: List of OrientationData objects
        output_file: Output file path (if None, don't save)

    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Extract data
    times = np.array([o.time for o in orientations]) / 1000.0  # Convert ps to ns
    theta_values = np.array([o.theta for o in orientations])
    phi_values = np.array([o.phi for o in orientations])
    z_values = np.array([o.z for o in orientations])

    # Tilt angle vs time
    ax1.plot(times, theta_values, 'b-', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Tilt angle θ (degrees)', fontsize=12)
    ax1.set_title('Orientation Time Series', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 180)

    # Add running average
    window = min(50, len(theta_values) // 10)
    if window > 1:
        theta_smooth = np.convolve(theta_values, np.ones(window)/window, mode='same')
        ax1.plot(times, theta_smooth, 'r-', linewidth=2, label=f'Running avg ({window} frames)')
        ax1.legend()

    # Rotation angle vs time
    ax2.plot(times, phi_values, 'g-', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Rotation angle φ (degrees)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 360)

    # Z-position vs time
    ax3.plot(times, z_values, 'purple', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Time (ns)', fontsize=12)
    ax3.set_ylabel('Z-distance from membrane (Å)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Membrane center')
    ax3.legend()

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved time series to {output_file}")

    return fig


def plot_tilt_angle_distribution(
    orientations: List[OrientationData],
    output_file: Optional[Path] = None,
    bins: int = 36
) -> Figure:
    """
    Plot histogram of tilt angles.

    Args:
        orientations: List of OrientationData objects
        output_file: Output file path (if None, don't save)
        bins: Number of histogram bins

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract tilt angles
    theta_values = np.array([o.theta for o in orientations])

    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        theta_values,
        bins=bins,
        range=(0, 180),
        density=True,
        alpha=0.7,
        color='blue',
        edgecolor='black'
    )

    # Add statistics
    mean_theta = np.mean(theta_values)
    std_theta = np.std(theta_values)
    median_theta = np.median(theta_values)

    ax.axvline(mean_theta, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_theta:.1f}°')
    ax.axvline(median_theta, color='green', linestyle='--', linewidth=2, label=f'Median: {median_theta:.1f}°')

    # Labels and formatting
    ax.set_xlabel('Tilt angle θ (degrees)', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'Tilt Angle Distribution\n(μ = {mean_theta:.1f}°, σ = {std_theta:.1f}°)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved tilt angle distribution to {output_file}")

    return fig


def plot_sasa_vs_tilt(
    orientations: List[OrientationData],
    output_file: Optional[Path] = None
) -> Optional[Figure]:
    """
    Plot SASA vs tilt angle correlation.

    Args:
        orientations: List of OrientationData objects
        output_file: Output file path (if None, don't save)

    Returns:
        Matplotlib Figure object or None if no SASA data
    """
    # Check if SASA data is available
    sasa_values = [o.sasa for o in orientations if o.sasa is not None]
    if not sasa_values:
        print("No SASA data available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    theta_values = np.array([o.theta for o in orientations if o.sasa is not None])
    sasa_values = np.array(sasa_values)

    # Scatter plot
    scatter = ax.scatter(
        theta_values,
        sasa_values,
        c=sasa_values,
        cmap='viridis',
        alpha=0.6,
        s=20
    )

    # Add trend line
    z = np.polyfit(theta_values, sasa_values, 1)
    p = np.poly1d(z)
    theta_range = np.linspace(theta_values.min(), theta_values.max(), 100)
    ax.plot(theta_range, p(theta_range), 'r--', linewidth=2, label='Linear fit')

    # Calculate correlation
    correlation = np.corrcoef(theta_values, sasa_values)[0, 1]

    # Labels and formatting
    ax.set_xlabel('Tilt angle θ (degrees)', fontsize=12)
    ax.set_ylabel('SASA (Ų)', fontsize=12)
    ax.set_title(f'SASA vs Tilt Angle\n(correlation: {correlation:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('SASA (Ų)', fontsize=12)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved SASA vs tilt plot to {output_file}")

    return fig


def generate_all_plots(
    orientations: List[OrientationData],
    landscape: FreeEnergyLandscape,
    output_dir: Path,
    prefix: str = "protein_orientation"
) -> Dict[str, Path]:
    """
    Generate all standard plots and save to output directory.

    Args:
        orientations: List of OrientationData objects
        landscape: FreeEnergyLandscape object
        output_dir: Output directory for plots
        prefix: Prefix for output file names

    Returns:
        Dictionary mapping plot type to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Free energy landscape
    output_files['landscape'] = output_dir / f"{prefix}_landscape.png"
    plot_free_energy_landscape(landscape, output_files['landscape'])
    plt.close()

    # Marginal distributions
    output_files['marginals'] = output_dir / f"{prefix}_marginals.png"
    plot_marginal_distributions(landscape, output_files['marginals'])
    plt.close()

    # Time series
    output_files['timeseries'] = output_dir / f"{prefix}_timeseries.png"
    plot_time_series(orientations, output_files['timeseries'])
    plt.close()

    # Tilt angle distribution
    output_files['distribution'] = output_dir / f"{prefix}_distribution.png"
    plot_tilt_angle_distribution(orientations, output_files['distribution'])
    plt.close()

    # SASA vs tilt (if available)
    if any(o.sasa is not None for o in orientations):
        output_files['sasa'] = output_dir / f"{prefix}_sasa_vs_tilt.png"
        plot_sasa_vs_tilt(orientations, output_files['sasa'])
        plt.close()

    print(f"\nGenerated {len(output_files)} plots in {output_dir}")

    return output_files
