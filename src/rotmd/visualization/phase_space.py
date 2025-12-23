"""
Phase Space Visualization (Upgraded)

This module provides proper phase space analysis for protein orientation dynamics.

**Key Features:**
- 2D phase portraits with density (hexbin, not scatter)
- Trajectory overlays on energy landscapes
- Vector field visualization from torques
- Poincaré sections for periodic dynamics
- Proper (angle, angular_velocity) representations

**Philosophy:**
Phase space plots should reveal:
1. Attractors (high-density regions)
2. Trajectories (time evolution)
3. Flow (vector fields)
4. Structure (separatrices, limit cycles)

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Optional, Tuple, Callable
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")


def plot_phase_portrait_2d(
    angle: np.ndarray,
    angular_velocity: np.ndarray,
    angle_label: str = 'θ',
    times: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    gridsize: int = 30,
    show_trajectory: bool = True,
    trajectory_alpha: float = 0.3,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot 2D phase portrait: (angle, angular_velocity) with density.

    This is the canonical phase space representation for 1D rotational dynamics.
    Uses hexbin to show density (attractors) + trajectory overlay for dynamics.

    Args:
        angle: (n_frames,) angle values in radians
        angular_velocity: (n_frames,) ω values in rad/ps
        angle_label: Label for angle (e.g., 'θ', 'ψ', 'φ')
        times: Optional (n_frames,) timestamps for trajectory coloring
        energy: Optional (n_frames,) energy for contour overlay
        gridsize: Hexbin grid resolution (higher = finer)
        show_trajectory: Overlay trajectory path
        trajectory_alpha: Transparency for trajectory
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> # Plot θ vs ω_θ phase portrait
        >>> plot_phase_portrait_2d(theta, omega_theta, angle_label='θ')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Density map (hexbin shows where trajectory spends time)
    hexbin = ax.hexbin(
        angle, angular_velocity,
        gridsize=gridsize,
        cmap='YlOrRd',
        mincnt=1,
        edgecolors='none',
        alpha=0.8,
        linewidths=0.2
    )

    # Colorbar for density
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Density (frames)', fontsize=11)

    # 2. Trajectory overlay (shows time evolution)
    if show_trajectory:
        if times is not None:
            # Color by time
            for i in range(len(angle) - 1):
                ax.plot(
                    angle[i:i+2], angular_velocity[i:i+2],
                    color=plt.cm.viridis(times[i] / times[-1]),
                    alpha=trajectory_alpha,
                    linewidth=1.5
                )
        else:
            # Single color
            ax.plot(angle, angular_velocity, 'b-', alpha=trajectory_alpha, linewidth=1)

    # 3. Labels and formatting
    ax.set_xlabel(f'{angle_label} (rad)', fontsize=13)
    ax.set_ylabel(f'ω_{angle_label[0]} (rad/ps)', fontsize=13)
    ax.set_title(f'Phase Portrait: ({angle_label}, ω_{angle_label[0]})',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Angle ticks in terms of π
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.close()


def plot_energy_phase_space(
    theta: np.ndarray,
    psi: np.ndarray,
    energy: np.ndarray,
    times: Optional[np.ndarray] = None,
    pmf: Optional[np.ndarray] = None,
    theta_bins: Optional[np.ndarray] = None,
    psi_bins: Optional[np.ndarray] = None,
    n_contours: int = 8,
    figsize: Tuple[float, float] = (12, 9),
    save_path: Optional[str] = None
) -> None:
    """
    Plot trajectory in (θ, ψ) space overlaid on energy landscape.

    Shows how the protein explores the energy surface over time.

    Args:
        theta: (n_frames,) θ angles in radians
        psi: (n_frames,) ψ angles in radians
        energy: (n_frames,) total energy in kcal/mol
        times: Optional (n_frames,) timestamps for trajectory coloring
        pmf: Optional (n_theta, n_psi) PMF for background contours
        theta_bins: Optional PMF bin centers for θ
        psi_bins: Optional PMF bin centers for ψ
        n_contours: Number of contour levels
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> plot_energy_phase_space(theta, psi, Etot, pmf=pmf_data)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)


    ax.contourf(
        psi_bins,
        theta_bins,
        pmf,
        levels=n_contours,
        cmap='bone',
        alpha=1.0
    )

    # Contour lines
    ax.contour(
        psi_bins,
        theta_bins,
        pmf,
        levels=n_contours,
        colors='black',
        alpha=0.8,
        linewidths=1.5
    )

# 2. Trajectory colored by time and/or energy if both are available
    scatter_time = ax.scatter(
        np.rad2deg(psi), np.rad2deg(theta),
        c=times,
        cmap='terrain',
        s=14,
        alpha=0.5,
        edgecolors='red',
        linewidths=0.5,
        label='Trajectory (Time)'
    )
    cbar_time = plt.colorbar(scatter_time, ax=ax, orientation='vertical')
    cbar_time.set_label('Time (ps)', fontsize=11)

    # Overlay: slightly larger, lower alpha, energy-colored
    scatter_energy = ax.scatter(
        np.rad2deg(psi), np.rad2deg(theta),
        c=energy[1::],
        cmap='autumn',
        s=10,
        alpha=0.15,
        edgecolors='white',
        linewidths=0.3,
        label='Trajectory (Energy)'
    )
    # Optional: Add a second colorbar for energy on a new axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="6%", pad=0.42)
    cbar_energy = plt.colorbar(scatter_energy, cax=cax, orientation='horizontal')
    # Move the colorbar axis label to the right and reduce tick label size for aesthetics
    cbar_energy.ax.xaxis.set_label_position('top')
    cbar_energy.ax.xaxis.set_ticks_position('top')
    cbar_energy.set_label('Energy (kcal/mol)', fontsize=11)


    # 3. Labels and formatting
    ax.set_xlabel('ψ (degrees)', fontsize=13)
    ax.set_ylabel('θ (degrees)', fontsize=13)
    ax.grid(True, alpha=0.2, linestyle='--')

    # Set standard angle ranges
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 180)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.close()


def plot_L_phase_space(
    L_parallel: np.ndarray,
    L_perp: np.ndarray,
    times: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    gridsize: int = 30,
    show_trajectory: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot angular momentum phase space: L_∥ vs L_⊥ with vector field.

    Phase space is a vector field - shows displacement vectors (dL_∥/dt, dL_⊥/dt)
    at each state, revealing flow, attractors, and dynamical structure.

    Args:
        L_parallel: (n_frames,) magnitude of L_∥ in amu·Å²/ps
        L_perp: (n_frames,) magnitude of L_⊥ in amu·Å²/ps
        times: Optional timestamps for trajectory coloring
        energy: Optional energy for coloring
        gridsize: Not used (kept for API compatibility)
        show_trajectory: Overlay trajectory with flow arrows
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> plot_L_phase_space(L['L_parallel_mag'], L['L_perp_mag'], times=times)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Compute velocity field: (dL_∥/dt, dL_⊥/dt)
    if len(L_parallel) > 1:
        # Finite differences for flow vectors
        dL_par = np.diff(L_parallel)
        dL_perp = np.diff(L_perp)
        dt = np.diff(times) if times is not None else np.ones(len(dL_par))

        # Velocity field
        v_par = dL_par / dt
        v_perp = dL_perp / dt

        # Mid-points for vector placement
        L_par_mid = (L_parallel[:-1] + L_parallel[1:]) / 2
        L_perp_mid = (L_perp[:-1] + L_perp[1:]) / 2

        # 2. Plot flow field with arrows (subsample for clarity)
        stride = max(1, len(L_par_mid) // 50)  # ~50 arrows max

        if times is not None:
            # Color by time
            colors = times[:-1:stride]
            norm = Normalize(vmin=times.min(), vmax=times.max())
            quiver = ax.quiver(
                L_par_mid[::stride], L_perp_mid[::stride],
                v_par[::stride], v_perp[::stride],
                colors, cmap='viridis', norm=norm,
                scale=None, scale_units='xy', angles='xy',
                alpha=0.7, width=0.004, headwidth=4, headlength=5
            )
            cbar = plt.colorbar(quiver, ax=ax)
            cbar.set_label('Time (ps)', fontsize=11)
        elif energy is not None:
            # Color by energy
            colors = energy[:-1:stride]
            norm = Normalize(vmin=energy.min(), vmax=energy.max())
            quiver = ax.quiver(
                L_par_mid[::stride], L_perp_mid[::stride],
                v_par[::stride], v_perp[::stride],
                colors, cmap='coolwarm', norm=norm,
                scale=None, scale_units='xy', angles='xy',
                alpha=0.7, width=0.004, headwidth=4, headlength=5
            )
            cbar = plt.colorbar(quiver, ax=ax)
            cbar.set_label('Energy (kcal/mol)', fontsize=11)
        else:
            # Single color
            ax.quiver(
                L_par_mid[::stride], L_perp_mid[::stride],
                v_par[::stride], v_perp[::stride],
                color='blue', scale=None, scale_units='xy', angles='xy',
                alpha=0.6, width=0.004, headwidth=4, headlength=5
            )

        # 3. Overlay trajectory path (faint line)
        ax.plot(L_parallel, L_perp, 'k-', alpha=0.15, linewidth=0.8, zorder=1)

    # 4. Add ratio reference lines: L_∥/L_⊥ = r
    L_par_max = L_parallel.max()
    L_perp_max = L_perp.max()
    max_val = max(L_par_max, L_perp_max)

    for ratio in [0.5, 1.0, 2.0]:
        L_perp_line = np.linspace(0, max_val, 100)
        L_par_line = ratio * L_perp_line
        ax.plot(L_par_line, L_perp_line, 'k--', alpha=0.25, linewidth=0.8, zorder=0)
        if ratio * max_val * 0.92 < max_val:
            ax.text(ratio * max_val * 0.92, max_val * 0.92, f'{ratio:.1f}',
                   fontsize=9, alpha=0.5, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    # 5. Labels
    ax.set_xlabel('L_∥ (spin) [amu·Å²/ps]', fontsize=13)
    ax.set_ylabel('L_⊥ (nutation) [amu·Å²/ps]', fontsize=13)
    ax.set_title('Angular Momentum Phase Space (Flow Field)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.close()


def plot_phase_portrait_with_vector_field(
    angle: np.ndarray,
    angular_velocity: np.ndarray,
    torque: np.ndarray,
    moment_of_inertia: float,
    angle_label: str = 'θ',
    times: Optional[np.ndarray] = None,
    n_arrows: int = 20,
    gridsize: int = 25,
    figsize: Tuple[float, float] = (12, 9),
    save_path: Optional[str] = None
) -> None:
    """
    Plot phase portrait with vector field showing dynamics.

    Vector field shows:
    - d(angle)/dt = ω (horizontal component)
    - d(ω)/dt = τ/I (vertical component)

    This reveals flow direction, attractors, and repellers.

    Args:
        angle: (n_frames,) angle in radians
        angular_velocity: (n_frames,) ω in rad/ps
        torque: (n_frames,) torque component in kcal/mol
        moment_of_inertia: I in amu·Å²
        angle_label: Label for angle
        times: Optional timestamps
        n_arrows: Grid resolution for vector field
        gridsize: Hexbin resolution
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> plot_phase_portrait_with_vector_field(
        ...     theta, omega_theta, tau_theta, I_theta
        ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Background density
    hexbin = ax.hexbin(
        angle, angular_velocity,
        gridsize=gridsize,
        cmap='coolwarm',
        mincnt=1,
        alpha=0.3
    )

    # 2. Vector field
    # Create grid
    angle_range = [angle.min(), angle.max()]
    omega_range = [angular_velocity.min(), angular_velocity.max()]

    theta_grid = np.linspace(angle_range[0], angle_range[1], n_arrows)
    omega_grid = np.linspace(omega_range[0], omega_range[1], n_arrows)
    THETA, OMEGA = np.meshgrid(theta_grid, omega_grid)

    # Interpolate torque onto grid (simple nearest-neighbor)
    # For proper vector field, you'd compute torque as function of (θ, ω)
    # Here we use average torque as approximation

    # Dynamics: dθ/dt = ω, dω/dt = τ/I
    U = OMEGA  # d(angle)/dt = ω

    # Convert torque from kcal/mol to proper units for angular acceleration
    # 1 kcal/mol = 4.184 kJ/mol = 6.947e-21 J = 6.947e-21 kg·m²/s²
    # Torque in amu·Å²/ps² = torque_kcal * 6.947e-21 / (1.66054e-27 * 1e-20)
    # Simplified: just use mean torque for now
    mean_torque = np.mean(torque)
    V = np.ones_like(OMEGA) * (mean_torque / moment_of_inertia)  # dω/dt = τ/I

    # Normalize arrows
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-10)
    V_norm = V / (magnitude + 1e-10)

    # Plot vector field
    ax.quiver(THETA, OMEGA, U_norm, V_norm,
             magnitude, cmap='seismic', alpha=0.3, scale=8, width=0.004)

    # 3. Trajectory overlay
    if times is not None:
        for i in range(len(angle) - 1):
            ax.plot(
                angle[i:i+2], angular_velocity[i:i+2],
                color='olive',
                alpha=0.2,
                linewidth=0.3
            )

    # 4. Labels
    ax.set_xlabel(f'{angle_label} (rad)', fontsize=13)
    ax.set_ylabel(f'ω_{angle_label[0]} (rad/ps)', fontsize=13)
    ax.set_title(f'Torque field for spin/nutation components', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.close()


def plot_poincare_section_improved(
    euler_angles: np.ndarray,
    angular_velocities: np.ndarray,
    section_angles: list = [0, np.pi/2, np.pi, 3*np.pi/2],
    tolerance: float = 0.05,
    gridsize: int = 30,
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot Poincaré sections with multiple crossing planes.

    Records (θ, ω_θ) whenever φ crosses specified angles.
    Uses hexbin to show structure (periodic orbits, chaos, attractors).

    Args:
        euler_angles: (n_frames, 3) Euler angles (φ, θ, ψ)
        angular_velocities: (n_frames, 3) angular velocities
        section_angles: List of φ values to take sections at
        tolerance: Crossing detection tolerance
        gridsize: Hexbin resolution
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> plot_poincare_section_improved(euler, omega)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    n_sections = len(section_angles)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    phi = euler_angles[1:-1:, 0]
    theta = euler_angles[1:-1:, 1]
    omega_theta = angular_velocities[:, 1]

    for idx, section_angle in enumerate(section_angles[:4]):
        ax = axes[idx]

        # Find crossings
        crossings = np.abs(phi - section_angle) < tolerance

        if np.sum(crossings) < 10:
            ax.text(0.5, 0.5, f'Insufficient crossings\n(n={np.sum(crossings)})',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'φ = {section_angle:.2f} rad')
            continue

        theta_cross = theta[crossings]
        omega_cross = omega_theta[crossings]

        # Hexbin for density
        hexbin = ax.hexbin(
            theta_cross, omega_cross,
            gridsize=gridsize,
            cmap='YlOrRd',
            mincnt=1
        )

        ax.set_xlabel('θ (rad)', fontsize=11)
        ax.set_ylabel('ω_θ (rad/ps)', fontsize=11)
        ax.set_title(f'Poincaré Section: φ = {section_angle:.2f} rad\n(n={np.sum(crossings)} crossings)',
                    fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.close()


def plot_multi_panel_summary(
    theta: np.ndarray,
    psi: np.ndarray,
    omega_theta: np.ndarray,
    omega_psi: np.ndarray,
    energy: np.ndarray,
    times: np.ndarray,
    gridsize: int = 25,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None
) -> None:
    """
    Multi-panel summary of phase space dynamics.

    Layout:
    - Top left: (θ, ω_θ) phase portrait
    - Top right: (ψ, ω_ψ) phase portrait
    - Bottom left: (θ, ψ) trajectory
    - Bottom right: Energy timeseries

    Args:
        theta: (n_frames,) θ angles
        psi: (n_frames,) ψ angles
        omega_theta: (n_frames,) ω_θ values
        omega_psi: (n_frames,) ω_ψ values
        energy: (n_frames,) total energy
        times: (n_frames,) timestamps
        gridsize: Hexbin resolution
        figsize: Figure size
        save_path: Optional save path

    Example:
        >>> plot_multi_panel_summary(theta, psi, omega_theta, omega_psi, Etot, times)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: (θ, ω_θ)
    ax1 = fig.add_subplot(gs[0, 0])
    hexbin1 = ax1.hexbin(theta, omega_theta, gridsize=gridsize, cmap='YlOrRd', mincnt=1)
    ax1.set_xlabel('θ (rad)', fontsize=12)
    ax1.set_ylabel('ω_θ (rad/ps)', fontsize=12)
    ax1.set_title('Phase Portrait: (θ, ω_θ)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel 2: (ψ, ω_ψ)
    ax2 = fig.add_subplot(gs[0, 1])
    hexbin2 = ax2.hexbin(psi, omega_psi, gridsize=gridsize, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('ψ (rad)', fontsize=12)
    ax2.set_ylabel('ω_ψ (rad/ps)', fontsize=12)
    ax2.set_title('Phase Portrait: (ψ, ω_ψ)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: (θ, ψ) trajectory
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(np.degrees(psi), np.degrees(theta),
                         c=times, cmap='viridis', s=5, alpha=0.6)
    cbar3 = plt.colorbar(scatter, ax=ax3)
    cbar3.set_label('Time (ps)', fontsize=10)
    ax3.set_xlabel('ψ (degrees)', fontsize=12)
    ax3.set_ylabel('θ (degrees)', fontsize=12)
    ax3.set_title('Trajectory: (θ, ψ)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 360)
    ax3.set_ylim(0, 180)

    # Panel 4: Energy timeseries
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times, energy, linewidth=1, alpha=0.7, color='darkblue')
    ax4.fill_between(times, energy, alpha=0.2, color='blue')
    ax4.set_xlabel('Time (ps)', fontsize=12)
    ax4.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax4.set_title('Energy Evolution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add statistics
    mean_E = np.mean(energy)
    std_E = np.std(energy)
    ax4.axhline(mean_E, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_E:.2f} ± {std_E:.2f} kcal/mol')
    ax4.legend(fontsize=10)

    plt.suptitle('Phase Space Summary', fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.close()


if __name__ == '__main__':
    print("Phase Space Visualization Module (Upgraded)")
    print("=" * 50)
    print()
    print("Key improvements:")
    print("  1. Hexbin density plots (not scatter)")
    print("  2. Trajectory overlays on energy landscapes")
    print("  3. Vector field visualization")
    print("  4. Improved Poincaré sections")
    print("  5. Multi-panel summaries")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.visualization.phase_space import \\")
    print("    plot_phase_portrait_2d, plot_energy_phase_space, plot_L_phase_space")
    print()
    print("# 2D phase portrait with density")
    print("plot_phase_portrait_2d(theta, omega_theta, angle_label='θ')")
    print()
    print("# Trajectory on energy landscape")
    print("plot_energy_phase_space(theta, psi, Etot, pmf=pmf_data)")
    print()
    print("# Angular momentum phase space")
    print("plot_L_phase_space(L['L_parallel_mag'], L['L_perp_mag'])")
