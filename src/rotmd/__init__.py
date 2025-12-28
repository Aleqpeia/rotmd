"""
rotmd - Rotational Molecular Dynamics Analysis Toolkit

A high-performance toolkit for analyzing protein rotational dynamics from MD trajectories.
Optimized with numba, JAX, xarray, and includes progress tracking with tqdm.

Main Features:
- Euler angle extraction and orientation analysis
- Angular momentum and torque calculations
- PMF computation and free energy landscapes
- Rotational diffusion analysis
- Langevin dynamics modeling with JAX/optax
- Comprehensive visualization tools
- Fast numba-compiled numerical kernels
- xarray-based data management with metadata

Quick Start:
    >>> from rotmd import analyze_trajectory
    >>> results = analyze_trajectory('system.gro', 'traj.trr')

For CLI usage:
    $ rotmd analyze system.gro traj.trr -o results/

Author: Mykyta Bobylyow
Date: 2025
"""

__version__ = "0.1.0"
__author__ = "Mykyta Bobylyow"

import numpy as np

# Core functionality
from .core.trajectory import TrajectoryData, load_trajectory, validate_trajectory
from .core.inertia import inertia_tensor, principal_axes
from .core.orientation import (
    extract_orientation_trajectory,
    rotation_matrix_to_euler_zyz,
    euler_zyz_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    compute_angular_displacement,
    unwrap_euler_angles,
)

# Observables
# Note: angular_momentum, angular_velocity, and torque modules were consolidated
# into observables/unified.py for the JAX migration. Use compute_all_observables instead.
# from .observables.angular_momentum import (
#     compute_angular_momentum,
#     decompose_angular_momentum,
# )
# from .observables.angular_velocity import (
#     angular_velocity_from_rotation_matrices,
#     compute_angular_velocity_from_trajectory
# )
# from .observables.torque import (
#     compute_torque,
#     validate_euler_equation
# )
from .observables.unified import compute_all_observables
from .observables.diffusion import (
    mean_squared_angular_displacement,
    extract_diffusion_coefficient,
    analyze_diffusion,
)

# Analysis
from .analysis.correlations import autocorrelation_function, cross_correlation_function
from .analysis.friction import extract_friction_from_acf, orientation_dependent_friction
from .analysis.pmf import (
    compute_pmf_1d,
    compute_pmf_2d,
    jacobian_euler_angles,
    free_energy_difference,
)
from .analysis import membrane

# Models
from .models.langevin import LangevinIntegrator, AnisotropicLangevin
from .models.energy import ElectrostaticEnergy, TotalEnergy, HydrophobicEnergy

# I/O
from .io.gromacs import load_gromacs_trajectory, detect_trajectory_contents
from .io.output import (
    save_results_json,
    load_results_json,
    save_results_npz,
    load_results_npz,
)
from .io.plumed import (
    generate_plumed_input,
    read_colvar_file,
    PlumedConfig,
    PlumedWriter,
)

# Configuration
from .config import AnalysisConfig, load_config_with_overrides

# Utilities
from .utils import (
    bootstrap_confidence_interval,
    block_average,
    circular_mean,
    circular_std,
)


# High-level API function
def analyze_trajectory(
    topology: str,
    trajectory: str,
    selection: str = "protein",
    output_dir: str = "analysis_results",
    temperature: float = 300.0,
    verbose: bool = True,
    save_plots: bool = False,
):
    """
    High-level function to perform complete orientation analysis.

    This is a convenience function that runs the full analysis pipeline:
    1. Load trajectory
    2. Extract Euler angles
    3. Compute PMF
    4. Analyze diffusion
    5. Optionally generate plots

    Args:
        topology: Topology file (.gro, .pdb)
        trajectory: Trajectory file (.trr, .xtc)
        selection: Atom selection (default: 'protein')
        output_dir: Output directory for results
        temperature: Temperature in Kelvin
        verbose: Print progress messages
        save_plots: Generate and save visualization plots

    Returns:
        results: Dictionary with all analysis results

    Example:
        >>> from rotmd import analyze_trajectory
        >>> results = analyze_trajectory('system.gro', 'traj.trr')
        >>> print(f"Diffusion coefficient: {results['D_msad']:.3f} rad²/ps")
        >>> print(f"Mean nutation angle: {np.degrees(results['theta_mean']):.1f}°")

    See Also:
        - For more control, use individual functions from submodules
        - For CLI usage: `rotmd analyze ...`
    """
    from pathlib import Path

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    if verbose:
        print("=" * 70)
        print("Protein Orientation Analysis")
        print("=" * 70)

    # Step 1: Load trajectory
    if verbose:
        print("\n[1/4] Loading trajectory...")

    traj_data = load_gromacs_trajectory(
        topology, trajectory, selection=selection, verbose=verbose
    )

    # Step 2: Extract Euler angles
    if verbose:
        print("\n[2/4] Extracting Euler angles...")

    euler_angles = extract_orientation_trajectory(
        traj_data["positions"], traj_data["masses"]
    )

    if verbose:
        print(f"  ✓ Extracted {len(euler_angles)} frames")

    # Step 3: Compute PMF
    if verbose:
        print("\n[3/4] Computing PMF...")

    # Extract theta and psi in degrees
    _, theta, psi = euler_angles.T  # phi not needed for 2D PMF
    theta_deg = np.degrees(theta)
    psi_deg = np.degrees(psi)
    print(max(theta_deg), max(psi_deg))

    pmf_result = compute_pmf_2d(
        theta_deg, psi_deg, theta_bins=18, psi_bins=36, temperature=temperature
    )

    pmf = pmf_result["pmf"]
    theta_bins = pmf_result["theta_centers"]
    psi_bins = pmf_result["psi_centers"]

    # Step 4: Analyze diffusion
    if verbose:
        print("\n[4/4] Analyzing diffusion...")

    diffusion_results = analyze_diffusion(
        euler_angles, traj_data["times"], verbose=verbose
    )

    # Compile results
    results = {
        "n_frames": len(euler_angles),
        "euler_angles": euler_angles,
        "times": traj_data["times"],
        "theta_mean": np.mean(euler_angles[:, 1]),
        "theta_std": np.std(euler_angles[:, 1]),
        "pmf": pmf,
        "theta_bins": theta_bins,
        "psi_bins": psi_bins,
        **diffusion_results,
    }

    # Save results
    save_results_npz(
        {
            "euler": euler_angles,
            "times": traj_data["times"],
            "pmf": pmf,
            "theta_bins": theta_bins,
            "psi_bins": psi_bins,
        },
        str(out_path / "trajectory_data.npz"),
    )

    save_results_json(
        {k: v for k, v in results.items() if not isinstance(v, np.ndarray)},
        str(out_path / "results.json"),
    )

    if verbose:
        print(f"\n✓ Results saved to {output_dir}/")

    # Generate plots
    if save_plots:
        try:
            from .visualization.surfaces import plot_pmf_heatmap
            from .visualization.phase_space import plot_trajectory_with_states

            if verbose:
                print("\nGenerating plots...")

            plots_dir = out_path / "plots"
            plots_dir.mkdir(exist_ok=True)

            # PMF
            plot_pmf_heatmap(
                pmf, theta_bins, psi_bins, save_path=str(plots_dir / "pmf.png")
            )

            # Trajectory
            plot_trajectory_with_states(
                euler_angles[:, 1],  # theta
                traj_data["times"],
                ylabel="θ (rad)",
                save_path=str(plots_dir / "theta_trajectory.png"),
            )

            if verbose:
                print(f"  ✓ Plots saved to {plots_dir}/")

        except ImportError as e:
            if verbose:
                print(f"  ✗ Could not generate plots: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)

    return results


__all__ = [
    # Version
    "__version__",
    "__author__",
    # High-level API
    "analyze_trajectory",
    # Core
    "TrajectoryData",
    "load_trajectory",
    "validate_trajectory",
    "inertia_tensor",
    "principal_axes",
    "extract_orientation_trajectory",
    "rotation_matrix_to_euler_zyz",
    "euler_zyz_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "compute_angular_displacement",
    "unwrap_euler_angles",
    # Observables
    "compute_all_observables",  # Unified API (replaces individual functions)
    # 'compute_angular_momentum',  # Moved to unified.py
    # 'decompose_angular_momentum',  # Moved to unified.py
    # 'angular_velocity_from_rotation_matrices',  # Moved to unified.py
    # 'compute_angular_velocity_from_trajectory',  # Moved to unified.py
    # 'compute_torque',  # Moved to unified.py
    # 'validate_euler_equation',  # Moved to unified.py
    "mean_squared_angular_displacement",
    "extract_diffusion_coefficient",
    "analyze_diffusion",
    # Analysis
    "autocorrelation_function",
    "cross_correlation_function",
    "extract_friction_from_acf",
    "orientation_dependent_friction",
    "compute_pmf_1d",
    "compute_pmf_2d",
    "jacobian_euler_angles",
    "free_energy_difference",
    "membrane",
    # Models
    "LangevinIntegrator",
    "AnisotropicLangevin",
    "TotalEnergy",
    "ElectrostaticEnergy",
    "HydrophobicEnergy",
    # I/O
    "load_gromacs_trajectory",
    "detect_trajectory_contents",
    "save_results_json",
    "load_results_json",
    "save_results_npz",
    "load_results_npz",
    "generate_plumed_input",
    "read_colvar_file",
    "PlumedConfig",
    "PlumedWriter",
    # Configuration
    "AnalysisConfig",
    "load_config_with_overrides",
    # Utils
    "bootstrap_confidence_interval",
    "block_average",
    "circular_mean",
    "circular_std",
]
