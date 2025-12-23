"""
Command-Line Interface for Protein Orientation Analysis

Provides user-friendly CLI commands for common analysis workflows.

Commands:
- analyze: Full orientation analysis pipeline
- extract: Extract Euler angles from trajectory
- pmf: Compute PMF from Euler angles
- diffusion: Analyze rotational diffusion
- compare: Compare two systems

Author: Mykyta Bobylyow
Date: 2025
"""

from ast import Add
from functools import *
import sys
import argparse
import json
from pathlib import Path
import numpy as np

from protein_orientation import compute_angular_velocity_from_trajectory, extract_orientation_trajectory, principal_axes
from protein_orientation.base import membrane_interface
from protein_orientation.observables import compute_angular_momentum_trajectory, compute_dL_dt, compute_torque_trajectory
from protein_orientation.observables.torque import torque_field, torque_field_from_pmf
from protein_orientation.visualization.phase_space import plot_energy_landscape_trajectory
from protein_orientation.visualization.spectra import plot_power_spectrum
from .analysis import compute_pmf_2d, extract_friction_from_acf


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog='protein-orientation',
        description='Protein Orientation Analysis Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global flag for configuration file
    parser.add_argument(
        '-c', '--config',
        metavar='FILE',
        help='Configuration file (TOML/YAML/JSON). CLI args override config values.'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init-config command (NEW)
    init_config_parser = subparsers.add_parser(
        'init-config',
        help='Generate default configuration file template'
    )
    init_config_parser.add_argument(
        '--format',
        choices=['toml', 'yaml', 'json'],
        default='toml',
        help='Configuration format (default: toml)'
    )
    init_config_parser.add_argument(
        '-o', '--output',
        default='protein_orientation.toml',
        help='Output file path (default: protein_orientation.toml)'
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run full orientation analysis pipeline'
    )
    analyze_parser.add_argument('topology', help='Topology file (.gro, .pdb)')
    analyze_parser.add_argument('trajectory', help='Trajectory file (.trr, .xtc)')
    analyze_parser.add_argument('-o', '--output', default='analysis_results',
                               help='Output directory (default: analysis_results)')
    analyze_parser.add_argument('--start', type=int, default=0,
                               help='Start frame (default: 0)')
    analyze_parser.add_argument('--stop', type=int, default=None,
                               help='Stop frame (default: all)')
    analyze_parser.add_argument('--step', type=int, default=1,
                               help='Frame step (default: 1)')
    analyze_parser.add_argument('--selection', default='protein',
                               help='Atom selection (default: protein)')
    analyze_parser.add_argument('--temperature', type=float, default=300.0,
                               help='Temperature in K (default: 300)')
    analyze_parser.add_argument('--plots', action='store_true',
                               help='Generate visualization plots')

    # extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract Euler angles and optional parameters from trajectory'
    )
    extract_parser.add_argument('topology', help='Topology file')
    extract_parser.add_argument('trajectory', help='Trajectory file')
    extract_parser.add_argument('-d', '--dir', required=True,
                               help='Output directory')
    extract_parser.add_argument('-f', '--filename', required=True,
                               help='Output file name (.npz or .h5)')
    extract_parser.add_argument('--start', type=int, default=0)
    extract_parser.add_argument('--stop', type=int, default=None)
    extract_parser.add_argument('--step', type=int, default=1)
    extract_parser.add_argument('--selection', default='protein')
    extract_parser.add_argument('--include-structural', action='store_true',
                               help='Extract structural parameters (RMSD, Rg, etc.)')
    extract_parser.add_argument('--include-energetics', action='store_true',
                               help='Extract energetic parameters (KE, temp, etc.)')
    extract_parser.add_argument('--reference-frame', type=int, default=0,
                               help='Reference frame for RMSD (default: 0)')

    # plot command (comprehensive visualization)
    plot_parser = subparsers.add_parser(
        'plot',
        help='Generate comprehensive visualization from analysis results'
    )
    plot_parser.add_argument('result', help='Analysis results directory')
    plot_parser.add_argument('-o', '--output', default=None,
                            help='Output directory for plots (default: results_dir/plots)')
    plot_parser.add_argument('--pmf', action='store_true',
                            help='Plot PMF heatmaps and contours')
    plot_parser.add_argument('--phase-space', action='store_true',
                            help='Plot phase space portraits')
    plot_parser.add_argument('--energy', action='store_true',
                            help='Plot energy landscapes and timeseries')
    plot_parser.add_argument('--angular-momentum', action='store_true',
                            help='Plot angular momentum phase space')
    plot_parser.add_argument('--spectra', action='store_true',
                            help='Plot autocorrelation and spectral analysis')
    plot_parser.add_argument('--metrics', action='store_true',
                            help='Plot RMSD and Rg time series')
    plot_parser.add_argument('--bifurcation', action='store_true',
                            help='Plot Poincaré bifurcation diagrams vs energy')
    plot_parser.add_argument('--residue-energy', action='store_true',
                            help='Plot per-residue energy analysis')
    plot_parser.add_argument('--summary', action='store_true',
                            help='Generate multi-panel summary plot')
    plot_parser.add_argument('--all', action='store_true',
                            help='Generate all available plots')
    plot_parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                            help='Output format (default: png)')
    plot_parser.add_argument('--dpi', type=int, default=300,
                            help='DPI for raster formats (default: 300)')

    # pmf command (kept for backward compatibility)
    pmf_parser = subparsers.add_parser(
        'pmf',
        help='Compute PMF from Euler angles'
    )
    pmf_parser.add_argument('input', help='Input Euler angles file (.npz)')
    pmf_parser.add_argument('-o', '--output', required=True,
                           help='Output PMF file (.npz)')
    pmf_parser.add_argument('--temperature', type=float, default=300.0)
    pmf_parser.add_argument('--bins', type=int, nargs=3, default=[36, 18, 36],
                           help='Bins for phi, theta, psi (default: 36 18 36)')
    pmf_parser.add_argument('--plot', action='store_true',
                           help='Generate PMF visualization')

    # diffusion command
    diffusion_parser = subparsers.add_parser(
        'diffusion',
        help='Analyze rotational diffusion'
    )
    diffusion_parser.add_argument('input', help='Input Euler angles file (.npz)')
    diffusion_parser.add_argument('-o', '--output', required=True,
                                 help='Output results file (.json)')
    diffusion_parser.add_argument('--plot', action='store_true')

    # compare command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare two systems'
    )
    compare_parser.add_argument('system1', help='System 1 analysis results (.json)')
    compare_parser.add_argument('system2', help='System 2 analysis results (.json)')
    compare_parser.add_argument('-o', '--output', default='comparison',
                               help='Output directory')
    compare_parser.add_argument('--labels', nargs=2, default=['System 1', 'System 2'],
                               help='Labels for systems')

    # detailed-analysis command (NEW)
    detailed_parser = subparsers.add_parser(
        'detailed-analysis',
        help='Run detailed mutation analysis (per-residue energy, N75K motion, Poincaré sections)'
    )
    detailed_parser.add_argument('wt_data', help='Wild-type analysis data (.npz)')
    detailed_parser.add_argument('mut_data', help='Mutant analysis data (.npz)')
    detailed_parser.add_argument('-o', '--output', required=True,
                                help='Output directory for all plots')
    detailed_parser.add_argument('--per-residue', action='store_true',
                                help='Generate per-residue energy plots')
    detailed_parser.add_argument('--n75k-motion', action='store_true',
                                help='Generate N75K residue motion plots')
    detailed_parser.add_argument('--poincare', action='store_true',
                                help='Generate Poincaré bifurcation plots')
    detailed_parser.add_argument('--all', action='store_true',
                                help='Generate all detailed analysis plots')
    detailed_parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='pdf',
                                help='Output format (default: pdf)')
    detailed_parser.add_argument('--n75-residue-id', type=int, default=75,
                                help='Residue ID for N75K mutation site (default: 75)')

    return parser


def cmd_init_config(args):
    """Generate default configuration file template."""
    from .config import AnalysisConfig

    print("=" * 60)
    print("Generating Configuration Template")
    print("=" * 60)

    # Create default config
    config = AnalysisConfig()

    # Determine output path and format
    output_path = Path(args.output)

    # If user didn't specify extension, use format flag
    if output_path.suffix == '':
        output_path = output_path.with_suffix(f'.{args.format}')

    # Save config
    try:
        config.save(str(output_path))

        print(f"\n✓ Configuration template saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Edit {output_path} with your analysis parameters")
        print(f"2. Run analysis with: protein-orientation analyze TOPO TRAJ --config {output_path}")
        print(f"\nNote: CLI arguments will override config file values")

    except Exception as e:
        print(f"\n✗ Error saving config: {e}", file=sys.stderr)
        sys.exit(1)


def _load_config_with_overrides(args):
    """
    Load config file and apply CLI argument overrides.

    Override hierarchy: CLI args > config file > defaults

    Returns:
        config: AnalysisConfig instance with all overrides applied
    """
    from .config import AnalysisConfig

    # Load config if provided
    if hasattr(args, 'config') and args.config:
        config = AnalysisConfig.load(args.config)
        print(f"✓ Loaded config from: {args.config}")
    else:
        config = AnalysisConfig()

    # Apply CLI overrides for common parameters
    # Only override if CLI arg was explicitly set (not default)

    # Trajectory parameters
    if hasattr(args, 'start') and args.start != 0:
        config.set('trajectory.start', args.start)

    if hasattr(args, 'stop') and args.stop is not None:
        config.set('trajectory.stop', args.stop)

    if hasattr(args, 'step') and args.step != 1:
        config.set('trajectory.step', args.step)

    if hasattr(args, 'selection') and args.selection != 'protein':
        config.set('trajectory.selection', args.selection)

    # Analysis parameters
    if hasattr(args, 'temperature') and args.temperature != 300.0:
        config.set('analysis.temperature', args.temperature)

    if hasattr(args, 'bins'):
        config.set('analysis.pmf_bins', args.bins)

    # Output parameters
    if hasattr(args, 'output'):
        config.set('output.directory', args.output)

    # Visualization parameters
    if hasattr(args, 'dpi'):
        config.set('visualization.dpi', args.dpi)

    if hasattr(args, 'format'):
        # Note: 'format' here is plot format, not output format
        # Keep them separate in config
        pass

    return config


def cmd_analyze(args):
    """Run full analysis pipeline."""
    from .io.gromacs import load_gromacs_trajectory
    from .core.orientation import (extract_orientation_trajectory, 
                                   euler_zyz_to_rotation_matrix,
                                   rotation_matrix_to_quaternion, rotation_matrix_to_euler_zyz)
    from .core.inertia import (principal_axes, asymmetry_parameter,
                               inertia_tensor, parallel_axis_theorem)
    from .observables import (compute_angular_momentum_trajectory, compute_angular_velocity_from_trajectory,
                              compute_spin_nutation_ratio, compute_dL_dt,
                              compute_torque_trajectory, decompose_torque,
                              acylindricity, asphericity, analyze_diffusion,
                              decompose_angular_momentum, decompose_angular_velocity)
    from .analysis import (compute_pmf_2d, 
                           compute_pmf_6d_projection,
                           angular_momentum_acf,
                           angular_velocity_acf,
                           jacobian_euler_angles,
                           fit_exponential_decay,
                           free_energy_difference,
                           orientation_dependent_friction)
    from .models.energy import (TotalEnergy, ElectrostaticEnergy, HydrophobicEnergy) 
    from .models.langevin import (LangevinIntegrator, AnisotropicLangevin)
    from .io.output import save_results_json, save_results_npz

    print("=" * 60)
    print("Protein Orientation Analysis")
    print("=" * 60)

    # Load config with CLI overrides
    config = _load_config_with_overrides(args)

    # Create output directory
    output_dir = Path(config.get('output.directory'))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load trajectory
    print("\n[1/5] Loading trajectory...")
    traj_data = load_gromacs_trajectory(
        args.topology,
        args.trajectory,
        selection=config.get('trajectory.selection'),
        start=config.get('trajectory.start'),
        stop=config.get('trajectory.stop'),
        step=config.get('trajectory.step'),
        verbose=True
    )
#     structural_results = compute_structural_trajectory(
#         traj_data['positions'],
#         traj_data['masses'],
#         reference=reference,
#         verbose=True
#     )
    
    normal = traj_data['normal']
    I = traj_data['inertia_tensor']
    n_frames = len(traj_data['times'])
    axes = np.zeros((n_frames, 3, 3))
    moments = np.zeros((n_frames, 3))
    for i in range(n_frames):
        moments[i], axes[i] = principal_axes(I[i])

    # Compute angular momentum (corrected physics)
    print("\n[2/5] Computing angular momentum...")
    L = compute_angular_momentum_trajectory(
        positions_traj=traj_data['positions'],
        velocities_traj=traj_data['velocities'],
        masses=traj_data['masses'],
        principal_axes_traj=axes,
        membrane_normal=normal,
        verbose=True
    )
    print("="*60,
        traj_data['times'],
        "="*60)
    # Compute torque
    dLdt = compute_dL_dt(L['L'], traj_data['times'])

    Etot = traj_data['Etot']
    Epol = traj_data['Epol']
    Enonpol = traj_data['Enonpol']
    E_residues = traj_data['per_residue']

    # Extract Euler angles
    print("\n[3/5] Extracting Euler angles...")
    euler_angles, R = extract_orientation_trajectory(
        traj_data['positions'],
        traj_data['masses']
    )

    # Save Euler angles
    euler_file = output_dir / 'euler_angles.npz'
    save_results_npz({
        'euler': euler_angles,
        'times': traj_data['times']
    }, str(euler_file))

    # Compute PMF
    print("\n[4/5] Computing PMF...")
    # Extract theta and psi in degrees
    phi, theta, psi = euler_angles.T
    omega_lab, omega_body = compute_angular_velocity_from_trajectory(
        traj_data['positions'],
        traj_data['velocities'],
        traj_data['masses'],
        interval=args.step,
        verbose=True
    )
    torques = compute_torque_trajectory(
        traj_data['positions'],
        traj_data['forces'],
        traj_data['masses'],
        principal_axes_traj=axes,
        membrane_normal=normal
    )
    theta = np.deg2rad(np.degrees(theta))
    psi = np.deg2rad(np.degrees(psi))
    pmf_result = compute_pmf_2d(theta, psi)

    pmf_file = output_dir / 'pmf.npz'
    save_results_npz({
        'pmf': pmf_result['pmf'],
        'theta_bins': pmf_result['theta_centers'],
        'psi_bins': pmf_result['psi_centers'],
        'theta_edges': pmf_result['theta_edges'],
        'psi_edges': pmf_result['psi_edges']
    }, str(pmf_file))

    # Analyze diffusion
    print("\n[4/5] Analyzing diffusion...")
    diffusion_results = analyze_diffusion(
        euler_angles,
        traj_data['times'],
        verbose=True
    )
    # Save results
    results_file = output_dir / 'results.json'
    energies_dict = {'total energy': Etot, 'electrostatics': Epol, 'non-polar interactions': Enonpol}
    save_results_json(diffusion_results, str(results_file))

    # Generate plots
    if args.plots:
        print("\n[5/5] Generating plots...")
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        try:
            from .visualization.surfaces import (plot_pmf_heatmap, plot_free_energy_landscape,
                                                 plot_pmf_contour, plot_torque_vector_field)
            from .visualization.spectra import (plot_autocorrelation, 
                                                plot_friction_extraction,
                                                plot_multiple_acfs,
                                                plot_correlation_comparison,
                                                plot_spectral_density)
            from .visualization.phase_space import (plot_L_phase_space, plot_multi_panel_summary, plot_phase_portrait_2d, 
                                                    plot_phase_with_flow, plot_poincare_section)
            # for k, v in torques.items():
            #     if isinstance(v, np.ndarray) and v.shape == (n_frames, 3):
            #         plot_trajectory_with_states(
            #             theta,
            #             times=traj_data['times'],
            #             save_path=str(plots_dir / f'{k}_states.png')
            #         )
            #     else:
            #         continue

            # for k, v in omega_body.items():
            #     if isinstance(v, np.ndarray) and v.shape == (n_frames, 3):
            #         plot_angular_velocity_distribution(
            #             v,
            #             save_path=str(plots_dir / f'omega_{k}_dist.png')
            #         )
            #     else:
            #         continue
            # for k, v in energies_dict.items():
            #     if isinstance(v, np.ndarray) and v.shape == (n_frames,):
            #         plot_3d_phase_space(
            #             L['L_mag'], L['spin_nutation_ratio'], v, traj_data['times'],
            #             save_path=str(plots_dir / f'{k}_phase_space.png')
            #         )           
            #     else:
            #         continue
            # for k, v in energies_dict.items():
            #     if isinstance(v, np.ndarray) and v.shape == (n_frames,):
            #         plot_energy_vs_angular_momentum(v, L['L_mag'], traj_data['times'],
            #             save_path=str(plots_dir / f'{k}_energy_vs_angular_momentum.png')
            #         )
            #     else:
            #         continue
            for k, v in L.items():
                if isinstance(v, np.ndarray) and v.shape == (n_frames, 3):
                    acf = angular_momentum_acf(v, traj_data['times'])
                    plot_friction_extraction(
                        times=acf['lags'], 
                        acf=acf['acf'],
                        friction=extract_friction_from_acf(
                                times=acf['lags'],
                                acf=acf['acf'],
                                moment_of_inertia=np.mean(moments)
                            )['gamma'],
                        save_path=str(plots_dir / f'{k}_friction.png')
                    )
                else: 
                    continue
            # PMF heatmap
            plot_pmf_heatmap(
                pmf_result['pmf'],
                pmf_result['theta_centers'],
                pmf_result['psi_centers'],
                save_path=str(plots_dir / 'pmf_heatmap.png')
            )
            plot_pmf_contour(
                pmf_result['pmf'],
                pmf_result['theta_centers'],
                pmf_result['psi_centers'],
                save_path=str(plots_dir / 'pmf_contour.png')
            )
            plot_torque_vector_field(
                torque_field=torque_field_from_pmf(pmf_data=pmf_result['pmf']),
                save_path=str(plots_dir / 'tau_empirical_vfield.png')
            )
            plot_torque_vector_field(
                torque_field=torque_field,
                save_path=str(plots_dir / 'tau_analytical_vfield.png')
            )


            print(f"  ✓ Plots saved to {plots_dir}/")
        except ImportError:
            print("  ✗ Matplotlib not available, skipping plots")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def cmd_extract(args):
    """Extract Euler angles and optional structural/energetic parameters."""
    from .io.gromacs import load_gromacs_trajectory
    from .core.orientation import extract_orientation_trajectory
    from .io.output import save_trajectory_data, save_results_npz

    print("=" * 60)
    print("Parameter Extraction")
    print("=" * 60)

    # Load config with CLI overrides
    config = _load_config_with_overrides(args)

    # Determine what to extract
    extract_structural = args.include_structural

    # Load trajectory
    print("\n[1/4] Loading trajectory...")
    traj_data = load_gromacs_trajectory(
        args.topology,
        args.trajectory,
        selection=config.get('trajectory.selection'),
        start=config.get('trajectory.start'),
        stop=config.get('trajectory.stop'),
        step=config.get('trajectory.step'),
        verbose=True
    )
    normal = traj_data['normal']
    I = traj_data['inertia_tensor']
    n_frames = len(traj_data['times'])
    axes = np.zeros((n_frames, 3, 3))
    moments = np.zeros((n_frames, 3))
    for i in range(n_frames):
        moments[i], axes[i] = principal_axes(I[i])

    # Compute angular momentum (corrected physics)
    print("\n[2/5] Computing angular momentum...")
    L = compute_angular_momentum_trajectory(
        positions_traj=traj_data['positions'],
        velocities_traj=traj_data['velocities'],
        masses=traj_data['masses'],
        principal_axes_traj=axes,
        membrane_normal=normal,
        verbose=True
    )
    # Compute torque
    dLdt, dLtimepoints = compute_dL_dt(L['L'], traj_data['times'])
    print(f"  ✓ Loaded {n_frames} frames")

    # Extract Euler angles and rotation matrices
    print("\n[2/4] Extracting orientation (rotation matrices)...")
    euler_angles, R = extract_orientation_trajectory(
        traj_data['positions'],
        traj_data['masses']
    )
    phi, theta, psi = euler_angles.T
    print(f"  ✓ Extracted orientation for {len(R)} frames")
    print(f"     - Rotation matrices: {R.shape}")
    print(f"     - Euler angles: {euler_angles.shape}")
    omega_lab, omega_body = compute_angular_velocity_from_trajectory(
        traj_data['positions'],
        traj_data['velocities'],
        traj_data['masses'],
        interval=args.step,
        verbose=True
    )
    torques = compute_torque_trajectory(
        traj_data['positions'],
        traj_data['forces'],
        traj_data['masses'],
        principal_axes_traj=axes,
        membrane_normal=normal
    )
    # Prepare output data
    output_data = {
        'phi': phi,
        'theta': theta,
        'psi': psi,
        'rotation_matrices': R,
        'positions': traj_data['positions'],
        'masses': traj_data['masses'],
        'velocities': traj_data['velocities'],
        'forces': traj_data['forces'],
        'times': traj_data['times'],
        'inertia_tensor': I,
        'normal': normal,
        'axes': axes,
        'moments': moments,
        'n_atoms': traj_data['n_atoms'],
        'Etot': traj_data['Etot'],
        'Epol': traj_data['Epol'],
        'Enonpol': traj_data['Enonpol'],
        'per_residue': traj_data['per_residue']
    }

    # Extract structural parameters if requested
    if extract_structural:
        print("\n[3/4] Extracting structural parameters...")
        from .observables.structural import compute_structural_trajectory

        # Use specified reference frame
        ref_frame = args.reference_frame
        if ref_frame >= n_frames:
            print(f"  ⚠ Warning: reference frame {ref_frame} >= {n_frames}, using frame 0")
            ref_frame = 0

        reference = traj_data['positions'][ref_frame]

        structural_results = compute_structural_trajectory(
            traj_data['positions'],
            traj_data['masses'],
            reference=reference,
            verbose=True
        )
        # Add to output
        output_data.update(L)
        output_data.update(omega_body)
        output_data.update(omega_lab)
        output_data.update(torques)
        output_data.update({
            'rmsd': structural_results['rmsd'],
            'rg': structural_results['rg'],
            'rg_components': structural_results['rg_components'],
            'asphericity': structural_results['asphericity'],
            'acylindricity': structural_results['acylindricity'],
            'end_to_end': structural_results['end_to_end']
        })
    else:
        print("\n[3/4] Skipping structural parameters (use --include-structural)")

    # Save
    output_dir = Path(args.dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_filename = args.filename

    try:
        print(f"\n[4/4] Saving extracted data...")
        save_results_npz(output_data, str(output_dir), output_filename)
        print(f"  ✓ Saved to {output_dir}/")
    except Exception as e:
        print(f"  ✗ Could not save with redundant format: {e}")
        print(f"  → Falling back to single file...")

        # Fallback: save as single npz file
        output_path = output_dir / output_filename
        np.savez(
            output_path,
            **output_data
        )
        print(f"  ✓ Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"Output directory: {output_dir}")
    print("\nExtracted parameters:")
    print(f"  • Euler angles: {euler_angles.shape}")
    print(f"  • Rotation matrices: {R.shape}")
    print(f"  • Times: {traj_data['times'].shape}")
    print(f"  • Angular momentum: L with {len(L)} components")
    print(f"  • Energies: Etot, Epol, Enonpol")
    if extract_structural:
        print(f"  • Structural: RMSD, Rg, shape parameters")
    print("=" * 60)


def cmd_plot(args):
    """Generate comprehensive visualizations from analysis results."""
    from .io.output import load_results_npz
    from .visualization.phase_space import (
        plot_phase_portrait_2d,
        plot_L_phase_space,
        plot_contour_density,
        plot_poincare_section,
        plot_multi_panel_summary
    )
    from .visualization.surfaces import (
        plot_pmf_heatmap,
        plot_pmf_contour,
        plot_torque_vector_field
    )
    from .visualization.spectra import (
        plot_autocorrelation,
        plot_friction_extraction
    )
    from .observables.torque import torque_field, torque_field_from_pmf
    from .analysis import angular_momentum_acf, extract_friction_from_acf

    print("=" * 60)
    print("Comprehensive Visualization")
    print("=" * 60)

    # Setup paths
    result = Path(args.result)

    output_dir = Path(args.output) if args.output else args.result / 'plots'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine what to plot
    plot_all = args.all
    plot_pmf = args.pmf or plot_all
    plot_phase = args.phase_space or plot_all
    plot_energy = args.energy or plot_all
    plot_L = args.angular_momentum or plot_all
    plot_spectra = args.spectra or plot_all
    plot_metrics = args.metrics or plot_all
    plot_bifurcation = args.bifurcation or plot_all
    plot_residue = args.residue_energy 
    plot_summary = args.summary or plot_all

    # Set matplotlib DPI
    import matplotlib
    matplotlib.rcParams['savefig.dpi'] = args.dpi
    matplotlib.rcParams['savefig.format'] = args.format

    print(f"\nLoading data from {result}...")

    # Factory pattern: load .npz and create variables from keys
    npz_data = np.load(result, allow_pickle=True)
    print(npz_data.files)

    # Create namespace object that acts like a module with dict-like iteration
    class DataNamespace:
        """Simple namespace to hold loaded variables with dict-like items()."""
        def __init__(self, npz_file):
            # Auto-create attributes from all keys
            for key in npz_file.files:
                setattr(self, key, npz_file[key])
            self._keys = npz_file.files
            self._npz_file = npz_file  # Keep npz file for key-value access

        def has(self, *keys):
            """Check if all keys exist."""
            return all(k in self._keys for k in keys)

        def get(self, key, default=None):
            """Get attribute with default."""
            return getattr(self, key, default)

        def items(self):
            """Yield (key, value) pairs, like dict.items()."""
            for key in self._keys:
                yield key, getattr(self, key)

    # Create data namespace - now you can use data.phi, data.theta, etc.
    data = DataNamespace(npz_data)
    print(f"  ✓ Loaded {len(data._keys)} variables from npz file")
    def convert_theta_angle_mirrored(theta_old_deg: np.ndarray) -> np.ndarray:
        """
        Mirror theta: 160 -> 20, restrict to [0, 90] degrees.

        Args:
            theta_old_deg: Input theta angle(s) in degrees.
        Returns:
            Mirrored theta, with values clipped to [0, 90].
        """
        mirrored = np.abs(180.0 - theta_old_deg)
        # Restrict domain to [0, 90]
        mirrored_clipped = np.clip(mirrored, 0, 90)
        return mirrored_clipped
    # Reconstruct euler_angles from components if needed
    euler_angles = np.column_stack([data.phi, data.theta, data.psi])
    theta = convert_theta_angle_mirrored(np.rad2deg(data.theta))
    print(theta, np.rad2deg(data.theta))
    theta = np.deg2rad(theta)
    psi = data.psi
    phi = data.phi
    print(f"  ✓ Reconstructed euler_angles from (phi, theta, psi)")
    # Compute PMF
    print(f"  Computing PMF from {len(theta)} frames...")
    pmf_data = compute_pmf_2d(theta, psi, theta_bins=36, psi_bins=36)

    # Check availability of optional data
    L_available = data.has('L')
    E_available = data.has('Etot') or data.has('Epol', 'Enonpol')
    omega_available = data.has('omega_body') or data.has('omega_lab')

    if L_available:
        print(f"  ✓ Angular momentum available")
    if E_available:
        print(f"  ✓ Energy data available")
    if omega_available:
        print(f"  ✓ Angular velocity available")
    if data.has('moments'):
        print(f"  ✓ Moments of inertia available")

    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print(f"{'=' * 60}\n")

    plot_count = 0

    # 1. PMF Visualizations
    if plot_pmf:
        print("[1/9] PMF visualizations...")
        try:
            # Heatmap
            plot_pmf_heatmap(pmf_data['pmf'], pmf_data['theta_centers'], pmf_data['psi_centers'],
                save_path=str(output_dir / f'pmf_heatmap.{args.format}'))
            plot_count += 1

            # Contour
            plot_pmf_contour(pmf_data['pmf'], pmf_data['theta_centers'], pmf_data['psi_centers'],
                save_path=str(output_dir / f'pmf_contour.{args.format}'), n_levels=18)
            plot_count += 1

            plot_count += 1


            print(f"  ✓ Generated PMF plots")
        except Exception as e:
            print(f"  ✗ PMF plotting failed: {e}")

    # 2. Phase Space Portraits
    if plot_phase and omega_available:
        print("[2/9] Phase space portraits...")
        try:
            omega_theta = data.nutation_magnitude
            omega_psi = data.spin_magnitude
            omega = data.omega_body
            V = data.Etot
            I = data.moments
            torques = data.tau_total
            print(euler_angles.shape, omega.shape, 
                  V.shape, torques.shape, I.shape)
            # Standard phase portraits with density
            plot_phase_portrait_2d(euler_angles[1:-1:,1], omega[:,1], 
                angle_index=1, scalar_metric='potential',
                energy=V[1:-1], angle_label='θ',
                save_path=str(output_dir / f'phase_portrait_theta_test.{args.format}'))
            plot_phase_portrait_2d(euler_angles[1:-1:,2], omega[:,2], 
                angle_index=2, scalar_metric='potential', 
                energy=V[1:-1], angle_label='ψ',
                save_path=str(output_dir / f'phase_portrait_psi_test.{args.format}'))
            plot_count += 2

            # Phase portraits WITH vector fields (if torque data available)
            from .visualization.phase_space import plot_phase_portrait_with_vector_field

            # θ/θ' with torque field
            plot_phase_portrait_with_vector_field(
                euler_angles[1:-1:, 1], omega[:,1],
                torques=torques[1:-1:,1],
                moments_of_inertia=I[1:-1,1], angle_index=1, # Itheta
                angle_label='θ', component_label='θ',
                save_path=str(output_dir / f'phase_portrait_theta_attract.{args.format}'))

            # ψ/ψ' with torque field
            plot_phase_portrait_with_vector_field(
                euler_angles[1:-1:,2], omega[:,2],
                torques=torques[1:-1:,2],
                moments_of_inertia=I[1:-1,2], angle_index=2, # Ipsi
                angle_label='ψ', component_label='ψ',
                save_path=str(output_dir / f'phase_portrait_psi_attract.{args.format}'))
            # ψ/ω` with torque field
            plot_phase_portrait_with_vector_field(
                euler_angles[1:-1:,2], omega[:,2],
                torques=torques[1:-1:,2],
                moments_of_inertia=I[1:-1,2], angle_index=2, # Ipsi
                angle_label='ψ', component_label='total',energy=V[1:-1],
                save_path=str(output_dir / f'phase_portrait_psi-total_attract.{args.format}'))
            # θ/ω` with torque field
            plot_phase_portrait_with_vector_field(
                euler_angles[1:-1:,1], omega[:,1],
                torques=torques[1:-1:,1],
                moments_of_inertia=I[1:-1,1], angle_index=1, # Itheta
                angle_label='θ', component_label='total', energy=V[1:-1],
                save_path=str(output_dir / f'phase_portrait_theta-total_attract.{args.format}'))


            plot_count += 2
            print(f"  ✓ Generated phase portraits with vector fields")

            # Poincaré sections
            plot_poincare_section(euler_angles, omega,
                section_angles=[x for x in np.arange(0, 2*np.pi, np.pi/3)],
                save_path=str(output_dir / f'poincare_sections.{args.format}'))
            plot_count += 1
            print(f"  ✓ Generated phase space portraits")
        except Exception as e:
            print(f"  ✗ Phase space plotting failed: {e}")

    # 3. Energy Landscapes
    if plot_energy and E_available:
        print("[3/9] Energy landscapes...")
        try:
            plot_energy_landscape_trajectory(theta, psi, data.Etot, times=data.times,
                pmf=pmf_data['pmf'], theta_bins=pmf_data['theta_centers'], psi_bins=pmf_data['psi_centers'],
                save_path=str(output_dir / f'energy_landscape.{args.format}'))
            plot_count += 1
            print(f"  ✓ Generated energy landscapes")
        except Exception as e:
            print(f"  ✗ Energy landscape plotting failed: {e}")

    # 4. Angular Momentum Phase Space
    if plot_L and L_available:
        print("[4/9] Angular momentum phase space...")
        try:
            omega = data.nutation_magnitude
            L = data.L_perp_mag[1:-1]
            plot_L_phase_space(
                omega, L, variable_type='ω', component_label='∥',
                save_path=str(output_dir / f'L_portrait.{args.format}')
            )
            plot_count += 1
            print(f"  ✓ Generated angular momentum phase space")
        except Exception as e:
            print(f"  ✗ Angular momentum plotting failed: {e}")

    # 5. Spectral Analysis
    if plot_spectra and L_available:
        print("[5/6] Spectral analysis...")
        try:
            L_vec = data.L 
            acf_result = angular_momentum_acf(L_vec, data.times)
            plot_autocorrelation(acf_result['lags'], acf_result['acf'],
                save_path=str(output_dir / f'acf_angular_momentum.{args.format}'))
            plot_count += 1

            if data.has('moments'):
                friction_result = extract_friction_from_acf(acf_result['lags'], acf_result['acf'],
                    moment_of_inertia=np.mean(data.moments))
                plot_friction_extraction(acf_result['lags'], acf_result['acf'], friction_result['gamma'],
                    save_path=str(output_dir / f'friction_extraction.{args.format}'))
                plot_count += 1
            print(f"  ✓ Generated spectral analysis plots")
        except Exception as e:
            print(f"  ✗ Spectral analysis plotting failed: {e}")

    # 6. Structural Metrics (RMSD, Rg)
    if plot_metrics and data.has('rmsd', 'rg'):
        print("[6/9] Structural metrics...")
        try:
            from .visualization.metrics import (
                plot_rmsd_timeseries,
                plot_rg_timeseries,
                plot_multi_metric_panel
            )

            # RMSD time series
            plot_rmsd_timeseries(
                data.times, data.rmsd,
                title='RMSD vs Time',
                save_path=str(output_dir / f'rmsd_timeseries.{args.format}'),
                save_formats=(args.format,)
            )
            plot_count += 1

            # Rg time series
            plot_rg_timeseries(
                data.times, data.rg,
                title='Radius of Gyration vs Time',
                save_path=str(output_dir / f'rg_timeseries.{args.format}'),
                save_formats=(args.format,)
            )
            plot_count += 1

            # Multi-metric panel
            metrics = {
                'rmsd': data.rmsd,
                'rg': data.rg,
            }
            if data.has('asphericity'):
                metrics['asphericity'] = data.asphericity
            if data.has('acylindricity'):
                metrics['acylindricity'] = data.acylindricity

            plot_multi_metric_panel(
                data.times, metrics, metrics,  # Same system, no comparison
                metric_names=list(metrics.keys()),
                wt_label='System',
                mut_label='System',
                save_path=str(output_dir / f'metrics_panel.{args.format}'),
                save_formats=(args.format,)
            )
            plot_count += 1

            print(f"  ✓ Generated structural metrics plots")
        except Exception as e:
            print(f"  ✗ Metrics plotting failed: {e}")

    # 7. Bifurcation Diagrams
    if plot_bifurcation and omega_available and E_available:
        print("[7/9] Bifurcation analysis...")
        try:
            from .visualization.bifurcation import (
                compute_poincare_section,
                plot_poincare_bifurcation,
                plot_multi_section_bifurcation
            )

            # Get omega for one component (e.g., theta)
            omega_phi = data.spin_magnitude # theta component
            Q = data.Enonpol

            # Truncate angles and energy to match omega length (omega uses finite diff)
            n_omega_phi = len(omega_phi)
            phi_trunc = phi[:n_omega_phi]
            theta_trunc = theta[:n_omega_phi]
            Q_trunc = Q[:n_omega_phi]

            # Compute Poincaré section at theta=0
            omega_cross, E_cross = compute_poincare_section(
                phi_trunc, omega_phi, Q_trunc,
                section_angle=0.0,
                tolerance=0.05
            )

            if len(omega_cross) > 0:
                print(omega_cross.shape, E_cross.shape)
                # Single bifurcation diagram
                plot_poincare_bifurcation(
                    E_cross, omega_cross,
                    title='Bifurcation Diagram (φ=0 section)',
                    xlabel='Q-factor (non-polar interaction) (kcal/mol)',
                    ylabel='Angular Velocity (rad/ps)',
                    save_path=str(output_dir / f'bifurcation_nonpol.{args.format}'),
                    save_formats=(args.format,)
                )
                plot_count += 1

                # Multi-section bifurcation
                plot_multi_section_bifurcation(
                    theta_trunc, omega_phi, Q_trunc,
                    section_angles=[0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi/2],
                    angle_labels=['φ=0', 'φ=π/6', 'φ=π/3', 'φ=π/2', 'φ=2π/3', 'φ=5π/6', 'φ=π'],
                    save_path=str(output_dir / f'bifurcation_multi_nonpolar.{args.format}'),
                    save_formats=(args.format,)
                )
                plot_count += 1

                print(f"  ✓ Generated bifurcation diagrams ({len(omega_cross)} crossings)")
            else:
                print(f"  ⚠ No Poincaré crossings found")

        except Exception as e:
            print(f"  ✗ Bifurcation plotting failed: {e}")

    # 8. Hopf Bifurcation Analysis
    if plot_bifurcation and omega_available and E_available:
        print("[8/10] Hopf bifurcation analysis...")
        try:
            from .analysis.hopf_bifurcation import (
                detect_hopf_bifurcation,
                plot_hopf_bifurcation_diagram
            )

            # Use theta component for perpendicular configuration analysis
            omega_theta = data.omega_body[:, 1]

            # Truncate theta and Etot to match omega length (omega uses finite diff, loses 2 frames)
            n_omega = len(omega_theta)
            theta_trunc = theta[:n_omega]
            Etot_trunc = data.Etot[:n_omega]

            # Detect Hopf bifurcation
            hopf_result = detect_hopf_bifurcation(
                theta_trunc, omega_theta, Etot_trunc,
                n_energy_bins=10,
                verbose=True
            )

            # Plot diagnostic diagram
            plot_hopf_bifurcation_diagram(
                theta_trunc, omega_theta, Etot_trunc, hopf_result,
                save_path=str(output_dir / f'hopf_bifurcation.{args.format}'),
                save_formats=(args.format,)
            )
            plot_count += 1

            # Print summary
            if hopf_result.has_hopf:
                print(f"  ✓ Hopf bifurcation detected at E = {hopf_result.bifurcation_energy:.2f} kcal/mol")
                print(f"     Type: {hopf_result.bifurcation_type}")
                if hopf_result.has_limit_cycle:
                    print(f"     Limit cycle: amplitude = {hopf_result.cycle_amplitude:.3f} rad, "
                          f"frequency = {hopf_result.cycle_frequency:.3f} rad/ps")
            else:
                print(f"  ⚠ No Hopf bifurcation detected")
                print(f"     Fixed point stable: {hopf_result.fixed_point_stable}")

        except Exception as e:
            print(f"  ✗ Hopf bifurcation analysis failed: {e}")

    # 9. Per-Residue Energy Analysis
    if plot_residue and data.has('per_residue'):
        print("[9/10] Per-residue energy analysis...")
        try:
            from .analysis.per_residue_energy import (
                ResidueEnergyProfile,
                identify_hotspot_residues,
                plot_residue_energy_profile
            )

            per_residue_data = data.per_residue
            import MDAnalysis as mda

            # Create map from residue indices to residue names
            projname = str(output_dir)[:-23]
            u = mda.Universe(topology=f'{projname}.tpr')
            protein = u.select_atoms('protein')
            residx_to_name = {i: res.resname for i, res in enumerate(protein.residues)}

            # Collect all residue indices present across frames
            residue_indices = set()
            for frame_dict in per_residue_data:
                residue_indices.update(frame_dict.keys())
            residue_indices = sorted(residue_indices)

            # Construct ResidueEnergyProfile for each residue (fix placement of append!)
            profiles = []
            for i in residue_indices:
                # Gather energy trajectory for this residue across all frames
                energies = []
                for frame_dict in per_residue_data:
                    if i in frame_dict:
                        # Assume tuple is (energy, ...)
                        energy_val = frame_dict[i][2]
                        energies.append(energy_val)
                if energies:
                    resname = residx_to_name.get(i, f"UNK{i}")
                    profile = ResidueEnergyProfile(
                        residue_id=int(i + 1),  # MDAnalysis residue ids start from 1
                        residue_name=str(resname),
                        energy_series=np.array(energies),
                        mean_energy=float(np.mean(energies)),
                        std_energy=float(np.std(energies)),
                        max_energy=float(np.max(energies)),
                        min_energy=float(np.min(energies))
                    )
                    profiles.append(profile)

            # Sort by energy standard deviation (most variable at top)
            profiles.sort(key=lambda p: abs(p.std_energy), reverse=True)

            # Plot top residues
            plot_residue_energy_profile(
                profiles,
                title='Per-Residue Energy Profile',
                top_n=20,
                save_path=str(output_dir / f'residue_energy.{args.format}'),
                save_formats=(args.format,)
            )
            plot_count += 1

            # Identify hotspots
            hotspots = identify_hotspot_residues(profiles, threshold_percentile=90)
            print(f"  ✓ Generated per-residue energy plot ({len(hotspots)} hotspots)")
            print(f"     Top 3 hotspots: {', '.join([f'Res{p.residue_id}' for p in hotspots[:3]])}")

        except Exception as e:
            print(f"  ✗ Per-residue energy plotting failed: {e}")

    # 10. Multi-Panel Summary
    if plot_summary and omega_available and E_available:
        print("[10/10] Multi-panel summary...")
        try:
            omega_theta = data.nutation_magnitude
            omega_psi = data.spin_magnitude
            print(omega_theta.size, theta.size)
            plot_multi_panel_summary(theta[1:-1:], psi[1:-1:], omega_theta, omega_psi,
                data.Etot[1:-1:], data.times[1:-1:], save_path=str(output_dir / f'summary.{args.format}'))
            plot_count += 1
            print(f"  ✓ Generated multi-panel summary")
        except Exception as e:
            print(f"  ✗ Summary plotting failed: {e}")

    # Generate JSON analysis report
    print(f"\n{'=' * 60}")
    print("Generating analysis report...")
    print(f"{'=' * 60}\n")

    import json
    report = {
        'system': {
            'n_frames': int(len(theta)),
            'time_range_ps': [float(data.times[0]), float(data.times[-1])] if data.has('times') else None,
            'dt_ps': float(data.times[1] - data.times[0]) if data.has('times') and len(data.times) > 1 else None,
        },
        'orientation': {
            'theta_mean_deg': float(np.rad2deg(np.mean(theta))),
            'theta_std_deg': float(np.rad2deg(np.std(theta))),
            'theta_min_deg': float(np.rad2deg(np.min(theta))),
            'theta_max_deg': float(np.rad2deg(np.max(theta))),
            'psi_mean_deg': float(np.rad2deg(np.mean(psi))),
            'psi_std_deg': float(np.rad2deg(np.std(psi))),
        },
        'pmf': {
            'min_kcal_mol': float(np.min(pmf_data['pmf'])),
            'max_kcal_mol': float(np.max(pmf_data['pmf'])),
            'barrier_height_kcal_mol': float(np.max(pmf_data['pmf']) - np.min(pmf_data['pmf'])),
        }
    }

    if E_available:
        report['energy'] = {
            'total_mean': float(np.mean(data.Etot)),
            'total_std': float(np.std(data.Etot)),
            'polar_mean': float(np.mean(data.Epol)),
            'nonpolar_mean': float(np.mean(data.Enonpol)),
            'polar_fraction': float(np.mean(data.Epol) / np.mean(data.Etot)) if np.mean(data.Etot) != 0 else 0,
        }

    # Fill angular momentum dictionary with all relevant fields present in the data
    if L_available:
        L_dict = dict(
            L=data.L,
            L_parallel=data.L_parallel,
            L_perp=data.L_perp,
            L_mag=data.L_mag,
            L_parallel_mag=data.L_parallel_mag,
            L_perp_mag=data.L_perp_mag,
            L_z=data.L_z,
            L_z_mag=data.L_z_mag,
            spin_magnitude=data.spin_magnitude,
            nutation_magnitude=data.nutation_magnitude,
            spin_nutation_ratio=data.spin_nutation_ratio
        )
        report['angular_momentum'] = {
            'L_total_mean': float(np.mean(L_dict['L_mag'])),
            'L_parallel_mean': float(np.mean(L_dict['L_parallel_mag'])),
            'L_perp_mean': float(np.mean(L_dict['L_perp_mag'])),
            'spin_nutation_ratio': float(np.mean(L_dict['spin_nutation_ratio'])),
            'L_z_mean': float(np.mean(L_dict['L_z_mag'])),
            'spin_mean': float(np.mean(L_dict['spin_magnitude'])),
            'nutation_mean': float(np.mean(L_dict['nutation_magnitude'])),
        }

    if omega_available:
        # Fill omega_body and derived dictionaries with relevant angular velocity data
        omega_body = dict(
            omega_body=data.omega_body if hasattr(data, "omega_body") else None,
            omega_body_spin=data.omega_body_spin if hasattr(data, "omega_body_spin") else None,
            omega_body_nutation=data.omega_body_nutation if hasattr(data, "omega_body_nutation") else None,
            spin_magnitude=data.spin_magnitude if hasattr(data, "spin_magnitude") else None,
            nutation_magnitude=data.nutation_magnitude if hasattr(data, "nutation_magnitude") else None,
        )
        omega_lab = dict(
            omega_lab=data.omega_lab if hasattr(data, "omega_lab") else None,
            omega_magnitude=data.omega_magnitude if hasattr(data, "omega_magnitude") else None,
        )
        report['angular_velocity'] = {
            'omega_spin_mean_rad_ps': float(np.mean(np.abs(omega_body['omega_body_spin']))) if omega_body.get('omega_body_spin') is not None else None,
            'omega_nutation_mean_rad_ps': float(np.mean(np.abs(omega_body['omega_body_nutation']))) if omega_body.get('omega_body_nutation') is not None else None,
            'omega_lab_mean_rad_ps': float(np.mean(np.abs(omega_lab['omega_lab']))) if omega_lab.get('omega_lab') is not None else None,
            'omega_lab_mag_mean_rad_ps': float(np.mean(np.abs(omega_lab['omega_magnitude']))) if omega_lab.get('omega_magnitude') is not None else None,
            'spin_magnitude_mean_rad_ps': float(np.mean(omega_body['spin_magnitude'])) if omega_body.get('spin_magnitude') is not None else None,
            'nutation_magnitude_mean_rad_ps': float(np.mean(omega_body['nutation_magnitude'])) if omega_body.get('nutation_magnitude') is not None else None,
        }

    if data.has('moments'):
        moments_mean = np.mean(data.moments, axis=0)
        report['inertia'] = {
            'I_moments_amu_A2': [float(moments_mean[0]), float(moments_mean[1]), float(moments_mean[2])],
            'asymmetry_param': float((moments_mean[1] - moments_mean[2]) / moments_mean[0]) if moments_mean[0] != 0 else 0,
        }

    # Save JSON report
    report_path = output_dir / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Saved: {report_path}")

    # Print key findings
    print(f"\n{'=' * 60}")
    print("KEY FINDINGS")
    print(f"{'=' * 60}")
    print(f"Frames: {report['system']['n_frames']}")
    print(f"Orientation: θ = {report['orientation']['theta_mean_deg']:.1f}° ± {report['orientation']['theta_std_deg']:.1f}°")
    print(f"PMF barrier: {report['pmf']['barrier_height_kcal_mol']:.2f} kcal/mol")
    if E_available:
        print(f"Energy: {report['energy']['total_mean']:.1f} ± {report['energy']['total_std']:.1f} kcal/mol")
        print(f"  Polar/Total: {report['energy']['polar_fraction']*100:.1f}%")
    if L_available:
        print(f"Spin/Nutation: {report['angular_momentum']['spin_nutation_ratio']:.2f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Visualization complete!")
    print(f"Generated {plot_count} plots")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def cmd_pmf(args):
    """Compute PMF from Euler angles."""
    from .io.output import load_results_npz, save_results_npz
    from .analysis.pmf import compute_pmf_2d

    print("Computing PMF...")

    # Load Euler angles
    data = load_results_npz(args.input)
    euler_angles = data['euler']

    # Extract theta and psi in degrees
    phi, theta, psi = euler_angles.T
    theta_deg = np.degrees(theta)
    psi_deg = np.degrees(psi)

    # Load config with CLI overrides
    config = _load_config_with_overrides(args)

    # Compute PMF
    bins = config.get('analysis.pmf_bins')
    pmf_result = compute_pmf_2d(
        theta_deg,
        psi_deg,
        theta_bins=bins[1],  # theta bins
        psi_bins=bins[2],    # psi bins
        temperature=config.get('analysis.temperature')
    )

    # Save
    save_results_npz({
        'pmf': pmf_result['pmf'],
        'theta_bins': pmf_result['theta_centers'],
        'psi_bins': pmf_result['psi_centers'],
        'theta_edges': pmf_result['theta_edges'],
        'psi_edges': pmf_result['psi_edges']
    }, args.output)

    print(f"✓ PMF saved to {args.output}")

    # Plot if requested
    if args.plot:
        try:
            from .visualization.surfaces import plot_pmf_heatmap
            plot_path = Path(args.output).with_suffix('.png')
            plot_pmf_heatmap(
                pmf_result['pmf'],
                pmf_result['theta_centers'],
                pmf_result['psi_centers'],
                save_path=str(plot_path)
            )
        except ImportError:
            print("  ✗ Matplotlib not available for plotting")


def cmd_diffusion(args):
    """Analyze rotational diffusion."""
    from .io.output import load_results_npz, save_results_json
    from .observables.diffusion import analyze_diffusion

    print("Analyzing rotational diffusion...")

    # Load data
    data = load_results_npz(args.input)
    euler_angles = data['euler']
    times = data['times']

    # Analyze
    results = analyze_diffusion(euler_angles, times, verbose=True)

    # Save
    save_results_json(results, args.output)

    print(f"✓ Results saved to {args.output}")


def cmd_compare(args):
    """Compare two systems."""
    from .io.output import load_results_json, save_results_json

    print("Comparing systems...")

    # Load results
    results1 = load_results_json(args.system1)
    results2 = load_results_json(args.system2)

    # Create comparison
    comparison = {
        'system1': {
            'label': args.labels[0],
            'data': results1
        },
        'system2': {
            'label': args.labels[1],
            'data': results2
        },
        'differences': {}
    }

    # Compute differences for common keys
    for key in results1.keys():
        if key in results2:
            val1 = results1[key]
            val2 = results2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                percent = 100 * diff / val1 if val1 != 0 else np.inf
                comparison['differences'][key] = {
                    'absolute': diff,
                    'percent': percent
                }

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    save_results_json(comparison, str(output_dir / 'comparison.json'))

    print(f"✓ Comparison saved to {output_dir}/comparison.json")

    # Print summary
    print("\nComparison Summary:")
    print("=" * 60)
    for key, diff_data in comparison['differences'].items():
        print(f"{key:20s}: Δ = {diff_data['absolute']:+.3e} ({diff_data['percent']:+.1f}%)")


def cmd_detailed_analysis(args):
    """Run detailed mutation analysis with per-residue energy, N75K motion, and Poincaré sections."""
    import matplotlib
    matplotlib.rcParams['savefig.format'] = args.format

    print("=" * 60)
    print("DETAILED MUTATION ANALYSIS")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine what to generate
    gen_all = args.all
    gen_residue = args.per_residue or gen_all
    gen_motion = args.n75k_motion or gen_all
    gen_poincare = args.poincare or gen_all

    # Load data
    print(f"\nLoading data...")
    print(f"  WT: {args.wt_data}")
    print(f"  Mutant: {args.mut_data}")

    data_wt = np.load(args.wt_data, allow_pickle=True)
    data_mut = np.load(args.mut_data, allow_pickle=True)

    # Convert to dict
    wt_dict = {key: data_wt[key] for key in data_wt.files}
    mut_dict = {key: data_mut[key] for key in data_mut.files}

    print(f"  ✓ Loaded WT: {len(wt_dict)} arrays")
    print(f"  ✓ Loaded Mutant: {len(mut_dict)} arrays")

    plot_count = 0

    # 1. Per-Residue Energy Analysis
    if gen_residue:
        print("\n" + "=" * 60)
        print("[1/3] Per-Residue Energy Analysis")
        print("=" * 60)

        residue_dir = output_dir / 'per_residue_energy'
        residue_dir.mkdir(exist_ok=True, parents=True)

        try:
            from .analysis.per_residue_energy import (
                compute_residue_energies,
                identify_hotspot_residues,
                compare_residue_energies,
                plot_residue_energy_profile,
                plot_residue_energy_comparison
            )

            if 'per_residue' in wt_dict and 'per_residue' in mut_dict:
                print("\n  Extracting energy profiles from per_residue data...")

                # Get real residue names from topology if available
                try:
                    import MDAnalysis as mda
                    wt_tpr = str(Path(args.wt_data).parent.parent / "wt.tpr")
                    mut_tpr = str(Path(args.mut_data).parent.parent / "n75k.tpr")

                    u_wt = mda.Universe(wt_tpr)
                    residx_to_name_wt = {res.resid: res.resname for res in u_wt.select_atoms('protein').residues}
                    print(f"  ✓ Loaded WT topology")

                    u_mut = mda.Universe(mut_tpr)
                    residx_to_name_mut = {res.resid: res.resname for res in u_mut.select_atoms('protein').residues}
                    print(f"  ✓ Loaded mutant topology")
                except Exception as e:
                    print(f"  ⚠ Could not load topology: {e}")
                    residx_to_name_wt = {}
                    residx_to_name_mut = {}

                # Build profiles from per_residue data
                from .analysis.per_residue_energy import ResidueEnergyProfile

                profiles_wt = []
                residue_ids_wt = sorted(set(k for frame in wt_dict['per_residue'] for k in frame.keys()))
                for res_id in residue_ids_wt:
                    energies = [frame[res_id][2] for frame in wt_dict['per_residue'] if res_id in frame]
                    if energies:
                        profiles_wt.append(ResidueEnergyProfile(
                            residue_id=int(res_id),
                            residue_name=residx_to_name_wt.get(res_id, f'RES{res_id}'),
                            energy_series=np.array(energies),
                            mean_energy=float(np.mean(energies)),
                            std_energy=float(np.std(energies)),
                            max_energy=float(np.max(energies)),
                            min_energy=float(np.min(energies))
                        ))

                profiles_mut = []
                residue_ids_mut = sorted(set(k for frame in mut_dict['per_residue'] for k in frame.keys()))
                for res_id in residue_ids_mut:
                    energies = [frame[res_id][2] for frame in mut_dict['per_residue'] if res_id in frame]
                    if energies:
                        profiles_mut.append(ResidueEnergyProfile(
                            residue_id=int(res_id),
                            residue_name=residx_to_name_mut.get(res_id, f'RES{res_id}'),
                            energy_series=np.array(energies),
                            mean_energy=float(np.mean(energies)),
                            std_energy=float(np.std(energies)),
                            max_energy=float(np.max(energies)),
                            min_energy=float(np.min(energies))
                        ))

                # Sort by std (most variable)
                profiles_wt.sort(key=lambda p: abs(p.std_energy), reverse=True)
                profiles_mut.sort(key=lambda p: abs(p.std_energy), reverse=True)

                print(f"  ✓ Analyzed {len(profiles_wt)} WT residues, {len(profiles_mut)} mutant residues")

                # Identify hotspots
                hotspots_wt = identify_hotspot_residues(profiles_wt, threshold_percentile=90)
                hotspots_mut = identify_hotspot_residues(profiles_mut, threshold_percentile=90)

                print(f"  ✓ Found {len(hotspots_wt)} WT hotspots, {len(hotspots_mut)} mutant hotspots")

                # Plot
                print("\n  Generating plots...")

                plot_residue_energy_profile(
                    profiles_wt, title='Wild-Type Per-Residue Energy', top_n=20,
                    save_path=str(residue_dir / f'residue_energy_wt'),
                    save_formats=(args.format,)
                )
                plot_count += 1

                plot_residue_energy_profile(
                    profiles_mut, title='N75K Per-Residue Energy', top_n=20,
                    save_path=str(residue_dir / f'residue_energy_n75k'),
                    save_formats=(args.format,)
                )
                plot_count += 1

                plot_residue_energy_comparison(
                    profiles_wt, profiles_mut,
                    wt_label='WT', mut_label='N75K', top_n=20,
                    save_path=str(residue_dir / f'residue_energy_comparison'),
                    save_formats=(args.format,)
                )
                plot_count += 1

                print(f"  ✓ Saved plots to {residue_dir}/")

            else:
                print("  ⚠ No position data available, skipping per-residue analysis")

        except Exception as e:
            print(f"  ✗ Per-residue analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # 2. N75K Residue Motion Analysis
    if gen_motion:
        print("\n" + "=" * 60)
        print("[2/3] N75K Residue Motion Analysis")
        print("=" * 60)

        motion_dir = output_dir / 'n75k_motion'
        motion_dir.mkdir(exist_ok=True, parents=True)

        try:
            import matplotlib.pyplot as plt

            # Use tilt angle as proxy for mutation effect
            if 'theta' in wt_dict and 'theta' in mut_dict:
                print("\n  Analyzing tilt angle changes...")

                theta_wt = wt_dict['theta']
                theta_mut = mut_dict['theta']
                times_wt = wt_dict.get('times', np.arange(len(theta_wt)))
                times_mut = mut_dict.get('times', np.arange(len(theta_mut)))

                # Create plot
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))

                # Time series
                axes[0].plot(times_wt, np.degrees(theta_wt), label='WT', alpha=0.7, lw=1, color='steelblue')
                axes[0].plot(times_mut, np.degrees(theta_mut), label='N75K', alpha=0.7, lw=1, color='coral')
                axes[0].set_ylabel('Tilt Angle (°)', fontsize=12)
                axes[0].set_title('Tilt Angle Time Series', fontsize=14)
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)

                # Difference (only where both have data)
                min_len = min(len(theta_wt), len(theta_mut))
                theta_wt_trunc = theta_wt[:min_len]
                theta_mut_trunc = theta_mut[:min_len]
                times_trunc = times_wt[:min_len]
                diff = np.degrees(theta_mut_trunc - theta_wt_trunc)
                axes[1].plot(times_trunc, diff, color='red', lw=1, alpha=0.7)
                axes[1].axhline(y=0, color='k', linestyle='--', lw=1)
                axes[1].set_xlabel('Time (ns)', fontsize=12)
                axes[1].set_ylabel('Δθ (N75K - WT) (°)', fontsize=12)
                axes[1].set_title('Mutation-Induced Tilt Change', fontsize=14)
                axes[1].grid(True, alpha=0.3)

                fig.tight_layout()

                # Save
                save_path = motion_dir / f'n75k_tilt_effect.{args.format}'
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_count += 1

                print(f"\n  Tilt angle statistics:")
                print(f"    WT: {np.degrees(np.mean(theta_wt)):.2f} ± {np.degrees(np.std(theta_wt)):.2f}°")
                print(f"    N75K: {np.degrees(np.mean(theta_mut)):.2f} ± {np.degrees(np.std(theta_mut)):.2f}°")
                print(f"    Δθ: {np.degrees(np.mean(diff)):.2f} ± {np.degrees(np.std(diff)):.2f}°")
                print(f"  ✓ Saved plot to {save_path}")

            else:
                print("  ⚠ No theta data available, skipping motion analysis")

        except Exception as e:
            print(f"  ✗ Motion analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. Poincaré Bifurcation Analysis
    if gen_poincare:
        print("\n" + "=" * 60)
        print("[3/3] Poincaré Bifurcation Analysis")
        print("=" * 60)

        poincare_dir = output_dir / 'poincare_sections'
        poincare_dir.mkdir(exist_ok=True, parents=True)

        try:
            from .visualization.bifurcation import (
                compute_poincare_section,
                plot_poincare_bifurcation,
                plot_multi_section_bifurcation
            )
            import matplotlib.pyplot as plt

            # Check for required data
            required = ['theta', 'omega_body', 'Etot']
            if all(k in wt_dict for k in required) and all(k in mut_dict for k in required):
                print("\n  Computing Poincaré sections...")

                # WT analysis
                theta_wt = wt_dict['theta']
                omega_wt = wt_dict['omega_body'][:, 1]  # θ component
                E_wt = wt_dict['Etot']

                omega_cross_wt, E_cross_wt = compute_poincare_section(
                    theta_wt, omega_wt, E_wt,
                    section_angle=0.0, tolerance=0.05
                )
                print(f"  ✓ WT: Found {len(omega_cross_wt)} crossings")

                # Mutant analysis
                theta_mut = mut_dict['theta']
                omega_mut = mut_dict['omega_body'][:, 1]
                E_mut = mut_dict['Etot']

                omega_cross_mut, E_cross_mut = compute_poincare_section(
                    theta_mut, omega_mut, E_mut,
                    section_angle=0.0, tolerance=0.05
                )
                print(f"  ✓ N75K: Found {len(omega_cross_mut)} crossings")

                print("\n  Generating plots...")

                # Individual bifurcation diagrams
                if len(omega_cross_wt) > 0:
                    plot_poincare_bifurcation(
                        E_cross_wt, omega_cross_wt,
                        title='Wild-Type: Poincaré Bifurcation Diagram',
                        save_path=str(poincare_dir / f'bifurcation_wt'),
                        save_formats=(args.format,)
                    )
                    plot_count += 1

                if len(omega_cross_mut) > 0:
                    plot_poincare_bifurcation(
                        E_cross_mut, omega_cross_mut,
                        title='N75K: Poincaré Bifurcation Diagram',
                        save_path=str(poincare_dir / f'bifurcation_n75k'),
                        save_formats=(args.format,)
                    )
                    plot_count += 1

                # Comparison plot
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

                if len(omega_cross_wt) > 0:
                    axes[0].scatter(E_cross_wt, omega_cross_wt, s=1, alpha=0.3, c='steelblue')
                    axes[0].set_xlabel('Total Energy (kcal/mol)', fontsize=12)
                    axes[0].set_ylabel('Angular Velocity at θ=0 (rad/ps)', fontsize=12)
                    axes[0].set_title('Wild-Type', fontsize=14)
                    axes[0].grid(True, alpha=0.3)

                if len(omega_cross_mut) > 0:
                    axes[1].scatter(E_cross_mut, omega_cross_mut, s=1, alpha=0.3, c='coral')
                    axes[1].set_xlabel('Total Energy (kcal/mol)', fontsize=12)
                    axes[1].set_title('N75K Mutant', fontsize=14)
                    axes[1].grid(True, alpha=0.3)

                fig.tight_layout()
                save_path = poincare_dir / f'bifurcation_comparison.{args.format}'
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_count += 1

                print(f"  ✓ Saved plots to {poincare_dir}/")

            else:
                print(f"  ⚠ Missing required data ({required}), skipping Poincaré analysis")

        except Exception as e:
            print(f"  ✗ Poincaré analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {plot_count} plots")
    print(f"Output directory: {output_dir}")
    print("\nGenerated plots:")
    if gen_residue:
        print("  per_residue_energy/")
        print("    - residue_energy_wt.*")
        print("    - residue_energy_n75k.*")
        print("    - residue_energy_comparison.*")
    if gen_motion:
        print("  n75k_motion/")
        print("    - n75k_tilt_effect.*")
    if gen_poincare:
        print("  poincare_sections/")
        print("    - bifurcation_wt.*")
        print("    - bifurcation_n75k.*")
        print("    - bifurcation_comparison.*")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'init-config':
            cmd_init_config(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'extract':
            cmd_extract(args)
        elif args.command == 'plot':
            cmd_plot(args)
        elif args.command == 'pmf':
            cmd_pmf(args)
        elif args.command == 'diffusion':
            cmd_diffusion(args)
        elif args.command == 'compare':
            cmd_compare(args)
        elif args.command == 'detailed-analysis':
            cmd_detailed_analysis(args)
        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
