"""
Functional CLI for rotmd

Ultra-simplified command-line interface using functional class-based architecture.
Clean, composable, minimal boilerplate.

Author: Mykyta Bobylyow
Date: 2025
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Functional imports
from rotmd.core.functional import Pipeline, Maybe
from rotmd.core.observables_classes import compute_all_observables_functional


@dataclass
class Config:
    """Configuration from command line."""
    topology: Path
    trajectory: Path
    output_dir: Path
    output_file: str
    start: int = 0
    stop: Optional[int] = None
    step: int = 1
    selection: str = "protein"
    format: str = "npz"
    include_structural: bool = False
    reference_frame: int = 0


def parse_args() -> Maybe[Config]:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        prog='rotmd',
        description='rotmd - Functional Rotational MD Analysis'
    )
    parser.add_argument('topology', type=Path)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('-d', '--dir', required=True, type=Path, dest='output_dir')
    parser.add_argument('-f', '--filename', required=True, dest='output_file')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=None)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--selection', default='protein')
    parser.add_argument('--format', choices=['npz', 'netcdf'], default='npz')
    parser.add_argument('--include-structural', action='store_true')
    parser.add_argument('--reference-frame', type=int, default=0)

    try:
        args = parser.parse_args()
        return Maybe.of(Config(**vars(args)))
    except:
        parser.print_help()
        return Maybe.nothing()


def load_trajectory(config: Config):
    """Load trajectory from files."""
    from rotmd.io.gromacs import load_gromacs_trajectory
    from tqdm import tqdm

    print(f"Loading {config.trajectory}...")

    return load_gromacs_trajectory(
        str(config.topology),
        str(config.trajectory),
        selection=config.selection,
        start=config.start,
        stop=config.stop,
        step=config.step,
        verbose=True
    )


def extract_orientation(traj_data):
    """Extract Euler angles and rotation matrices."""
    from rotmd.core.orientation import extract_orientation_trajectory

    print("Extracting orientation...")
    euler_angles, R = extract_orientation_trajectory(
        traj_data['positions'],
        traj_data['masses']
    )

    return {
        **traj_data,
        'euler_angles': euler_angles,
        'rotation_matrices': R,
        'phi': euler_angles[:, 0],
        'theta': euler_angles[:, 1],
        'psi': euler_angles[:, 2]
    }


def compute_principal_axes(traj_data):
    """Compute principal axes for all frames."""
    from rotmd.core.inertia import principal_axes
    from tqdm import tqdm

    print("Computing principal axes...")
    I = traj_data['inertia_tensor']
    n_frames = len(traj_data['times'])

    axes = np.zeros((n_frames, 3, 3))
    moments = np.zeros((n_frames, 3))

    for i in tqdm(range(n_frames), desc="Principal axes"):
        moments[i], axes[i] = principal_axes(I[i])

    return {
        **traj_data,
        'axes': axes,
        'moments': moments
    }


def compute_observables(traj_data):
    """Compute all observables using functional class-based approach."""
    print("Computing observables...")

    obs = compute_all_observables_functional(
        positions=traj_data['positions'],
        velocities=traj_data['velocities'],
        forces=traj_data['forces'],
        masses=traj_data['masses'],
        inertia_tensors=traj_data['inertia_tensor'],
        principal_axes=traj_data['axes'],
        membrane_normal=traj_data['normal'],
        times=traj_data['times'],
        validate=True
    )

    return {**traj_data, 'observables': obs}


def add_structural_params(config: Config):
    """Conditionally add structural parameters."""
    def _add(traj_data):
        if not config.include_structural:
            return traj_data

        print("Extracting structural parameters...")
        from rotmd.observables.structural import compute_structural_trajectory

        ref_frame = min(config.reference_frame, len(traj_data['times']) - 1)
        reference = traj_data['positions'][ref_frame]

        structural = compute_structural_trajectory(
            traj_data['positions'],
            traj_data['masses'],
            reference=reference,
            verbose=True
        )

        return {**traj_data, 'structural': structural}

    return _add


def save_results(config: Config):
    """Save results to file."""
    def _save(traj_data):
        print(f"\nSaving to {config.format}...")

        config.output_dir.mkdir(exist_ok=True, parents=True)
        output_path = config.output_dir / config.output_file

        if config.format == 'npz':
            # Prepare NPZ data
            data_dict = {
                'phi': traj_data['phi'],
                'theta': traj_data['theta'],
                'psi': traj_data['psi'],
                'rotation_matrices': traj_data['rotation_matrices'],
                'times': traj_data['times'],
                'positions': traj_data['positions'],
                'masses': traj_data['masses'],
                'velocities': traj_data['velocities'],
                'forces': traj_data['forces'],
                'inertia_tensor': traj_data['inertia_tensor'],
                'normal': traj_data['normal'],
                'axes': traj_data['axes'],
                'moments': traj_data['moments'],
                'Etot': traj_data['Etot'],
                'Epol': traj_data['Epol'],
                'Enonpol': traj_data['Enonpol'],
            }

            # Add observables
            for name, obs in traj_data['observables'].items():
                data_dict.update(obs.to_dict())

            # Add structural if present
            if 'structural' in traj_data:
                data_dict.update(traj_data['structural'])

            np.savez_compressed(output_path, **data_dict)

        elif config.format == 'netcdf':
            import xarray as xr

            # Create xarray Dataset
            ds = xr.Dataset()

            # Add observables
            for name, obs in traj_data['observables'].items():
                ds.update(obs.to_xarray())

            # Add angles
            ds['phi'] = xr.DataArray(traj_data['phi'], coords={'time': traj_data['times']})
            ds['theta'] = xr.DataArray(traj_data['theta'], coords={'time': traj_data['times']})
            ds['psi'] = xr.DataArray(traj_data['psi'], coords={'time': traj_data['times']})

            output_path = output_path.with_suffix('.nc')
            ds.to_netcdf(output_path)

        print(f"✓ Saved to {output_path}")
        return traj_data

    return _save


def print_summary(traj_data):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)

    obs = traj_data['observables']

    print(f"\nOrientation:")
    print(f"  θ: {np.rad2deg(traj_data['theta'].mean()):.1f}° ± {np.rad2deg(traj_data['theta'].std()):.1f}°")
    print(f"  ψ: {np.rad2deg(traj_data['psi'].mean()):.1f}° ± {np.rad2deg(traj_data['psi'].std()):.1f}°")

    print(f"\nAngular Momentum:")
    print(f"  Mean |L|: {obs['L'].magnitude.mean:.3f} {obs['L'].units}")
    print(f"  Spin: {obs['L'].parallel.magnitude.mean:.3f}")
    print(f"  Nutation: {obs['L'].perp.magnitude.mean:.3f}")
    print(f"  Spin/Nutation: {obs['L'].spin_nutation_ratio.mean:.3f}")

    print(f"\nTorque:")
    print(f"  Mean |τ|: {obs['tau'].magnitude.mean:.3f} {obs['tau'].units}")

    print(f"\nAngular Velocity:")
    print(f"  Mean |ω|: {obs['omega'].magnitude.mean:.3f} {obs['omega'].units}")

    print("\n" + "=" * 60)

    return traj_data


def extract_pipeline(config: Config):
    """
    Main extraction pipeline using functional composition.

    Pure functional pipeline - each step returns new data dict.
    """
    return (Pipeline(config)
        .map(load_trajectory)
        .map(extract_orientation)
        .map(compute_principal_axes)
        .map(compute_observables)
        .map(add_structural_params(config))
        .map(save_results(config))
        .map(print_summary)
        .value)


def main():
    """CLI entry point using Maybe monad."""
    result = (parse_args()
        .bind(lambda cfg: Maybe.of(extract_pipeline(cfg)))
        .or_else(None))

    sys.exit(0 if result is not None else 1)


if __name__ == '__main__':
    main()
