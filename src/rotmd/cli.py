"""
Functional CLI for rotmd

Ultra-simplified command-line interface using functional class-based architecture.
Clean, composable, minimal boilerplate.

Commands:
- extract: Extract orientation from MD trajectory
- plumed: Generate PLUMED input files for GROMACS

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
from rotmd.core import kernels as K


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
    include_energy: bool = False
    membrane_sel: str = "resname CHL1"
    n_workers: Optional[int] = None
    reference_frame: int = 0
    chunked: bool = False
    chunk_size: int = 1000


def parse_args() -> Maybe[Config]:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="rotmd", description="rotmd - Functional Rotational MD Analysis"
    )
    parser.add_argument("topology", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("-d", "--dir", required=True, type=Path, dest="output_dir")
    parser.add_argument("-f", "--filename", required=True, dest="output_file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--selection", default="protein")
    parser.add_argument("--format", choices=["npz", "netcdf"], default="npz")
    parser.add_argument("--include-structural", action="store_true")
    parser.add_argument(
        "--include-energy",
        action="store_true",
        help="Compute energy terms (slower, requires freesasa)",
    )
    parser.add_argument(
        "--membrane-sel",
        default="resname CHL1",
        help="MDAnalysis selection for membrane (default: resname CHL1)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for energy computation (default: CPU count)",
    )
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked processing for large trajectories (>10k frames)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Frames per chunk (default: 1000)"
    )

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
        verbose=True,
    )


def extract_orientation(traj_data):
    """Extract Euler angles and rotation matrices."""
    from rotmd.core.orientation import extract_orientation_trajectory

    print("Extracting orientation...")
    euler_angles, R = extract_orientation_trajectory(
        traj_data["positions"], traj_data["masses"]
    )

    return {
        **traj_data,
        "euler_angles": euler_angles,
        "rotation_matrices": R,
        "phi": euler_angles[:, 0],
        "theta": euler_angles[:, 1],
        "psi": euler_angles[:, 2],
    }


def compute_principal_axes(traj_data):
    """Compute principal axes for all frames."""
    from rotmd.core.inertia import principal_axes
    from tqdm import tqdm

    print("Computing principal axes...")

    # Handle both full trajectory and chunked data
    if "inertia_tensor" in traj_data:
        # Full trajectory from load_gromacs_trajectory
        I = traj_data["inertia_tensor"]
    else:
        # Chunk from load_gromacs_trajectory_chunked - compute inertia tensors
        print("  Computing inertia tensors...")
        com = np.mean(traj_data["positions"], axis=1)
        I = K.inertia_tensor_batch(traj_data["positions"], traj_data["masses"], com)

    n_frames = len(traj_data["times"])

    axes = np.zeros((n_frames, 3, 3))
    moments = np.zeros((n_frames, 3))

    for i in tqdm(range(n_frames), desc="Principal axes"):
        moments[i], axes[i] = principal_axes(I[i])

    return {
        **traj_data,
        "axes": axes,
        "moments": moments,
        "inertia_tensor": I,  # Store for observables computation
    }


def compute_observables(traj_data):
    """Compute all observables using functional class-based approach."""
    print("Computing observables...")

    # Handle both full trajectory and chunked data
    membrane_normal = traj_data.get("normal", np.array([0, 0, 1]))  # Default to z-axis

    obs = compute_all_observables_functional(
        positions=traj_data["positions"],
        velocities=traj_data["velocities"],
        forces=traj_data["forces"],
        masses=traj_data["masses"],
        inertia_tensors=traj_data["inertia_tensor"],
        principal_axes=traj_data["axes"],
        membrane_normal=membrane_normal,
        times=traj_data["times"],
        validate=False,  # Skip validation for chunked processing (faster)
    )

    return {**traj_data, "observables": obs}


def add_energy(config: Config):
    """Conditionally compute energy terms with parallel processing."""

    def _add(traj_data):
        if not config.include_energy:
            return traj_data

        print("Computing energies (parallel)...")
        from rotmd.io.gromacs import compute_trajectory_energies

        energies = compute_trajectory_energies(
            str(config.topology),
            str(config.trajectory),
            selection=config.selection,
            membrane_sel=config.membrane_sel,
            start=config.start,
            stop=config.stop,
            step=config.step,
            n_workers=config.n_workers,
            verbose=True,
        )

        return {**traj_data, **energies}

    return _add


def add_structural_params(config: Config):
    """Conditionally add structural parameters."""

    def _add(traj_data):
        if not config.include_structural:
            return traj_data

        print("Extracting structural parameters...")
        from rotmd.observables.structural import compute_structural_trajectory

        ref_frame = min(config.reference_frame, len(traj_data["times"]) - 1)
        reference = traj_data["positions"][ref_frame]

        structural = compute_structural_trajectory(
            traj_data["positions"],
            traj_data["masses"],
            reference=reference,
            verbose=True,
        )

        return {**traj_data, "structural": structural}

    return _add


def save_results(config: Config):
    """Save results to file."""

    def _save(traj_data):
        print(f"\nSaving to {config.format}...")

        config.output_dir.mkdir(exist_ok=True, parents=True)
        output_path = config.output_dir / config.output_file

        if config.format == "npz":
            # Prepare NPZ data - core fields always present
            data_dict = {
                "phi": traj_data["phi"],
                "theta": traj_data["theta"],
                "psi": traj_data["psi"],
                "rotation_matrices": traj_data["rotation_matrices"],
                "times": traj_data["times"],
                "positions": traj_data["positions"],
                "masses": traj_data["masses"],
                "inertia_tensor": traj_data["inertia_tensor"],
                "axes": traj_data["axes"],
                "moments": traj_data["moments"],
            }

            # Optional fields - add if present
            for key in [
                "velocities",
                "forces",
                "com",
                "normal",
                "Etot",
                "Epol",
                "Enonpol",
            ]:
                if key in traj_data and traj_data[key] is not None:
                    data_dict[key] = traj_data[key]

            # Add observables
            for name, obs in traj_data["observables"].items():
                data_dict.update(obs.to_dict())

            # Add structural if present
            if "structural" in traj_data:
                data_dict.update(traj_data["structural"])

            np.savez_compressed(output_path, **data_dict)

        elif config.format == "netcdf":
            import xarray as xr

            # Create xarray Dataset
            ds = xr.Dataset()

            # Add observables
            for name, obs in traj_data["observables"].items():
                ds.update(obs.to_xarray())

            # Add angles
            ds["phi"] = xr.DataArray(
                traj_data["phi"], coords={"time": traj_data["times"]}
            )
            ds["theta"] = xr.DataArray(
                traj_data["theta"], coords={"time": traj_data["times"]}
            )
            ds["psi"] = xr.DataArray(
                traj_data["psi"], coords={"time": traj_data["times"]}
            )

            output_path = output_path.with_suffix(".nc")
            ds.to_netcdf(output_path)

        print(f"✓ Saved to {output_path}")
        return traj_data

    return _save


def print_summary(traj_data):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)

    obs = traj_data["observables"]

    print(f"\nOrientation:")
    print(
        f"  θ: {np.rad2deg(traj_data['theta'].mean()):.1f}° ± {np.rad2deg(traj_data['theta'].std()):.1f}°"
    )
    print(
        f"  ψ: {np.rad2deg(traj_data['psi'].mean()):.1f}° ± {np.rad2deg(traj_data['psi'].std()):.1f}°"
    )

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
    return (
        Pipeline(config)
        .map(load_trajectory)
        .map(extract_orientation)
        .map(compute_principal_axes)
        .map(compute_observables)
        .map(add_structural_params(config))
        .map(add_energy(config))
        .map(save_results(config))
        .map(print_summary)
        .value
    )


class ChunkedPipeline:
    """Pipeline for processing trajectory chunks with accumulation."""

    def __init__(self, chunk_generator, config: Config):
        self.chunks = chunk_generator
        self.config = config
        self.accumulated = {}
        self.frame_count = 0

    def process(self, step_func):
        """Apply processing step to each chunk."""
        processed_chunks = []
        for chunk in self.chunks:
            print(f"Processing chunk {self.frame_count // 1000 + 1}...")
            processed = step_func(chunk)
            processed_chunks.append(processed)
            self.frame_count += len(chunk["positions"])
            print(f"  ✓ {self.frame_count} frames processed")

        self.chunks = iter(processed_chunks)
        return self

    def save(self):
        """Save accumulated results."""
        print("Saving results...")
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        results_file = self.config.output_dir / f"{self.config.output_file}.npz"

        # Collect and concatenate all chunks
        all_data = {}
        for chunk in self.chunks:
            # Extract Euler angles
            if "phi" in chunk:
                for key in ["phi", "theta", "psi", "times"]:
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(chunk[key])

            # Extract observables if present
            if "observables" in chunk:
                obs = chunk["observables"]
                for obs_name in obs:
                    obs_data = obs[obs_name]
                    key = obs_name
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(obs_data)

        # Concatenate arrays
        final_results = {}
        for k, v in all_data.items():
            if v and isinstance(v[0], np.ndarray):
                final_results[k] = np.concatenate(v, axis=0)
            else:
                final_results[k] = v

        np.savez_compressed(results_file, **final_results)
        print(f"✓ Saved to {results_file}")

        self.results = final_results
        return self

    def summarize(self):
        """Print summary statistics."""
        print(f"\nProcessed {self.frame_count} total frames")
        return self.results


def extract_pipeline_chunked(config: Config):
    """
    Chunked extraction pipeline for large trajectories.

    Processes trajectory in memory-efficient chunks using functional composition.
    """
    from rotmd.io.gromacs import load_gromacs_trajectory_chunked

    print(f"Using chunked processing ({config.chunk_size} frames/chunk)...")

    chunk_loader = load_gromacs_trajectory_chunked(
        str(config.topology),
        str(config.trajectory),
        selection=config.selection,
        chunk_size=config.chunk_size,
        start=config.start,
        stop=config.stop,
        step=config.step,
        verbose=True,
    )

    return (
        ChunkedPipeline(chunk_loader, config)
        .process(extract_orientation)
        .process(compute_principal_axes)
        .process(compute_observables)
        .save()
        .summarize()
    )


def plumed_command():
    """Generate PLUMED input file for GROMACS simulations."""
    import argparse
    from rotmd.io.plumed import generate_plumed_input

    parser = argparse.ArgumentParser(
        prog="rotmd plumed",
        description="Generate PLUMED input file for orientation CVs",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output plumed.dat file path"
    )
    parser.add_argument(
        "-a", "--atoms", required=True, help='Protein atom selection (e.g., "1-1000")'
    )
    parser.add_argument(
        "--cvs",
        nargs="+",
        default=["theta", "psi"],
        choices=["phi", "theta", "psi"],
        help="Collective variables to include",
    )
    parser.add_argument(
        "--metad", action="store_true", help="Enable metadynamics biasing"
    )
    parser.add_argument(
        "--hills-height",
        type=float,
        default=1.2,
        help="Metadynamics hill height (kJ/mol)",
    )
    parser.add_argument(
        "--hills-stride", type=int, default=500, help="Deposit hill every N steps"
    )
    parser.add_argument(
        "--wall-min", type=float, default=None, help="Lower wall for theta (degrees)"
    )
    parser.add_argument(
        "--wall-max", type=float, default=None, help="Upper wall for theta (degrees)"
    )
    parser.add_argument(
        "--stride", type=int, default=100, help="Output stride for COLVAR file"
    )

    args = parser.parse_args(sys.argv[2:])  # Skip 'rotmd plumed'

    print("Generating PLUMED input file...")
    print(f"  Protein atoms: {args.atoms}")
    print(f"  CVs: {', '.join(args.cvs)}")
    if args.metad:
        print(f"  Metadynamics: enabled (height={args.hills_height} kJ/mol)")
    if args.wall_min or args.wall_max:
        print(f"  Walls: theta ∈ [{args.wall_min or 0}, {args.wall_max or 180}]°")

    generate_plumed_input(
        output_path=args.output,
        protein_atoms=args.atoms,
        cv_names=args.cvs,
        metadynamics=args.metad,
        hills_height=args.hills_height,
        hills_stride=args.hills_stride,
        wall_theta_min=args.wall_min,
        wall_theta_max=args.wall_max,
        output_stride=args.stride,
    )


def main():
    """CLI entry point with subcommands."""
    import argparse

    # Main parser
    parser = argparse.ArgumentParser(
        prog="rotmd", description="rotmd - Rotational MD Analysis Toolkit"
    )

    # Check if subcommand specified
    if len(sys.argv) < 2:
        # Old behavior: default to extract
        result = (
            parse_args()
            .bind(
                lambda cfg: Maybe.of(
                    extract_pipeline_chunked(cfg)
                    if cfg.chunked
                    else extract_pipeline(cfg)
                )
            )
            .or_else(None)
        )
        sys.exit(0 if result is not None else 1)

    # Subcommand handling
    subcommand = sys.argv[1]

    if subcommand == "plumed":
        plumed_command()
    elif subcommand in ["-h", "--help"]:
        parser.print_help()
        print("\nSubcommands:")
        print("  extract    Extract orientation from MD trajectory (default)")
        print("  plumed     Generate PLUMED input files")
        print("\nRun 'rotmd <subcommand> -h' for subcommand help")
    else:
        # Treat as old-style extract command
        result = (
            parse_args()
            .bind(
                lambda cfg: Maybe.of(
                    extract_pipeline_chunked(cfg)
                    if cfg.chunked
                    else extract_pipeline(cfg)
                )
            )
            .or_else(None)
        )
        sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
