"""Chunk-as-unit CLI for the rotmd extract pipeline.

One invocation processes one trajectory slice (one chunk in HPC array
parlance) and writes one .npz with the canonical schema. The pipeline does
not stream multiple chunks internally — concatenation is a separate
``rotmd merge`` command, so the natural unit of SLURM array work matches
the natural unit of output.

Schema written per chunk:

    time_ps          (n_frames,)            absolute trajectory time, ps
    masses           (n_atoms,)             amu
    positions        (n_frames, n_atoms, 3) Å
    velocities       (n_frames, n_atoms, 3) Å/ps         (if available)
    forces           (n_frames, n_atoms, 3) kJ/(mol·nm)  (if available)
    com              (n_frames, 3)          Å
    inertia_tensor   (n_frames, 3, 3)       amu·Å²
    axes             (n_frames, 3, 3)       principal axes (columns)
    moments          (n_frames, 3)          principal moments
    phi, theta, psi  (n_frames,)            ZYZ Euler, rad
    tilt             (n_frames,)            tilt vs membrane surface, rad [0, π/2]
                                            (π/2 = axis ∥ normal, 0 = axis in plane)
    rotation_matrices (n_frames, 3, 3)
    {L,tau,omega,dLdt}_{vector,parallel,perp,z_component}            (n_frames, 3)
    {L,tau,omega,dLdt}_{magnitude,parallel_mag,perp_mag,z_mag}       (n_frames,)
    rmsd             (n_frames,)            Å, Kabsch-aligned vs --reference
    rg               (n_frames,)            Å
    rg_components    (n_frames, 3)
    asphericity, acylindricity, end_to_end  (n_frames,)
    E_total, E_pol, E_nonpol                 (n_frames,)             kcal/mol
    E_per_residue    (n_frames, n_residues)
    K_trans          (n_frames,)            kcal/mol
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExtractConfig:
    topology: Path
    trajectory: Path
    reference: Path
    output: Path
    selection: str
    membrane_sel: str
    start: int
    stop: int | None
    step: int
    n_workers: int | None
    force: bool


def _add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("topology", type=Path, help="Topology (.gro/.pdb/.tpr)")
    p.add_argument("trajectory", type=Path, help="Trajectory (.trr/.xtc) — one chunk")
    p.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference structure for RMSD (must match --selection); "
             "shared across all chunks of one trajectory",
    )
    p.add_argument("-o", "--output", required=True, type=Path, help="Output .npz path")
    p.add_argument("--selection", default="protein", help="MDAnalysis selection")
    p.add_argument(
        "--membrane-sel",
        default="resname CHL1",
        help="Selection used for membrane center/normal",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Parallel workers for energy SASA (default: cpu_count)",
    )
    p.add_argument(
        "--force", action="store_true", help="Overwrite existing output (default: skip)"
    )


def _load_reference(config: ExtractConfig, masses: np.ndarray) -> np.ndarray:
    """Frame-0 positions of --reference under --selection, COM-centered.

    Centering matches load_gromacs_trajectory(center=True) so RMSD is
    translation-invariant by construction.
    """
    import MDAnalysis as mda

    u = mda.Universe(str(config.topology), str(config.reference))
    atoms = u.select_atoms(config.selection)
    if len(atoms) != len(masses):
        raise ValueError(
            f"--reference selection has {len(atoms)} atoms but trajectory "
            f"selection has {len(masses)} — selections must match"
        )
    pos = atoms.positions.copy()
    com = np.average(pos, weights=masses, axis=0)
    return pos - com


def _flatten_observables(obs: dict) -> dict[str, np.ndarray]:
    """Flatten {name: VectorObservable} into npz-serializable arrays."""
    out: dict[str, np.ndarray] = {}
    for name, v in obs.items():
        out[f"{name}_vector"] = np.asarray(v.vector)
        out[f"{name}_parallel"] = np.asarray(v.parallel)
        out[f"{name}_perp"] = np.asarray(v.perp)
        out[f"{name}_z_component"] = np.asarray(v.z_component)
        out[f"{name}_magnitude"] = np.asarray(v.magnitude)
        out[f"{name}_parallel_mag"] = np.asarray(v.parallel_mag)
        out[f"{name}_perp_mag"] = np.asarray(v.perp_mag)
        out[f"{name}_z_mag"] = np.asarray(v.z_mag)
    return out


def extract(config: ExtractConfig) -> Path:
    if config.output.exists() and not config.force:
        print(f"✓ {config.output} exists — skipping (use --force to overwrite)")
        return config.output

    from rotmd.core.orientation import extract_orientation_trajectory, membrane_tilt_angle
    from rotmd.core.inertia import principal_axes
    from rotmd.io.gromacs import load_gromacs_trajectory, compute_trajectory_energies
    from rotmd.io.output import save_npz
    from rotmd.observables.unified import compute_all_observables
    from rotmd.observables.structural import compute_structural_trajectory
    from rotmd.observables.energetics import kinetic_energy_translational

    print(f"[1/6] Loading {config.trajectory.name} [{config.start}:{config.stop}:{config.step}]")
    traj = load_gromacs_trajectory(
        str(config.topology),
        str(config.trajectory),
        selection=config.selection,
        start=config.start,
        stop=config.stop,
        step=config.step,
        verbose=False,
    )
    n_frames = traj["n_frames"]
    print(f"      {n_frames} frames, {traj['n_atoms']} atoms")

    print("[2/6] Reference + structural (RMSD, Rg)")
    reference = _load_reference(config, traj["masses"])
    structural = compute_structural_trajectory(
        traj["positions"], traj["masses"], reference=reference, verbose=False
    )

    print("[3/6] Orientation (ZYZ Euler, rotation matrices)")
    euler, R = extract_orientation_trajectory(traj["positions"], traj["masses"])

    print("[4/6] Principal axes")
    axes = np.zeros((n_frames, 3, 3))
    moments = np.zeros((n_frames, 3))
    for i in range(n_frames):
        moments[i], axes[i] = principal_axes(traj["inertia_tensor"][i])

    print("[5/6] Vector observables (L, τ, ω, dL/dt)")
    if traj["velocities"] is None or traj["forces"] is None:
        raise RuntimeError(
            "trajectory lacks velocities and/or forces — required for "
            "L/τ/ω. Re-run gmx mdrun with nstvout/nstfout > 0."
        )
    membrane_normal = np.array([0.0, 0.0, 1.0])
    obs = compute_all_observables(
        positions=traj["positions"],
        velocities=traj["velocities"],
        forces=traj["forces"],
        masses=traj["masses"],
        inertia_tensors=traj["inertia_tensor"],
        principal_axes=axes,
        membrane_normal=membrane_normal,
        times=traj["times"],
        verbose=False,
    )

    print("[6/6] Energies (SASA + simplified electrostatic, parallel)")
    energies = compute_trajectory_energies(
        str(config.topology),
        str(config.trajectory),
        selection=config.selection,
        membrane_sel=config.membrane_sel,
        start=config.start,
        stop=config.stop,
        step=config.step,
        n_workers=config.n_workers,
        verbose=False,
    )
    K_trans = np.array(
        [kinetic_energy_translational(traj["velocities"][i], traj["masses"]) for i in range(n_frames)]
    )

    data: dict[str, np.ndarray] = {
        "time_ps": traj["times"].astype(np.float64),
        "masses": traj["masses"],
        "positions": traj["positions"],
        "com": traj["com"],
        "inertia_tensor": traj["inertia_tensor"],
        "axes": axes,
        "moments": moments,
        "phi": euler[:, 0],
        "theta": euler[:, 1],
        "psi": euler[:, 2],
        # Convenience tilt vs. the membrane surface: 90° = principal axis
        # collinear with the membrane normal, 0° = lying in the plane.
        "tilt": membrane_tilt_angle(euler[:, 1]),
        "rotation_matrices": R,
        "rmsd": structural["rmsd"],
        "rg": structural["rg"],
        "rg_components": structural["rg_components"],
        "asphericity": structural["asphericity"],
        "acylindricity": structural["acylindricity"],
        "end_to_end": structural["end_to_end"],
        "E_total": energies["Etot"],
        "E_pol": energies["Epol"],
        "E_nonpol": energies["Enonpol"],
        "E_per_residue": energies["per_residue"],
        "K_trans": K_trans,
    }
    if traj["velocities"] is not None:
        data["velocities"] = traj["velocities"]
    if traj["forces"] is not None:
        data["forces"] = traj["forces"]
    data.update(_flatten_observables(obs))

    out = save_npz(config.output, data)
    size_mb = out.stat().st_size / (1024**2)
    print(f"✓ {out}  ({size_mb:.1f} MB, {n_frames} frames)")
    return out


def merge(inputs: list[Path], output: Path, force: bool) -> Path:
    from rotmd.io.output import merge_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output
    inputs = sorted(inputs)
    print(f"Merging {len(inputs)} chunks → {output}")
    out = merge_npz(inputs, output)
    size_mb = out.stat().st_size / (1024**2)
    print(f"✓ {out}  ({size_mb:.1f} MB)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rotmd", description="rotmd — chunked extract pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract one chunk → one .npz")
    _add_extract_args(p_extract)

    p_merge = sub.add_parser("merge", help="Concatenate chunk .npz files along time")
    p_merge.add_argument("inputs", nargs="+", type=Path)
    p_merge.add_argument("-o", "--output", required=True, type=Path)
    p_merge.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "extract":
        cfg = ExtractConfig(
            topology=args.topology,
            trajectory=args.trajectory,
            reference=args.reference,
            output=args.output,
            selection=args.selection,
            membrane_sel=args.membrane_sel,
            start=args.start,
            stop=args.stop,
            step=args.step,
            n_workers=args.n_workers,
            force=args.force,
        )
        extract(cfg)
        return 0
    if args.cmd == "merge":
        merge(args.inputs, args.output, args.force)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
