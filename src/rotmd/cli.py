"""Chunk-as-unit CLI for the rotmd extract pipeline.

One invocation processes one trajectory slice (one chunk in HPC array
parlance) and writes one .npz with the canonical schema. The pipeline does
not stream multiple chunks internally — concatenation is a separate
``rotmd merge`` command, so the natural unit of SLURM array work matches
the natural unit of output.

Schema written per chunk::

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
    ca_coords        (n_frames, n_ca, 3)    Å, float32, COM-centred but NOT
                                            rotationally aligned — downstream
                                            stages choose their own reference
    ca_resids        (n_ca,)                int32, resid per CA (time-invariant)
    ca_align_mask    (n_ca,)                bool, --sel-align core   (time-invariant)
    rmsd_dom         (n_frames, n_domains)  Å, per-domain, each self-superposed
    E_total, E_pol, E_nonpol                 (n_frames,)             kcal/mol
    E_per_residue    (n_frames, n_residues)
    K_trans          (n_frames,)            kcal/mol

Optional stages degrade rather than fail, because the useful subset differs by
trajectory type. A positions-only ``.xtc`` still gives orientation, structural
and CA output (so DCCM/DSSP/RMSF all work); ``L/τ/ω/dL/dt`` and ``K_trans``
additionally need velocities+forces (a ``.trr``), and the ``E_*`` fields
additionally need ``freesasa``. Each chunk records what it actually contains in
its ``<chunk>.meta.json`` sidecar.
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
    # --- T1 additions -------------------------------------------------------
    sel_ca: str = "name CA"
    sel_align: str = ""
    domains: str = ""
    system: str = ""
    replica: int | None = None
    no_energy: bool = False


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
    p.add_argument(
        "--sel-ca",
        default="name CA",
        help="CA selection, applied WITHIN --selection. Drives ca_coords/resids "
             "and everything downstream (DCCM, RMSF). Default: 'name CA'",
    )
    p.add_argument(
        "--sel-align",
        default="",
        help="Stable-core subset of --sel-ca used as the superposition reference "
             "downstream (e.g. 'resid 20-90'). Empty = use all CA atoms. Excluding "
             "flexible termini here keeps their wobble out of every later alignment.",
    )
    p.add_argument(
        "--domains",
        default="",
        help="Per-domain RMSD spec: 'EF1:20-35,EF2:56-70' or a .json file. "
             "'+' joins discontiguous spans (N-lobe:1-40+55-70).",
    )
    p.add_argument("--system", default="", help="System label recorded in meta.json (e.g. wt)")
    p.add_argument("--replica", type=int, default=None, help="Replica index recorded in meta.json")
    p.add_argument(
        "--no-energy",
        action="store_true",
        help="Skip the freesasa energy stage. Implied automatically when freesasa "
             "is unavailable or the trajectory carries no velocities.",
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


def _ca_indexing(config: ExtractConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate the CA subset inside ``--selection``.

    Returns ``(ca_mask, ca_resids, align_mask)`` where ``ca_mask`` indexes the
    extracted atom axis and ``align_mask`` indexes the CA axis.

    Both masks are stored in the chunk so downstream stages never need the
    topology again: ``ca_mask`` slices ``positions`` to ``ca_coords`` here, and
    ``align_mask`` tells DCCM which CA atoms define the superposition core.
    """
    import MDAnalysis as mda

    u = mda.Universe(str(config.topology))
    sel = u.select_atoms(config.selection)
    ca = sel.select_atoms(config.sel_ca)
    if len(ca) == 0:
        raise ValueError(
            f"--sel-ca {config.sel_ca!r} selects no atoms within --selection "
            f"{config.selection!r} ({len(sel)} atoms)"
        )

    ca_mask = np.isin(sel.indices, ca.indices)
    if config.sel_align:
        align = ca.select_atoms(config.sel_align)
        if len(align) == 0:
            raise ValueError(
                f"--sel-align {config.sel_align!r} selects no atoms within "
                f"--sel-ca {config.sel_ca!r} ({len(ca)} CA atoms)"
            )
        align_mask = np.isin(ca.indices, align.indices)
    else:
        align_mask = np.ones(len(ca), dtype=bool)

    return ca_mask, ca.resids.astype(np.int32), align_mask


def _freesasa_available() -> bool:
    """Whether the energy stage can run at all.

    freesasa is sdist-only on Linux (needs a compiler), so it is routinely
    absent on an air-gapped cluster where the rest of rotmd installs fine from
    wheels. Extract degrades to a no-energy chunk rather than failing the whole
    SLURM array over an optional stage.
    """
    from importlib.util import find_spec

    return find_spec("freesasa") is not None


def _check_membrane_sel(config: ExtractConfig) -> None:
    """Fail fast on a membrane selection that matches nothing.

    The energy stage resolves ``--membrane-sel`` only at phase 7/7, so without
    this an hour of loading, structural and orientation work is thrown away at
    the very last step — and nothing is written. Checked here against the
    topology alone (cheap, no trajectory) so a typo'd selection costs seconds,
    matching how ``_ca_indexing`` already validates its own selections.
    """
    import MDAnalysis as mda

    u = mda.Universe(str(config.topology))
    if len(u.select_atoms(config.membrane_sel)) == 0:
        raise ValueError(
            f"--membrane-sel {config.membrane_sel!r} selects no atoms in "
            f"{config.topology.name}. Pass the selection matching your lipids, "
            f"or --no-energy if this system has no membrane."
        )


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

    from rotmd.analysis.domains import domain_masks, parse_domains
    from rotmd.core.inertia import principal_axes
    from rotmd.core.orientation import extract_orientation_trajectory, membrane_tilt_angle
    from rotmd.io.gromacs import compute_trajectory_energies, load_gromacs_trajectory
    from rotmd.io.meta import meta_path, write_json
    from rotmd.io.output import save_npz
    from rotmd.observables.energetics import kinetic_energy_translational
    from rotmd.observables.structural import compute_rmsd_trajectory, compute_structural_trajectory
    from rotmd.observables.unified import compute_all_observables

    # Validate every selection before the expensive work, not after it.
    if not config.no_energy and _freesasa_available():
        _check_membrane_sel(config)

    print(f"[1/7] Loading {config.trajectory.name} [{config.start}:{config.stop}:{config.step}]")
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

    print("[2/7] CA indexing (ca_coords, alignment core)")
    ca_mask, ca_resids, align_mask = _ca_indexing(config)
    # float32 halves the on-disk cost of the largest per-chunk array; CA
    # coordinates are ~0.001 Å-precise in the trajectory anyway, far below
    # float32's ~1e-4 Å resolution at protein length scales.
    ca_coords = traj["positions"][:, ca_mask, :].astype(np.float32)
    print(f"      {ca_coords.shape[1]} CA atoms, {int(align_mask.sum())} in the alignment core")

    print("[3/7] Reference + structural (RMSD, Rg) + per-domain RMSD")
    reference = _load_reference(config, traj["masses"])
    structural = compute_structural_trajectory(
        traj["positions"], traj["masses"], reference=reference, verbose=False
    )
    domains = parse_domains(config.domains)
    domain_names = list(domains)
    if domains:
        masks = domain_masks(ca_resids, domains)
        ref_ca = reference[ca_mask]
        ca_masses = traj["masses"][ca_mask]
        # Each domain is superposed on itself, so the column reports that
        # domain's internal deformation rather than its rigid-body motion
        # relative to the rest of the protein.
        rmsd_dom = np.column_stack([
            compute_rmsd_trajectory(
                ca_coords[:, masks[name]], ref_ca[masks[name]], ca_masses[masks[name]], align=True
            )
            for name in domain_names
        ])
        print(f"      domains: {', '.join(f'{n}({int(masks[n].sum())})' for n in domain_names)}")
    else:
        rmsd_dom = np.zeros((n_frames, 0))

    print("[4/7] Orientation (ZYZ Euler, rotation matrices)")
    euler, R = extract_orientation_trajectory(traj["positions"], traj["masses"])

    print("[5/7] Principal axes")
    axes = np.zeros((n_frames, 3, 3))
    moments = np.zeros((n_frames, 3))
    for i in range(n_frames):
        moments[i], axes[i] = principal_axes(traj["inertia_tensor"][i])

    # L/τ/ω need velocities *and* forces, i.e. a .trr written with
    # nstvout/nstfout > 0. A positions-only .xtc still yields everything the
    # DCCM / structural / DSSP path needs, so this degrades instead of raising.
    has_dynamics = traj["velocities"] is not None and traj["forces"] is not None
    obs = None
    if has_dynamics:
        print("[6/7] Vector observables (L, τ, ω, dL/dt)")
        obs = compute_all_observables(
            positions=traj["positions"],
            velocities=traj["velocities"],
            forces=traj["forces"],
            masses=traj["masses"],
            inertia_tensors=traj["inertia_tensor"],
            principal_axes=axes,
            membrane_normal=np.array([0.0, 0.0, 1.0]),
            times=traj["times"],
            verbose=False,
            com=traj["com"],
        )
    else:
        print("[6/7] Vector observables — SKIPPED (no velocities/forces in trajectory)")

    energies = None
    if config.no_energy:
        print("[7/7] Energies — SKIPPED (--no-energy)")
    elif not _freesasa_available():
        print("[7/7] Energies — SKIPPED (freesasa not installed)")
    else:
        print("[7/7] Energies (SASA + simplified electrostatic, parallel)")
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
        # --- T1: CA subsystem, consumed by dccm / rmsf without re-reading the
        # trajectory. Time-invariant keys (ca_resids, ca_align_mask) are stored
        # once per chunk and deduplicated by `rotmd merge`.
        "ca_coords": ca_coords,
        "ca_resids": ca_resids,
        "ca_align_mask": align_mask,
        "rmsd_dom": rmsd_dom,
    }
    if traj["velocities"] is not None:
        data["velocities"] = traj["velocities"]
        data["K_trans"] = np.array([
            kinetic_energy_translational(traj["velocities"][i], traj["masses"])
            for i in range(n_frames)
        ])
    if traj["forces"] is not None:
        data["forces"] = traj["forces"]
    if obs is not None:
        data.update(_flatten_observables(obs))
    if energies is not None:
        data["E_total"] = energies["Etot"]
        data["E_pol"] = energies["Epol"]
        data["E_nonpol"] = energies["Enonpol"]
        data["E_per_residue"] = energies["per_residue"]

    out = save_npz(config.output, data)

    times = traj["times"]
    meta = {
        "system": config.system or None,
        "replica": config.replica,
        "n_frames": int(n_frames),
        "n_atoms": int(traj["n_atoms"]),
        "dt_ps": float(np.median(np.diff(times))) if n_frames > 1 else None,
        "t_start_ps": float(times[0]),
        "t_end_ps": float(times[-1]),
        "selection": config.selection,
        "sel_ca": config.sel_ca,
        "sel_align": config.sel_align or config.sel_ca,
        "n_ca": int(ca_coords.shape[1]),
        "n_align": int(align_mask.sum()),
        "domains": {n: [list(s) for s in domains[n]] for n in domain_names},
        "domain_names": domain_names,
        "has_velocities": bool(traj["velocities"] is not None),
        "has_forces": bool(traj["forces"] is not None),
        "has_energy": energies is not None,
        "source": {
            "topology": str(config.topology),
            "trajectory": str(config.trajectory),
            "reference": str(config.reference),
            "start": config.start,
            "stop": config.stop,
            "step": config.step,
        },
        # Filled in by `rotmd equilibrate` (T3); every later stage reads it.
        "window": None,
    }
    write_json(meta_path(out), meta)

    size_mb = out.stat().st_size / (1024**2)
    print(f"✓ {out}  ({size_mb:.1f} MB, {n_frames} frames)")
    print(f"✓ {meta_path(out)}")
    return out


def merge(inputs: list[Path], output: Path, force: bool) -> Path:
    from rotmd.io.meta import merge_meta, meta_path, read_json, write_json
    from rotmd.io.output import load_npz, merge_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output
    inputs = sorted(inputs)
    print(f"Merging {len(inputs)} chunks → {output}")
    out = merge_npz(inputs, output)
    size_mb = out.stat().st_size / (1024**2)
    print(f"✓ {out}  ({size_mb:.1f} MB)")

    # Collapse the per-chunk sidecars into one per-replica sidecar, so
    # `rotmd equilibrate` has a single meta.json to extend. Chunks predating
    # T1 have no sidecar; that is not fatal, the merged npz is still valid.
    metas = [read_json(meta_path(p)) for p in inputs if meta_path(p).exists()]
    if metas:
        merged_meta = merge_meta(metas, load_npz(out)["time_ps"])
        write_json(meta_path(out), merged_meta)
        print(f"✓ {meta_path(out)}")
    return out


def equilibrate(
    inputs: list[Path],
    output: Path,
    column: str,
    nskip: int | None,
    method: str,
    pool: bool,
    force: bool,
) -> Path:
    """T3 — decide the production window and record it in ``window.json``."""
    from rotmd.analysis.equilibration import build_window, pool_windows
    from rotmd.io.meta import meta_path, read_json, update_meta, write_json
    from rotmd.io.output import load_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    if pool:
        windows = [read_json(p) for p in sorted(inputs)]
        result = pool_windows(windows)
        write_json(output, result)
        print(
            f"✓ {output}  (pooled {result['n_replicas']} replicas → "
            f"t0 = {result['t0_ps']:.1f} ps, the latest across replicas)"
        )
        if not result["all_cross_checks_agree"]:
            print("  ! at least one replica's cross-check disagreed — inspect before pooling")
        return output

    if len(inputs) != 1:
        raise SystemExit(
            f"equilibrate takes exactly one merged .npz (got {len(inputs)}); "
            "use --pool to combine per-replica window.json files"
        )

    src = inputs[0]
    data = load_npz(src)
    if column not in data:
        scalars = sorted(k for k, v in data.items() if v.ndim == 1 and k != "time_ps")
        raise SystemExit(f"{src} has no column {column!r}. Available 1-D columns: {scalars}")

    window = build_window(
        data["time_ps"], data[column], column=column, nskip=nskip, method=method
    )
    window["source_npz"] = str(src)
    write_json(output, window)

    # Extend the replica sidecar so the window travels with the data.
    sidecar = meta_path(src)
    if sidecar.exists():
        update_meta(sidecar, window=window)

    xc = window["cross_check"]
    print(
        f"✓ {output}\n"
        f"  t0 = {window['t0_ps']:.1f} ps (frame {window['t0_index']}), "
        f"discarding {window['frac_discarded']:.1%} of {window['n_frames']} frames\n"
        f"  g = {window['g']:.2f}, N_eff = {window['n_eff']:.0f}  [{window['method']}]\n"
        f"  cross-check ({xc['method']}): t0 = {xc['t0_ps']:.1f} ps"
    )
    if not xc["agree"]:
        print(
            f"  ! estimators disagree by {xc['rel_diff']:.1%} of the trajectory "
            f"(> {xc['tolerance']:.0%}) — plot the order parameter before trusting t0"
        )
    return output


def dccm(
    input_npz: Path, window_path: Path, output: Path, force: bool, max_blocks: int = 50
) -> Path:
    """T4 — dynamic cross-correlation map over the production window."""
    from rotmd.analysis.dccm import compute_dccm
    from rotmd.analysis.equilibration import apply_window
    from rotmd.io.meta import read_json
    from rotmd.io.output import load_npz, save_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    data = load_npz(input_npz)
    missing = [k for k in ("time_ps", "ca_coords", "ca_resids", "ca_align_mask") if k not in data]
    if missing:
        raise SystemExit(
            f"{input_npz} lacks {missing} — it predates the CA arrays. "
            "Re-run `rotmd extract` to regenerate it."
        )

    window = read_json(window_path)
    frame_mask = apply_window(data["time_ps"], window)

    # Blocks must be at least `g` frames long to count as independent samples
    # for `rotmd compare`'s bootstrap; cap the count so each block still has
    # enough frames to give a stable per-block map.
    n_used = int(frame_mask.sum())
    g = float(window.get("g", 1.0)) or 1.0
    n_blocks = int(max(2, min(max_blocks, n_used // max(1, int(np.ceil(g))))))

    result = compute_dccm(
        data["ca_coords"],
        data["ca_align_mask"],
        data["ca_resids"],
        frame_mask=frame_mask,
        n_blocks=n_blocks,
    )
    result["t0_ps"] = np.float64(window["t0_ps"])
    save_npz(output, result)

    n_used = int(result["frames_used"])
    off_diag = result["dcc"][~np.eye(len(result["dcc"]), dtype=bool)]
    print(
        f"✓ {output}\n"
        f"  {result['dcc'].shape[0]} CA residues, {n_used}/{len(data['time_ps'])} frames "
        f"(t0 = {window['t0_ps']:.1f} ps)\n"
        f"  superposition converged in {int(result['align_iterations'])} iterations on "
        f"{int(result['align_mask'].sum())} core atoms\n"
        f"  off-diagonal correlation: mean {off_diag.mean():+.3f}, "
        f"range [{off_diag.min():+.3f}, {off_diag.max():+.3f}]\n"
        f"  {len(result['dcc_blocks'])} bootstrap blocks stored (g = {g:.1f})"
    )
    return output


def dssp(
    topology: Path,
    trajectory: Path,
    window_path: Path | None,
    output: Path,
    selection: str,
    step: int,
    force: bool,
) -> Path:
    """T8 — per-residue secondary-structure occupancy over the window."""
    from rotmd.analysis.dssp import DSSP_CODES, compute_dssp
    from rotmd.io.meta import read_json
    from rotmd.io.output import save_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    t0 = float(read_json(window_path)["t0_ps"]) if window_path else None
    result = compute_dssp(
        str(topology), str(trajectory), t0_ps=t0, selection=selection, step=step
    )
    save_npz(output, result)

    occ = result["occupancy"]
    mean_by_class = occ.mean(axis=0)
    print(
        f"✓ {output}\n"
        f"  {occ.shape[0]} residues, {int(result['frames_used'])} frames "
        f"(from frame {int(result['start_frame'])})\n"
        "  mean occupancy: "
        + ", ".join(f"{code}={mean_by_class[i]:.2f}" for i, code in enumerate(DSSP_CODES))
    )
    return output


def local(
    topology: Path,
    trajectory: Path,
    site: int,
    output: Path,
    window_path: Path | None,
    merged: Path | None,
    cutoff: float,
    radius: float,
    step: int,
    force: bool,
) -> Path:
    """T5a — salt bridges, H-bonds and RMSF around the mutation site."""
    from rotmd.analysis.dssp import first_frame_at_or_after
    from rotmd.analysis.equilibration import apply_window
    from rotmd.analysis.local import hydrogen_bonds, rmsf_per_residue, salt_bridges
    from rotmd.io.meta import read_json
    from rotmd.io.output import load_npz, save_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    import MDAnalysis as mda

    universe = mda.Universe(str(topology), str(trajectory))
    window = read_json(window_path) if window_path else None
    start = first_frame_at_or_after(universe, float(window["t0_ps"])) if window else 0

    bridges = salt_bridges(universe, site, cutoff=cutoff, start=start, step=step)
    payload: dict[str, np.ndarray] = {
        "sb_resids": bridges["resids"],
        "sb_occupancy": bridges["occupancy"],
        "sb_any_occupancy": bridges["any_occupancy"],
        "sb_cutoff": bridges["cutoff"],
        "n_frames": bridges["n_frames"],
        "site": np.int64(site),
    }

    try:
        hbonds = hydrogen_bonds(universe, site, radius=radius, start=start, step=step)
        payload["hb_pairs"] = hbonds["pairs"]
        payload["hb_occupancy"] = hbonds["occupancy"]
    except ValueError as exc:
        print(f"  ! hydrogen bonds skipped: {exc}")

    # RMSF comes from the extracted CA arrays so it uses the same superposition
    # (and the same window) as the DCCM it will be read alongside.
    if merged is not None:
        data = load_npz(merged)
        frame_mask = apply_window(data["time_ps"], window) if window else None
        rmsf = rmsf_per_residue(
            data["ca_coords"], data["ca_align_mask"], data["ca_resids"], frame_mask
        )
        payload["rmsf"] = rmsf["rmsf"]
        payload["rmsf_resids"] = rmsf["resids"]

    save_npz(output, payload)

    top = [
        f"{int(r)}:{o:.0%}"
        for r, o in zip(bridges["resids"], bridges["occupancy"], strict=True)
        if o > 0
    ][:5]
    print(
        f"✓ {output}\n"
        f"  site {site}: engaged with a carboxylate in "
        f"{float(bridges['any_occupancy']):.1%} of {int(bridges['n_frames'])} frames\n"
        f"  top partners: {', '.join(top) if top else 'none within cutoff'}"
    )
    if "rmsf" in payload:
        print(f"  RMSF: mean {payload['rmsf'].mean():.2f} Å, max {payload['rmsf'].max():.2f} Å")
    return output


def apbs(
    topology: Path,
    trajectory: Path,
    output: Path,
    window_path: Path | None,
    n_samples: int,
    selection: str,
    forcefield: str,
    workdir: Path | None,
    force: bool,
) -> Path:
    """T5c — ensemble-averaged Poisson-Boltzmann electrostatics."""
    from rotmd.analysis.apbs import compute_pb_ensemble
    from rotmd.io.meta import read_json
    from rotmd.io.output import save_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    t0 = float(read_json(window_path)["t0_ps"]) if window_path else None
    result = compute_pb_ensemble(
        str(topology), str(trajectory), t0_ps=t0, n_samples=n_samples,
        selection=selection, forcefield=forcefield,
        workdir=str(workdir) if workdir else None,
    )
    save_npz(output, {k: v for k, v in result.items() if k != "workdir"})

    print(
        f"✓ {output}\n"
        f"  ΔG_elec = {float(result['mean_kcal']):.1f} ± {float(result['sem_kcal']):.1f} "
        f"kcal/mol  (mean ± SEM over {int(result['n_frames_used'])} frames)\n"
        f"  scratch: {result['workdir']}"
    )
    return output


def coulomb(
    structure: Path,
    trajectory: Path,
    topology_top: Path,
    site: int,
    output: Path,
    window_path: Path | None,
    radius: float,
    selection: str,
    workdir: Path | None,
    force: bool,
) -> Path:
    """T5b — site/shell Coulomb + LJ decomposition from a GROMACS rerun."""
    import tempfile

    from rotmd.analysis.coulomb import coulomb_decomposition
    from rotmd.io.meta import read_json
    from rotmd.io.output import save_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    g = 1.0
    if window_path:
        window = read_json(window_path)
        g = float(window.get("g", 1.0))

    scratch = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rotmd-rerun-"))
    result = coulomb_decomposition(
        topology_top, structure, trajectory, site=site,
        workdir=scratch, g=g, radius=radius, selection=selection,
    )
    save_npz(output, result)

    lines = [
        f"✓ {output}",
        f"  site {site}: {int(result['group_site_atoms'])} atoms vs "
        f"{int(result['group_shell_atoms'])} shell atoms within {radius} Å, "
        f"{int(result['n_frames'])} frames",
    ]
    for kind, label in (("Coul_SR", "Coulomb (SR)"), ("LJ_SR", "Lennard-Jones (SR)")):
        if f"{kind}_mean" in result:
            mean = float(result[f"{kind}_mean"])
            err = float(result[f"{kind}_stderr"])
            marker = "" if abs(mean) > 2 * err else "   (not resolved above the noise)"
            lines.append(
                f"  {label:20s} {mean:9.2f} ± {err:.2f} kJ/mol "
                f"[{int(result[f'{kind}_n_blocks'])} blocks × "
                f"{int(result[f'{kind}_block_size'])} frames]{marker}"
            )
    print("\n".join(lines))
    return output


def compare(
    system_a: list[Path],
    system_b: list[Path],
    output: Path,
    label_a: str,
    label_b: str,
    site: int | None,
    exclude_within: int,
    n_boot: int,
    alpha: float,
    seed: int,
    figure: Path | None,
    force: bool,
    dssp_a: list[Path] | None = None,
    dssp_b: list[Path] | None = None,
    dssp_figure: Path | None = None,
) -> Path:
    """T6 — ΔDCCM between two systems, with bootstrap significance."""
    from rotmd.analysis.compare import compare_dccm
    from rotmd.io.output import load_npz, save_npz

    if output.exists() and not force:
        print(f"✓ {output} exists — skipping (use --force to overwrite)")
        return output

    def _load(paths: list[Path], label: str):
        maps, blocks, resids = [], [], None
        for path in sorted(paths):
            data = load_npz(path)
            maps.append(data["dcc"])
            if "dcc_blocks" in data:
                blocks.append(data["dcc_blocks"])
            if resids is None:
                resids = data["resids"]
            elif not np.array_equal(resids, data["resids"]):
                raise SystemExit(
                    f"{label}: {path} has different resids from the first replica — "
                    "all replicas must share one residue numbering"
                )
        return maps, blocks, resids

    maps_a, blocks_a, resids_a = _load(system_a, label_a)
    maps_b, blocks_b, resids_b = _load(system_b, label_b)
    if not np.array_equal(resids_a, resids_b):
        raise SystemExit(
            f"{label_a} and {label_b} have different residue numbering — "
            "they cannot be compared cell by cell"
        )

    if not (blocks_a and blocks_b):
        print("  ! no dcc_blocks found — reporting ΔDCCM without significance. "
              "Re-run `rotmd dccm` to regenerate them.")

    result = compare_dccm(
        maps_a, maps_b, resids_a,
        blocks_a=blocks_a or None, blocks_b=blocks_b or None,
        site=site, exclude_within=exclude_within,
        n_boot=n_boot, alpha=alpha, seed=seed,
    )
    save_npz(output, result)

    delta = result["ddccm"]
    off = ~np.eye(len(delta), dtype=bool)
    lines = [
        f"✓ {output}",
        f"  {label_a}: {len(maps_a)} replicas   {label_b}: {len(maps_b)} replicas",
        f"  |ΔDCCM| max {np.abs(delta[off]).max():.3f}, mean {np.abs(delta[off]).mean():.3f}",
    ]
    if "significant" in result:
        frac = float(result["significant"][off].mean())
        lines.append(
            f"  {frac:.1%} of off-diagonal cells significant at alpha={alpha} "
            f"({n_boot} block-bootstrap resamples)"
        )
    if "max_distal_change" in result:
        lines.append(
            f"  strongest significant distal change (|i − {site}| ≥ {exclude_within}): "
            f"{float(result['max_distal_change']):.3f}"
        )
    print("\n".join(lines))

    if figure is not None:
        from rotmd.analysis.plots import plot_ddccm

        plot_ddccm(result, figure, label_a=label_a, label_b=label_b,
                   title=f"{label_b} vs {label_a}")
        print(f"✓ {figure}")

    # T8 hand-off: per-residue Δoccupancy, averaged over each system's replicas.
    if dssp_a and dssp_b:
        from rotmd.analysis.dssp import delta_occupancy

        def _mean_occupancy(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
            loaded = [load_npz(p) for p in sorted(paths)]
            return (
                np.mean([d["occupancy"] for d in loaded], axis=0),
                loaded[0]["resids"],
            )

        occ_a, dssp_resids = _mean_occupancy(dssp_a)
        occ_b, _ = _mean_occupancy(dssp_b)
        delta_occ = delta_occupancy(occ_a, occ_b)

        save_npz(
            output.with_name(f"{output.stem}_dssp.npz"),
            {
                "occupancy_a": occ_a, "occupancy_b": occ_b,
                "delta_occupancy": delta_occ, "resids": dssp_resids,
            },
        )
        biggest = int(np.argmax(np.abs(delta_occ).max(axis=1)))
        print(
            f"✓ {output.with_name(f'{output.stem}_dssp.npz')}\n"
            f"  largest Δoccupancy at resid {int(dssp_resids[biggest])}: "
            f"{np.abs(delta_occ[biggest]).max():+.2f}"
        )

        if dssp_figure is not None:
            from rotmd.analysis.plots import plot_dssp_delta

            plot_dssp_delta(
                occ_a, occ_b, dssp_resids, dssp_figure,
                label_a=label_a, label_b=label_b, site=site,
                title=f"Secondary structure — {label_b} vs {label_a}",
            )
            print(f"✓ {dssp_figure}")

    return output


def methods(
    mdp: list[str], topology: Path | None, structure: Path | None, outdir: Path, force: bool
) -> Path:
    """T7 — render methods.json + methods.md from the simulation inputs."""
    from rotmd.analysis.methods import build_methods
    from rotmd.io.meta import write_json

    json_path = outdir / "methods.json"
    md_path = outdir / "methods.md"
    if json_path.exists() and not force:
        print(f"✓ {json_path} exists — skipping (use --force to overwrite)")
        return json_path

    # Accept "name=path" to label stages explicitly, or a bare path whose stem
    # becomes the label (em.mdp -> "em").
    files: dict[str, Path] = {}
    for item in mdp:
        name, sep, path = item.partition("=")
        chosen = Path(path) if sep else Path(name)
        if not chosen.exists():
            raise SystemExit(f"mdp file not found: {chosen}")
        files[name if sep else chosen.stem] = chosen

    payload, markdown = build_methods(files, topology, structure)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(json_path, payload)
    md_path.write_text(markdown)

    print(f"✓ {json_path}\n✓ {md_path}\n  stages: {', '.join(files)}")
    return json_path


def plot_equil(
    inputs: list[Path], outdir: Path, window_path: Path | None, force: bool
) -> Path:
    """T2 — equilibration figures: RMSD / Rg / per-domain with t0 marked."""
    from rotmd.analysis.plots import plot_equilibration, plot_replica_overlay
    from rotmd.io.meta import meta_path, read_json
    from rotmd.io.output import load_npz

    window = read_json(window_path) if window_path else None
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = []
    written = []
    for src in sorted(inputs):
        data = load_npz(src)
        sidecar = meta_path(src)
        meta = read_json(sidecar) if sidecar.exists() else {}
        label = "/".join(
            str(meta[k]) for k in ("system", "replica") if meta.get(k) is not None
        ) or src.stem

        target = outdir / f"equil_{src.stem}.png"
        if target.exists() and not force:
            print(f"✓ {target} exists — skipping (use --force to overwrite)")
        else:
            plot_equilibration(
                data, target, window=window,
                domain_names=meta.get("domain_names"), title=label,
            )
            written.append(target)
            print(f"✓ {target}")
        datasets.append((label, data))

    if len(datasets) > 1:
        overlay = outdir / "equil_overlay.png"
        if overlay.exists() and not force:
            print(f"✓ {overlay} exists — skipping (use --force to overwrite)")
        else:
            plot_replica_overlay(datasets, overlay, window=window, title="Replica overlay")
            written.append(overlay)
            print(f"✓ {overlay}")

    return written[0] if written else outdir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rotmd", description="rotmd — chunked extract pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract one chunk → one .npz")
    _add_extract_args(p_extract)

    p_merge = sub.add_parser("merge", help="Concatenate chunk .npz files along time")
    p_merge.add_argument("inputs", nargs="+", type=Path)
    p_merge.add_argument("-o", "--output", required=True, type=Path)
    p_merge.add_argument("--force", action="store_true")

    p_eq = sub.add_parser(
        "equilibrate",
        help="Detect the production window → window.json",
        description="Detect where equilibration ends, so every later analysis "
                    "slices the same explicit, recorded frame range.",
    )
    p_eq.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One merged .npz, or several window.json files with --pool",
    )
    p_eq.add_argument("-o", "--output", required=True, type=Path, help="Output window.json")
    p_eq.add_argument(
        "--column", default="rmsd", help="1-D order parameter to analyse (default: rmsd)"
    )
    p_eq.add_argument(
        "--nskip", type=int, default=None, help="Stride over candidate t0 (default: ~200 candidates)"
    )
    p_eq.add_argument(
        "--method",
        default="auto",
        choices=["auto", "pymbar", "native"],
        help="Estimator for the statistical inefficiency (default: pymbar if installed)",
    )
    p_eq.add_argument(
        "--pool",
        action="store_true",
        help="Combine per-replica window.json files into one conservative common window",
    )
    p_eq.add_argument("--force", action="store_true")

    p_dccm = sub.add_parser(
        "dccm",
        help="Dynamic cross-correlation map over the production window",
        description="CA-CA displacement correlations after superposing every "
                    "frame on the average structure over the alignment core.",
    )
    p_dccm.add_argument("input", type=Path, help="Merged .npz carrying ca_coords")
    p_dccm.add_argument(
        "--window", required=True, type=Path, help="window.json from `rotmd equilibrate`"
    )
    p_dccm.add_argument("-o", "--output", required=True, type=Path)
    p_dccm.add_argument("--force", action="store_true")

    p_methods = sub.add_parser(
        "methods",
        help="Auto-generate methods.json + methods.md from .mdp/topology",
        description="Render the methods section from the files that actually "
                    "produced the trajectories, so it cannot drift from them.",
    )
    p_methods.add_argument(
        "--mdp",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="An .mdp stage; repeatable. 'nvt=nvt.mdp' labels it, a bare path "
             "uses the filename stem. Order given is the order reported.",
    )
    p_methods.add_argument("--topology", type=Path, help="topol.top (force field, water, ions)")
    p_methods.add_argument("--structure", type=Path, help="Any structure/tpr for atom count + box")
    p_methods.add_argument("-o", "--outdir", required=True, type=Path)
    p_methods.add_argument("--force", action="store_true")

    p_plot = sub.add_parser(
        "plot-equil",
        help="Equilibration figures (RMSD/Rg/per-domain) with t0 marked",
        description="One figure per replica, plus a replica overlay when given "
                    "more than one input.",
    )
    p_plot.add_argument("inputs", nargs="+", type=Path, help="Merged .npz files")
    p_plot.add_argument("-o", "--outdir", required=True, type=Path)
    p_plot.add_argument(
        "--window", type=Path, help="window.json — draws t0 and shades the discarded region"
    )
    p_plot.add_argument("--force", action="store_true")

    p_cmp = sub.add_parser(
        "compare",
        help="ΔDCCM between two systems with bootstrap significance",
        description="Pool replicas per system in Fisher-z space, difference the "
                    "maps, and attach a block-bootstrap CI to every cell.",
    )
    p_cmp.add_argument("--a", nargs="+", required=True, type=Path, metavar="DCCM_NPZ",
                       help="Reference system replicas (e.g. WT)")
    p_cmp.add_argument("--b", nargs="+", required=True, type=Path, metavar="DCCM_NPZ",
                       help="Comparison system replicas (e.g. the mutant)")
    p_cmp.add_argument("-o", "--output", required=True, type=Path)
    p_cmp.add_argument("--label-a", default="A")
    p_cmp.add_argument("--label-b", default="B")
    p_cmp.add_argument("--site", type=int, help="Mutation site for the distal view (e.g. 75)")
    p_cmp.add_argument("--exclude-within", type=int, default=5,
                       help="Residues within this distance of --site are masked (default: 5)")
    p_cmp.add_argument("--n-boot", type=int, default=500)
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument("--seed", type=int, default=0, help="Bootstrap seed (keeps runs reproducible)")
    p_cmp.add_argument("--figure", type=Path, help="Also render the ΔDCCM heatmap here")
    p_cmp.add_argument("--dssp-a", nargs="+", type=Path, help="dssp.npz replicas for system A")
    p_cmp.add_argument("--dssp-b", nargs="+", type=Path, help="dssp.npz replicas for system B")
    p_cmp.add_argument("--dssp-figure", type=Path, help="Render the Δoccupancy figure here")
    p_cmp.add_argument("--force", action="store_true")

    p_dssp = sub.add_parser(
        "dssp",
        help="Per-residue secondary-structure occupancy over the window",
        description="Runs MDAnalysis' pure-Python DSSP (no external binary) and "
                    "reduces per-frame assignments to per-residue occupancy.",
    )
    p_dssp.add_argument("topology", type=Path)
    p_dssp.add_argument("trajectory", type=Path)
    p_dssp.add_argument("-o", "--output", required=True, type=Path)
    p_dssp.add_argument("--window", type=Path, help="window.json — analyse only post-t0 frames")
    p_dssp.add_argument("--selection", default="protein")
    p_dssp.add_argument("--step", type=int, default=1, help="Frame stride (DSSP is the slow part)")
    p_dssp.add_argument("--force", action="store_true")

    p_local = sub.add_parser(
        "local",
        help="Salt bridges / H-bonds / RMSF around the mutation site",
        description="Heavy-atom salt-bridge occupancy per partner residue, the "
                    "local hydrogen-bond network, and windowed RMSF.",
    )
    p_local.add_argument("topology", type=Path)
    p_local.add_argument("trajectory", type=Path)
    p_local.add_argument("--site", required=True, type=int, help="Residue of interest (e.g. 75)")
    p_local.add_argument("-o", "--output", required=True, type=Path)
    p_local.add_argument("--window", type=Path, help="window.json — analyse only post-t0 frames")
    p_local.add_argument("--merged", type=Path, help="Merged .npz, adds RMSF from ca_coords")
    p_local.add_argument("--cutoff", type=float, default=4.0, help="Salt-bridge cutoff Å (default 4)")
    p_local.add_argument("--radius", type=float, default=10.0, help="H-bond shell radius Å")
    p_local.add_argument("--step", type=int, default=1)
    p_local.add_argument("--force", action="store_true")

    p_apbs = sub.add_parser(
        "apbs",
        help="Ensemble Poisson-Boltzmann electrostatics (PDB2PQR + APBS)",
        description="Polar solvation free energy averaged over frames sampled "
                    "from the production window. Needs the external pdb2pqr30 "
                    "and apbs binaries.",
    )
    p_apbs.add_argument("topology", type=Path)
    p_apbs.add_argument("trajectory", type=Path)
    p_apbs.add_argument("-o", "--output", required=True, type=Path)
    p_apbs.add_argument("--window", type=Path, help="window.json — sample only post-t0 frames")
    p_apbs.add_argument("--n-samples", type=int, default=50,
                        help="Frames to sample across the window (default 50)")
    p_apbs.add_argument("--selection", default="protein")
    p_apbs.add_argument("--forcefield", default="CHARMM", help="PDB2PQR force field")
    p_apbs.add_argument("--workdir", type=Path, help="Scratch dir (default: a temp dir)")
    p_apbs.add_argument("--force", action="store_true")

    p_coul = sub.add_parser(
        "coulomb",
        help="Per-residue Coulomb/LJ decomposition via `gmx mdrun -rerun`",
        description="Re-evaluates the force field on existing frames with "
                    "energygrps set, giving the site-shell Coulomb and LJ "
                    "energies with block-averaged errors. Needs the topology "
                    "that produced the trajectory.",
    )
    p_coul.add_argument("structure", type=Path, help="Structure matching the topology (.gro)")
    p_coul.add_argument("trajectory", type=Path, help="Frames to re-evaluate (.xtc/.trr)")
    p_coul.add_argument("--top", required=True, type=Path, dest="topology_top",
                        help="topol.top matching the trajectory atom-for-atom")
    p_coul.add_argument("--site", required=True, type=int)
    p_coul.add_argument("-o", "--output", required=True, type=Path)
    p_coul.add_argument("--window", type=Path, help="window.json — supplies g for block errors")
    p_coul.add_argument("--radius", type=float, default=10.0, help="Shell radius Å (default 10)")
    p_coul.add_argument("--selection", default="protein")
    p_coul.add_argument("--workdir", type=Path)
    p_coul.add_argument("--force", action="store_true")

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
            sel_ca=args.sel_ca,
            sel_align=args.sel_align,
            domains=args.domains,
            system=args.system,
            replica=args.replica,
            no_energy=args.no_energy,
        )
        extract(cfg)
        return 0
    if args.cmd == "merge":
        merge(args.inputs, args.output, args.force)
        return 0
    if args.cmd == "equilibrate":
        equilibrate(
            args.inputs,
            args.output,
            column=args.column,
            nskip=args.nskip,
            method=args.method,
            pool=args.pool,
            force=args.force,
        )
        return 0
    if args.cmd == "dccm":
        dccm(args.input, args.window, args.output, args.force)
        return 0
    if args.cmd == "methods":
        if not args.mdp:
            raise SystemExit("methods needs at least one --mdp")
        methods(args.mdp, args.topology, args.structure, args.outdir, args.force)
        return 0
    if args.cmd == "plot-equil":
        plot_equil(args.inputs, args.outdir, args.window, args.force)
        return 0
    if args.cmd == "compare":
        compare(
            args.a, args.b, args.output,
            label_a=args.label_a, label_b=args.label_b,
            site=args.site, exclude_within=args.exclude_within,
            n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
            figure=args.figure, force=args.force,
            dssp_a=args.dssp_a, dssp_b=args.dssp_b, dssp_figure=args.dssp_figure,
        )
        return 0
    if args.cmd == "dssp":
        dssp(
            args.topology, args.trajectory, args.window, args.output,
            selection=args.selection, step=args.step, force=args.force,
        )
        return 0
    if args.cmd == "coulomb":
        coulomb(
            args.structure, args.trajectory, args.topology_top, args.site, args.output,
            window_path=args.window, radius=args.radius, selection=args.selection,
            workdir=args.workdir, force=args.force,
        )
        return 0
    if args.cmd == "apbs":
        apbs(
            args.topology, args.trajectory, args.output,
            window_path=args.window, n_samples=args.n_samples,
            selection=args.selection, forcefield=args.forcefield,
            workdir=args.workdir, force=args.force,
        )
        return 0
    if args.cmd == "local":
        local(
            args.topology, args.trajectory, args.site, args.output,
            window_path=args.window, merged=args.merged, cutoff=args.cutoff,
            radius=args.radius, step=args.step, force=args.force,
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
