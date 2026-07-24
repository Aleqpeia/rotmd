"""
GROMACS File I/O

This module provides utilities for reading GROMACS trajectory and topology files
via MDAnalysis, with specialized handling for protein orientation analysis.

Key Features:
- XTC/TRR trajectory readers
- GRO/PDB topology readers
- Automatic protein selection
- Velocity/force detection
- Parallel batch computation via numba kernels

Streaming/chunked readers for very large trajectories live in
``rotmd.analysis.io``; the extract CLI loads one chunk per invocation whole.

Author: Mykyta Bobylyow
Date: 2025
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List

import numpy as np

from rotmd.core import kernels as K  # Parallel batch functions

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    warnings.warn("MDAnalysis not available. Install with: pip install MDAnalysis")


def load_gromacs_trajectory(
    topology: str,
    trajectory: str,
    selection: str = "protein",
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    align_to_first: bool = False,
    center: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Load GROMACS trajectory for orientation analysis.

    Args:
        topology: Topology file (.gro, .pdb, .tpr)
        trajectory: Trajectory file (.xtc, .trr)
        selection: MDAnalysis selection string
        start: First frame index
        stop: Last frame index (None = all)
        step: Frame step size
        align_to_first: Align all frames to first frame
        center: Center at origin
        verbose: Print progress

    Returns:
        data: Dictionary with:
            - positions: (n_frames, n_atoms, 3) in Å
            - masses: (n_atoms,) in amu
            - times: (n_frames,) in ps
            - velocities: Optional (n_frames, n_atoms, 3) in Å/ps
            - forces: Optional (n_frames, n_atoms, 3) in kJ/(mol·nm)
            - has_velocities: bool
            - has_forces: bool
            - n_frames: int
            - n_atoms: int

    Raises:
        ImportError: If MDAnalysis not available
        ValueError: If selection returns zero atoms

    Example:
        >>> data = load_gromacs_trajectory('system.gro', 'traj.trr')
        >>> print(f"Loaded {data['n_frames']} frames, {data['n_atoms']} atoms")
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis required. Install with: pip install MDAnalysis")

    if verbose:
        print(f"Loading GROMACS trajectory...")
        print(f"  Topology: {topology}")
        print(f"  Trajectory: {trajectory}")
        print(f"  Selection: '{selection}'")

    # Load universe
    u = mda.Universe(topology, trajectory)
    atoms = u.select_atoms(selection)

    if len(atoms) == 0:
        raise ValueError(f"Selection '{selection}' returned 0 atoms")

    if verbose:
        print(f"  Selected {len(atoms)} atoms")

    # Check data availability
    has_velocities = False
    try:
        _ = atoms.velocities
        has_velocities = True
        if verbose:
            print(f"  ✓ Velocities available")
    except (AttributeError, mda.exceptions.NoDataError):
        if verbose:
            print(f"  ✗ Velocities NOT available")

    has_forces = False
    try:
        _ = atoms.forces
        has_forces = True
        if verbose:
            print(f"  ✓ Forces available")
    except (AttributeError, mda.exceptions.NoDataError):
        if verbose:
            print(f"  ✗ Forces NOT available")

    # ==========================================================================
    # PHASE 1: Sequential data loading (MDAnalysis I/O limitation)
    # ==========================================================================
    positions_list = []
    velocities_list = [] if has_velocities else None
    forces_list = [] if has_forces else None
    times_list = []

    n_frames = len(u.trajectory[start:stop:step])
    if n_frames == 0:
        # Fail here, with the numbers, rather than letting an empty list become
        # a 1-D (0,) array downstream: the numba kernels then report
        # "Unknown attribute 'shape' of type float64", which says nothing about
        # the actual cause — a --start past the end of this trajectory.
        raise ValueError(
            f"[{start}:{stop}:{step}] selects 0 of the {len(u.trajectory)} frames in "
            f"{trajectory}. --start ({start}) is at or past the end of this "
            f"trajectory, so there is nothing to extract."
        )
    if verbose:
        print(f"  Phase 1: Loading {n_frames} frames (sequential I/O)...")

    # Reference for alignment
    ref_positions = None
    masses = atoms.masses.copy()

    for ts in u.trajectory[start:stop:step]:
        if verbose and (ts.frame % max(1, n_frames // 10) == 0):
            print(f"    Frame {ts.frame}/{n_frames}")

        pos = atoms.positions.copy()

        # Center at origin
        if center:
            com = np.average(pos, weights=masses, axis=0)
            pos -= com

        # Align to first frame
        if align_to_first:
            if ref_positions is None:
                ref_positions = pos.copy()
            else:
                R = _rotation_matrix_align(pos, ref_positions, masses)
                pos = pos @ R.T

        positions_list.append(pos)

        if has_velocities:
            velocities_list.append(atoms.velocities.copy())

        if has_forces:
            forces_list.append(atoms.forces.copy())

        times_list.append(ts.time)

    # Convert to arrays
    positions = np.array(positions_list)
    times = np.array(times_list)
    velocities = np.array(velocities_list) if has_velocities else None
    forces = np.array(forces_list) if has_forces else None

    if verbose:
        print(f"  ✓ Loaded {n_frames} frames")

    # ==========================================================================
    # PHASE 2: Parallel batch computation (numba prange)
    # ==========================================================================
    if verbose:
        print(f"  Phase 2: Computing observables (parallel)...")

    # Compute COM for all frames in parallel
    com_batch = K.compute_com_batch(positions, masses)

    # Compute inertia tensors for all frames in parallel
    inertia_batch = K.inertia_tensor_batch(positions, masses, com_batch)

    if verbose:
        print(f"  ✓ Computed inertia tensors ({n_frames} frames)")

    # ==========================================================================
    # Return structured data
    # ==========================================================================
    data = {
        "positions": positions,
        "masses": masses,
        "times": times,
        "velocities": velocities,
        "forces": forces,
        "com": com_batch,
        "inertia_tensor": inertia_batch,
        "has_velocities": has_velocities,
        "has_forces": has_forces,
        "n_frames": n_frames,
        "n_atoms": len(atoms),
    }

    return data


def _compute_energy_chunk(args: Tuple) -> List[Dict]:
    """
    Worker function for parallel energy computation.

    Each worker creates its own MDAnalysis Universe to avoid pickling issues.

    Args:
        args: Tuple of (topology, trajectory, selection, frame_indices, membrane_center_z)

    Returns:
        List of energy dicts for each frame in the chunk
    """
    topology, trajectory, selection, frame_indices, membrane_center_z = args

    # Each worker creates its own Universe (can't pickle MDAnalysis objects)
    import MDAnalysis as mda
    from rotmd.models.energy import TotalEnergy

    u = mda.Universe(topology, trajectory)
    atoms = u.select_atoms(selection)
    energy_calc = TotalEnergy()

    results = []
    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        energy = energy_calc.calculate(
            protein_atoms=atoms, membrane_center_z=membrane_center_z
        )
        results.append(
            {
                "frame_idx": frame_idx,
                "total": energy["total"],
                "electrostatic": energy["electrostatic"],
                "hydrophobic": energy["hydrophobic"],
                "per_residue": energy["per_residue"],
            }
        )

    return results


def compute_trajectory_energies(
    topology: str,
    trajectory: str,
    selection: str = "protein",
    membrane_sel: str = "resname CHL1",
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compute energies for trajectory frames with parallel processing.

    This function is intentionally separate from load_gromacs_trajectory to allow
    fast extraction without the energy computation bottleneck.

    Parallelization: Uses ProcessPoolExecutor to distribute frame computation
    across N CPU cores. Each worker creates its own MDAnalysis Universe.

    Args:
        topology: Topology file (.gro, .pdb, .tpr)
        trajectory: Trajectory file (.xtc, .trr)
        selection: MDAnalysis selection string for protein
        membrane_sel: MDAnalysis selection string for membrane
        start: First frame index
        stop: Last frame index (None = all)
        step: Frame step size
        n_workers: Number of parallel workers (default: CPU count)
        verbose: Print progress

    Returns:
        energies: Dictionary with:
            - Etot: (n_frames,) total energy in kcal/mol
            - Epol: (n_frames,) electrostatic energy
            - Enonpol: (n_frames,) hydrophobic energy
            - per_residue: (n_frames, n_residues) per-residue energies
            - normal: (3,) membrane normal vector

    Example:
        >>> energies = compute_trajectory_energies('system.tpr', 'traj.trr', n_workers=8)
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis required")

    from rotmd.base import membrane_interface

    u = mda.Universe(topology, trajectory)

    # Get frame indices
    frame_indices = list(range(len(u.trajectory)))[start:stop:step]
    n_frames = len(frame_indices)

    if verbose:
        print(f"Computing energies for {n_frames} frames...")

    # Get membrane center once (assumes static membrane)
    membrane_center_z = membrane_interface.get_membrane_center_z(
        u, membrane_sel=membrane_sel, method="density"
    )
    normal = membrane_interface.get_membrane_normal(u, membrane_sel=membrane_sel)

    # Determine number of workers
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    n_workers = min(n_workers, n_frames)  # Don't use more workers than frames

    if verbose:
        print(f"  Using {n_workers} parallel workers")

    # Split frames into chunks for each worker
    chunk_size = (n_frames + n_workers - 1) // n_workers
    chunks = [frame_indices[i : i + chunk_size] for i in range(0, n_frames, chunk_size)]

    # Prepare worker arguments
    worker_args = [
        (topology, trajectory, selection, chunk, membrane_center_z) for chunk in chunks
    ]

    # Parallel execution
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_energy_chunk, args): i
            for i, args in enumerate(worker_args)
        }

        completed = 0
        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.extend(chunk_results)
            completed += 1
            if verbose:
                print(f"  Completed chunk {completed}/{len(chunks)}")

    # Sort by frame index (parallel execution may complete out of order)
    all_results.sort(key=lambda x: x["frame_idx"])

    # Extract arrays
    energy_list = [r["total"] for r in all_results]
    Epolar_list = [r["electrostatic"] for r in all_results]
    Enonpol_list = [r["hydrophobic"] for r in all_results]

    # Per-residue energies come back as {resid: (hydro, elec, total)} dicts whose
    # key set varies frame to frame (the electrostatic cutoff drops residues far
    # from the membrane), so project every frame onto one fixed, sorted resid axis
    # to build the (n_frames, n_residues) matrix promised by the schema. The
    # stored value is the per-residue *total*, consistent with Etot.
    resids = np.unique(u.select_atoms(selection).residues.resids)
    resid_to_col = {int(r): c for c, r in enumerate(resids)}
    per_residue = np.zeros((n_frames, len(resids)), dtype=np.float64)
    for frame_row, r in enumerate(all_results):
        for resid, terms in r["per_residue"].items():
            col = resid_to_col.get(int(resid))
            if col is not None:
                per_residue[frame_row, col] = terms[2]

    if verbose:
        print(f"✓ Computed energies")
        print(f"\nEnergy Summary:")
        print(
            f"  Total: {np.mean(energy_list):.2f} ± {np.std(energy_list):.2f} kcal/mol"
        )
        print(
            f"  Electrostatic: {np.mean(Epolar_list):.2f} ± {np.std(Epolar_list):.2f} kcal/mol"
        )
        print(
            f"  Hydrophobic: {np.mean(Enonpol_list):.2f} ± {np.std(Enonpol_list):.2f} kcal/mol"
        )

    return {
        "Etot": np.array(energy_list),
        "Epol": np.array(Epolar_list),
        "Enonpol": np.array(Enonpol_list),
        "per_residue": per_residue,
        "normal": normal,
    }


def _rotation_matrix_align(
    mobile: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Compute optimal rotation matrix to align mobile to target.

    Uses Kabsch algorithm.

    Args:
        mobile: (n_atoms, 3) positions to align
        target: (n_atoms, 3) reference positions
        weights: (n_atoms,) atomic masses

    Returns:
        R: (3, 3) rotation matrix
    """
    # Weighted covariance matrix
    W = weights[:, np.newaxis]
    H = (mobile * W).T @ target

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    return R


def detect_trajectory_contents(
    trajectory: str, verbose: bool = True
) -> Dict[str, bool]:
    """
    Detect what data is available in trajectory file.

    Args:
        trajectory: Trajectory file path
        verbose: Print results

    Returns:
        contents: Dictionary with boolean flags:
            - has_positions: Always True for valid trajectory
            - has_velocities: Velocity data present
            - has_forces: Force data present
            - is_trr: .trr format (full precision)
            - is_xtc: .xtc format (compressed)

    Example:
        >>> contents = detect_trajectory_contents('traj.trr')
        >>> if contents['has_forces']:
        ...     print("Can compute torques!")
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis required")

    # Detect format from extension
    is_trr = trajectory.lower().endswith(".trr")
    is_xtc = trajectory.lower().endswith(".xtc")

    # XTC files never have velocities/forces
    if is_xtc:
        contents = {
            "has_positions": True,
            "has_velocities": False,
            "has_forces": False,
            "is_trr": False,
            "is_xtc": True,
        }

        if verbose:
            print(f"Trajectory: {trajectory}")
            print(f"  Format: XTC (compressed)")
            print(f"  ✓ Positions")
            print(f"  ✗ Velocities")
            print(f"  ✗ Forces")

        return contents

    # For TRR, need to check actual file
    try:
        # Create minimal universe
        u = mda.Universe(trajectory)

        # Check first frame
        ts = u.trajectory[0]

        has_positions = True  # Always true if file loaded
        has_velocities = hasattr(ts, "has_velocities") and ts.has_velocities
        has_forces = hasattr(ts, "has_forces") and ts.has_forces

        contents = {
            "has_positions": has_positions,
            "has_velocities": has_velocities,
            "has_forces": has_forces,
            "is_trr": is_trr,
            "is_xtc": is_xtc,
        }

        if verbose:
            print(f"Trajectory: {trajectory}")
            print(f"  Format: {'TRR (full precision)' if is_trr else 'Unknown'}")
            print(f"  {'✓' if has_positions else '✗'} Positions")
            print(f"  {'✓' if has_velocities else '✗'} Velocities")
            print(f"  {'✓' if has_forces else '✗'} Forces")

        return contents

    except Exception as e:
        if verbose:
            print(f"Error reading trajectory: {e}")
        return {
            "has_positions": False,
            "has_velocities": False,
            "has_forces": False,
            "is_trr": is_trr,
            "is_xtc": is_xtc,
        }


def extract_frame(
    topology: str, trajectory: str, frame_idx: int, selection: str = "protein"
) -> Dict:
    """
    Extract single frame from trajectory.

    Args:
        topology: Topology file
        trajectory: Trajectory file
        frame_idx: Frame index to extract
        selection: Atom selection

    Returns:
        frame_data: Dictionary with positions, velocities, forces for single frame

    Example:
        >>> frame = extract_frame('system.gro', 'traj.trr', frame_idx=100)
        >>> print(frame['positions'].shape)  # (n_atoms, 3)
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis required")

    u = mda.Universe(topology, trajectory)
    atoms = u.select_atoms(selection)

    ts = u.trajectory[frame_idx]

    frame_data = {
        "positions": atoms.positions.copy(),
        "masses": atoms.masses.copy(),
        "time": ts.time,
    }

    try:
        frame_data["velocities"] = atoms.velocities.copy()
    except (AttributeError, mda.exceptions.NoDataError):
        frame_data["velocities"] = None

    try:
        frame_data["forces"] = atoms.forces.copy()
    except (AttributeError, mda.exceptions.NoDataError):
        frame_data["forces"] = None

    return frame_data
