"""
GROMACS File I/O

This module provides utilities for reading GROMACS trajectory and topology files
via MDAnalysis, with specialized handling for protein orientation analysis.

Key Features:
- XTC/TRR trajectory readers
- GRO/PDB topology readers
- Automatic protein selection
- Velocity/force detection
- Memory-efficient chunked reading
- Parallel batch computation via numba kernels

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from rotmd.core import kernels as K  # Parallel batch functions


try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

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
    per_residue_list = [r["per_residue"] for r in all_results]

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
        "per_residue": np.array(per_residue_list),
        "normal": normal,
    }


def load_gromacs_trajectory_chunked(
    topology: str,
    trajectory: str,
    selection: str = "protein",
    chunk_size: int = 1000,  # Process 1000 frames at a time
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    verbose: bool = False,
):
    """
    Load trajectory in chunks to avoid memory overflow.
    Yields processed chunks instead of loading entire trajectory.
    """
    import MDAnalysis as mda
    from tqdm import tqdm

    u = mda.Universe(topology, trajectory)
    protein = u.select_atoms(selection)
    n_atoms = protein.n_atoms

    # Determine frame range
    total_frames = len(u.trajectory)
    if stop is None:
        stop = total_frames

    frames_to_process = range(start, min(stop, total_frames), step)

    # Process in chunks
    chunk_positions = []
    chunk_velocities = []
    chunk_forces = []
    chunk_times = []

    for frame_idx in tqdm(frames_to_process, desc="Loading frames"):
        u.trajectory[frame_idx]
        ts = u.trajectory.ts

        chunk_positions.append(protein.positions.copy())
        chunk_velocities.append(
            protein.velocities.copy()
            if ts.has_velocities
            else np.zeros_like(protein.positions)
        )
        chunk_forces.append(
            protein.forces.copy() if ts.has_forces else np.zeros_like(protein.positions)
        )
        chunk_times.append(ts.time)

        # Yield chunk when it reaches size limit
        if len(chunk_positions) >= chunk_size:
            yield {
                "positions": np.array(chunk_positions),
                "velocities": np.array(chunk_velocities),
                "forces": np.array(chunk_forces),
                "times": np.array(chunk_times),
                "masses": protein.masses.copy(),
                "n_atoms": n_atoms,
            }

            # Reset for next chunk
            chunk_positions = []
            chunk_velocities = []
            chunk_forces = []
            chunk_times = []

    # Yield remaining frames
    if chunk_positions:
        yield {
            "positions": np.array(chunk_positions),
            "velocities": np.array(chunk_velocities),
            "forces": np.array(chunk_forces),
            "times": np.array(chunk_times),
            "masses": protein.masses.copy(),
            "n_atoms": n_atoms,
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


def chunked_trajectory_reader(
    topology: str,
    trajectory: str,
    chunk_size: int = 1000,
    selection: str = "protein",
    **kwargs,
):
    """
    Generator for reading trajectory in chunks.

    Useful for very large trajectories that don't fit in memory.

    Args:
        topology: Topology file
        trajectory: Trajectory file
        chunk_size: Frames per chunk
        selection: Atom selection
        **kwargs: Additional arguments for load_gromacs_trajectory

    Yields:
        data: Dictionary for each chunk (same format as load_gromacs_trajectory)

    Example:
        >>> for chunk in chunked_trajectory_reader('system.gro', 'traj.xtc', chunk_size=500):
        ...     # Process chunk
        ...     euler = extract_orientation_trajectory(chunk['positions'], chunk['masses'])
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis required")

    # Get total number of frames
    u = mda.Universe(topology, trajectory)
    total_frames = len(u.trajectory)

    # Read in chunks
    start = kwargs.get("start", 0)
    stop = kwargs.get("stop", total_frames)
    step = kwargs.get("step", 1)

    for chunk_start in range(start, stop, chunk_size * step):
        chunk_stop = min(chunk_start + chunk_size * step, stop)

        chunk_data = load_gromacs_trajectory(
            topology,
            trajectory,
            selection=selection,
            start=chunk_start,
            stop=chunk_stop,
            step=step,
            verbose=False,
            **{k: v for k, v in kwargs.items() if k not in ["start", "stop", "step"]},
        )

        yield chunk_data


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
