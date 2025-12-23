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

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

from rotmd.models.energy import TotalEnergy
from rotmd.base import membrane_interface
from rotmd.core.inertia import inertia_tensor


try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    warnings.warn("MDAnalysis not available. Install with: pip install MDAnalysis")


def load_gromacs_trajectory(topology: str,
                            trajectory: str,
                            selection: str = "protein",
                            start: int = 0,
                            stop: Optional[int] = None,
                            step: int = 1,
                            align_to_first: bool = False,
                            center: bool = True,
                            verbose: bool = True) -> Dict:
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

    # Load data
    positions_list = []
    velocities_list = [] if has_velocities else None
    forces_list = [] if has_forces else None
    times_list = []
    masses_list = []
    inertia_list = []
    energy_list = []
    Epolar_list = []
    Enonpol_list = []
    per_residue_list = []

    n_frames = len(u.trajectory[start:stop:step])
    if verbose:
        print(f"  Loading {n_frames} frames...")

    # Reference for alignment
    ref_positions = None

    for ts in u.trajectory[start:stop:step]:
        if verbose and (ts.frame % max(1, n_frames // 10) == 0):
            print(f"    Frame {ts.frame}/{n_frames}")

        pos = atoms.positions.copy()
        mass = atoms.masses.copy()
        I = inertia_tensor(pos, mass)
        # Center at origin
        if center:
            com = np.average(pos, weights=atoms.masses, axis=0)
            pos -= com

        # Align to first frame
        if align_to_first:
            if ref_positions is None:
                ref_positions = pos.copy()
            else:
                # Rotation-only alignment
                R = _rotation_matrix_align(pos, ref_positions, atoms.masses)
                pos = pos @ R.T
        energy = TotalEnergy().calculate(protein_atoms=atoms.copy(), 
                                    membrane_center_z=membrane_interface.get_membrane_center_z(u.copy(), 
                                    membrane_sel="resname CHL1", method='density'))
        positions_list.append(pos)
        masses_list.append(mass)
        energy_list.append(energy['total'])
        Epolar_list.append(energy['electrostatic'])
        Enonpol_list.append(energy['hydrophobic'])
        per_residue_list.append(energy['per_residue'])
        inertia_list.append(I)
        if has_velocities:
            velocities_list.append(atoms.velocities.copy())

        if has_forces:
            forces_list.append(atoms.forces.copy())

        times_list.append(ts.time)  # Already in ps for GROMACS

    if verbose:
        print(f"  ✓ Loaded {len(positions_list)} frames")
        print(f"\nEnergy Summary (across trajectory):")
        print(f"  Total Energy: {np.mean(energy_list):.2f} ± {np.std(energy_list):.2f} kcal/mol")
        print(f"  Electrostatic: {np.mean(Epolar_list):.2f} ± {np.std(Epolar_list):.2f} kcal/mol")
        print(f"  Hydrophobic: {np.mean(Enonpol_list)} ± {np.std(Enonpol_list):} kcal/mol")

    normal = membrane_interface.get_membrane_normal(u, membrane_sel="resname CHL1") 
    # Convert to arrays
    data = {
        'positions': np.array(positions_list),
        'masses': atoms.masses.copy(),
        'masses_list': np.array(masses_list),
        'times': np.array(times_list),
        'velocities': np.array(velocities_list) if has_velocities else None,
        'forces': np.array(forces_list) if has_forces else None,
        'inertia_tensor': np.array(inertia_list),
        'normal': normal,
        'has_velocities': has_velocities,
        'has_forces': has_forces,
        'n_frames': len(positions_list),
        'n_atoms': len(atoms),
        'Etot': np.array(energy_list),
        'Epol': np.array(Epolar_list),
        'Enonpol': np.array(Enonpol_list),
        'per_residue': np.array(per_residue_list)
    }

    return data


def _rotation_matrix_align(mobile: np.ndarray,
                           target: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
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


def chunked_trajectory_reader(topology: str,
                              trajectory: str,
                              chunk_size: int = 1000,
                              selection: str = "protein",
                              **kwargs):
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
    start = kwargs.get('start', 0)
    stop = kwargs.get('stop', total_frames)
    step = kwargs.get('step', 1)

    for chunk_start in range(start, stop, chunk_size * step):
        chunk_stop = min(chunk_start + chunk_size * step, stop)

        chunk_data = load_gromacs_trajectory(
            topology, trajectory,
            selection=selection,
            start=chunk_start,
            stop=chunk_stop,
            step=step,
            verbose=False,
            **{k: v for k, v in kwargs.items() if k not in ['start', 'stop', 'step']}
        )

        yield chunk_data


def detect_trajectory_contents(trajectory: str,
                               verbose: bool = True) -> Dict[str, bool]:
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
    is_trr = trajectory.lower().endswith('.trr')
    is_xtc = trajectory.lower().endswith('.xtc')

    # XTC files never have velocities/forces
    if is_xtc:
        contents = {
            'has_positions': True,
            'has_velocities': False,
            'has_forces': False,
            'is_trr': False,
            'is_xtc': True
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
        has_velocities = hasattr(ts, 'has_velocities') and ts.has_velocities
        has_forces = hasattr(ts, 'has_forces') and ts.has_forces

        contents = {
            'has_positions': has_positions,
            'has_velocities': has_velocities,
            'has_forces': has_forces,
            'is_trr': is_trr,
            'is_xtc': is_xtc
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
            'has_positions': False,
            'has_velocities': False,
            'has_forces': False,
            'is_trr': is_trr,
            'is_xtc': is_xtc
        }


def extract_frame(topology: str,
                 trajectory: str,
                 frame_idx: int,
                 selection: str = "protein") -> Dict:
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
        'positions': atoms.positions.copy(),
        'masses': atoms.masses.copy(),
        'time': ts.time
    }

    try:
        frame_data['velocities'] = atoms.velocities.copy()
    except (AttributeError, mda.exceptions.NoDataError):
        frame_data['velocities'] = None

    try:
        frame_data['forces'] = atoms.forces.copy()
    except (AttributeError, mda.exceptions.NoDataError):
        frame_data['forces'] = None

    return frame_data


if __name__ == '__main__':
    # Example usage
    print("GROMACS I/O Module")
    print("==================")
    print()
    print("Example usage:")
    print()
    print("from rotmd.io.gromacs import load_gromacs_trajectory, detect_trajectory_contents")
    print()
    print("# Check what's in trajectory")
    print("contents = detect_trajectory_contents('traj.trr')")
    print()
    print("# Load trajectory")
    print("data = load_gromacs_trajectory('system.gro', 'traj.trr')")
    print()
    print("print(f'Loaded {data[\"n_frames\"]} frames')")
    print("print(f'Has velocities: {data[\"has_velocities\"]}')")
