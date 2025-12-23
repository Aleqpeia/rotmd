"""
Trajectory I/O and Preprocessing

This module provides utilities for loading and preprocessing MD trajectories
for protein orientation analysis.

Key Features:
- MDAnalysis integration for .trr/.xtc files
- Automatic detection of velocities/forces
- Protein alignment and centering
- Frame selection and slicing
- Data validation

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

try:
    import MDAnalysis as mda
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    warnings.warn("MDAnalysis not available. Install with: pip install MDAnalysis")


class TrajectoryData:
    """Container for trajectory data"""

    def __init__(self,
                 positions: np.ndarray,
                 masses: np.ndarray,
                 times: np.ndarray,
                 velocities: Optional[np.ndarray] = None,
                 forces: Optional[np.ndarray] = None):
        """
        Initialize trajectory data container.

        Args:
            positions: (n_frames, n_atoms, 3) positions in Å
            masses: (n_atoms,) atomic masses in amu
            times: (n_frames,) timestamps in ps
            velocities: Optional (n_frames, n_atoms, 3) velocities in Å/ps
            forces: Optional (n_frames, n_atoms, 3) forces in kcal/(mol·Å)
        """
        self.positions = positions
        self.masses = masses
        self.times = times
        self.velocities = velocities
        self.forces = forces

        self.n_frames = len(positions)
        self.n_atoms = len(masses)
        self.has_velocities = velocities is not None
        self.has_forces = forces is not None

    def __repr__(self):
        return (f"TrajectoryData(n_frames={self.n_frames}, n_atoms={self.n_atoms}, "
                f"velocities={'✓' if self.has_velocities else '✗'}, "
                f"forces={'✓' if self.has_forces else '✗'})")


def load_trajectory(topology: str,
                    trajectory: str,
                    selection: str = "protein",
                    start: int = 0,
                    stop: Optional[int] = None,
                    step: int = 1,
                    center: bool = True,
                    verbose: bool = True) -> TrajectoryData:
    """
    Load trajectory from MDAnalysis.

    Args:
        topology: Topology file (.gro, .pdb, etc.)
        trajectory: Trajectory file (.trr, .xtc, etc.)
        selection: MDAnalysis selection string
        start: First frame index
        stop: Last frame index (None = all)
        step: Frame step size
        center: Whether to center protein at origin
        verbose: Print progress

    Returns:
        TrajectoryData object with loaded data

    Raises:
        ImportError: If MDAnalysis not available
        ValueError: If selection returns zero atoms
    """
    if not HAS_MDANALYSIS:
        raise ImportError("MDAnalysis required. Install with: pip install MDAnalysis")

    if verbose:
        print(f"Loading trajectory...")
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

    n_frames = len(u.trajectory[start:stop:step])
    if verbose:
        print(f"  Loading {n_frames} frames...")

    for i, ts in enumerate(u.trajectory[start:stop:step]):
        if verbose and (i % max(1, n_frames // 10) == 0):
            print(f"    Frame {i}/{n_frames}")

        pos = atoms.positions.copy()

        # Center if requested
        if center:
            com = np.average(pos, weights=atoms.masses, axis=0)
            pos -= com

        positions_list.append(pos)

        if has_velocities:
            velocities_list.append(atoms.velocities.copy())

        if has_forces:
            forces_list.append(atoms.forces.copy())

        times_list.append(ts.time / 1000.0)  # fs → ps

    if verbose:
        print(f"  ✓ Loaded {len(positions_list)} frames")

    # Convert to arrays
    return TrajectoryData(
        positions=np.array(positions_list),
        masses=atoms.masses.copy(),
        times=np.array(times_list),
        velocities=np.array(velocities_list) if has_velocities else None,
        forces=np.array(forces_list) if has_forces else None
    )


def validate_trajectory(traj: TrajectoryData,
                        require_velocities: bool = False,
                        require_forces: bool = False,
                        verbose: bool = True) -> bool:
    """
    Validate trajectory data for analysis.

    Args:
        traj: TrajectoryData object
        require_velocities: Whether velocities are required
        require_forces: Whether forces are required
        verbose: Print validation results

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if verbose:
        print("Validating trajectory...")

    # Check basic requirements
    if traj.n_frames < 2:
        raise ValueError("Need at least 2 frames for analysis")

    if traj.n_atoms < 3:
        raise ValueError("Need at least 3 atoms for orientation analysis")

    # Check velocities
    if require_velocities and not traj.has_velocities:
        raise ValueError("Velocities required but not available")

    # Check forces
    if require_forces and not traj.has_forces:
        raise ValueError("Forces required but not available")

    # Check for NaNs
    if np.any(np.isnan(traj.positions)):
        raise ValueError("NaN values found in positions")

    if traj.has_velocities and np.any(np.isnan(traj.velocities)):
        raise ValueError("NaN values found in velocities")

    if traj.has_forces and np.any(np.isnan(traj.forces)):
        raise ValueError("NaN values found in forces")

    if verbose:
        print(f"  ✓ Trajectory valid")
        print(f"    Frames: {traj.n_frames}")
        print(f"    Atoms: {traj.n_atoms}")
        print(f"    Time range: {traj.times[0]:.1f} - {traj.times[-1]:.1f} ps")

    return True


if __name__ == '__main__':
    # Example usage
    print("Trajectory I/O Module")
    print("=====================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.core.trajectory import load_trajectory")
    print()
    print("traj = load_trajectory(")
    print("    topology='system.gro',")
    print("    trajectory='traj.trr',")
    print("    selection='protein',")
    print("    start=0,")
    print("    stop=1000,")
    print("    step=10")
    print(")")
    print()
    print(f"print(traj)  # {TrajectoryData.__doc__.split(chr(10))[0]}")
