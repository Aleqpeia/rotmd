"""Streaming trajectory readers (analyze worktree).

The extract CLI processes one chunk per invocation and loads it whole
(:func:`rotmd.io.gromacs.load_gromacs_trajectory`), so it never needs to stream.
These generators are for downstream analysis over very large trajectories that
do not fit in memory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rotmd.io.gromacs import load_gromacs_trajectory


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


def chunked_trajectory_reader(
        topology: str,
        trajectory: str,
        chunk_size: int = 1000,
        selection: str = "protein",
        **kwargs,
):
    """
    Generator for reading trajectory in chunks.

    Useful for very large trajectories that don't fit in memory. Each yielded
    chunk has the same format as
    :func:`rotmd.io.gromacs.load_gromacs_trajectory` (COM + inertia tensors
    included), since it delegates to it per slice.

    Args:
        topology: Topology file
        trajectory: Trajectory file
        chunk_size: Frames per chunk
        selection: Atom selection
        **kwargs: Additional arguments for load_gromacs_trajectory

    Yields:
        data: Dictionary for each chunk (same format as load_gromacs_trajectory)
    """
    import MDAnalysis as mda

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
