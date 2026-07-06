"""
I/O for the rotmd extract pipeline.

Reading GROMACS trajectories/topologies and computing per-frame energies, plus
the compressed ``.npz`` chunk format that is the extract↔analyze contract.
Streaming/chunked readers for oversized trajectories live in
:mod:`rotmd.analysis.io`.
"""

from .gromacs import (
    load_gromacs_trajectory,
    compute_trajectory_energies,
    detect_trajectory_contents,
    extract_frame,
)
from .output import (
    save_npz,
    load_npz,
    merge_npz,
)

__all__ = [
    # GROMACS I/O
    "load_gromacs_trajectory",
    "compute_trajectory_energies",
    "detect_trajectory_contents",
    "extract_frame",
    # NPZ chunk I/O
    "save_npz",
    "load_npz",
    "merge_npz",
]
