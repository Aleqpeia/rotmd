"""
I/O Module for Protein Orientation Analysis

Provides utilities for reading GROMACS trajectories and saving analysis results.
"""

from .gromacs import (
    load_gromacs_trajectory,
    chunked_trajectory_reader,
    detect_trajectory_contents,
    extract_frame
)

from .output import (
    save_results_json,
    load_results_json,
    save_results_npz,
    load_results_npz,
    save_results_hdf5,
    load_results_hdf5,
    save_trajectory_data,
    load_trajectory_data,
    export_to_csv
)

__all__ = [
    # GROMACS I/O
    'load_gromacs_trajectory',
    'chunked_trajectory_reader',
    'detect_trajectory_contents',
    'extract_frame',
    # Output handling
    'save_results_json',
    'load_results_json',
    'save_results_npz',
    'load_results_npz',
    'save_results_hdf5',
    'load_results_hdf5',
    'save_trajectory_data',
    'load_trajectory_data',
    'export_to_csv',
]
