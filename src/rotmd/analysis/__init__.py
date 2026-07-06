"""
rotmd.analysis — downstream analysis of the extract pipeline's output.

The extract worktree (``rotmd extract``) writes per-chunk ``.npz`` files of raw
observables. This package holds the *analyze*-stage helpers that build on that
output but are not part of the extract hot path: orientation post-processing
(quaternions, angular displacement, autocorrelation, angle derivatives),
inertia-derived quantities, vector-observable serialization/statistics
(including optional xarray/NetCDF export), and streaming trajectory readers for
data too large to load whole.

Kept separate so the extract scope stays minimal and dependency-light (e.g. no
``xarray`` import unless analysis actually uses it).
"""

from .inertia import parallel_axis_theorem, asymmetry_parameter
from .io import load_gromacs_trajectory_chunked, chunked_trajectory_reader
from .observables import (
    observable_to_dict,
    observable_to_xarray,
    observable_mean,
    observable_std,
    compute_spin_nutation_ratio,
)
from .orientation import (
    euler_zyz_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    extract_orientation,
    compute_angular_displacement,
    unwrap_euler_angles,
    compute_orientation_time_derivative,
    orientation_autocorrelation,
)

__all__ = [
    # orientation
    "euler_zyz_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "extract_orientation",
    "compute_angular_displacement",
    "unwrap_euler_angles",
    "compute_orientation_time_derivative",
    "orientation_autocorrelation",
    # vector observables
    "observable_to_dict",
    "observable_to_xarray",
    "observable_mean",
    "observable_std",
    "compute_spin_nutation_ratio",
    # inertia
    "parallel_axis_theorem",
    "asymmetry_parameter",
    # streaming io
    "load_gromacs_trajectory_chunked",
    "chunked_trajectory_reader",
]
