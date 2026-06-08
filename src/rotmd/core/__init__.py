#!/usr/bin/env python
"""
Core geometric, kinematic, and numerical machinery for rotmd.

This package is the foundation the rest of ``rotmd`` builds on. It is
organised into a few clearly separated concerns, and this ``__init__``
declares the *public* surface — the symbols other parts of the package
(and downstream users) are expected to import from ``rotmd.core``:

1. Rigid-body geometry  (:mod:`rotmd.core.inertia`)
       center of mass, inertia tensor, principal axes / moments.

2. Orientation kinematics  (:mod:`rotmd.core.orientation`)
       rotation-matrix / Euler / quaternion conversions and per-frame
       orientation extraction from trajectories.

3. Trajectory loading  (:mod:`rotmd.core.trajectory`)
       ``TrajectoryData`` plus loading / validation helpers.

4. Observables  (:mod:`rotmd.core.observables_classes`)
       The canonical, immutable, lazily-evaluated observable system
       (``Observable``, ``VectorObservable``, ``AngularMomentum``,
       ``Torque``, ``AngularVelocity``) and ``compute_all_observables_functional``.

5. Low-level vector math  (:mod:`rotmd.core.vector_observables`)
       The decomposition / magnitude / cross-product primitives that the
       observable classes are built on. Exposed here as plain functions.

6. Backend dispatch  (:mod:`rotmd.core.kernels`)
       Runtime selection of the numerical kernel backend (numba today,
       jax planned). Import the module as ``from rotmd.core import kernels``
       for ``kernels.compute_*`` calls, or use the control functions
       re-exported below.

7. Functional utilities  (:mod:`rotmd.core.functional`)
       Small composition helpers (``Pipeline``, ``Maybe``, ``pipe`` …)
       used to keep the analysis code declarative.

Note on a known redundancy
---------------------------
There are currently two ``VectorObservable`` types: the immutable one in
``observables_classes`` (re-exported here and used by the CLI) and an
older procedural one in ``vector_observables`` (still used by
``rotmd.observables.unified``). Only the immutable one is part of the
``rotmd.core`` public API; the procedural variant remains reachable via
its submodule until the two are consolidated. See ``SESSION.md``.
"""

# -----------------------------------------------------------------------------
# 1. Rigid-body geometry
# -----------------------------------------------------------------------------
from .inertia import (
    compute_center_of_mass,
    inertia_tensor,
    principal_axes,
    principal_moments,
)

# -----------------------------------------------------------------------------
# 2. Orientation kinematics
# -----------------------------------------------------------------------------
from .orientation import (
    rotation_matrix_to_euler_zyz,
    euler_zyz_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    extract_orientation,
    extract_orientation_trajectory,
    compute_angular_displacement,
    unwrap_euler_angles,
    compute_orientation_time_derivative,
    orientation_autocorrelation,
)

# -----------------------------------------------------------------------------
# 3. Trajectory loading
# -----------------------------------------------------------------------------
from .trajectory import (
    TrajectoryData,
    load_trajectory,
    validate_trajectory,
)

# -----------------------------------------------------------------------------
# 4. Observables (canonical immutable system)
# -----------------------------------------------------------------------------
from .observables_classes import (
    Observable,
    VectorObservable,
    AngularMomentum,
    Torque,
    AngularVelocity,
    validate_eulers_equation,
    compute_all_observables_functional,
)

# -----------------------------------------------------------------------------
# 5. Low-level vector math primitives
# -----------------------------------------------------------------------------
from .vector_observables import (
    decompose_vector_parallel,
    compute_magnitudes,
    compute_cross_product_trajectory,
)

# -----------------------------------------------------------------------------
# 6. Backend dispatch (control surface; import the module for kernel calls)
# -----------------------------------------------------------------------------
from . import kernels
from .kernels import (
    set_backend,
    get_backend,
    get_available_backends,
    print_backend_info,
)

# -----------------------------------------------------------------------------
# 7. Functional utilities
# -----------------------------------------------------------------------------
from .functional import (
    compose,
    pipe,
    Pipeline,
    Lazy,
    lazy,
    memoize,
    curry,
    Maybe,
)

__all__ = [
    # Geometry
    "compute_center_of_mass",
    "inertia_tensor",
    "principal_axes",
    "principal_moments",
    # Orientation
    "rotation_matrix_to_euler_zyz",
    "euler_zyz_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "extract_orientation",
    "extract_orientation_trajectory",
    "compute_angular_displacement",
    "unwrap_euler_angles",
    "compute_orientation_time_derivative",
    "orientation_autocorrelation",
    # Trajectory
    "TrajectoryData",
    "load_trajectory",
    "validate_trajectory",
    # Observables
    "Observable",
    "VectorObservable",
    "AngularMomentum",
    "Torque",
    "AngularVelocity",
    "validate_eulers_equation",
    "compute_all_observables_functional",
    # Vector math primitives
    "decompose_vector_parallel",
    "compute_magnitudes",
    "compute_cross_product_trajectory",
    # Backend dispatch
    "kernels",
    "set_backend",
    "get_backend",
    "get_available_backends",
    "print_backend_info",
    # Functional utilities
    "compose",
    "pipe",
    "Pipeline",
    "Lazy",
    "lazy",
    "memoize",
    "curry",
    "Maybe",
]
