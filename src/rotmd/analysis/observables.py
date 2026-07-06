"""Vector-observable analysis helpers (analyze worktree).

Post-processing conveniences for the :class:`~rotmd.core.vector_observables.VectorObservable`
objects produced by the extract pipeline: serialization, summary statistics,
xarray/NetCDF export, and derived ratios. These used to live as methods on the
dataclass; they were pulled out so the extract scope carries no ``xarray``
dependency (imported lazily below) and the dataclass stays a plain container.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from rotmd.core.vector_observables import VectorObservable


def observable_to_dict(obs: VectorObservable) -> Dict[str, np.ndarray]:
    """Flatten a VectorObservable into descriptive NPZ-friendly keys."""
    return {
        f"{obs.name}": obs.vector,
        f"{obs.name}_parallel": obs.parallel,
        f"{obs.name}_perp": obs.perp,
        f"{obs.name}_z": obs.z_component,
        f"{obs.name}_mag": obs.magnitude,
        f"{obs.name}_parallel_mag": obs.parallel_mag,
        f"{obs.name}_perp_mag": obs.perp_mag,
        f"{obs.name}_z_mag": obs.z_mag,
    }


def observable_to_xarray(obs: VectorObservable):
    """Convert a VectorObservable to an :class:`xarray.Dataset` for NetCDF export.

    ``xarray`` is imported lazily so importing this module (or the extract
    pipeline) does not require it.
    """
    import xarray as xr

    if obs.times is not None:
        coords = {"time": obs.times, "component": ["x", "y", "z"]}
        dim = "time"
    else:
        coords = {"frame": np.arange(len(obs.vector)), "component": ["x", "y", "z"]}
        dim = "frame"

    return xr.Dataset(
        {
            obs.name: ([dim, "component"], obs.vector),
            f"{obs.name}_mag": ([dim], obs.magnitude),
            f"{obs.name}_parallel_mag": ([dim], obs.parallel_mag),
            f"{obs.name}_perp_mag": ([dim], obs.perp_mag),
        },
        coords=coords,
    )


def observable_mean(obs: VectorObservable) -> float:
    """Mean magnitude of the observable."""
    return float(np.mean(obs.magnitude))


def observable_std(obs: VectorObservable) -> float:
    """Standard deviation of the observable magnitude."""
    return float(np.std(obs.magnitude))


def compute_spin_nutation_ratio(observable: VectorObservable) -> np.ndarray:
    """
    Compute ratio of spin (parallel) to nutation (perpendicular) magnitude.

    spin/nutation ratio = |v_∥| / |v_⊥|

    Args:
        observable: VectorObservable with decomposition

    Returns:
        ratio: (n_frames,) spin/nutation ratio
    """
    return observable.parallel_mag / (observable.perp_mag + 1e-10)
