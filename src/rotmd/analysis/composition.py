"""
Composition Metadata and Tidy Count-Table Assembly
==================================================

``rotmd`` has no first-class notion of a "system" or "bilayer composition":
trajectories are just file paths. To run a regression *across* compositions we
therefore need a thin metadata layer that answers, for each replica, "what was
in the bilayer?" and "where is its data?". A small YAML manifest provides that,
and this module turns a manifest into a tidy long-format table of counts ready
for :mod:`rotmd.analysis.regression`.

A manifest looks like::

    spacing_deg: 15.0
    coordinates: [tilt, azimuth, geodesic]
    contacts:
      lipid_sel: "resname POPI DMPI"
      cutoff: 6.0
      threshold: 0.5
    hbonds:
      sel_a: "protein"
      sel_b: "resname POPI DMPI"
      d_a_cutoff: 3.0
      d_h_a_angle_cutoff: 150.0
      threshold: 0.5
    systems:
      - label: simple
        replica: 1
        npz: data/simple_r1.npz
        covariates: {has_PIP2: 0, has_CHOL: 0}
      - label: complex
        replica: 1
        npz: data/complex_r1.npz
        topology: data/complex_r1.tpr      # only needed for contacts
        trajectory: data/complex_r1.xtc
        covariates: {has_PIP2: 1, has_CHOL: 1}

.. warning::

   With only the two compositions described in the project (``simple`` =
   DPPC/DOPC and ``complex`` = DPPC/DOPC/CHOL/PIP2), ``has_PIP2`` and
   ``has_CHOL`` are perfectly collinear: every PIP2-containing replica also
   contains cholesterol. A regression can then only estimate the *combined*
   "complex vs simple" effect, not a PIP2-specific one. To separate them you
   must add a third composition (e.g. CHOL without PIP2). The covariate
   columns are kept general so the model generalises the moment you do.

Author: rotmd contributors
Date: 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rotmd.analysis.crossings import (
    CrossingCounts,
    count_angle_series_crossings,
    count_orientation_crossings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


# Map a coordinate name to the NPZ angle key and whether it wraps. Used for the
# fast "replay from extracted angles" path when rotation matrices are absent.
_ANGLE_KEYS: dict[str, tuple[str, bool]] = {
    "theta": ("theta", False),
    "phi": ("phi", True),
    "psi": ("psi", True),
}


@dataclass
class SystemRecord:
    """One replica trajectory plus the bilayer metadata describing it."""

    label: str
    replica: str | int
    covariates: dict[str, float] = field(default_factory=dict)
    npz: Path | None = None
    topology: Path | None = None
    trajectory: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SystemRecord:
        def _path(key: str) -> Path | None:
            value = raw.get(key)
            return Path(value) if value else None

        return cls(
            label=str(raw["label"]),
            replica=raw.get("replica", 1),
            covariates={k: float(v) for k, v in (raw.get("covariates") or {}).items()},
            npz=_path("npz"),
            topology=_path("topology"),
            trajectory=_path("trajectory"),
        )


@dataclass
class Manifest:
    """A parsed crossings manifest: global options plus per-system records."""

    systems: list[SystemRecord]
    spacing_deg: float = 15.0
    coordinates: list[str] = field(default_factory=lambda: ["tilt", "azimuth", "geodesic"])
    min_step_deg: float = 0.0
    reference_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    body_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    contacts: dict[str, Any] | None = None
    hbonds: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Manifest:
        systems = [SystemRecord.from_dict(s) for s in raw.get("systems", [])]
        if not systems:
            raise ValueError("manifest contains no 'systems'")
        return cls(
            systems=systems,
            spacing_deg=float(raw.get("spacing_deg", 15.0)),
            coordinates=list(raw.get("coordinates", ["tilt", "azimuth", "geodesic"])),
            min_step_deg=float(raw.get("min_step_deg", 0.0)),
            reference_axis=tuple(raw.get("reference_axis", (0.0, 0.0, 1.0))),  # type: ignore[arg-type]
            body_axis=tuple(raw.get("body_axis", (0.0, 0.0, 1.0))),  # type: ignore[arg-type]
            contacts=raw.get("contacts"),
            hbonds=raw.get("hbonds"),
        )


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a YAML crossings manifest."""
    import yaml

    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"manifest {path} must be a YAML mapping")
    return Manifest.from_dict(raw)


# ==============================================================================
# Per-record count extraction
# ==============================================================================
#
# For a single replica we compute one CrossingCounts per requested coordinate.
# We prefer rotation matrices (the full-SO(3) path -> tilt/azimuth/geodesic) and
# fall back to replaying a bare Euler angle (theta/phi/psi) when only those are
# stored. This lets the same manifest serve both rich and minimal NPZ outputs.


def _counts_for_record(record: SystemRecord, manifest: Manifest) -> list[CrossingCounts]:
    if record.npz is None:
        raise ValueError(f"system {record.label}/{record.replica} has no 'npz'")

    data = np.load(record.npz)
    times = data["times"] if "times" in data else np.arange(_first_len(data), dtype=float)

    has_matrices = "rotation_matrices" in data
    results: list[CrossingCounts] = []
    for coord in manifest.coordinates:
        if coord in _ANGLE_KEYS and coord in data:
            # Minimal path: a stored Euler angle in radians -> degrees.
            key, periodic = _ANGLE_KEYS[coord]
            results.append(
                count_angle_series_crossings(
                    np.degrees(data[key]),
                    times,
                    coordinate=coord,
                    spacing_deg=manifest.spacing_deg,
                    periodic=periodic,
                    min_step_deg=manifest.min_step_deg,
                )
            )
        elif has_matrices:
            results.append(
                count_orientation_crossings(
                    data["rotation_matrices"],
                    times,
                    coordinate=coord,
                    spacing_deg=manifest.spacing_deg,
                    body_axis=manifest.body_axis,
                    reference_axis=manifest.reference_axis,
                    min_step_deg=manifest.min_step_deg,
                )
            )
        else:
            raise ValueError(
                f"coordinate {coord!r} needs 'rotation_matrices' (or a stored "
                f"angle) in {record.npz}"
            )
    return results


def _first_len(data: Any) -> int:
    for key in ("theta", "phi", "psi", "rotation_matrices"):
        if key in data:
            return len(data[key])
    raise ValueError("NPZ has none of theta/phi/psi/rotation_matrices")


def _trajectory_counts_for_record(record: SystemRecord, manifest: Manifest):
    """Compute trajectory-derived event counts (contacts and/or H-bonds).

    Both observables reduce to a per-frame integer series that
    :func:`~rotmd.analysis.binding.count_binding_events` turns into
    binding/unbinding events, so they share one ``BindingCounts`` container;
    the caller tags each with a ``coordinate`` label (``"contacts"`` vs
    ``"hbonds"``) so the regression keeps their rates separate.

    Returns a list of ``(coordinate, BindingCounts)`` pairs, possibly empty.

    The MDAnalysis ``Universe`` is built **once** and reused for both analyses:
    trajectories are large and reloading per observable would dominate runtime.
    Records lacking a raw trajectory are silently skipped so a manifest can mix
    contact/H-bond-enabled and orientation-only systems.
    """
    if not (manifest.contacts or manifest.hbonds):
        return []
    if record.topology is None or record.trajectory is None:
        return []

    import MDAnalysis as mda

    from rotmd.analysis.binding import (
        contact_timeseries,
        count_binding_events,
        hbond_timeseries,
    )

    universe = mda.Universe(str(record.topology), str(record.trajectory))
    results: list[tuple[str, Any]] = []

    if manifest.contacts:
        cfg = manifest.contacts
        times, contacts = contact_timeseries(
            universe,
            protein_sel=cfg.get("protein_sel", "protein"),
            lipid_sel=cfg.get("lipid_sel", "resname POPI POP2 PIP2 DMPI DOPI"),
            cutoff=float(cfg.get("cutoff", 6.0)),
            level=cfg.get("level", "residue"),
        )
        results.append(
            (
                "contacts",
                count_binding_events(
                    contacts,
                    times,
                    threshold=float(cfg.get("threshold", 0.5)),
                    selection=cfg.get("lipid_sel", "lipid"),
                ),
            )
        )

    if manifest.hbonds:
        cfg = manifest.hbonds
        times, counts = hbond_timeseries(
            universe,
            sel_a=cfg.get("sel_a", "protein"),
            sel_b=cfg.get("sel_b", "resname POPI POP2 PIP2 DMPI DOPI"),
            d_a_cutoff=float(cfg.get("d_a_cutoff", 3.0)),
            d_h_a_angle_cutoff=float(cfg.get("d_h_a_angle_cutoff", 150.0)),
            hydrogens_sel=cfg.get("hydrogens_sel"),
            acceptors_sel=cfg.get("acceptors_sel"),
        )
        results.append(
            (
                "hbonds",
                count_binding_events(
                    counts,
                    times,
                    threshold=float(cfg.get("threshold", 0.5)),
                    selection=cfg.get("sel_b", "hbond"),
                ),
            )
        )

    return results


# ==============================================================================
# Tidy table assembly
# ==============================================================================


def build_count_table(manifest: Manifest, verbose: bool = True) -> pd.DataFrame:
    """
    Assemble a tidy long-format count table from a manifest.

    Each row is one (replica, coordinate, response) observation. ``response``
    is ``"crossings"`` (grid crossings), ``"direction_changes"`` (turning
    points) or ``"binding"`` (binding/unbinding events). Binding events come in
    two flavours distinguished by ``coordinate``: ``"contacts"`` (lipid
    proximity) and ``"hbonds"`` (directional hydrogen bonds), so the regression
    fits a separate rate for each. ``count`` is the Poisson response and
    ``exposure`` the offset (observation time). The composition ``label`` and
    any ``covariates`` columns are the predictors.

    Args:
        manifest: A parsed :class:`Manifest`.
        verbose: Print progress per system.

    Returns:
        A ``pandas.DataFrame`` with columns ``label, replica, coordinate,
        response, count, exposure, n_frames`` plus one column per covariate.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    covariate_keys: set[str] = set()
    for record in manifest.systems:
        covariate_keys.update(record.covariates)

    for record in manifest.systems:
        if verbose:
            print(f"  counting {record.label}/replica {record.replica} ...")

        base = {
            "label": record.label,
            "replica": record.replica,
            **{k: record.covariates.get(k, np.nan) for k in covariate_keys},
        }

        for counts in _counts_for_record(record, manifest):
            rows.append(
                {
                    **base,
                    "coordinate": counts.coordinate,
                    "response": "crossings",
                    "count": counts.n_crossings,
                    "exposure": counts.exposure,
                    "n_frames": counts.n_frames,
                }
            )
            rows.append(
                {
                    **base,
                    "coordinate": counts.coordinate,
                    "response": "direction_changes",
                    "count": counts.n_direction_changes,
                    "exposure": counts.exposure,
                    "n_frames": counts.n_frames,
                }
            )

        for coordinate, binding in _trajectory_counts_for_record(record, manifest):
            rows.append(
                {
                    **base,
                    "coordinate": coordinate,
                    "response": "binding",
                    "count": binding.n_binding_events + binding.n_unbinding_events,
                    "exposure": binding.exposure,
                    "n_frames": binding.n_frames,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no counts produced; check manifest paths and coordinates")
    # Guard against zero exposure that would make log(exposure) undefined.
    if (frame["exposure"] <= 0).any():
        raise ValueError(
            "non-positive exposure encountered; ensure 'times' spans >1 frame"
        )
    return frame
