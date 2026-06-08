"""
Protein-Lipid Contact and Binding-Event Counts
==============================================

To ask whether a bilayer composition *stabilises binding* (not just
orientation) we need a binding observable. Here we build a per-frame count of
how many lipid residues are in contact with the protein, then turn it into
count data in the same spirit as :mod:`rotmd.analysis.crossings`:

    A binding "event" is a threshold crossing of the contact number -- the
    protein engaging a new lipid (up-crossing) or releasing one (down). The
    rate of such events measures how *labile* the interface is, while the
    bound fraction measures how *occupied* it is.

These contact series require the actual trajectory (an MDAnalysis
``Universe``), unlike the orientation crossings which can be replayed from the
extracted angle arrays. MDAnalysis is imported lazily so importing this module
never forces the (heavy) dependency until a contact calculation is requested.

Author: rotmd contributors
Date: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rotmd.analysis.crossings import count_level_crossings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import MDAnalysis as mda


@dataclass(frozen=True)
class BindingCounts:
    """Binding-event counts derived from a protein-lipid contact series."""

    selection: str
    n_binding_events: int
    n_unbinding_events: int
    bound_fraction: float
    mean_contacts: float
    n_frames: int
    exposure: float
    threshold: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "selection": self.selection,
            "n_binding_events": int(self.n_binding_events),
            "n_unbinding_events": int(self.n_unbinding_events),
            "n_binding_total": int(self.n_binding_events + self.n_unbinding_events),
            "bound_fraction": float(self.bound_fraction),
            "mean_contacts": float(self.mean_contacts),
            "n_frames": int(self.n_frames),
            "exposure": float(self.exposure),
            "threshold": float(self.threshold),
        }


def contact_timeseries(
    universe: mda.Universe,
    protein_sel: str = "protein",
    lipid_sel: str = "resname POPI POP2 PIP2 DMPI DOPI",
    *,
    cutoff: float = 6.0,
    level: str = "residue",
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Number of lipid contacts with the protein per frame.

    For each frame we count how many lipid *residues* (or atoms) have at least
    one atom within ``cutoff`` of the protein. Residue-level counting is the
    default because it answers "how many lipids are bound", which is the
    physically meaningful binding quantity; atom-level counting inflates with
    interface area and is offered mainly for fine-grained interface work.

    Args:
        universe: An MDAnalysis ``Universe`` with the trajectory loaded.
        protein_sel: Selection string for the protein.
        lipid_sel: Selection string for the lipid species of interest. The
            default targets common phosphoinositide residue names; override it
            for your force field / naming.
        cutoff: Contact distance in angstrom.
        level: ``"residue"`` (count unique lipid residues) or ``"atom"``.
        start, stop, step: Frame slice forwarded to the trajectory iterator.

    Returns:
        ``(times, contacts)`` with ``times`` in ps and ``contacts`` an integer
        per-frame count.

    Raises:
        ValueError: If either selection matches no atoms, or ``level`` is
            invalid.
    """
    if level not in ("residue", "atom"):
        raise ValueError("level must be 'residue' or 'atom'")

    protein = universe.select_atoms(protein_sel)
    if protein.n_atoms == 0:
        raise ValueError(f"protein selection matched no atoms: {protein_sel!r}")

    # An *updating* selection re-evaluates the distance query every frame, so we
    # always see the lipids currently within the cutoff of the protein.
    near = universe.select_atoms(
        f"({lipid_sel}) and around {cutoff} ({protein_sel})", updating=True
    )
    # Validate that the lipid pool exists at all (independent of the cutoff).
    if universe.select_atoms(lipid_sel).n_atoms == 0:
        raise ValueError(f"lipid selection matched no atoms: {lipid_sel!r}")

    times = []
    contacts = []
    for _ in universe.trajectory[start:stop:step]:
        times.append(universe.trajectory.time)
        contacts.append(near.n_residues if level == "residue" else near.n_atoms)

    return np.asarray(times, dtype=float), np.asarray(contacts, dtype=int)


def count_binding_events(
    contacts: np.ndarray,
    times: np.ndarray,
    *,
    threshold: float = 0.5,
    selection: str = "lipid",
) -> BindingCounts:
    """
    Count binding / unbinding events as threshold crossings of a contact series.

    With the default ``threshold=0.5`` an up-crossing marks the transition from
    *no* contact to *some* contact (a binding event) and a down-crossing the
    reverse. Raise the threshold to study turnover of a multi-lipid interface
    (e.g. ``threshold=2.5`` counts movements across the "3+ lipids bound" line).

    Args:
        contacts: (n_frames,) per-frame contact counts.
        times: (n_frames,) timestamps for the exposure offset.
        threshold: Contact-count level defining a binding event.
        selection: Label stored on the result (usually the lipid species).

    Returns:
        A ``BindingCounts``.
    """
    contacts = np.asarray(contacts, dtype=float)
    times = np.asarray(times, dtype=float)
    _, n_up, n_down = count_level_crossings(contacts, threshold)
    bound_fraction = float(np.mean(contacts > threshold)) if contacts.size else 0.0
    exposure = float(times[-1] - times[0]) if times.size >= 2 else 0.0
    return BindingCounts(
        selection=selection,
        n_binding_events=n_up,
        n_unbinding_events=n_down,
        bound_fraction=bound_fraction,
        mean_contacts=float(np.mean(contacts)) if contacts.size else 0.0,
        n_frames=int(contacts.size),
        exposure=exposure,
        threshold=float(threshold),
    )
