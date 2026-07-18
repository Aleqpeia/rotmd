"""Local environment of the mutation site (T5a).

Salt bridges, hydrogen bonds and RMSF around residue 75. For an Asn->Lys
substitution this is the most direct mechanistic readout available: a neutral
side chain is replaced by a formal +1, and the question is simply whether that
charge finds a carboxylate partner, which one, and for what fraction of the
trajectory.

Occupancies are reported per partner residue, not just as a total, because
"K75 is salt-bridged 80% of the time" and "K75 spends 40% on D92 and 40% on
E118" are different mechanisms with different mutational predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Side-chain carboxylate oxygens (CHARMM and AMBER naming both covered).
ANION_SELECTION = "(resname ASP GLU) and name OD1 OD2 OE1 OE2"
# Lysine ammonium / arginine guanidinium nitrogens.
CATION_SELECTION = "(resname LYS and name NZ) or (resname ARG and name NH1 NH2 NE)"


def _frame_indices(universe, start: int, stop: int | None, step: int) -> range:
    return range(*slice(start, stop, step).indices(len(universe.trajectory)))


def contact_occupancy(
    universe,
    probe_selection: str,
    partner_selection: str,
    cutoff: float = 4.0,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> dict[str, Any]:
    """Fraction of frames each partner residue is within ``cutoff`` of the probe.

    A contact is counted when *any* probe atom is within ``cutoff`` of *any*
    atom of that partner residue — the standard heavy-atom salt-bridge
    criterion, which deliberately ignores which particular oxygen is involved
    because carboxylates rotate freely.

    Distances honour periodic boundaries when the trajectory carries a box;
    without that, a partner across the boundary reads as tens of Å away and the
    interaction silently disappears from the table.
    """
    from MDAnalysis.lib.distances import distance_array

    probe = universe.select_atoms(probe_selection)
    partners = universe.select_atoms(partner_selection)
    if len(probe) == 0:
        raise ValueError(f"probe selection {probe_selection!r} matched no atoms")
    if len(partners) == 0:
        raise ValueError(f"partner selection {partner_selection!r} matched no atoms")

    # Column index -> partner residue, so per-atom distances collapse per residue.
    partner_resids = partners.resids
    unique_resids = np.unique(partner_resids)
    columns = [partner_resids == resid for resid in unique_resids]

    hits = np.zeros(len(unique_resids), dtype=np.int64)
    engaged_frames = 0
    n_frames = 0

    for index in _frame_indices(universe, start, stop, step):
        ts = universe.trajectory[index]
        dist = distance_array(probe.positions, partners.positions, box=ts.dimensions)
        close = dist.min(axis=0) <= cutoff
        for col, mask in enumerate(columns):
            if close[mask].any():
                hits[col] += 1
        engaged_frames += bool(close.any())
        n_frames += 1

    if n_frames == 0:
        raise ValueError("no frames selected")

    occupancy = hits / n_frames
    order = np.argsort(-occupancy)
    return {
        "resids": unique_resids[order].astype(np.int32),
        "occupancy": occupancy[order],
        "n_frames": np.int64(n_frames),
        "cutoff": np.float64(cutoff),
        # Fraction of frames with at least one partner engaged: the "is the
        # charge satisfied at all?" number. Tracked in the same pass, and NOT
        # the sum of the per-partner column, since several partners can be
        # engaged simultaneously.
        "any_occupancy": np.float64(engaged_frames / n_frames),
    }


def _has_hydrogens(universe) -> bool:
    """Whether the topology carries explicit hydrogens.

    Checked by name first and only then by element, because a ``.gro`` (and
    several other formats) has no ``elements`` attribute at all — asking
    MDAnalysis for ``element H`` on one raises AttributeError rather than
    returning an empty group.
    """
    if len(universe.select_atoms("name H*")) > 0:
        return True
    if hasattr(universe.atoms, "elements"):
        return len(universe.select_atoms("element H")) > 0
    return False


def salt_bridges(
    universe,
    site: int,
    cutoff: float = 4.0,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> dict[str, Any]:
    """Salt-bridge occupancy between the site's side chain and carboxylates."""
    probe = f"resid {site} and ({CATION_SELECTION})"
    if len(universe.select_atoms(probe)) == 0:
        # WT Asn75 has no charged nitrogen; fall back to its side-chain polar
        # atoms so the same table is produced for both systems and the columns
        # stay comparable.
        probe = f"resid {site} and name ND1 ND2 NE2 OD1 OE1 NZ"
    return contact_occupancy(
        universe, probe, ANION_SELECTION, cutoff=cutoff, start=start, stop=stop, step=step
    )


def rmsf_per_residue(
    ca_coords: np.ndarray,
    align_mask: np.ndarray,
    resids: np.ndarray,
    frame_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-CA RMSF on the window, using the same superposition as the DCCM.

    Shares :func:`rotmd.analysis.dccm.average_structure` deliberately: an RMSF
    computed against a different reference than the correlation map would not
    be comparable with it, and the two are read together.
    """
    from rotmd.analysis.dccm import average_structure

    ca_coords = np.asarray(ca_coords)
    align_mask = np.asarray(align_mask, dtype=bool)
    if frame_mask is not None:
        ca_coords = ca_coords[np.asarray(frame_mask, dtype=bool)]
    if len(ca_coords) < 2:
        raise ValueError(f"need at least 2 frames, got {len(ca_coords)}")

    aligned, mean_structure, _ = average_structure(ca_coords, align_mask)
    displacements = aligned - mean_structure
    return {
        "rmsf": np.sqrt(np.mean(np.sum(displacements**2, axis=2), axis=0)),
        "resids": np.asarray(resids, dtype=np.int32),
        "frames_used": np.int64(len(ca_coords)),
    }


def hydrogen_bonds(
    universe,
    site: int,
    radius: float = 10.0,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    d_a_cutoff: float = 3.5,
    d_h_a_angle: float = 150.0,
) -> dict[str, Any]:
    """Hydrogen-bond occupancy within ``radius`` of the site.

    Wraps ``MDAnalysis.analysis.hydrogenbonds``. Requires explicit hydrogens —
    a heavy-atom-only trajectory cannot support the donor-H-acceptor angle
    criterion, and this raises rather than silently reporting zero bonds.
    """
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
        HydrogenBondAnalysis,
    )

    if not _has_hydrogens(universe):
        raise ValueError(
            "no hydrogens in the trajectory — hydrogen-bond analysis needs them; "
            "use salt_bridges() for a heavy-atom criterion instead"
        )

    shell = f"(around {radius} (resid {site})) or (resid {site})"
    try:
        analysis = HydrogenBondAnalysis(
            universe,
            between=[shell, shell],
            d_a_cutoff=d_a_cutoff,
            d_h_a_angle_cutoff=d_h_a_angle,
            update_selections=False,
        )
        analysis.run(start=start, stop=stop, step=step)
    except (ValueError, AttributeError) as exc:
        # MDAnalysis identifies donors/acceptors from partial charges and bonds,
        # neither of which a .gro/.pdb topology carries. Say so, because the fix
        # is a different topology file rather than a different analysis.
        raise ValueError(
            f"{exc}. Hydrogen-bond detection needs partial charges and bonds — "
            "pass a .tpr as the topology instead of a .gro/.pdb. Salt bridges "
            "are unaffected (heavy-atom distance criterion)."
        ) from exc

    results = np.asarray(analysis.results.hbonds)
    n_frames = analysis.n_frames
    if results.size == 0:
        return {"pairs": np.zeros((0, 2), np.int32),
                "occupancy": np.zeros(0), "n_frames": np.int64(n_frames)}

    # Columns: frame, donor_ix, hydrogen_ix, acceptor_ix, distance, angle.
    donors = universe.atoms[results[:, 1].astype(int)].resids
    acceptors = universe.atoms[results[:, 3].astype(int)].resids
    pairs = np.sort(np.stack([donors, acceptors], axis=1), axis=1)

    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    occupancy = counts / n_frames
    order = np.argsort(-occupancy)
    return {
        "pairs": unique[order].astype(np.int32),
        "occupancy": occupancy[order],
        "n_frames": np.int64(n_frames),
    }


def compare_occupancy(
    table_a: dict[str, Any], table_b: dict[str, Any]
) -> list[dict[str, Any]]:
    """Gained/lost/retained partners between two occupancy tables."""
    occ_a = dict(zip(table_a["resids"].tolist(), table_a["occupancy"].tolist(), strict=True))
    occ_b = dict(zip(table_b["resids"].tolist(), table_b["occupancy"].tolist(), strict=True))

    rows = []
    for resid in sorted(set(occ_a) | set(occ_b)):
        a = occ_a.get(resid, 0.0)
        b = occ_b.get(resid, 0.0)
        if a == 0.0 and b == 0.0:
            continue
        rows.append({
            "resid": int(resid),
            "occupancy_a": a,
            "occupancy_b": b,
            "delta": b - a,
            "status": "gained" if a == 0.0 else "lost" if b == 0.0 else "retained",
        })
    return sorted(rows, key=lambda r: -abs(r["delta"]))
