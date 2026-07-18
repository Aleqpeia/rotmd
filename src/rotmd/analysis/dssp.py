"""Secondary-structure occupancy (T8).

Per-frame DSSP over the production window, reduced to per-residue occupancy.
Helix<->coil transitions near the mutation site and in the EF-hand loops are
the expected structural signature of a charge change, and occupancy is how you
see them: a residue that is helical 95% of the time in WT and 40% in the mutant
is a concrete, local, quantitative claim.

Uses ``MDAnalysis.analysis.dssp``, which is pure Python and needs no external
``dssp``/``mkdssp`` binary — the plan's requirement of no new heavy dependency.
It is one of the two stages that must re-read the trajectory, because DSSP
needs backbone N/C/O atoms that the CA-only extract output does not carry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# MDAnalysis' DSSP implementation reports the simplified three-state alphabet.
DSSP_CODES = ("-", "H", "E")
DSSP_NAMES = {"-": "coil", "H": "helix", "E": "sheet"}


def encode(codes: np.ndarray) -> np.ndarray:
    """Map an array of DSSP characters to indices into :data:`DSSP_CODES`."""
    codes = np.asarray(codes, dtype="<U1")
    out = np.zeros(codes.shape, dtype=np.int8)
    for index, letter in enumerate(DSSP_CODES):
        out[codes == letter] = index
    return out


def occupancy_from_codes(encoded: np.ndarray, n_classes: int = len(DSSP_CODES)) -> np.ndarray:
    """``(n_frames, n_res)`` integer codes -> ``(n_res, n_classes)`` fractions.

    Rows sum to exactly 1 by construction (every frame assigns each residue
    exactly one state), which is the plan's acceptance criterion.
    """
    encoded = np.asarray(encoded)
    if encoded.ndim != 2:
        raise ValueError(f"expected (n_frames, n_res) codes, got shape {encoded.shape}")

    n_frames, n_res = encoded.shape
    counts = np.zeros((n_res, n_classes), dtype=np.float64)
    for cls in range(n_classes):
        counts[:, cls] = (encoded == cls).sum(axis=0)
    return counts / n_frames


def first_frame_at_or_after(universe, t0_ps: float) -> int:
    """Index of the first frame with ``time >= t0_ps``.

    Derived from the frame spacing rather than by scanning, so opening a
    100k-frame trajectory to find the window start does not cost a full pass.
    Falls back to scanning when the reader reports no usable ``dt``.
    """
    traj = universe.trajectory
    first_time = traj[0].time
    dt = float(getattr(traj, "dt", 0.0) or 0.0)

    if dt > 0:
        index = int(np.ceil((float(t0_ps) - first_time) / dt))
        return int(np.clip(index, 0, len(traj) - 1))

    for ts in traj:
        if ts.time >= t0_ps:
            return int(ts.frame)
    return len(traj) - 1


def compute_dssp(
    topology: str,
    trajectory: str,
    t0_ps: float | None = None,
    selection: str = "protein",
    step: int = 1,
    guess_hydrogens: bool = True,
) -> dict[str, Any]:
    """Run DSSP over the window and reduce to codes + occupancy."""
    import MDAnalysis as mda
    from MDAnalysis.analysis.dssp import DSSP

    universe = mda.Universe(topology, trajectory)
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"selection {selection!r} matched no atoms")

    start = first_frame_at_or_after(universe, t0_ps) if t0_ps is not None else 0

    run = DSSP(atoms, guess_hydrogens=guess_hydrogens).run(start=start, step=step)
    letters = np.asarray(run.results.dssp)
    encoded = encode(letters)
    occupancy = occupancy_from_codes(encoded)

    return {
        "codes": encoded,
        "occupancy": occupancy,
        "resids": atoms.residues.resids.astype(np.int32),
        "frames_used": np.int64(encoded.shape[0]),
        "start_frame": np.int64(start),
        "t0_ps": np.float64(t0_ps if t0_ps is not None else np.nan),
    }


def delta_occupancy(occ_a: np.ndarray, occ_b: np.ndarray) -> np.ndarray:
    """``occ_b - occ_a``, per residue and class."""
    occ_a = np.asarray(occ_a, dtype=np.float64)
    occ_b = np.asarray(occ_b, dtype=np.float64)
    if occ_a.shape != occ_b.shape:
        raise ValueError(f"occupancies differ in shape: {occ_a.shape} vs {occ_b.shape}")
    return occ_b - occ_a
