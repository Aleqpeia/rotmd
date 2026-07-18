"""Dynamic cross-correlation maps (T4).

``C_ij = <Δr_i · Δr_j> / sqrt(<|Δr_i|²> <|Δr_j|²>)`` over CA displacements from
the mean structure: +1 = two residues always move together, -1 = always in
opposition, 0 = uncorrelated. This is the map that answers "does the mutation
rewire coupling to residues far from site 75?".

Why the alignment matters here more than usual
----------------------------------------------
Every frame is first RMS-fit onto the *average* structure using only the
``--sel-align`` core. Skipping that, or fitting on a set that includes flexible
termini, leaves whole-molecule tumbling in the displacements — and global
rotation correlates every atom with every other atom, producing a map that is
uniformly positive and says nothing about internal dynamics. rotmd exists to
study exactly that rotational motion, so it is worth being explicit: this stage
deliberately projects it out, because here it is the contaminant.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _superpose(coords: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Kabsch-fit every frame onto ``reference`` using only ``mask`` atoms.

    The rotation is *determined* by the core atoms but *applied* to all of
    them, which is the point: the flexible regions are then measured in the
    frame of the stable core rather than being allowed to define it.

    Batched over frames — a Python loop with a 3x3 SVD per frame is the
    bottleneck at 10^4-10^5 frames, and numpy's SVD is already batched.
    """
    core = coords[:, mask]                                    # (F, M, 3)
    core_com = core.mean(axis=1, keepdims=True)               # (F, 1, 3)
    ref_core = reference[mask]
    ref_com = ref_core.mean(axis=0)                           # (3,)

    p = core - core_com                                       # (F, M, 3)
    q = ref_core - ref_com                                    # (M, 3)

    # Covariance H = P^T Q per frame, then R = V diag(1,1,d) U^T with
    # d = sign(det(V U^T)) guarding against an improper (mirror) rotation.
    h = np.einsum("fmi,mj->fij", p, q)
    u, _s, vt = np.linalg.svd(h)
    v = np.swapaxes(vt, 1, 2)
    ut = np.swapaxes(u, 1, 2)
    d = np.sign(np.linalg.det(v @ ut))
    eye = np.zeros((len(coords), 3, 3))
    eye[:, 0, 0] = eye[:, 1, 1] = 1.0
    eye[:, 2, 2] = d
    rot = v @ eye @ ut                                        # (F, 3, 3)

    return np.einsum("fij,fnj->fni", rot, coords - core_com) + ref_com


def average_structure(
    coords: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iteratively superpose onto the running average until self-consistent.

    Fitting to frame 0 would make the map depend on which frame happened to be
    first. Fitting to the average is well defined, but the average itself
    depends on the fit, so it is iterated to a fixed point (typically 2-4
    rounds).

    Returns ``(aligned_coords, mean_structure, n_iterations)``.
    """
    reference = coords[0].astype(np.float64)
    aligned = coords.astype(np.float64)

    # noqa B007: `iteration` is read after the loop, to report how many rounds
    # convergence needed — not unused, just not used inside the body.
    for iteration in range(1, max_iter + 1):  # noqa: B007
        aligned = _superpose(coords.astype(np.float64), reference, mask)
        new_reference = aligned.mean(axis=0)
        shift = float(np.sqrt(np.mean(np.sum((new_reference - reference) ** 2, axis=1))))
        reference = new_reference
        if shift < tol:
            break

    return aligned, reference, iteration


def _covariance(displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(cov_ij, var_i)`` from ``(n_frames, n_atoms, 3)`` displacements.

    Accumulated as three ``(N,F)x(F,N)`` matrix products rather than one
    einsum: identical arithmetic, but it lands in BLAS and runs roughly an
    order of magnitude faster. That matters because the block bootstrap
    evaluates this hundreds of times.

    ``var_i = <|Δr_i|²>`` is exactly ``cov_ii``, so it comes free.
    """
    n_frames = displacements.shape[0]
    n_atoms = displacements.shape[1]
    cov = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for axis in range(3):
        component = np.ascontiguousarray(displacements[:, :, axis], dtype=np.float64)
        cov += component.T @ component
    cov /= n_frames
    return cov, np.diagonal(cov).copy()


def cross_correlation(displacements: np.ndarray) -> np.ndarray:
    """Normalised DCCM from ``(n_frames, n_atoms, 3)`` displacements."""
    cov, var = _covariance(displacements)

    norm = np.sqrt(np.outer(var, var))
    # An atom that never moves has zero variance and no defined correlation;
    # report 0 rather than letting a 0/0 nan propagate through the whole map.
    with np.errstate(invalid="ignore", divide="ignore"):
        dcc = np.where(norm > 0, cov / norm, 0.0)

    # Force exact symmetry: the einsum is symmetric analytically but not always
    # to the last bit, and downstream code asserts symmetry.
    dcc = 0.5 * (dcc + dcc.T)
    np.fill_diagonal(dcc, np.where(var > 0, 1.0, 0.0))
    return dcc


def block_dccms(displacements: np.ndarray, n_blocks: int) -> np.ndarray:
    """One DCCM per contiguous block of frames, shape ``(n_blocks, N, N)``.

    These are what makes a *login-site* significance test possible: `compare`
    can block-bootstrap over them without ever touching a trajectory again.
    Blocks must be at least as long as the statistical inefficiency ``g`` for
    the resampling to treat genuinely independent units, which is why the
    caller derives ``n_blocks`` from the window's ``g``.
    """
    n_frames = displacements.shape[0]
    n_blocks = int(max(1, min(n_blocks, n_frames // 2)))
    edges = np.linspace(0, n_frames, n_blocks + 1).astype(int)
    return np.stack([
        cross_correlation(displacements[lo:hi]).astype(np.float32)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        if hi - lo >= 2
    ])


def compute_dccm(
    ca_coords: np.ndarray,
    align_mask: np.ndarray,
    resids: np.ndarray,
    frame_mask: np.ndarray | None = None,
    n_blocks: int = 0,
) -> dict[str, Any]:
    """Full T4 computation for one replica.

    Args:
        ca_coords: (n_frames, n_ca, 3) CA trajectory.
        align_mask: (n_ca,) bool, the superposition core.
        resids: (n_ca,) residue ids, carried through so every downstream
            figure and comparison is keyed to residues rather than indices.
        frame_mask: (n_frames,) bool selecting the production window.

    Returns a dict of arrays ready for ``save_npz``.
    """
    ca_coords = np.asarray(ca_coords)
    if ca_coords.ndim != 3 or ca_coords.shape[2] != 3:
        raise ValueError(f"ca_coords must be (n_frames, n_ca, 3), got {ca_coords.shape}")

    align_mask = np.asarray(align_mask, dtype=bool)
    if align_mask.shape != (ca_coords.shape[1],):
        raise ValueError(
            f"align_mask {align_mask.shape} does not match n_ca {ca_coords.shape[1]}"
        )
    if not align_mask.any():
        raise ValueError("align_mask selects no atoms — cannot define a superposition core")

    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        ca_coords = ca_coords[frame_mask]
    n_frames = len(ca_coords)
    if n_frames < 2:
        raise ValueError(f"need at least 2 frames in the window, got {n_frames}")

    aligned, mean_structure, n_iter = average_structure(ca_coords, align_mask)
    displacements = aligned - mean_structure
    dcc = cross_correlation(displacements)

    # RMSF falls out of the same displacements for free, and T5a/T6 want it
    # on exactly this alignment and window.
    rmsf = np.sqrt(np.mean(np.sum(displacements**2, axis=2), axis=0))

    result = {
        "dcc": dcc,
        "rmsf": rmsf,
        "resids": np.asarray(resids, dtype=np.int32),
        "mean_structure": mean_structure,
        "align_mask": align_mask,
        "frames_used": np.int64(n_frames),
        "align_iterations": np.int64(n_iter),
    }
    if n_blocks:
        result["dcc_blocks"] = block_dccms(displacements, n_blocks)
    return result
