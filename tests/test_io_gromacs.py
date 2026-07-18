"""Tests for io.gromacs against small synthetic GROMACS topology+TRR pairs."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

from rotmd.io.gromacs import (  # noqa: E402
    compute_trajectory_energies,
    detect_trajectory_contents,
    extract_frame,
    load_gromacs_trajectory,
)
from _helpers import write_synthetic_gromacs  # noqa: E402


def test_load_basic_shapes_and_capabilities(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=5, n_residues=4)
    data = load_gromacs_trajectory(top, trj, selection="all", verbose=False)
    assert data["n_frames"] == 5
    assert data["n_atoms"] == info["n_atoms"]
    assert data["positions"].shape == (5, info["n_atoms"], 3)
    assert data["has_velocities"] is True
    assert data["has_forces"] is True
    # COM and inertia are computed for every frame.
    assert data["com"].shape == (5, 3)
    assert data["inertia_tensor"].shape == (5, 3, 3)


def test_load_centers_positions_at_origin(tmp_path):
    top, trj, _ = write_synthetic_gromacs(tmp_path, n_frames=3)
    data = load_gromacs_trajectory(top, trj, selection="all", center=True, verbose=False)
    # With center=True each frame's mass-weighted COM is the origin.
    for f in range(data["n_frames"]):
        com = np.average(data["positions"][f], weights=data["masses"], axis=0)
        assert com == pytest.approx(np.zeros(3), abs=1e-4)


def test_load_respects_frame_slicing(tmp_path):
    top, trj, _ = write_synthetic_gromacs(tmp_path, n_frames=6)
    data = load_gromacs_trajectory(top, trj, selection="all", start=1, stop=5, step=2, verbose=False)
    # frames 1 and 3 -> 2 frames
    assert data["n_frames"] == 2
    assert data["times"].shape == (2,)


def test_load_zero_frame_slice_raises(tmp_path):
    """A --start past the end must say so, not fail later inside numba.

    An empty frame list becomes a 1-D ``(0,)`` array, and the numba kernels
    then reject it with "Unknown attribute 'shape' of type float64" — a message
    that points at the kernel instead of at the real cause.
    """
    top, trj, _ = write_synthetic_gromacs(tmp_path, n_frames=6)
    with pytest.raises(ValueError, match="selects 0 of the 6 frames"):
        load_gromacs_trajectory(top, trj, selection="all", start=1001, step=10, verbose=False)


def test_load_zero_atom_selection_raises(tmp_path):
    top, trj, _ = write_synthetic_gromacs(tmp_path, n_frames=2)
    with pytest.raises(ValueError, match="0 atoms"):
        load_gromacs_trajectory(top, trj, selection="resname NOPE", verbose=False)


def test_trajectory_without_forces(tmp_path):
    top, trj, _ = write_synthetic_gromacs(
        tmp_path, n_frames=3, with_velocities=True, with_forces=False
    )
    data = load_gromacs_trajectory(top, trj, selection="all", verbose=False)
    assert data["has_forces"] is False
    assert data["forces"] is None


def test_detect_trajectory_contents_trr(tmp_path):
    _, trj, _ = write_synthetic_gromacs(tmp_path, n_frames=2)
    contents = detect_trajectory_contents(trj, verbose=False)
    assert contents["is_trr"] is True
    assert contents["has_velocities"] is True
    assert contents["has_forces"] is True


def test_detect_trajectory_contents_xtc_shortcut(tmp_path):
    # XTC never carries velocities/forces; the function shortcuts on extension.
    contents = detect_trajectory_contents("whatever.xtc", verbose=False)
    assert contents["is_xtc"] is True
    assert contents["has_velocities"] is False
    assert contents["has_forces"] is False


def test_extract_single_frame(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=4)
    frame = extract_frame(top, trj, frame_idx=2, selection="all")
    assert frame["positions"].shape == (info["n_atoms"], 3)
    assert frame["velocities"] is not None
    assert frame["forces"] is not None


def test_load_verbose_and_align(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=4)
    # verbose=True + align_to_first exercise the progress prints and the
    # Kabsch alignment branch.
    data = load_gromacs_trajectory(
        top, trj, selection="all", align_to_first=True, verbose=True
    )
    assert data["positions"].shape == (4, info["n_atoms"], 3)


def test_compute_trajectory_energies_real_end_to_end(tmp_path):
    """Exercise the real energy stage (no mock).

    The CLI test mocks ``compute_trajectory_energies``, so this is the only
    coverage of the actual model import (``models.energy.TotalEnergy``) and of
    the per-residue projection. It guards two regressions that the mocked path
    hid: the ``AtomGroup`` import in the energy model, and ``per_residue`` being
    emitted as a dict-of-tuples instead of an ``(n_frames, n_residues)`` matrix.
    """
    n_frames, n_residues = 4, 4
    top, trj, info = write_synthetic_gromacs(
        tmp_path, n_frames=n_frames, n_residues=n_residues
    )
    energies = compute_trajectory_energies(
        top,
        trj,
        selection="all",
        membrane_sel="resname ARG",
        n_workers=1,
        verbose=False,
    )

    for key in ("Etot", "Epol", "Enonpol"):
        assert energies[key].shape == (n_frames,)
        assert np.issubdtype(energies[key].dtype, np.floating)

    per_residue = energies["per_residue"]
    assert per_residue.shape == (n_frames, n_residues)
    assert np.issubdtype(per_residue.dtype, np.floating)
    # It must be a real numeric matrix, not an object array of dicts.
    assert per_residue.dtype != object
    assert energies["normal"].shape == (3,)
