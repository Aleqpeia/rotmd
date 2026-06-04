"""Tests for io.gromacs against small synthetic GROMACS topology+TRR pairs."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

from rotmd.io.gromacs import (  # noqa: E402
    chunked_trajectory_reader,
    detect_trajectory_contents,
    extract_frame,
    load_gromacs_trajectory,
    load_gromacs_trajectory_chunked,
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


def test_chunked_reader_yields_all_frames(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=7)
    chunks = list(
        load_gromacs_trajectory_chunked(top, trj, selection="all", chunk_size=3, verbose=False)
    )
    total = sum(c["positions"].shape[0] for c in chunks)
    assert total == 7
    assert all(c["n_atoms"] == info["n_atoms"] for c in chunks)


def test_load_verbose_and_align(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=4)
    # verbose=True + align_to_first exercise the progress prints and the
    # Kabsch alignment branch.
    data = load_gromacs_trajectory(
        top, trj, selection="all", align_to_first=True, verbose=True
    )
    assert data["positions"].shape == (4, info["n_atoms"], 3)


def test_chunked_trajectory_reader_generator(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=6)
    chunks = list(
        chunked_trajectory_reader(top, trj, chunk_size=4, selection="all")
    )
    total = sum(c["n_frames"] for c in chunks)
    assert total == 6
