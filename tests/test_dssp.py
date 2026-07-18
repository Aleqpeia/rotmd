"""T8: secondary-structure occupancy.

Runs against a genuine backbone built at ideal torsions, so DSSP has real
N/C/O geometry to work with: an alpha-helix must come out helical and an
extended strand must not.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

from _helpers import build_backbone, write_protein_trajectory
from rotmd.analysis.dssp import (
    DSSP_CODES,
    compute_dssp,
    delta_occupancy,
    encode,
    first_frame_at_or_after,
    occupancy_from_codes,
)

# ---------------------------------------------------------------------------
# Encoding / occupancy arithmetic
# ---------------------------------------------------------------------------

def test_encode_maps_the_three_state_alphabet():
    codes = np.array([["-", "H", "E"], ["H", "H", "-"]])
    np.testing.assert_array_equal(encode(codes), [[0, 1, 2], [1, 1, 0]])


def test_encode_treats_unknown_letters_as_coil():
    # MDAnalysis only emits '-', 'H', 'E'; anything else must not crash.
    np.testing.assert_array_equal(encode(np.array([["X", "H"]])), [[0, 1]])


def test_occupancy_rows_sum_to_one():
    """The plan's acceptance criterion for T8."""
    rng = np.random.default_rng(0)
    encoded = rng.integers(0, 3, size=(200, 17))
    occ = occupancy_from_codes(encoded)
    assert occ.shape == (17, 3)
    np.testing.assert_allclose(occ.sum(axis=1), 1.0, atol=1e-12)


def test_occupancy_counts_are_correct():
    encoded = np.array([[1, 0], [1, 0], [0, 2], [1, 2]])  # 4 frames, 2 residues
    occ = occupancy_from_codes(encoded)
    np.testing.assert_allclose(occ[0], [0.25, 0.75, 0.0])   # residue 0: 3x H, 1x coil
    np.testing.assert_allclose(occ[1], [0.5, 0.0, 0.5])     # residue 1: 2x coil, 2x sheet


def test_occupancy_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(n_frames, n_res\)"):
        occupancy_from_codes(np.zeros(10))


def test_delta_occupancy_and_shape_guard():
    a = np.array([[0.2, 0.8, 0.0]])
    b = np.array([[0.5, 0.5, 0.0]])
    np.testing.assert_allclose(delta_occupancy(a, b), [[0.3, -0.3, 0.0]])
    with pytest.raises(ValueError, match="differ in shape"):
        delta_occupancy(a, np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# The backbone fixture itself
# ---------------------------------------------------------------------------

def test_ideal_helix_geometry_is_physical():
    """If the fixture geometry is wrong, every DSSP assertion below is vacuous."""
    n, ca, c, o = build_backbone(12)
    ca_spacing = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    np.testing.assert_allclose(ca_spacing, 3.8, atol=0.05)
    # The defining i -> i+4 backbone hydrogen bond of an alpha-helix.
    assert 2.7 < np.linalg.norm(o[0] - n[4]) < 3.3


# ---------------------------------------------------------------------------
# DSSP end to end
# ---------------------------------------------------------------------------

def test_helix_is_assigned_helix(tmp_path):
    top, trj, info = write_protein_trajectory(tmp_path, n_res=18, n_frames=3)
    result = compute_dssp(top, trj)

    assert result["codes"].shape == (3, info["n_res"])
    assert result["occupancy"].shape == (info["n_res"], 3)
    np.testing.assert_allclose(result["occupancy"].sum(axis=1), 1.0, atol=1e-12)

    helix = DSSP_CODES.index("H")
    # Termini cannot form the i,i+4 bond, so score the interior.
    interior = result["occupancy"][2:-2, helix]
    assert interior.mean() > 0.9, f"helix occupancy only {interior.mean():.2f}"


def test_extended_strand_is_not_assigned_helix(tmp_path):
    # Beta-region torsions: the same code path must NOT report a helix.
    top, trj, _ = write_protein_trajectory(
        tmp_path, n_res=18, n_frames=3, phi=-139.0, psi=135.0
    )
    occ = compute_dssp(top, trj)["occupancy"]
    helix = DSSP_CODES.index("H")
    assert occ[:, helix].mean() < 0.1


def test_resids_are_carried_through(tmp_path):
    top, trj, info = write_protein_trajectory(tmp_path, n_res=10, n_frames=2)
    result = compute_dssp(top, trj)
    np.testing.assert_array_equal(result["resids"], np.arange(1, info["n_res"] + 1))


def test_window_restricts_the_frames_analysed(tmp_path):
    top, trj, _ = write_protein_trajectory(tmp_path, n_res=12, n_frames=10)
    # dt = 10 ps, so t0 = 50 ps should drop the first five frames.
    result = compute_dssp(top, trj, t0_ps=50.0)
    assert int(result["start_frame"]) == 5
    assert int(result["frames_used"]) == 5
    assert float(result["t0_ps"]) == 50.0


def test_step_subsamples_frames(tmp_path):
    top, trj, _ = write_protein_trajectory(tmp_path, n_res=12, n_frames=10)
    assert int(compute_dssp(top, trj, step=2)["frames_used"]) == 5


def test_empty_selection_raises(tmp_path):
    top, trj, _ = write_protein_trajectory(tmp_path, n_res=8, n_frames=2)
    with pytest.raises(ValueError, match="matched no atoms"):
        compute_dssp(top, trj, selection="resname ZZZ")


def test_first_frame_lookup(tmp_path):
    import MDAnalysis as mda

    top, trj, _ = write_protein_trajectory(tmp_path, n_res=8, n_frames=10)
    u = mda.Universe(top, trj)
    assert first_frame_at_or_after(u, 0.0) == 0
    assert first_frame_at_or_after(u, 30.0) == 3
    assert first_frame_at_or_after(u, 35.0) == 4        # rounds up to the next frame
    assert first_frame_at_or_after(u, 1e9) == 9         # clamped to the last frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_dssp_end_to_end(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import write_json
    from rotmd.io.output import load_npz

    top, trj, info = write_protein_trajectory(tmp_path, n_res=16, n_frames=6)
    window = write_json(tmp_path / "window.json", {"t0_ps": 20.0})
    out = tmp_path / "dssp.npz"

    assert main(["dssp", top, trj, "-o", str(out), "--window", str(window)]) == 0

    result = load_npz(out)
    assert result["occupancy"].shape == (info["n_res"], 3)
    np.testing.assert_allclose(result["occupancy"].sum(axis=1), 1.0, atol=1e-12)
    assert int(result["frames_used"]) == 4          # frames at t >= 20 ps


def test_cli_dssp_without_window_uses_all_frames(tmp_path):
    from rotmd.cli import main
    from rotmd.io.output import load_npz

    top, trj, _ = write_protein_trajectory(tmp_path, n_res=12, n_frames=5)
    out = tmp_path / "dssp.npz"
    main(["dssp", top, trj, "-o", str(out)])
    assert int(load_npz(out)["frames_used"]) == 5


def test_cli_compare_produces_delta_occupancy(tmp_path):
    """The T8 -> T6 hand-off: Δoccupancy plus its figure."""
    from rotmd.cli import main
    from rotmd.io.output import load_npz, save_npz

    n_res = 12
    resids = np.arange(1, n_res + 1, dtype=np.int32)

    def _dccm(name, seed):
        rng = np.random.default_rng(seed)
        m = rng.normal(0, 0.1, size=(n_res, n_res))
        m = 0.5 * (m + m.T)
        np.fill_diagonal(m, 1.0)
        return save_npz(tmp_path / name, {"dcc": m, "resids": resids})

    def _dssp(name, helix_frac):
        occ = np.zeros((n_res, 3))
        occ[:, 1] = helix_frac
        occ[:, 0] = 1.0 - helix_frac
        return save_npz(tmp_path / name, {"occupancy": occ, "resids": resids})

    a = [_dccm("a.npz", 0)]
    b = [_dccm("b.npz", 1)]
    da = [_dssp("da.npz", 0.9)]
    db = [_dssp("db.npz", 0.4)]
    out = tmp_path / "cmp.npz"
    fig = tmp_path / "dssp.png"

    rc = main([
        "compare", "--a", str(a[0]), "--b", str(b[0]), "-o", str(out),
        "--dssp-a", str(da[0]), "--dssp-b", str(db[0]),
        "--dssp-figure", str(fig), "--site", "6",
    ])
    assert rc == 0

    delta = load_npz(out.with_name("cmp_dssp.npz"))
    np.testing.assert_allclose(delta["delta_occupancy"][:, 1], -0.5, atol=1e-12)
    assert fig.exists() and fig.stat().st_size > 5000
