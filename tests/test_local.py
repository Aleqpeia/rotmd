"""T5a: salt bridges, H-bonds and RMSF around the mutation site.

The salt-bridge fixture places a LYS NZ at *exactly* known distances from two
carboxylates on a per-frame basis, so the expected occupancy is an arithmetic
fact (3 of 5 frames = 60%) rather than something read back off the code.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

from rotmd.analysis.local import (
    compare_occupancy,
    contact_occupancy,
    rmsf_per_residue,
    salt_bridges,
)


def _charged_universe(nz_distances, asp_far=20.0, box=200.0):
    """LYS75 NZ + ASP92 + GLU118, with NZ->ASP92 distance set per frame.

    ``nz_distances[i]`` is the NZ...OD1 separation in frame ``i``. GLU118 is
    parked ``asp_far`` Å away so it never contacts unless a test moves it.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u = mda.Universe.empty(
        5, n_residues=3, atom_resindex=[0, 1, 1, 2, 2], trajectory=True
    )
    u.add_TopologyAttr("names", ["NZ", "OD1", "OD2", "OE1", "OE2"])
    u.add_TopologyAttr("types", ["N", "O", "O", "O", "O"])
    u.add_TopologyAttr("elements", ["N", "O", "O", "O", "O"])
    # resnames/resids are per-residue attributes, not per-atom.
    u.add_TopologyAttr("resnames", ["LYS", "ASP", "GLU"])
    u.add_TopologyAttr("resids", [75, 92, 118])
    u.add_TopologyAttr("segids", ["A"])

    n_frames = len(nz_distances)
    coords = np.zeros((n_frames, 5, 3))
    for i, d in enumerate(nz_distances):
        coords[i, 0] = [0.0, 0.0, 0.0]          # NZ at the origin
        coords[i, 1] = [d, 0.0, 0.0]            # ASP92 OD1 at the requested distance
        coords[i, 2] = [d + 2.0, 0.0, 0.0]      # OD2, always further out
        coords[i, 3] = [asp_far, 0.0, 0.0]      # GLU118
        coords[i, 4] = [asp_far + 2.0, 0.0, 0.0]

    u.load_new(coords, format=MemoryReader, dt=10.0)
    u.dimensions = [box, box, box, 90.0, 90.0, 90.0]
    return u


# ---------------------------------------------------------------------------
# contact_occupancy / salt_bridges
# ---------------------------------------------------------------------------

def test_occupancy_is_the_fraction_of_frames_within_cutoff():
    # 3 of 5 frames inside 4.0 Å -> exactly 60%.
    u = _charged_universe([3.0, 3.5, 3.9, 5.0, 8.0])
    table = salt_bridges(u, site=75, cutoff=4.0)

    row = dict(zip(table["resids"].tolist(), table["occupancy"].tolist()))
    assert row[92] == pytest.approx(0.6)
    assert row[118] == pytest.approx(0.0)
    assert float(table["any_occupancy"]) == pytest.approx(0.6)
    assert int(table["n_frames"]) == 5


def test_cutoff_boundary_is_inclusive():
    u = _charged_universe([4.0, 4.0001])
    occ = dict(zip(*(salt_bridges(u, 75, cutoff=4.0)[k].tolist()
                     for k in ("resids", "occupancy"))))
    assert occ[92] == pytest.approx(0.5)   # only the exactly-4.0 frame counts


def test_always_and_never_bound_extremes():
    assert float(salt_bridges(_charged_universe([2.5] * 4), 75)["any_occupancy"]) == 1.0
    assert float(salt_bridges(_charged_universe([12.0] * 4), 75)["any_occupancy"]) == 0.0


def test_partners_are_sorted_by_occupancy():
    u = _charged_universe([3.0, 3.0, 9.0], asp_far=3.2)   # GLU118 now also close
    table = salt_bridges(u, site=75, cutoff=4.0)
    assert list(table["occupancy"]) == sorted(table["occupancy"], reverse=True)


def test_any_occupancy_is_not_the_sum_of_partners():
    """Two partners engaged in the same frame must not double-count."""
    u = _charged_universe([3.0, 3.0, 3.0], asp_far=3.5)   # both bound every frame
    table = salt_bridges(u, site=75, cutoff=4.0)
    assert table["occupancy"].sum() == pytest.approx(2.0)   # per-partner
    assert float(table["any_occupancy"]) == pytest.approx(1.0)  # per-frame


def test_step_subsamples_frames():
    u = _charged_universe([3.0, 9.0, 3.0, 9.0, 3.0, 9.0])
    table = salt_bridges(u, site=75, cutoff=4.0, step=2)
    assert int(table["n_frames"]) == 3
    assert float(table["any_occupancy"]) == pytest.approx(1.0)  # frames 0,2,4 all bound


def test_start_frame_respects_the_window():
    u = _charged_universe([3.0, 3.0, 9.0, 9.0])
    assert float(salt_bridges(u, 75, start=2)["any_occupancy"]) == 0.0
    assert float(salt_bridges(u, 75, start=0)["any_occupancy"]) == 0.5


def test_periodic_boundaries_are_honoured():
    """A partner across the box edge is close, not 190 Å away."""
    u = _charged_universe([197.0], box=200.0)   # 197 Å ≡ 3 Å through the boundary
    assert float(salt_bridges(u, 75, cutoff=4.0)["any_occupancy"]) == 1.0


def test_wt_site_without_a_charged_nitrogen_still_reports():
    """Asn75 has no NZ; the fallback probe must keep the table comparable."""
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u = mda.Universe.empty(3, n_residues=2, atom_resindex=[0, 1, 1], trajectory=True)
    u.add_TopologyAttr("names", ["ND2", "OD1", "OD2"])
    u.add_TopologyAttr("elements", ["N", "O", "O"])
    u.add_TopologyAttr("resnames", ["ASN", "ASP"])
    u.add_TopologyAttr("resids", [75, 92])
    u.load_new(np.array([[[0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]]], dtype=float),
               format=MemoryReader)
    u.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]

    table = salt_bridges(u, site=75, cutoff=4.0)
    assert float(table["any_occupancy"]) == 1.0


def test_hydrogen_detection_survives_a_topology_without_elements():
    """Regression: a .gro carries no `elements`, and `element H` then raises.

    Found by running against real GRO-based data — every synthetic fixture here
    sets an elements attribute, which hid it.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    from rotmd.analysis.local import _has_hydrogens

    u = mda.Universe.empty(2, n_residues=1, atom_resindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("names", ["N", "HN"])       # names only, no elements
    u.add_TopologyAttr("resnames", ["ALA"])
    u.add_TopologyAttr("resids", [1])
    u.load_new(np.zeros((1, 2, 3)), format=MemoryReader)

    assert not hasattr(u.atoms, "elements")
    assert _has_hydrogens(u) is True               # must not raise AttributeError


def test_hydrogen_detection_false_for_a_heavy_atom_only_system():
    u = _charged_universe([3.0])                   # NZ/OD/OE only, no hydrogens
    from rotmd.analysis.local import _has_hydrogens

    assert _has_hydrogens(u) is False


def test_empty_selections_raise():
    u = _charged_universe([3.0])
    with pytest.raises(ValueError, match="probe selection"):
        contact_occupancy(u, "resname ZZZ", "resname ASP")
    with pytest.raises(ValueError, match="partner selection"):
        contact_occupancy(u, "resname LYS", "resname ZZZ")


# ---------------------------------------------------------------------------
# compare_occupancy
# ---------------------------------------------------------------------------

def test_compare_occupancy_classifies_gained_lost_retained():
    a = {"resids": np.array([92, 118]), "occupancy": np.array([0.8, 0.5])}
    b = {"resids": np.array([92, 130]), "occupancy": np.array([0.2, 0.9])}

    rows = {r["resid"]: r for r in compare_occupancy(a, b)}
    assert rows[92]["status"] == "retained"
    assert rows[92]["delta"] == pytest.approx(-0.6)
    assert rows[118]["status"] == "lost"
    assert rows[130]["status"] == "gained"


def test_compare_occupancy_sorts_by_absolute_change():
    a = {"resids": np.array([1, 2]), "occupancy": np.array([0.5, 0.5])}
    b = {"resids": np.array([1, 2]), "occupancy": np.array([0.55, 0.95])}
    assert [r["resid"] for r in compare_occupancy(a, b)] == [2, 1]


# ---------------------------------------------------------------------------
# RMSF
# ---------------------------------------------------------------------------

def test_rmsf_is_zero_for_a_rigid_trajectory():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(8, 3)) * 5.0
    coords = np.repeat(ref[None], 20, axis=0)
    out = rmsf_per_residue(coords, np.ones(8, dtype=bool), np.arange(1, 9))
    assert out["rmsf"].max() < 1e-9


def _one_mobile_residue(n_frames=2000, amplitude=0.5, seed=1):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(6, 3)) * 5.0
    coords = np.repeat(ref[None], n_frames, axis=0).astype(float)
    coords[:, 3] += rng.normal(size=(n_frames, 3)) * amplitude
    return coords


def test_rmsf_tracks_imposed_amplitude():
    coords = _one_mobile_residue()
    out = rmsf_per_residue(coords, np.ones(6, dtype=bool), np.arange(1, 7))
    assert out["rmsf"][3] == out["rmsf"].max()
    # Fitting on all six atoms lets the mobile one drag the superposition, so
    # the static atoms pick up some apparent motion too — the contrast is real
    # but blunted. See the next test for the fix.
    assert out["rmsf"][3] > 3 * np.median(out["rmsf"])


def test_excluding_the_mobile_residue_from_the_align_core_sharpens_rmsf():
    """Why --sel-align exists: a flexible region must not define the frame."""
    coords = _one_mobile_residue()
    core = np.ones(6, dtype=bool)
    core[3] = False                                    # fit on the static atoms only

    all_atoms = rmsf_per_residue(coords, np.ones(6, dtype=bool), np.arange(1, 7))["rmsf"]
    stable_core = rmsf_per_residue(coords, core, np.arange(1, 7))["rmsf"]

    # The genuinely static residues now read as static...
    assert stable_core[core].max() < all_atoms[core].max() / 5
    # ...and the mobile one stands out far more clearly.
    assert stable_core[3] / np.median(stable_core[core]) > 20


def test_rmsf_honours_the_frame_mask():
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(100, 5, 3))
    mask = np.zeros(100, dtype=bool)
    mask[60:] = True
    assert int(rmsf_per_residue(coords, np.ones(5, dtype=bool),
                                np.arange(5), mask)["frames_used"]) == 40


def test_rmsf_needs_two_frames():
    with pytest.raises(ValueError, match="at least 2 frames"):
        rmsf_per_residue(np.zeros((1, 4, 3)), np.ones(4, dtype=bool), np.arange(4))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_local_end_to_end(tmp_path):
    import MDAnalysis as mda

    from rotmd.cli import main
    from rotmd.io.output import load_npz, save_npz

    u = _charged_universe([3.0, 3.0, 9.0, 9.0, 3.0])
    top = tmp_path / "sys.pdb"
    trj = tmp_path / "sys.xtc"
    u.atoms.write(str(top))
    with mda.Writer(str(trj), n_atoms=len(u.atoms)) as w:
        for _ts in u.trajectory:
            w.write(u.atoms)

    merged = save_npz(tmp_path / "merged.npz", {
        "time_ps": np.arange(5) * 10.0,
        "ca_coords": np.repeat(np.zeros((1, 3, 3)), 5, axis=0).astype(np.float32)
        + np.random.default_rng(0).normal(size=(5, 3, 3)) * 0.1,
        "ca_resids": np.array([75, 92, 118], dtype=np.int32),
        "ca_align_mask": np.ones(3, dtype=bool),
    })

    out = tmp_path / "local.npz"
    rc = main([
        "local", str(top), str(trj), "--site", "75",
        "-o", str(out), "--merged", str(merged),
    ])
    assert rc == 0

    result = load_npz(out)
    assert float(result["sb_any_occupancy"]) == pytest.approx(0.6)
    assert int(result["site"]) == 75
    assert result["rmsf"].shape == (3,)
