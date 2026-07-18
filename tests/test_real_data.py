"""Integration tests against the real hippocalcin N75K trajectory.

Synthetic fixtures cannot reproduce everything real MD does — the missing
`elements` attribute on a GRO topology, a positions-only XTC, 191 real residues
with real secondary structure. These tests run the actual pipeline on a
protein-only slice of the 420 ns production trajectory.

They skip unless the fixture exists, so the suite stays green on a machine
without the data. Build it with::

    python - <<'PY'
    import MDAnalysis as mda, os
    u = mda.Universe("~/Data/n75kcomplex.gro", "~/Data/n75kcomplex.xtc")
    p = u.select_atoms("protein")
    os.makedirs(os.path.expanduser("~/.cache/rotmd-testdata"), exist_ok=True)
    p.write(os.path.expanduser("~/.cache/rotmd-testdata/n75k_prot.gro"))
    with mda.Writer(os.path.expanduser("~/.cache/rotmd-testdata/n75k_prot.xtc"),
                    n_atoms=len(p)) as w:
        for ts in u.trajectory[::20]:
            w.write(p)
    PY
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

DATA = Path.home() / ".cache/rotmd-testdata"
GRO = DATA / "n75k_prot.gro"
XTC = DATA / "n75k_prot.xtc"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (GRO.exists() and XTC.exists()),
                       reason="real-data fixture not built (see module docstring)"),
]

N_CA = 191          # hippocalcin, resid 3-193
SITE = 75           # the N75K mutation


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """Run a real `rotmd extract` once and share it across the module."""
    from rotmd import cli

    out = tmp_path_factory.mktemp("real") / "n75k.npz"
    cfg = cli.ExtractConfig(
        topology=GRO, trajectory=XTC, reference=GRO, output=out,
        selection="protein", membrane_sel="resname CHL1",
        start=0, stop=200, step=1, n_workers=1, force=True,
        sel_ca="name CA", sel_align="resid 20-180",
        domains="N-lobe:3-95,C-lobe:96-193",
        system="n75k", replica=1, no_energy=True,
    )
    return cli.extract(cfg)


def test_extract_shapes_match_the_real_protein(extracted):
    from rotmd.io.output import load_npz

    data = load_npz(extracted)
    assert data["ca_coords"].shape == (200, N_CA, 3)
    assert data["ca_resids"].shape == (N_CA,)
    assert data["ca_resids"].min() == 3 and data["ca_resids"].max() == 193
    assert data["rmsd_dom"].shape == (200, 2)          # N-lobe, C-lobe
    # The alignment core excludes the flexible termini.
    assert 0 < int(data["ca_align_mask"].sum()) < N_CA


def test_positions_only_xtc_degrades_instead_of_failing(extracted):
    """The real .xtc has no velocities — extract must still produce a chunk.

    Before T1 this raised RuntimeError, which would have blocked the entire
    DCCM/DSSP path on exactly this file.
    """
    from rotmd.io.meta import meta_path, read_json
    from rotmd.io.output import load_npz

    data = load_npz(extracted)
    assert "L_vector" not in data
    assert "K_trans" not in data
    assert data["rmsd"].shape == (200,)                # structural path intact

    meta = read_json(meta_path(extracted))
    assert meta["has_velocities"] is False
    assert meta["n_ca"] == N_CA
    assert meta["system"] == "n75k"


def test_domain_rmsd_differs_between_lobes(extracted):
    """Per-domain RMSD exists precisely because lobes need not agree."""
    from rotmd.io.output import load_npz

    dom = load_npz(extracted)["rmsd_dom"]
    assert np.all(dom >= 0)
    assert not np.allclose(dom[:, 0], dom[:, 1])


def test_equilibrate_finds_a_window_inside_the_trajectory(extracted, tmp_path):
    from rotmd.analysis.equilibration import build_window
    from rotmd.io.output import load_npz

    data = load_npz(extracted)
    window = build_window(data["time_ps"], data["rmsd"], column="rmsd", method="native")

    assert data["time_ps"][0] <= window["t0_ps"] <= data["time_ps"][-1]
    assert window["g"] >= 1.0
    assert 0 < window["n_frames_used"] <= 200


def test_dccm_on_real_coordinates(extracted):
    from rotmd.analysis.dccm import compute_dccm
    from rotmd.io.output import load_npz

    data = load_npz(extracted)
    result = compute_dccm(data["ca_coords"], data["ca_align_mask"], data["ca_resids"])
    dcc = result["dcc"]

    assert dcc.shape == (N_CA, N_CA)
    np.testing.assert_allclose(dcc, dcc.T, atol=0)
    np.testing.assert_allclose(np.diag(dcc), 1.0, atol=1e-12)
    assert dcc.min() >= -1.0 and dcc.max() <= 1.0

    # Sequence neighbours are physically bonded and must move together far more
    # than an arbitrary distant pair — a real-structure sanity check.
    neighbours = np.mean([dcc[i, i + 1] for i in range(N_CA - 1)])
    distant = np.mean([dcc[i, i + 100] for i in range(N_CA - 100)])
    assert neighbours > 0.5
    assert neighbours > distant

    assert result["rmsf"].shape == (N_CA,)
    assert result["rmsf"].min() > 0


def test_dssp_reports_a_mostly_helical_protein():
    """Hippocalcin is an EF-hand protein: helix should dominate."""
    from rotmd.analysis.dssp import DSSP_CODES, compute_dssp

    result = compute_dssp(str(GRO), str(XTC), step=100)
    occ = result["occupancy"]

    assert occ.shape == (N_CA, 3)
    np.testing.assert_allclose(occ.sum(axis=1), 1.0, atol=1e-12)
    helix = occ[:, DSSP_CODES.index("H")].mean()
    sheet = occ[:, DSSP_CODES.index("E")].mean()
    assert helix > 0.35
    assert helix > sheet


def test_site_75_is_a_lysine_with_a_charged_nitrogen():
    """Confirms the fixture really is the N75K mutant."""
    import MDAnalysis as mda

    u = mda.Universe(str(GRO))
    site = u.select_atoms(f"protein and resid {SITE}")
    assert set(site.resnames) == {"LYS"}
    assert len(u.select_atoms(f"resid {SITE} and name NZ")) == 1


def test_salt_bridge_occupancy_is_a_valid_fraction():
    import MDAnalysis as mda

    from rotmd.analysis.local import salt_bridges

    u = mda.Universe(str(GRO), str(XTC))
    table = salt_bridges(u, site=SITE, cutoff=4.0, start=0, stop=60, step=10)

    assert 0.0 <= float(table["any_occupancy"]) <= 1.0
    assert np.all((table["occupancy"] >= 0) & (table["occupancy"] <= 1))
    # Partners can only be Asp/Glu, never the lysine itself.
    assert SITE not in table["resids"].tolist()


def test_hydrogen_bond_analysis_reports_why_a_gro_cannot_work():
    """A .gro has no charges; the error must say to use a .tpr, not just fail."""
    import MDAnalysis as mda

    from rotmd.analysis.local import hydrogen_bonds

    u = mda.Universe(str(GRO), str(XTC))
    with pytest.raises(ValueError, match=r"\.tpr"):
        hydrogen_bonds(u, site=SITE, start=0, stop=2)
