"""T5b: per-residue Coulomb/LJ decomposition via a GROMACS rerun.

Index/mdp generation, .edr term matching and block-error statistics are tested
directly. The grompp + mdrun path is marked ``integration`` and skips without
a `gmx` binary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from rotmd.analysis.coulomb import (
    block_error,
    build_groups,
    select_pair_terms,
    write_index,
    write_rerun_mdp,
)

needs_gmx = pytest.mark.skipif(shutil.which("gmx") is None, reason="gmx not installed")


# ---------------------------------------------------------------------------
# Index file
# ---------------------------------------------------------------------------

def test_index_is_written_one_based(tmp_path):
    """GROMACS indexes from 1; MDAnalysis from 0. Off by one here is silent."""
    path = write_index({"site": np.array([0, 1, 2])}, tmp_path / "i.ndx")
    text = path.read_text()
    assert "[ site ]" in text
    assert "1" in text.split("\n")[1] and "3" in text.split("\n")[1]
    assert "0" not in text.split("\n")[1].split()


def test_index_wraps_long_groups(tmp_path):
    path = write_index({"big": np.arange(40)}, tmp_path / "i.ndx", per_line=15)
    body = [ln for ln in path.read_text().splitlines() if not ln.startswith("[")]
    assert len(body) == 3                       # 15 + 15 + 10
    assert len(body[0].split()) == 15
    numbers = [int(v) for line in body for v in line.split()]
    assert numbers == list(range(1, 41))


def test_index_writes_multiple_groups_in_order(tmp_path):
    path = write_index(
        {"site75": np.array([5, 6]), "shell": np.array([1, 2, 3])}, tmp_path / "i.ndx"
    )
    text = path.read_text()
    assert text.index("[ site75 ]") < text.index("[ shell ]")


def test_index_rejects_an_empty_group(tmp_path):
    with pytest.raises(ValueError, match="is empty"):
        write_index({"nothing": np.array([], dtype=int)}, tmp_path / "i.ndx")


# ---------------------------------------------------------------------------
# mdp
# ---------------------------------------------------------------------------

def test_rerun_mdp_does_no_dynamics(tmp_path):
    text = write_rerun_mdp(tmp_path / "r.mdp", ["site75", "shell"]).read_text()
    assert "nsteps          = 0" in text
    assert "energygrps      = site75 shell" in text


def test_rerun_mdp_carries_the_production_cutoffs(tmp_path):
    """Changing the cutoff changes what 'short-range Coulomb' even means."""
    text = write_rerun_mdp(
        tmp_path / "r.mdp", ["a", "b"], coulombtype="PME", rcoulomb=1.4, rvdw=1.4
    ).read_text()
    assert "coulombtype     = PME" in text
    assert "rcoulomb        = 1.4" in text
    assert "rvdw            = 1.4" in text


# ---------------------------------------------------------------------------
# Energy-term matching
# ---------------------------------------------------------------------------

def test_pair_terms_found_in_either_ordering():
    """GROMACS does not promise the order of A and B in 'Coul-SR:A-B'."""
    forward = {"Coul-SR:site75-shell": np.zeros(3), "LJ-SR:site75-shell": np.zeros(3)}
    reverse = {"Coul-SR:shell-site75": np.zeros(3)}

    assert select_pair_terms(forward, "site75", "shell")["Coul-SR"] == "Coul-SR:site75-shell"
    assert select_pair_terms(reverse, "site75", "shell")["Coul-SR"] == "Coul-SR:shell-site75"


def test_pair_terms_absent_returns_empty():
    assert select_pair_terms({"Potential": np.zeros(2)}, "a", "b") == {}


# ---------------------------------------------------------------------------
# Block error
# ---------------------------------------------------------------------------

def test_block_error_mean_is_the_plain_mean():
    values = np.arange(100.0)
    assert block_error(values, g=10)["mean"] == pytest.approx(values.mean())


def test_block_error_uses_ceil_g_as_the_block_size():
    stats = block_error(np.zeros(100), g=4.2)
    assert stats["block_size"] == 5
    assert stats["n_blocks"] == 20


def test_block_error_exceeds_the_naive_standard_error_for_correlated_data():
    """The reason block averaging is used at all.

    A correlated series has fewer independent samples than frames, so the naive
    SEM is too small; block averaging over g-length blocks recovers the honest
    error bar.
    """
    rng = np.random.default_rng(0)
    phi, n = 0.9, 4000
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()

    g = (1 + phi) / (1 - phi)                       # = 19
    naive = x.std(ddof=1) / np.sqrt(n)
    blocked = block_error(x, g=g)["stderr"]
    assert blocked > 2 * naive


def test_block_error_is_nan_when_there_are_too_few_blocks():
    """Better an explicit NaN than a confidently tiny error from one block."""
    stats = block_error(np.arange(10.0), g=100)
    assert np.isnan(stats["stderr"])
    assert stats["n_blocks"] < 2
    assert not np.isnan(stats["mean"])


def test_block_error_of_independent_data_matches_the_sem():
    rng = np.random.default_rng(1)
    values = rng.normal(size=5000)
    stats = block_error(values, g=1.0)
    assert stats["stderr"] == pytest.approx(values.std(ddof=1) / np.sqrt(5000), rel=0.05)


# ---------------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------------

def _peptide_universe():
    """Ten glycines on a line, 5 Å apart, so shell membership is predictable."""
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    n_res, per_res = 10, 2
    n_atoms = n_res * per_res
    u = mda.Universe.empty(
        n_atoms, n_residues=n_res,
        atom_resindex=np.repeat(np.arange(n_res), per_res), trajectory=True,
    )
    u.add_TopologyAttr("names", ["N", "CA"] * n_res)
    u.add_TopologyAttr("types", ["N", "C"] * n_res)
    u.add_TopologyAttr("resnames", ["GLY"] * n_res)
    u.add_TopologyAttr("resids", list(range(1, n_res + 1)))
    u.add_TopologyAttr("segids", ["A"])

    coords = np.zeros((1, n_atoms, 3))
    for r in range(n_res):
        coords[0, r * per_res:(r + 1) * per_res, 0] = [r * 5.0, r * 5.0 + 1.0]
    u.load_new(coords, format=MemoryReader)
    return u


def test_shell_contains_whole_residues_only():
    """A half-included residue splits a charge group and biases the energy."""
    pytest.importorskip("MDAnalysis")
    u = _peptide_universe()
    groups = build_groups(u, site=5, radius=6.0, selection="all")

    shell_residues = u.atoms[groups["shell"]].residues
    for residue in shell_residues:
        assert set(residue.atoms.indices) <= set(groups["shell"].tolist())
    assert 5 not in shell_residues.resids       # the site itself is excluded


def test_larger_radius_grows_the_shell():
    pytest.importorskip("MDAnalysis")
    u = _peptide_universe()
    small = build_groups(u, site=5, radius=6.0, selection="all")["shell"]
    large = build_groups(u, site=5, radius=12.0, selection="all")["shell"]
    assert len(large) > len(small)


def test_unknown_site_raises():
    pytest.importorskip("MDAnalysis")
    with pytest.raises(ValueError, match="matched no atoms"):
        build_groups(_peptide_universe(), site=999, selection="all")


def test_isolated_site_raises():
    pytest.importorskip("MDAnalysis")
    with pytest.raises(ValueError, match="no residues within"):
        build_groups(_peptide_universe(), site=5, radius=0.5, selection="all")


# ---------------------------------------------------------------------------
# End to end (needs gmx)
# ---------------------------------------------------------------------------

@needs_gmx
@pytest.mark.integration
def test_rerun_end_to_end(tmp_path):
    """grompp + mdrun -rerun + panedr on a prebuilt real-protein system."""
    from rotmd.analysis.coulomb import coulomb_decomposition

    built = Path.home() / ".cache/rotmd-testdata/rerun_system"
    if not (built / "topol.top").exists():
        pytest.skip("GROMACS rerun fixture not built")

    result = coulomb_decomposition(
        topology_top=built / "topol.top",
        structure=built / "em.gro",
        trajectory=built / "md.xtc",
        site=75, workdir=tmp_path / "rerun", g=5.0, radius=10.0,
    )
    assert int(result["n_frames"]) > 5
    assert int(result["group_site_atoms"]) > 0
    assert int(result["group_shell_atoms"]) > int(result["group_site_atoms"])
    # A cationic lysine embedded in a net-negative protein must be attracted.
    assert float(result["Coul_SR_mean"]) < 0
    assert float(result["Coul_SR_stderr"]) > 0
