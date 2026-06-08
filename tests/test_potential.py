"""Tests for observables.potential.

The hydrophobic/SASA path needs the optional `freesasa` dependency, so those
tests are skipped when it is unavailable. The electrostatic model only needs
MDAnalysis and is exercised directly.
"""

from __future__ import annotations

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")

from rotmd.observables.potential import (  # noqa: E402
    HAS_FREESASA,
    RESIDUE_CHARGES,
    ElectrostaticEnergy,
)


def _universe(resnames, z_per_residue, atoms_per_residue=2):
    """A tiny MDAnalysis universe with one z-level per residue."""
    n_res = len(resnames)
    n_atoms = n_res * atoms_per_residue
    atom_resindex = np.repeat(np.arange(n_res), atoms_per_residue)
    u = mda.Universe.empty(
        n_atoms, n_residues=n_res, atom_resindex=atom_resindex, trajectory=True
    )
    u.add_TopologyAttr("names", ["CA"] * n_atoms)
    u.add_TopologyAttr("masses", np.ones(n_atoms))
    u.add_TopologyAttr("resnames", list(resnames))
    u.add_TopologyAttr("resids", list(range(1, n_res + 1)))
    pos = np.zeros((n_atoms, 3))
    for i, z in enumerate(z_per_residue):
        pos[atom_resindex == i, 2] = z
    u.atoms.positions = pos
    return u


def test_residue_charges_table():
    assert RESIDUE_CHARGES["ARG"] == pytest.approx(1.0)
    assert RESIDUE_CHARGES["ASP"] == pytest.approx(-1.0)
    assert RESIDUE_CHARGES["ALA"] == pytest.approx(0.0)


def test_calculate_residue_charge_lookup():
    u = _universe(["ARG", "ASP", "ALA"], [5.0, 5.0, 5.0])
    e = ElectrostaticEnergy()
    charges = [e.calculate_residue_charge(r) for r in u.residues]
    assert charges == pytest.approx([1.0, -1.0, 0.0])


def test_membrane_interaction_skips_neutral_and_returns_finite():
    # Place charged residues right at the upper membrane surface (z = center+20).
    u = _universe(["ARG", "ASP", "ALA"], [5.0, 5.0, 5.0])
    e = ElectrostaticEnergy()
    total, per_res = e.calculate_membrane_interaction_energy(
        u.atoms, membrane_center_z=-15.0, protein_com_z=5.0
    )
    assert np.isfinite(total)
    # Neutral ALA (resid 3) contributes nothing; charged residues are present.
    assert 3 not in per_res
    assert set(per_res) == {1, 2}


def test_membrane_interaction_lower_surface_branch():
    # protein_com_z below the membrane center selects the lower surface; place
    # the charged residues right on it (z = center - 20).
    u = _universe(["ARG", "ASP"], [-25.0, -25.0])
    e = ElectrostaticEnergy()
    total, per_res = e.calculate_membrane_interaction_energy(
        u.atoms, membrane_center_z=-5.0, protein_com_z=-25.0
    )
    assert set(per_res) == {1, 2}
    assert np.isfinite(total)


def test_membrane_interaction_beyond_cutoff_is_skipped():
    # Charged residues far from either surface (> cutoff) contribute nothing.
    e = ElectrostaticEnergy(distance_cutoff=5.0)
    u = _universe(["ARG", "ASP"], [200.0, 200.0])
    total, per_res = e.calculate_membrane_interaction_energy(
        u.atoms, membrane_center_z=0.0, protein_com_z=200.0
    )
    assert total == pytest.approx(0.0)
    assert per_res == {}


def test_membrane_interaction_all_neutral_is_zero():
    u = _universe(["ALA", "GLY"], [5.0, 5.0])
    e = ElectrostaticEnergy()
    total, per_res = e.calculate_membrane_interaction_energy(
        u.atoms, membrane_center_z=-15.0, protein_com_z=5.0
    )
    assert total == pytest.approx(0.0)
    assert per_res == {}


def test_self_energy_symmetric_distribution():
    u = _universe(["ARG", "ASP"], [0.0, 4.0])
    e = ElectrostaticEnergy()
    total, per_res = e.calculate_protein_self_energy(u.atoms)
    # Energy is split evenly between the two charged residues.
    assert sum(per_res.values()) == pytest.approx(total, rel=1e-9)
    # Opposite charges attract -> negative Coulomb energy.
    assert total < 0


@pytest.mark.skipif(not HAS_FREESASA, reason="freesasa not installed")
def test_total_energy_constructs_with_freesasa():
    from rotmd.observables.potential import TotalEnergy

    calc = TotalEnergy()
    assert calc.hydrophobic_weight == pytest.approx(1.0)


@pytest.mark.skipif(HAS_FREESASA, reason="freesasa is installed")
def test_hydrophobic_energy_requires_freesasa():
    from rotmd.observables.potential import HydrophobicEnergy

    with pytest.raises(ImportError):
        HydrophobicEnergy()
