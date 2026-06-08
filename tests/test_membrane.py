"""Tests for core.membrane geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")

from rotmd.core.membrane import get_membrane_center_z, get_membrane_normal  # noqa: E402


def _universe_with_membrane(z_values, resname="CHL1"):
    n = len(z_values)
    u = mda.Universe.empty(n, n_residues=n, atom_resindex=np.arange(n), trajectory=True)
    u.add_TopologyAttr("names", ["O"] * n)
    u.add_TopologyAttr("masses", np.ones(n))
    u.add_TopologyAttr("resnames", [resname] * n)
    u.add_TopologyAttr("resids", list(range(1, n + 1)))
    pos = np.zeros((n, 3))
    pos[:, 2] = z_values
    u.atoms.positions = pos
    return u


def test_membrane_center_z_is_mean():
    u = _universe_with_membrane([10.0, 20.0, 30.0])
    assert get_membrane_center_z(u, membrane_sel="resname CHL1") == pytest.approx(20.0)


def test_membrane_center_z_empty_selection_raises():
    u = _universe_with_membrane([1.0, 2.0])
    with pytest.raises(ValueError, match="No atoms matched"):
        get_membrane_center_z(u, membrane_sel="resname NOPE")


def test_membrane_normal_is_z_axis():
    u = _universe_with_membrane([1.0, 2.0])
    assert get_membrane_normal(u) == pytest.approx([0.0, 0.0, 1.0])
