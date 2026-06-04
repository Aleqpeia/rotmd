"""Tests for observables.structural: RMSD, Rg, shape parameters."""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.observables.structural import (
    acylindricity,
    asphericity,
    compute_rmsd,
    compute_rmsd_trajectory,
    compute_structural_parameters,
    compute_structural_trajectory,
    end_to_end_distance,
    radius_of_gyration,
    radius_of_gyration_components,
)
from _helpers import ellipsoid_positions


def test_rmsd_identical_is_zero():
    pos = ellipsoid_positions(40, seed=1)
    assert compute_rmsd(pos, pos.copy()) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_invariant_under_rotation_when_aligned():
    pos = ellipsoid_positions(50, axes=(1.0, 2.0, 3.0), seed=2)
    # Rotate by a known matrix; aligned RMSD should be ~0.
    theta = 0.7
    Rz = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    rotated = pos @ Rz.T
    assert compute_rmsd(rotated, pos, align=True) == pytest.approx(0.0, abs=1e-7)
    # Without alignment the RMSD is generally nonzero.
    assert compute_rmsd(rotated, pos, align=False) > 1e-3


def test_rmsd_translation_invariance():
    pos = ellipsoid_positions(30, seed=3)
    shifted = pos + np.array([10.0, -5.0, 3.0])
    assert compute_rmsd(shifted, pos, align=False) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_rmsd(np.zeros((5, 3)), np.zeros((6, 3)))


def test_rmsd_trajectory_first_frame_zero():
    pos = ellipsoid_positions(20, seed=4)
    traj = np.stack([pos, pos + 0.5, pos + 1.0])
    rmsd = compute_rmsd_trajectory(traj, pos, align=False)
    assert rmsd[0] == pytest.approx(0.0, abs=1e-9)
    assert rmsd.shape == (3,)


def test_radius_of_gyration_known_value():
    # Two unit masses at +/- d along x: Rg = d.
    pos = np.array([[2.0, 0, 0], [-2.0, 0, 0]])
    masses = np.array([1.0, 1.0])
    assert radius_of_gyration(pos, masses) == pytest.approx(2.0)


def test_rg_components_pythagoras():
    pos = ellipsoid_positions(60, axes=(1.0, 2.0, 4.0), seed=5)
    masses = np.ones(60)
    Rg, comps = radius_of_gyration_components(pos, masses)
    # Rg² = Rg_x² + Rg_y² + Rg_z²
    assert Rg**2 == pytest.approx(np.sum(comps**2), rel=1e-9)
    # The elongated z-axis carries the largest component.
    assert comps[2] > comps[0]


def test_asphericity_sphere_vs_rod():
    rng = np.random.default_rng(6)
    sphere = rng.normal(size=(400, 3))
    masses = np.ones(400)
    b_sphere = asphericity(sphere, masses)
    rod = sphere * np.array([0.05, 0.05, 5.0])
    b_rod = asphericity(rod, masses)
    assert b_sphere < 0.1  # near-spherical
    assert b_rod > b_sphere  # elongated is more aspherical


def test_acylindricity_nonnegative_for_axisymmetric():
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(300, 3)) * np.array([1.0, 1.0, 4.0])
    c = acylindricity(pts, np.ones(300))
    assert c >= -1e-9


def test_end_to_end_default_and_indices():
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [4, 0, 0]])
    assert end_to_end_distance(pos) == pytest.approx(4.0)
    assert end_to_end_distance(pos, atom_indices=(0, 1)) == pytest.approx(1.0)


def test_structural_parameters_dict_keys():
    pos = ellipsoid_positions(40, seed=8)
    masses = np.ones(40)
    params = compute_structural_parameters(pos, masses, reference=pos)
    assert {"rmsd", "rg", "rg_x", "rg_y", "rg_z", "asphericity", "com"} <= set(params)
    assert params["rmsd"] == pytest.approx(0.0, abs=1e-7)


def test_structural_trajectory_arrays():
    pos = ellipsoid_positions(25, seed=9)
    traj = np.stack([pos, pos + 0.3, pos - 0.2])
    masses = np.ones(25)
    out = compute_structural_trajectory(traj, masses, reference=pos, verbose=False)
    assert out["rg"].shape == (3,)
    assert out["rg_components"].shape == (3, 3)
    assert "rmsd" in out
    # Without a reference, no rmsd key is produced.
    out2 = compute_structural_trajectory(traj, masses, reference=None, verbose=False)
    assert "rmsd" not in out2


def test_structural_trajectory_verbose_runs():
    pos = ellipsoid_positions(20, seed=10)
    traj = np.stack([pos, pos + 0.1])
    # verbose=True exercises the progress/summary print branch.
    out = compute_structural_trajectory(traj, np.ones(20), reference=pos, verbose=True)
    assert out["rmsd"].shape == (2,)
