"""Tests for observables.energetics: kinetic energy, temperature, virial."""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.observables.energetics import (
    compute_energetics,
    compute_energetics_trajectory,
    instantaneous_temperature,
    kinetic_energy_rotational,
    kinetic_energy_total,
    kinetic_energy_translational,
    potential_energy_from_forces,
    virial_tensor,
)

# 1 amu·Å²/ps² in kcal/mol — the conversion factor used throughout the module.
AMU_A2_PS2_TO_KCAL = 0.000143933


def test_translational_kinetic_energy_known():
    # Single atom, mass 2, speed (3,4,0) -> v²=25, K=0.5*2*25=25 amu·Å²/ps².
    vel = np.array([[3.0, 4.0, 0.0]])
    masses = np.array([2.0])
    expected = 25.0 * AMU_A2_PS2_TO_KCAL
    assert kinetic_energy_translational(vel, masses) == pytest.approx(expected)


def test_kinetic_energy_total_equals_translational():
    rng = np.random.default_rng(0)
    vel = rng.normal(size=(10, 3))
    masses = np.abs(rng.normal(size=10)) + 1
    assert kinetic_energy_total(vel, masses) == pytest.approx(
        kinetic_energy_translational(vel, masses)
    )


def test_kinetic_energy_zero_velocity():
    assert kinetic_energy_translational(np.zeros((5, 3)), np.ones(5)) == pytest.approx(0.0)


def test_rotational_kinetic_energy_nonnegative():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(30, 3)) * 3
    vel = rng.normal(size=(30, 3))
    masses = np.abs(rng.normal(size=30)) + 1
    assert kinetic_energy_rotational(pos, vel, masses) >= 0.0


def test_temperature_scales_with_kinetic_energy():
    rng = np.random.default_rng(2)
    vel = rng.normal(size=(50, 3))
    masses = np.ones(50)
    t1 = instantaneous_temperature(vel, masses)
    t2 = instantaneous_temperature(2 * vel, masses)
    # T ∝ K ∝ v², so doubling velocities quadruples the temperature.
    assert t2 == pytest.approx(4 * t1, rel=1e-9)
    assert t1 > 0


def test_temperature_degenerate_few_atoms_is_zero():
    # With <= 2 atoms, N_dof = 3N - 6 <= 0, so the guard returns 0.
    assert instantaneous_temperature(np.ones((2, 3)), np.ones(2)) == 0.0


def test_potential_energy_from_forces_sign():
    # Constant force +x, displacement +x -> work done against -F·Δr is negative.
    pos = np.array([[1.0, 0, 0]])
    ref = np.array([[0.0, 0, 0]])
    forces = np.array([[2.0, 0, 0]])
    assert potential_energy_from_forces(pos, forces, ref) == pytest.approx(-2.0)


def test_virial_tensor_shape_and_value():
    pos = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    forces = np.array([[0.0, 1.0, 0], [3.0, 0, 0]])
    W = virial_tensor(pos, forces)
    assert W.shape == (3, 3)
    # W = -Σ r ⊗ F ; W[0,1] = -(1*1 + 0*0) = -1
    assert W[0, 1] == pytest.approx(-1.0)


def test_compute_energetics_without_forces():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(10, 3))
    vel = rng.normal(size=(10, 3))
    masses = np.ones(10)
    e = compute_energetics(pos, vel, None, masses)
    assert e["U"] is None
    assert e["E_total"] is None
    assert e["K_trans"] >= 0


def test_compute_energetics_with_forces():
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(12, 3))
    vel = rng.normal(size=(12, 3))
    forces = rng.normal(size=(12, 3))
    masses = np.ones(12)
    e = compute_energetics(pos, vel, forces, masses, reference_positions=pos)
    assert e["E_total"] == pytest.approx(e["K_total"] + e["U"])
    assert e["virial"].shape == (3, 3)


def test_energetics_trajectory_arrays():
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(4, 15, 3))
    vel = rng.normal(size=(4, 15, 3))
    masses = np.ones(15)
    out = compute_energetics_trajectory(pos, vel, masses, verbose=False)
    assert out["K_trans"].shape == (4,)
    assert out["T"].shape == (4,)
    assert "E_total" not in out  # no forces/reference provided


def test_energetics_trajectory_with_forces_and_verbose():
    rng = np.random.default_rng(6)
    pos = rng.normal(size=(5, 12, 3))
    vel = rng.normal(size=(5, 12, 3))
    forces = rng.normal(size=(5, 12, 3))
    masses = np.ones(12)
    # verbose=True also exercises the summary/print branch.
    out = compute_energetics_trajectory(
        pos, vel, masses, forces=forces, reference_positions=pos[0], verbose=True
    )
    assert out["E_total"].shape == (5,)
    assert out["U"].shape == (5,)
    assert out["E_total"] == pytest.approx(out["K_total"] + out["U"])
