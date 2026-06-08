"""Tests for observables.unified — the L/tau/omega/dLdt stage of `extract`.

This module previously crashed (`decompose_vector_parallel() got an unexpected
keyword argument 'name'`) because it called the low-level decomposition helper
instead of the `create_vector_observable` factory. These tests lock in the
fixed behavior so the regression can't return.
"""

from __future__ import annotations

import numpy as np
import pytest

import rotmd.core.kernels as K
from rotmd.core.inertia import principal_axes
from rotmd.core.vector_observables import VectorObservable, create_vector_observable
from rotmd.observables.unified import (
    compute_all_observables,
    compute_angular_momentum,
    compute_angular_velocity_from_inertia,
    compute_time_derivative,
    compute_torque,
)
from _helpers import random_trajectory


def _system(seed=0, n_frames=8, n_atoms=24):
    sysd = random_trajectory(n_frames=n_frames, n_atoms=n_atoms, seed=seed)
    pos, m = sysd["positions"], sysd["masses"]
    com = K.compute_com_batch(pos, m)
    I = K.inertia_tensor_batch(pos, m, com)
    axes = np.zeros((n_frames, 3, 3))
    for f in range(n_frames):
        _, axes[f] = principal_axes(I[f])
    sysd["inertia"] = I
    sysd["axes"] = axes
    sysd["normal"] = np.array([0.0, 0.0, 1.0])
    return sysd


def test_compute_angular_momentum_matches_definition():
    s = _system(seed=1)
    obs = compute_angular_momentum(
        s["positions"], s["velocities"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    assert isinstance(obs, VectorObservable)
    # Reference L = Σ m (r-COM) × v
    com = K.compute_com_batch(s["positions"], s["masses"])
    r = s["positions"] - com[:, None, :]
    L_ref = np.einsum("a,faj->fj", s["masses"], np.cross(r, s["velocities"]))
    assert obs.vector == pytest.approx(L_ref, abs=1e-8)
    assert obs.parallel + obs.perp == pytest.approx(obs.vector, abs=1e-8)


def test_compute_torque_uses_unit_weights():
    s = _system(seed=2)
    obs = compute_torque(
        s["positions"], s["forces"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    com = K.compute_com_batch(s["positions"], s["masses"])
    r = s["positions"] - com[:, None, :]
    # Torque does NOT mass-weight: τ = Σ (r-COM) × F
    tau_ref = np.einsum("faj->fj", np.cross(r, s["forces"]))
    assert obs.vector == pytest.approx(tau_ref, abs=1e-8)


def test_angular_velocity_solves_inertia_system():
    s = _system(seed=3)
    L = compute_angular_momentum(
        s["positions"], s["velocities"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    omega = compute_angular_velocity_from_inertia(
        L, s["inertia"], s["axes"], s["normal"], s["times"]
    )
    # I @ omega should reproduce L frame-by-frame.
    for f in range(s["n_frames"]):
        assert s["inertia"][f] @ omega.vector[f] == pytest.approx(L.vector[f], abs=1e-7)


def test_time_derivative_without_axis_has_zero_components():
    s = _system(seed=4)
    L = compute_angular_momentum(
        s["positions"], s["velocities"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    dLdt = compute_time_derivative(L, s["times"])
    assert dLdt.vector == pytest.approx(np.gradient(L.vector, s["times"], axis=0, edge_order=2))
    # No reference axis -> component fields are placeholders (zero).
    assert dLdt.parallel_mag == pytest.approx(np.zeros(s["n_frames"]))


def test_time_derivative_with_axis_is_decomposed():
    s = _system(seed=5)
    L = compute_angular_momentum(
        s["positions"], s["velocities"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    dLdt = compute_time_derivative(
        L, s["times"], reference_axis=s["axes"][:, :, 0], membrane_normal=s["normal"]
    )
    assert dLdt.parallel + dLdt.perp == pytest.approx(dLdt.vector, abs=1e-8)
    assert np.any(dLdt.parallel_mag > 0)


def test_angular_velocity_singular_inertia_uses_lstsq():
    s = _system(seed=11, n_frames=3)
    L = compute_angular_momentum(
        s["positions"], s["velocities"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    # Force singular inertia tensors so the np.linalg.solve path raises and the
    # least-squares fallback is exercised.
    singular = np.zeros((s["n_frames"], 3, 3))
    omega = compute_angular_velocity_from_inertia(
        L, singular, s["axes"], s["normal"], s["times"]
    )
    assert omega.vector.shape == (s["n_frames"], 3)
    assert np.isfinite(omega.vector).all()


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_time_derivative_methods_recover_linear_slope(method):
    # For a linear-in-time vector field, every finite-difference scheme must
    # recover the exact slope at all frames with no NaN/inf (regression for the
    # forward/backward broadcast + division-by-zero bug).
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    slope = np.array([2.0, -3.0, 0.5])
    vec = times[:, None] * slope[None, :]
    obs = create_vector_observable(vec, np.array([1.0, 0, 0]), times=times, name="L")
    dOdt = compute_time_derivative(obs, times, method=method)
    assert np.isfinite(dOdt.vector).all()
    assert dOdt.vector == pytest.approx(np.tile(slope, (5, 1)), abs=1e-9)


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_time_derivative_onesided_nonuniform_spacing(method):
    # Non-uniform time steps must not reintroduce a zero denominator.
    times = np.array([0.0, 0.5, 2.0, 5.0])
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(4, 3))
    obs = create_vector_observable(vec, np.array([1.0, 0, 0]), times=times, name="L")
    dOdt = compute_time_derivative(obs, times, method=method)
    assert dOdt.vector.shape == (4, 3)
    assert np.isfinite(dOdt.vector).all()
    # Interior slopes equal the consecutive one-sided differences.
    expected_mid = (vec[2] - vec[1]) / (times[2] - times[1])
    idx = 1 if method == "forward" else 2
    assert dOdt.vector[idx] == pytest.approx(expected_mid, abs=1e-9)


def test_compute_time_derivative_rejects_unknown_method():
    s = _system(seed=6, n_frames=4)
    L = compute_angular_momentum(
        s["positions"], s["velocities"], s["masses"], s["axes"], s["normal"], s["times"]
    )
    with pytest.raises(ValueError):
        compute_time_derivative(L, s["times"], method="quintic")


def test_compute_all_observables_keys_and_shapes():
    s = _system(seed=7, n_frames=10, n_atoms=30)
    obs = compute_all_observables(
        s["positions"], s["velocities"], s["forces"], s["masses"],
        s["inertia"], s["axes"], s["normal"], s["times"], verbose=False,
    )
    assert set(obs) == {"L", "tau", "omega", "dLdt"}
    for name, o in obs.items():
        assert o.vector.shape == (10, 3)
        assert o.magnitude.shape == (10,)
        # Every observable carries a valid parallel/perp decomposition.
        assert o.parallel + o.perp == pytest.approx(o.vector, abs=1e-8)
    # dLdt is genuinely decomposed (regression for the placeholder-zeros bug).
    assert np.any(obs["dLdt"].parallel_mag > 0)


def test_compute_all_observables_verbose_runs():
    s = _system(seed=8, n_frames=6, n_atoms=20)
    # verbose=True exercises the per-observable summary + Euler-validation prints.
    obs = compute_all_observables(
        s["positions"], s["velocities"], s["forces"], s["masses"],
        s["inertia"], s["axes"], s["normal"], s["times"], verbose=True,
    )
    assert set(obs) == {"L", "tau", "omega", "dLdt"}
