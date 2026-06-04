"""Tests for core.vector_observables: decomposition, magnitudes, factory."""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.core.vector_observables import (
    VectorObservable,
    compute_cross_product_trajectory,
    compute_magnitudes,
    compute_spin_nutation_ratio,
    create_vector_observable,
    decompose_vector_parallel,
)


def test_decompose_static_reference_reconstructs():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(30, 3))
    ref = np.array([0.0, 0.0, 1.0])
    par, perp = decompose_vector_parallel(vectors, ref)
    assert par + perp == pytest.approx(vectors, abs=1e-9)
    assert perp[:, 2] == pytest.approx(np.zeros(30), abs=1e-8)


def test_decompose_timevarying_reference_dispatch():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(20, 3))
    refs = rng.normal(size=(20, 3))  # ndim == 2 -> batch path
    par, perp = decompose_vector_parallel(vectors, refs)
    assert par + perp == pytest.approx(vectors, abs=1e-9)


def test_compute_magnitudes():
    v = np.array([[3.0, 4.0, 0.0], [5.0, 12.0, 0.0]])
    assert compute_magnitudes(v) == pytest.approx([5.0, 13.0])


def test_cross_product_trajectory_reference():
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(6, 12, 3))
    vel = rng.normal(size=(6, 12, 3))
    masses = np.abs(rng.normal(size=12)) + 1
    com = np.einsum("a,fai->fi", masses, pos) / masses.sum()
    got = compute_cross_product_trajectory(pos, vel, masses, com)
    r = pos - com[:, None, :]
    expected = np.einsum("a,faj->fj", masses, np.cross(r, vel))
    assert got == pytest.approx(expected, abs=1e-9)


def test_create_vector_observable_fields_and_invariants():
    rng = np.random.default_rng(3)
    vec = rng.normal(size=(15, 3))
    axis = np.array([1.0, 0.0, 0.0])
    obs = create_vector_observable(vec, axis, name="L")
    assert isinstance(obs, VectorObservable)
    assert obs.name == "L"
    # parallel + perp reconstructs the vector
    assert obs.parallel + obs.perp == pytest.approx(vec, abs=1e-8)
    # magnitudes are consistent with the stored vectors
    assert obs.magnitude == pytest.approx(np.linalg.norm(vec, axis=1))
    assert obs.parallel_mag == pytest.approx(np.linalg.norm(obs.parallel, axis=1))
    # parallel component lies along x (axis); its y,z are ~0
    assert obs.parallel[:, 1:] == pytest.approx(np.zeros((15, 2)), abs=1e-8)


def test_create_vector_observable_default_membrane_normal_is_z():
    vec = np.array([[1.0, 2.0, 3.0]])
    obs = create_vector_observable(vec, np.array([1.0, 0, 0]))
    # z_component is the projection onto default membrane normal (z-axis).
    assert obs.z_component == pytest.approx([[0.0, 0.0, 3.0]], abs=1e-8)
    assert obs.z_mag == pytest.approx([3.0], abs=1e-8)


def test_spin_nutation_ratio():
    obs = create_vector_observable(
        np.array([[3.0, 4.0, 0.0]]), np.array([1.0, 0.0, 0.0])
    )
    # parallel mag = 3 (x), perp mag = 4 (y) -> ratio 0.75
    ratio = compute_spin_nutation_ratio(obs)
    assert ratio == pytest.approx([0.75], rel=1e-6)


def test_vector_observable_to_xarray_with_and_without_times():
    xr = pytest.importorskip("xarray")
    vec = np.random.default_rng(0).normal(size=(8, 3))
    obs = create_vector_observable(
        vec, np.array([1.0, 0, 0]), times=np.arange(8.0), name="L"
    )
    ds = obs.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert "L" in ds and "L_mag" in ds
    assert "time" in ds.coords
    # Without times, frames are used as the coordinate instead.
    obs2 = create_vector_observable(vec, np.array([1.0, 0, 0]), name="L")
    ds2 = obs2.to_xarray()
    assert "frame" in ds2.coords


def test_vector_observable_to_dict_and_stats():
    vec = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    obs = create_vector_observable(vec, np.array([1.0, 0, 0]), name="tau")
    d = obs.to_dict()
    assert set(d) >= {"tau", "tau_parallel", "tau_perp", "tau_mag"}
    assert obs.mean() == pytest.approx(2.5)  # mean of [5, 0]
    assert obs.std() == pytest.approx(2.5)
