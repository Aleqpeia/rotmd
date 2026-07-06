"""Tests for core.vector_observables: decomposition, magnitudes, factory."""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.core.vector_observables import (
    VectorObservable,
    compute_cross_product_trajectory,
    compute_magnitudes,
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
