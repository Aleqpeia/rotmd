"""Tests for the optional-numba computational kernels.

The same code path runs whether numba JITs it or the no-op fallback leaves it as
plain Python, so these tests pin down the *numerics* against independent NumPy
references. A subprocess test additionally forces ``ROTMD_NUMBA=0`` to prove the
fallback produces identical results to whatever backend is active by default.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import rotmd.core.kernels as K
from _helpers import random_trajectory, reference_inertia_tensor


def test_com_batch_matches_weighted_average():
    sysd = random_trajectory(n_frames=6, n_atoms=20, seed=1)
    pos, m = sysd["positions"], sysd["masses"]
    got = K.compute_com_batch(pos, m)
    expected = np.einsum("a,fai->fi", m, pos) / m.sum()
    assert got.shape == (6, 3)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_com_single_matches_reference():
    sysd = random_trajectory(n_frames=1, n_atoms=15, seed=2)
    pos, m = sysd["positions"][0], sysd["masses"]
    got = K.compute_com_single(pos, m)
    assert got == pytest.approx(np.average(pos, weights=m, axis=0))


def test_inertia_tensor_batch_matches_reference_and_is_symmetric():
    sysd = random_trajectory(n_frames=5, n_atoms=30, seed=4)
    pos, m = sysd["positions"], sysd["masses"]
    com = K.compute_com_batch(pos, m)
    I = K.inertia_tensor_batch(pos, m, com)
    assert I.shape == (5, 3, 3)
    for f in range(5):
        ref = reference_inertia_tensor(pos[f], m)
        assert I[f] == pytest.approx(ref, rel=1e-8, abs=1e-8)
        assert I[f] == pytest.approx(I[f].T)  # symmetric


def test_cross_product_trajectory_matches_numpy_cross():
    sysd = random_trajectory(n_frames=7, n_atoms=18, seed=5)
    pos, vel, m = sysd["positions"], sysd["velocities"], sysd["masses"]
    com = K.compute_com_batch(pos, m)
    got = K.cross_product_trajectory(pos, vel, m, com)
    # Reference: Σ_i m_i (r_i - COM) × v_i
    r = pos - com[:, None, :]
    expected = np.einsum("a,faj->fj", m, np.cross(r, vel))
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_magnitude_batch_2d_and_3d():
    rng = np.random.default_rng(6)
    v2 = rng.normal(size=(10, 3))
    assert K.compute_magnitude_batch(v2) == pytest.approx(np.linalg.norm(v2, axis=1))
    v3 = rng.normal(size=(10, 4, 3))
    assert K.compute_magnitude_batch(v3) == pytest.approx(np.linalg.norm(v3, axis=2))


def test_decompose_static_ref_orthogonality():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(50, 3))
    ref = np.array([0.0, 0.0, 1.0])
    par, perp = K.decompose_vector_batch_static_ref(vectors, ref)
    # Parallel + perpendicular reconstructs the original vector.
    assert par + perp == pytest.approx(vectors)
    # Perp has no z-component; parallel is purely along z.
    assert perp[:, 2] == pytest.approx(np.zeros(50), abs=1e-9)
    assert par[:, :2] == pytest.approx(np.zeros((50, 2)), abs=1e-9)


def test_decompose_batch_timevarying_reference():
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(40, 3))
    refs = rng.normal(size=(40, 3))
    par, perp = K.decompose_vector_batch(vectors, refs)
    assert par + perp == pytest.approx(vectors, abs=1e-9)
    # Each perpendicular component is orthogonal to its reference axis.
    dots = np.einsum("fi,fi->f", perp, refs)
    assert dots == pytest.approx(np.zeros(40), abs=1e-7)


def test_decompose_single_orthogonality():
    par, perp = K.decompose_vector_single(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 5.0])
    )
    # The kernel regularizes the reference norm with +1e-10, so allow a small
    # absolute tolerance rather than the default near-exact one.
    assert par == pytest.approx([0.0, 0.0, 3.0], abs=1e-8)
    assert perp == pytest.approx([1.0, 2.0, 0.0], abs=1e-8)


def test_solve_linear_batch():
    rng = np.random.default_rng(9)
    n = 6
    A = rng.normal(size=(n, 3, 3)) + 3 * np.eye(3)  # diagonally dominant
    b = rng.normal(size=(n, 3))
    x = K.solve_linear_batch(A, b)
    for i in range(n):
        assert A[i] @ x[i] == pytest.approx(b[i], abs=1e-8)


def test_time_derivative_central_and_edges():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    data = np.stack([times**2, 2 * times, np.ones_like(times)], axis=1)
    deriv = K.time_derivative_batch(data, times)
    # Interior point of x²: central difference at t=1 -> (4-0)/2 = 2
    assert deriv[1, 0] == pytest.approx(2.0)
    # Linear column 2t: derivative is exactly 2 everywhere, edges included.
    assert deriv[:, 1] == pytest.approx(np.full(4, 2.0))


def test_principal_axes_batch_eigenvalues_sorted():
    sysd = random_trajectory(n_frames=4, n_atoms=40, seed=10)
    pos, m = sysd["positions"], sysd["masses"]
    com = K.compute_com_batch(pos, m)
    I = K.inertia_tensor_batch(pos, m, com)
    moments, axes = K.principal_axes_batch(I)
    assert moments.shape == (4, 3)
    assert axes.shape == (4, 3, 3)
    for f in range(4):
        assert np.all(np.diff(moments[f]) >= -1e-9)  # ascending


def test_process_in_chunks_matches_full():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(25, 3))

    def double(x):
        return 2 * x

    out = K.process_in_chunks(double, data, chunk_size=7)
    assert out == pytest.approx(2 * data)


def test_fallback_matches_active_backend_in_subprocess():
    """Force the pure-NumPy fallback (ROTMD_NUMBA=0) and compare to default."""
    snippet = (
        "import numpy as np, rotmd.core.kernels as K;"
        "rng=np.random.default_rng(123);"
        "pos=rng.normal(size=(5,16,3));vel=rng.normal(size=(5,16,3));"
        "m=np.abs(rng.normal(size=16))+1;com=K.compute_com_batch(pos,m);"
        "I=K.inertia_tensor_batch(pos,m,com);"
        "L=K.cross_product_trajectory(pos,vel,m,com);"
        "print(K.NUMBA_AVAILABLE, float(I.sum()), float(L.sum()), float(com.sum()))"
    )
    env = dict(os.environ)

    def run(flag):
        e = dict(env)
        e["ROTMD_NUMBA"] = flag
        out = subprocess.check_output([sys.executable, "-c", snippet], env=e)
        parts = out.decode().strip().splitlines()[-1].split()
        return parts[0], np.array([float(x) for x in parts[1:]])

    avail_off, vals_off = run("0")
    avail_on, vals_on = run("1")
    # The fallback must really be disabling numba.
    assert avail_off == "False"
    # Numba parallel reductions may reorder float sums, so compare with a
    # tolerance rather than bit-for-bit.
    assert vals_off == pytest.approx(vals_on, rel=1e-9, abs=1e-9)


def test_jit_fallback_decorator_is_transparent():
    """When numba is absent, ``jit`` must return the function unchanged."""
    if K.NUMBA_AVAILABLE:
        pytest.skip("numba active; fallback decorator not in use")

    @K.jit(nopython=True)
    def f(x):
        return x + 1

    assert f(41) == 42
    assert K.prange is range
