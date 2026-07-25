"""PMF Jacobian correctness and unit-convention regression test.

Angles here are radians throughout, matching ``rotmd.core.orientation``'s ZYZ
output. ``pmf.py`` used to mix radians (``compute_pmf_2d``) and degrees
(``compute_pmf_1d``/``compute_pmf_6d_projection``) internally, which silently
produced nonsense from the very same array depending which function you
called — these tests pin both the physical correctness of the Jacobian and
that every function bins the same array the same way.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.analysis.pmf import compute_pmf_1d, compute_pmf_2d, compute_pmf_6d_projection
from rotmd.core.orientation import membrane_tilt_angle


def _isotropic_theta(n: int, seed: int = 0) -> np.ndarray:
    """theta for points uniform on the sphere: theta = arccos(U(-1, 1))."""
    rng = np.random.default_rng(seed)
    return np.arccos(rng.uniform(-1.0, 1.0, size=n))


def test_compute_pmf_1d_theta_is_flat_for_isotropic_sampling():
    """Isotropic sampling is exactly what sin(theta) corrects for.

    A biased Jacobian (or one applied in the wrong units) makes low-theta
    bins look artificially favourable; the correct one flattens the PMF to
    within sampling noise.
    """
    theta = _isotropic_theta(200_000)
    result = compute_pmf_1d(theta, bins=20, coordinate_type="theta")
    populated = result["pmf"][~np.isnan(result["pmf"])]
    assert populated.std() < 0.1  # kcal/mol; empirically ~0.01 at this sample size


def test_compute_pmf_2d_and_1d_bin_theta_over_the_same_range():
    """Both must bin theta over [0, pi] radians — not one in degrees."""
    theta = _isotropic_theta(50_000)
    psi = np.random.default_rng(1).uniform(0, 2 * np.pi, size=50_000)

    result_2d = compute_pmf_2d(theta, psi, theta_bins=10, psi_bins=12)
    result_1d = compute_pmf_1d(theta, bins=10, coordinate_type="theta")

    np.testing.assert_allclose(result_2d["theta_edges"], result_1d["edges"])
    assert result_1d["edges"][-1] == pytest.approx(np.pi)


def test_compute_pmf_1d_tilt_is_flat_for_isotropic_sampling():
    """tilt folds theta and pi-theta together (they're the same physical
    orientation — the principal axis is a headless line), and sin(tilt) is
    the measure for *that* folded coordinate — same functional form as
    sin(theta), restricted to the acute domain. Isotropic sampling should
    come out flat under it too, exactly as for theta.
    """
    theta = _isotropic_theta(200_000)
    tilt = membrane_tilt_angle(theta)
    result = compute_pmf_1d(tilt, bins=20, coordinate_type="tilt")
    populated = result["pmf"][~np.isnan(result["pmf"])]
    assert populated.std() < 0.1


def test_compute_pmf_2d_tilt_matches_1d_range_and_is_half_theta():
    theta = _isotropic_theta(50_000)
    tilt = membrane_tilt_angle(theta)
    psi = np.random.default_rng(1).uniform(0, 2 * np.pi, size=50_000)

    result_2d = compute_pmf_2d(tilt, psi, theta_bins=10, psi_bins=12, angle_kind="tilt")
    result_1d = compute_pmf_1d(tilt, bins=10, coordinate_type="tilt")

    np.testing.assert_allclose(result_2d["theta_edges"], result_1d["edges"])
    assert result_1d["edges"][-1] == pytest.approx(np.pi / 2)


def test_compute_pmf_6d_projection_theta_edges_are_radians():
    theta = _isotropic_theta(20_000)
    psi = np.random.default_rng(2).uniform(0, 2 * np.pi, size=20_000)
    omega = np.random.default_rng(3).normal(scale=0.1, size=(20_000, 3))

    proj = compute_pmf_6d_projection(
        theta, psi, omega, theta_bins=8, psi_bins=9, omega_bins=5
    )
    theta_edges = proj["pmf_2d_theta_omega"]["theta_edges"]
    assert theta_edges[0] == pytest.approx(0.0)
    assert theta_edges[-1] == pytest.approx(np.pi)
