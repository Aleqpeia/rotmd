"""Tests for core.inertia: inertia tensor, principal axes, parallel-axis theorem."""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import ellipsoid_positions, reference_inertia_tensor
from rotmd.core.inertia import (
    compute_center_of_mass,
    inertia_tensor,
    principal_axes,
    principal_moments,
)


def test_com_simple():
    pos = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    masses = np.array([1.0, 3.0])
    # Weighted toward the heavier atom at x=2 -> x = (0*1 + 2*3)/4 = 1.5
    assert compute_center_of_mass(pos, masses) == pytest.approx([1.5, 0, 0])


def test_inertia_matches_reference():
    pos = ellipsoid_positions(80, axes=(1.0, 2.0, 3.0), seed=1)
    masses = np.linspace(1, 5, 80)
    I = inertia_tensor(pos, masses)
    assert I == pytest.approx(reference_inertia_tensor(pos, masses), rel=1e-9, abs=1e-9)
    assert I == pytest.approx(I.T)  # symmetric


def test_inertia_diagonal_of_point_on_axis():
    # Single unit mass at (0,0,1): I_xx = I_yy = 1, I_zz = 0.
    pos = np.array([[0.0, 0.0, 1.0]])
    masses = np.array([1.0])
    I = inertia_tensor(pos, masses, center_of_mass=np.zeros(3))
    assert I == pytest.approx(np.diag([1.0, 1.0, 0.0]))


def test_inertia_shape_validation():
    with pytest.raises(ValueError):
        inertia_tensor(np.zeros((5, 3)), np.ones(4))  # mismatched lengths
    with pytest.raises(ValueError):
        inertia_tensor(np.zeros((5, 2)), np.ones(5))  # not 3D


def test_principal_axes_diagonal_input():
    I = np.diag([100.0, 200.0, 300.0])
    moments, axes = principal_axes(I)
    assert moments == pytest.approx([100, 200, 300])
    # Axes form a proper rotation (det = +1) and are orthonormal.
    assert np.linalg.det(axes) == pytest.approx(1.0)
    assert axes.T @ axes == pytest.approx(np.eye(3))


def test_principal_axes_diagonalizes():
    pos = ellipsoid_positions(100, axes=(1.0, 2.0, 4.0), seed=2)
    masses = np.ones(100)
    I = inertia_tensor(pos, masses)
    moments, axes = principal_axes(I)
    # R^T I R should be diagonal with the sorted moments on the diagonal.
    diag = axes.T @ I @ axes
    off = diag - np.diag(np.diag(diag))
    assert off == pytest.approx(np.zeros((3, 3)), abs=1e-7)
    assert np.diag(diag) == pytest.approx(moments, rel=1e-7)


def test_principal_axes_rejects_nonsymmetric():
    with pytest.raises(ValueError):
        principal_axes(np.array([[1.0, 2.0, 0], [0, 1, 0], [0, 0, 1]]))
    with pytest.raises(ValueError):
        principal_axes(np.eye(2))  # wrong shape


def test_principal_moments_convenience():
    pos = ellipsoid_positions(70, axes=(1.0, 1.0, 5.0), seed=5)
    masses = np.ones(70)
    moments, axes = principal_moments(pos, masses)
    I = inertia_tensor(pos, masses)
    m2, _ = principal_axes(I)
    assert moments == pytest.approx(m2)
    # Elongated along z -> smallest moment is about the long axis.
    assert moments[0] < moments[2]
