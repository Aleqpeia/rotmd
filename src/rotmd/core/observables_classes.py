"""
Functional Class-Based Observable System

Elegant, composable classes for physical observables following functional principles:
- Immutability
- Lazy evaluation
- Method chaining
- Automatic decomposition

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from functools import cached_property
import xarray as xr

from rotmd.core.functional import Lazy, Pipeline, Maybe
from rotmd.core.vector_observables import (
    decompose_vector_parallel,
    compute_magnitudes,
    compute_cross_product_trajectory
)


# =============================================================================
# Base Observable Class
# =============================================================================

@dataclass(frozen=True)  # Immutable
class Observable:
    """
    Base class for all observables.

    Immutable and uses lazy evaluation for expensive computations.
    """
    times: np.ndarray
    _data: np.ndarray = field(repr=False)
    name: str = "observable"
    units: str = ""

    @cached_property
    def data(self) -> np.ndarray:
        """Get observable data (cached)."""
        return self._data

    @cached_property
    def mean(self) -> float:
        """Mean value."""
        return float(np.mean(self.data))

    @cached_property
    def std(self) -> float:
        """Standard deviation."""
        return float(np.std(self.data))

    @cached_property
    def min(self) -> float:
        """Minimum value."""
        return float(np.min(self.data))

    @cached_property
    def max(self) -> float:
        """Maximum value."""
        return float(np.max(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> 'Observable':
        """Transform data."""
        return Observable(
            times=self.times,
            _data=func(self.data),
            name=f"{func.__name__}({self.name})",
            units=self.units
        )

    def to_xarray(self) -> xr.DataArray:
        """Convert to xarray."""
        return xr.DataArray(
            self.data,
            coords={'time': self.times},
            dims=['time'],
            name=self.name,
            attrs={'units': self.units}
        )


# =============================================================================
# Vector Observable (3D vectors with decomposition)
# =============================================================================

@dataclass(frozen=True)
class VectorObservable(Observable):
    """
    3D vector observable with automatic decomposition.

    Immutable, uses lazy evaluation, supports functional composition.

    Example:
        >>> L = AngularMomentum.from_trajectory(positions, velocities, masses, axes, normal)
        >>> print(L.magnitude.mean)  # Lazy computed
        >>> print(L.parallel.magnitude.mean)  # Lazy decomposition
        >>> L.to_xarray().to_netcdf('L.nc')
    """
    _data: np.ndarray = field(repr=False)  # (n_frames, 3)
    reference_axis: np.ndarray = field(repr=False)  # (n_frames, 3) or (3,)
    membrane_normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]), repr=False)

    @cached_property
    def data(self) -> np.ndarray:
        """Full 3D vector field."""
        return self._data

    @cached_property
    def magnitude(self) -> Observable:
        """Magnitude |v| as scalar observable."""
        mag = compute_magnitudes(self.data)
        return Observable(
            times=self.times,
            _data=mag,
            name=f"|{self.name}|",
            units=self.units
        )

    @cached_property
    def parallel(self) -> 'VectorObservable':
        """Component parallel to reference axis (spin)."""
        v_par, _ = decompose_vector_parallel(self.data, self.reference_axis)
        return VectorObservable(
            times=self.times,
            _data=v_par,
            name=f"{self.name}_∥",
            units=self.units,
            reference_axis=self.reference_axis,
            membrane_normal=self.membrane_normal
        )

    @cached_property
    def perp(self) -> 'VectorObservable':
        """Component perpendicular to reference axis (nutation)."""
        _, v_perp = decompose_vector_parallel(self.data, self.reference_axis)
        return VectorObservable(
            times=self.times,
            _data=v_perp,
            name=f"{self.name}_⊥",
            units=self.units,
            reference_axis=self.reference_axis,
            membrane_normal=self.membrane_normal
        )

    @cached_property
    def z_component(self) -> 'VectorObservable':
        """Component along membrane normal (z-axis)."""
        normal = self.membrane_normal if self.membrane_normal.ndim == 1 else self.membrane_normal[0]
        normal_traj = np.broadcast_to(normal, self.data.shape).copy()
        v_z, _ = decompose_vector_parallel(self.data, normal_traj)
        return VectorObservable(
            times=self.times,
            _data=v_z,
            name=f"{self.name}_z",
            units=self.units,
            reference_axis=self.reference_axis,
            membrane_normal=self.membrane_normal
        )

    @cached_property
    def spin_nutation_ratio(self) -> Observable:
        """Ratio of spin to nutation magnitude."""
        ratio = self.parallel.magnitude.data / (self.perp.magnitude.data + 1e-10)
        return Observable(
            times=self.times,
            _data=ratio,
            name=f"{self.name}_spin/nutation",
            units=""
        )

    def derivative(self, method: str = 'central') -> 'VectorObservable':
        """Time derivative d/dt."""
        dt = np.gradient(self.data, self.times, axis=0, edge_order=2)
        return VectorObservable(
            times=self.times,
            _data=dt,
            name=f"d{self.name}/dt",
            units=f"{self.units}/ps",
            reference_axis=self.reference_axis,
            membrane_normal=self.membrane_normal
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to flat dictionary (for saving)."""
        return {
            f'{self.name}': self.data,
            f'{self.name}_mag': self.magnitude.data,
            f'{self.name}_parallel': self.parallel.data,
            f'{self.name}_parallel_mag': self.parallel.magnitude.data,
            f'{self.name}_perp': self.perp.data,
            f'{self.name}_perp_mag': self.perp.magnitude.data,
            f'{self.name}_z': self.z_component.data,
            f'{self.name}_z_mag': self.z_component.magnitude.data,
        }

    def to_xarray(self) -> xr.Dataset:
        """Convert to xarray Dataset with all components."""
        coords = {'time': self.times, 'component': ['x', 'y', 'z']}

        return xr.Dataset({
            self.name: (['time', 'component'], self.data),
            f'{self.name}_mag': (['time'], self.magnitude.data),
            f'{self.name}_parallel_mag': (['time'], self.parallel.magnitude.data),
            f'{self.name}_perp_mag': (['time'], self.perp.magnitude.data),
        }, coords=coords, attrs={'units': self.units})


# =============================================================================
# Specialized Observable Classes
# =============================================================================

@dataclass(frozen=True)
class AngularMomentum(VectorObservable):
    """
    Angular momentum L = Σ m (r - COM) × v

    Example:
        >>> L = AngularMomentum.from_trajectory(
        ...     positions, velocities, masses, principal_axes, normal, times
        ... )
        >>> print(f"Mean |L|: {L.magnitude.mean:.3f}")
        >>> print(f"Spin/Nutation: {L.spin_nutation_ratio.mean:.3f}")
    """
    name: str = field(default="L", init=False)
    units: str = field(default="amu·Ų/ps", init=False)

    @classmethod
    def from_trajectory(
        cls,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        principal_axes: np.ndarray,
        membrane_normal: np.ndarray,
        times: np.ndarray
    ) -> 'AngularMomentum':
        """Compute from trajectory data."""
        from rotmd.core.inertia import compute_center_of_mass

        # Compute COM
        n_frames = len(times)
        com = np.array([compute_center_of_mass(positions[i], masses) for i in range(n_frames)])

        # Compute L using numba kernel
        L_data = compute_cross_product_trajectory(positions, velocities, masses, com)

        # Extract principal axis (longest = index 0)
        ref_axis = principal_axes[:, :, 0]

        return cls(
            times=times,
            _data=L_data,
            reference_axis=ref_axis,
            membrane_normal=membrane_normal
        )


@dataclass(frozen=True)
class Torque(VectorObservable):
    """
    Torque τ = Σ (r - COM) × F

    Example:
        >>> tau = Torque.from_trajectory(
        ...     positions, forces, masses, principal_axes, normal, times
        ... )
        >>> # Validate Euler's equation
        >>> dLdt = L.derivative()
        >>> error = np.mean(np.abs(tau.magnitude.data - dLdt.magnitude.data))
    """
    name: str = field(default="τ", init=False)
    units: str = field(default="amu·Ų/ps²", init=False)

    @classmethod
    def from_trajectory(
        cls,
        positions: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        principal_axes: np.ndarray,
        membrane_normal: np.ndarray,
        times: np.ndarray
    ) -> 'Torque':
        """Compute from trajectory data."""
        from rotmd.core.inertia import compute_center_of_mass

        # Compute COM
        n_frames = len(times)
        com = np.array([compute_center_of_mass(positions[i], masses) for i in range(n_frames)])

        # Compute τ (no mass weighting for torque)
        masses_ones = np.ones_like(masses)
        tau_data = compute_cross_product_trajectory(positions, forces, masses_ones, com)

        # Extract principal axis
        ref_axis = principal_axes[:, :, 0]

        return cls(
            times=times,
            _data=tau_data,
            reference_axis=ref_axis,
            membrane_normal=membrane_normal
        )


@dataclass(frozen=True)
class AngularVelocity(VectorObservable):
    """
    Angular velocity ω (from L = I·ω or from rotation matrices)

    Example:
        >>> omega = AngularVelocity.from_angular_momentum(L, inertia_tensors)
        >>> print(f"Mean |ω|: {omega.magnitude.mean:.3f} rad/ps")
    """
    name: str = field(default="ω", init=False)
    units: str = field(default="rad/ps", init=False)

    @classmethod
    def from_angular_momentum(
        cls,
        L: AngularMomentum,
        inertia_tensors: np.ndarray
    ) -> 'AngularVelocity':
        """Compute from L = I·ω."""
        n_frames = len(L)
        omega_data = np.zeros((n_frames, 3))

        # Solve ω = I^{-1}·L
        for i in range(n_frames):
            try:
                omega_data[i] = np.linalg.solve(inertia_tensors[i], L.data[i])
            except np.linalg.LinAlgError:
                omega_data[i] = np.linalg.lstsq(inertia_tensors[i], L.data[i], rcond=None)[0]

        return cls(
            times=L.times,
            _data=omega_data,
            reference_axis=L.reference_axis,
            membrane_normal=L.membrane_normal
        )


# =============================================================================
# Physics Validation
# =============================================================================

def validate_eulers_equation(
    L: AngularMomentum,
    tau: Torque,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Validate Euler's equation: dL/dt = τ

    Returns:
        Dictionary with error metrics
    """
    dLdt = L.derivative()

    errors = {
        'mean_absolute_error': float(np.mean(np.abs(tau.data - dLdt.data))),
        'max_absolute_error': float(np.max(np.abs(tau.data - dLdt.data))),
        'relative_error': float(np.mean(np.abs(tau.data - dLdt.data) / (np.abs(tau.data) + 1e-10))),
        'magnitude_correlation': float(np.corrcoef(
            tau.magnitude.data, dLdt.magnitude.data
        )[0, 1])
    }

    if verbose:
        print("Euler's Equation Validation (dL/dt = τ):")
        print(f"  Mean absolute error: {errors['mean_absolute_error']:.3e}")
        print(f"  Max absolute error: {errors['max_absolute_error']:.3e}")
        print(f"  Relative error: {errors['relative_error']*100:.2f}%")
        print(f"  Magnitude correlation: {errors['magnitude_correlation']:.4f}")

    return errors


# =============================================================================
# Factory Function (for convenience)
# =============================================================================

def compute_all_observables_functional(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    inertia_tensors: np.ndarray,
    principal_axes: np.ndarray,
    membrane_normal: np.ndarray,
    times: np.ndarray,
    validate: bool = True
) -> Dict[str, VectorObservable]:
    """
    Compute all observables using functional class-based approach.

    Returns immutable observable objects with lazy decomposition.

    Example:
        >>> obs = compute_all_observables_functional(...)
        >>> print(obs['L'].parallel.magnitude.mean)  # Lazy evaluation
        >>> obs['L'].to_xarray().to_netcdf('L.nc')
    """
    # Compute observables (all immutable)
    L = AngularMomentum.from_trajectory(
        positions, velocities, masses, principal_axes, membrane_normal, times
    )

    tau = Torque.from_trajectory(
        positions, forces, masses, principal_axes, membrane_normal, times
    )

    omega = AngularVelocity.from_angular_momentum(L, inertia_tensors)

    # Validate physics
    if validate:
        validate_eulers_equation(L, tau, verbose=True)

    return {
        'L': L,
        'tau': tau,
        'omega': omega,
        'dLdt': L.derivative()
    }
