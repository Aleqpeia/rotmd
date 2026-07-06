"""
Unified Observable Computation

This module provides high-level functions for computing all vector observables
(L, ω, τ) using the common framework from core.vector_observables.

All functions are optimized with numba and return VectorObservable objects.

Author: Mykyta Bobylyow
Date: 2025
"""

from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

import rotmd.core.kernels as K
from rotmd.core.vector_observables import (
    compute_cross_product_trajectory,
    create_vector_observable,
    VectorObservable
)


# =============================================================================
# Angular Momentum
# =============================================================================

def compute_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    principal_axes: np.ndarray,
    membrane_normal: np.ndarray,
    times: Optional[np.ndarray] = None,
        verbose: bool = False,
        com: Optional[np.ndarray] = None,
) -> VectorObservable:
    """
    Compute angular momentum L = Σ m_i (r_i - COM) × v_i.

    Uses numba-optimized kernels for maximum performance.

    Args:
        positions: (n_frames, n_atoms, 3) positions
        velocities: (n_frames, n_atoms, 3) velocities
        masses: (n_atoms,) atomic masses
        principal_axes: (n_frames, 3, 3) principal axes trajectory
        membrane_normal: (3,) membrane normal vector
        times: (n_frames,) optional time points
        verbose: Show progress bar
        com: (n_frames, 3) precomputed center of mass. When ``None`` it is
            computed here via the batch kernel; the caller (the extract CLI)
            passes the COM already computed by the loader to avoid recomputing.

    Returns:
        VectorObservable containing:
            - L: total angular momentum
            - L_parallel: spin component
            - L_perp: nutation component
            - L_z: membrane-plane component
            - magnitudes of all components

    Example:
        >>> L_obs = compute_angular_momentum(
        ...     positions, velocities, masses, axes, normal, times
        ... )
        >>> print(f"Mean |L|: {L_obs.magnitude.mean()}")
        >>> print(f"Spin/Nutation: {L_obs.parallel_mag.mean() / L_obs.perp_mag.mean()}")
    """
    if verbose:
        print("Computing angular momentum...")

    if com is None:
        com = K.compute_com_batch(positions, masses)

    # Compute L = Σ m (r - COM) × v using numba kernel
    L = compute_cross_product_trajectory(positions, velocities, masses, com)

    # Principal axes are sorted by ascending moment, so column 0 is the
    # smallest-moment axis (the long/spin axis for an elongated body). We
    # decompose L into spin (parallel) and nutation (perpendicular) about it.
    principal_axis = principal_axes[:, :, 0]  # (n_frames, 3)

    # Build the decomposed VectorObservable (parallel/perp + membrane-z).
    return create_vector_observable(
        L, principal_axis, membrane_normal, times, name='L'
    )


# =============================================================================
# Torque
# =============================================================================

def compute_torque(
    positions: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    principal_axes: np.ndarray,
    membrane_normal: np.ndarray,
    times: Optional[np.ndarray] = None,
        verbose: bool = False,
        com: Optional[np.ndarray] = None,
) -> VectorObservable:
    """
    Compute torque τ = Σ (r_i - COM) × F_i.

    Uses numba-optimized kernels for maximum performance.

    Args:
        positions: (n_frames, n_atoms, 3) positions
        forces: (n_frames, n_atoms, 3) forces
        masses: (n_atoms,) atomic masses
        principal_axes: (n_frames, 3, 3) principal axes trajectory
        membrane_normal: (3,) membrane normal vector
        times: (n_frames,) optional time points
        verbose: Show progress bar
        com: (n_frames, 3) precomputed center of mass. When ``None`` it is
            computed here via the batch kernel.

    Returns:
        VectorObservable containing:
            - tau: total torque
            - tau_parallel: spin torque
            - tau_perp: nutation torque
            - tau_z: membrane-plane torque
            - magnitudes of all components

    Example:
        >>> tau_obs = compute_torque(
        ...     positions, forces, masses, axes, normal, times
        ... )
        >>> # Validate Euler's equation: dL/dt = τ
        >>> dLdt = np.gradient(L_obs.vector, times, axis=0)
        >>> correlation = np.corrcoef(tau_obs.vector.flatten(), dLdt.flatten())[0,1]
    """
    if verbose:
        print("Computing torque...")

    if com is None:
        com = K.compute_com_batch(positions, masses)

    # Compute τ = Σ (r - COM) × F using numba kernel
    # Note: Forces don't need mass weighting
    masses_ones = np.ones_like(masses)  # No mass weighting for torque
    tau = compute_cross_product_trajectory(positions, forces, masses_ones, com)

    # Extract principal axis (smallest-moment / long axis)
    principal_axis = principal_axes[:, :, 0]

    # Build the decomposed VectorObservable (parallel/perp + membrane-z).
    return create_vector_observable(
        tau, principal_axis, membrane_normal, times, name='tau'
    )


# =============================================================================
# Angular Velocity
# =============================================================================

def compute_angular_velocity_from_inertia(
    angular_momentum: VectorObservable,
    inertia_tensors: np.ndarray,
    principal_axes: np.ndarray,
    membrane_normal: np.ndarray,
    times: Optional[np.ndarray] = None,
    verbose: bool = False
) -> VectorObservable:
    """
    Compute angular velocity ω from L = I·ω.

    Uses the relation: ω = I^{-1} · L

    Args:
        angular_momentum: VectorObservable for L
        inertia_tensors: (n_frames, 3, 3) inertia tensors
        principal_axes: (n_frames, 3, 3) principal axes
        membrane_normal: (3,) membrane normal
        times: (n_frames,) time points
        verbose: Show progress

    Returns:
        VectorObservable for angular velocity

    Example:
        >>> L_obs = compute_angular_momentum(...)
        >>> omega_obs = compute_angular_velocity_from_inertia(
        ...     L_obs, inertia_tensors, axes, normal
        ... )
    """
    if verbose:
        print("Computing angular velocity from L = I·ω...")

    n_frames = len(angular_momentum.vector)
    omega = np.zeros((n_frames, 3))

    # Solve ω = I^{-1} · L for each frame
    iterator = tqdm(range(n_frames), desc="ω = I⁻¹·L") if verbose else range(n_frames)
    for i in iterator:
        try:
            omega[i] = np.linalg.solve(inertia_tensors[i], angular_momentum.vector[i])
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudoinverse
            omega[i] = np.linalg.lstsq(inertia_tensors[i], angular_momentum.vector[i], rcond=None)[0]

    # Extract principal axis (smallest-moment / long axis)
    principal_axis = principal_axes[:, :, 0]

    # Build the decomposed VectorObservable (parallel/perp + membrane-z).
    return create_vector_observable(
        omega, principal_axis, membrane_normal, times, name='omega'
    )


# =============================================================================
# Time Derivative: dL/dt
# =============================================================================

def compute_time_derivative(
    observable: VectorObservable,
    times: np.ndarray,
    method: str = 'central',
    reference_axis: Optional[np.ndarray] = None,
    membrane_normal: Optional[np.ndarray] = None,
) -> VectorObservable:
    """
    Compute time derivative dO/dt for any vector observable.

    Args:
        observable: VectorObservable to differentiate
        times: (n_frames,) time values
        method: 'central', 'forward', or 'backward'
        reference_axis: (n_frames, 3) or (3,) spin axis to decompose against.
            When provided, dO/dt is split into parallel/perp/z components the
            same way the source observable was. When ``None`` the component
            fields are left as zeros (only ``vector`` and ``magnitude`` are
            meaningful) — this keeps the function usable when no axis is handy.
        membrane_normal: (3,) membrane normal for the z-decomposition.

    Returns:
        VectorObservable for dO/dt

    Example:
        >>> dLdt = compute_time_derivative(L_obs, times, reference_axis=axes[:, :, 0])
        >>> # Check Euler's equation
        >>> error = np.linalg.norm(dLdt.vector - tau_obs.vector, axis=1)
    """
    if method == 'central':
        # Central difference: [f(t+dt) - f(t-dt)] / (2dt)
        dOdt = np.gradient(observable.vector, times, axis=0, edge_order=2)
    elif method in ('forward', 'backward'):
        # One-sided difference between consecutive frames:
        #   slope[i] = (v[i+1] - v[i]) / (t[i+1] - t[i])
        # This yields n-1 values; we duplicate one edge slope to keep length n.
        # The forward variant carries the last interior slope onto the final
        # frame; the backward variant carries the first onto the initial frame.
        # Using consecutive differences (rather than np.diff with prepend/append
        # of an endpoint) avoids a zero time-step — and hence division by zero —
        # at the duplicated edge.
        dt = np.diff(times)
        if np.any(dt == 0):
            raise ValueError("times must be strictly increasing (zero dt)")
        slope = np.diff(observable.vector, axis=0) / dt[:, np.newaxis]
        if method == 'forward':
            dOdt = np.concatenate([slope, slope[-1:]], axis=0)
        else:
            dOdt = np.concatenate([slope[0:1], slope], axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    name = f'd{observable.name}_dt'

    # Decompose against the same reference the source observable used, so the
    # dL/dt_{parallel,perp,z} columns of the chunk schema carry real values
    # instead of placeholder zeros.
    if reference_axis is not None:
        obs = create_vector_observable(
            dOdt, reference_axis, membrane_normal, times, name=name
        )
        return obs

    # No reference axis: only the full vector and its magnitude are defined.
    return VectorObservable(
        vector=dOdt,
        parallel=np.zeros_like(dOdt),
        perp=np.zeros_like(dOdt),
        z_component=np.zeros_like(dOdt),
        magnitude=np.linalg.norm(dOdt, axis=1),
        parallel_mag=np.zeros(len(dOdt)),
        perp_mag=np.zeros(len(dOdt)),
        z_mag=np.zeros(len(dOdt)),
        times=times,
        name=name,
    )


# =============================================================================
# Batch Computation
# =============================================================================

def compute_all_observables(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    inertia_tensors: np.ndarray,
    principal_axes: np.ndarray,
    membrane_normal: np.ndarray,
    times: np.ndarray,
        verbose: bool = True,
        com: Optional[np.ndarray] = None,
) -> Dict[str, VectorObservable]:
    """
    Compute all vector observables in one function call.

    Args:
        positions: (n_frames, n_atoms, 3)
        velocities: (n_frames, n_atoms, 3)
        forces: (n_frames, n_atoms, 3)
        masses: (n_atoms,)
        inertia_tensors: (n_frames, 3, 3)
        principal_axes: (n_frames, 3, 3)
        membrane_normal: (3,)
        times: (n_frames,)
        verbose: Show progress
        com: (n_frames, 3) precomputed center of mass, threaded to the L and τ
            computations so the COM is not recomputed. Computed on demand when
            ``None``.

    Returns:
        Dictionary with keys:
            - 'L': angular momentum
            - 'tau': torque
            - 'omega': angular velocity
            - 'dLdt': time derivative of L

    Example:
        >>> obs = compute_all_observables(
        ...     positions, velocities, forces, masses,
        ...     I_tensors, axes, normal, times
        ... )
        >>> # Validate Euler's equation
        >>> tau_error = np.mean(np.abs(obs['tau'].magnitude - obs['dLdt'].magnitude))
    """
    results = {}

    # Angular momentum
    results['L'] = compute_angular_momentum(
        positions, velocities, masses, principal_axes, membrane_normal, times, verbose, com
    )

    # Torque
    results['tau'] = compute_torque(
        positions, forces, masses, principal_axes, membrane_normal, times, verbose, com
    )

    # Angular velocity
    results['omega'] = compute_angular_velocity_from_inertia(
        results['L'], inertia_tensors, principal_axes, membrane_normal, times, verbose
    )

    # Time derivative of L, decomposed against the same long axis as L so the
    # dLdt_{parallel,perp,z} columns are populated (Euler's eqn: dL/dt ≈ τ).
    results['dLdt'] = compute_time_derivative(
        results['L'], times,
        reference_axis=principal_axes[:, :, 0],
        membrane_normal=membrane_normal,
    )

    if verbose:
        print("\n" + "="*60)
        print("Observable Summary:")
        print("="*60)
        for name, obs in results.items():
            print(f"{name:8s}: mean |{name}| = {obs.magnitude.mean():.3f}")

        # Validate Euler's equation: dL/dt = τ
        tau_validation = np.mean(np.abs(
            results['tau'].magnitude - results['dLdt'].magnitude
        ))
        print(f"\nEuler's equation validation: |τ - dL/dt| = {tau_validation:.3e}")
        print("="*60)

    return results
