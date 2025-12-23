#!/usr/bin/env python
"""
Torque Calculation and Analysis

This module computes torques from forces and validates energy models via
the fundamental relation τ = -∇V for conservative forces.

Mathematical Background:
========================

Torque from Forces:
-------------------
For a rigid body, torque τ about center of mass is:

    τ = Σ_α (r_α - r_COM) × F_α

where:
- r_α: position of atom α
- F_α: force on atom α
- r_COM: center of mass position

Connection to Angular Momentum:
-------------------------------
Euler's equation of rotational motion:

    dL/dt = τ

This fundamental relation connects torque to rate of change of angular momentum.

Torque from Energy Gradient (Conservative Forces):
--------------------------------------------------
For a potential V(θ, ψ, r), the torque is:

    τ = -∇V

In body-fixed coordinates:
    τ_θ = -∂V/∂θ
    τ_ψ = -∂V/∂ψ

Validation Test:
----------------
Given energy model E_total(θ, ψ) = α·SASA(θ, ψ) + E_elec(θ, ψ),
we can validate by checking:

    τ_computed (from forces) ≈ -∇E_total (from energy model)

If this holds, the energy model correctly captures the forces.

Decomposition:
--------------
Torque can be decomposed like angular momentum:
    τ_∥ = (τ · p) p    (spin torque, along protein axis)
    τ_⊥ = τ - τ_∥      (nutation torque, perpendicular)
    τ_z = (τ · n) n    (membrane-plane torque)

Physical Interpretation:
------------------------
- τ_∥: Drives spin (rotation around protein axis)
- τ_⊥: Drives nutation (wobbling)
- Large |τ|: Strong forces → fast dynamics
- Small |τ|: Weak forces → slow, constrained dynamics

For well-bound protein:
- Small |τ|: System near equilibrium
- τ ≈ 0: Minimal driving force

For poorly-bound protein:
- Large |τ|: System driven out of equilibrium
- Fluctuating τ: Unstable binding

References:
-----------
- Goldstein, Poole, Safko (2002). Classical Mechanics (3rd ed.), Chapter 5.
- Landau & Lifshitz (1976). Mechanics (3rd ed.), §9.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings

def torque_field(theta, psi):
    """
    Analytical torque for transmembrane protein.
    
    Physical model:
    - Hydrophobic mismatch drives τ_θ
    - Electrostatic drives τ_ψ
    """
    # Hydrophobic torque (restores transmembrane orientation)
    theta_eq = np.pi/2  # Transmembrane = 90°
    k_hydro = 2.0  # kT
    tau_theta = -k_hydro * np.sin(2 * (theta - theta_eq))
    
    # Electrostatic torque (ψ-dependent for charged proteins)
    # Example: dipole in field
    E_field = 0.5  # kT
    tau_psi = -E_field * np.sin(psi)
    
    return tau_theta, tau_psi

from typing import Callable, Dict
import numpy as np
def torque_field_from_pmf(
    pmf_data: Dict[str, np.ndarray]
) -> Callable[[float, float], tuple]:
    """
    Returns a function that estimates the torque field (τ_θ, τ_ψ) from a PMF.
    Args:
        pmf_data: Dict with keys 'theta_centers', 'psi_centers', and 'pmf'.
            PMF should be shape (n_theta, n_psi), angles in radians.
    Returns:
        torque_field(theta, psi): Callable that returns (tau_theta, tau_psi).
    Notes:
        - Uses bicubic spline interpolation (RectBivariateSpline)
        - Input angles (theta, psi) should be in radians and within the grids.
    """
    from scipy.interpolate import RectBivariateSpline
    theta_centers = pmf_data.get("theta_centers")
    psi_centers = pmf_data.get("psi_centers")
    pmf = pmf_data.get("pmf")
    # Defensive: ensure 1D grids
    theta_centers = np.asarray(theta_centers).flatten()
    psi_centers = np.asarray(psi_centers).flatten()
    pmf = np.asarray(pmf)

    # Interpolator expects (x, y, z) with increasing x and y
    pmf_interp = RectBivariateSpline(theta_centers, psi_centers, pmf)

    def torque_field(theta: float, psi: float):
        """
        Returns the (tau_theta, tau_psi) torque at given (theta, psi).
        Args:
            theta: Tilt angle in radians (scalar or array)
            psi: Spin angle in radians (scalar or array)

        Returns:
            tau_theta, tau_psi (same shape as inputs)
        """
        # dU/dtheta = ∂PMF/∂θ, dU/dpsi = ∂PMF/∂ψ
        dU_dtheta = pmf_interp(theta, psi, dx=1, dy=0, grid=False)
        dU_dpsi = pmf_interp(theta, psi, dx=0, dy=1, grid=False)
        return -dU_dtheta, -dU_dpsi

    return torque_field

def compute_torque(
    positions: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    center_of_mass: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute torque τ = Σ (r_α - r_COM) × F_α.

    Args:
        positions: Atomic positions, shape (N, 3) in Å
        forces: Atomic forces, shape (N, 3) in kcal/(mol·Å)
        masses: Atomic masses, shape (N,) in amu
        center_of_mass: COM position (3,). If None, computed automatically.

    Returns:
        Torque τ, shape (3,) in kcal·Å/mol

    Mathematical Formula:
        τ = Σ_α (r_α - r_COM) × F_α

    Example:
        >>> positions = np.random.randn(100, 3)
        >>> forces = np.random.randn(100, 3) * 0.1
        >>> masses = np.ones(100)
        >>> tau = compute_torque(positions, forces, masses)
        >>> print(f"|τ| = {np.linalg.norm(tau):.2f} kcal·Å/mol")
    """
    if len(positions) != len(forces) or len(positions) != len(masses):
        raise ValueError("positions, forces, and masses must have same length")

    if positions.shape[1] != 3 or forces.shape[1] != 3:
        raise ValueError("positions and forces must have shape (N, 3)")

    # Compute center of mass if not provided
    if center_of_mass is None:
        center_of_mass = np.average(positions, weights=masses, axis=0)

    # Relative positions
    r = positions - center_of_mass

    # Torque: τ = Σ (r × F)
    tau = np.sum(np.cross(r, forces), axis=0)

    return tau


def decompose_torque(
    tau: np.ndarray,
    principal_axis: np.ndarray,
    membrane_normal: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Optional[np.ndarray]]:
    """
    Decompose torque into spin and nutation components.

    Args:
        tau: Torque vector, shape (3,) in kcal·Å/mol
        principal_axis: Protein principal axis p, shape (3,)
        membrane_normal: Membrane normal n, shape (3,). If None, no z-component.

    Returns:
        Tuple of:
            - tau_parallel: Spin torque τ_∥ = (τ · p) p, shape (3,)
            - tau_perp: Nutation torque τ_⊥ = τ - τ_∥, shape (3,)
            - magnitudes: Dict with |τ|, |τ_∥|, |τ_⊥|, |τ_z|
            - tau_z: Membrane-normal torque (τ · n) n, shape (3,) or None

    Physical Interpretation:
        - τ_∥: Drives spin around protein axis
        - τ_⊥: Drives nutation (wobbling)
        - |τ_∥| >> |τ_⊥|: Spin-dominated dynamics
        - |τ_⊥| >> |τ_∥|: Nutation-dominated dynamics

    Example:
        >>> tau = np.array([10, 5, 20])  # kcal·Å/mol
        >>> p = np.array([0, 0, 1])
        >>> tau_par, tau_perp, mags, _ = decompose_torque(tau, p)
        >>> print(f"|τ_∥| = {mags['tau_parallel']:.2f}")
        >>> print(f"|τ_⊥| = {mags['tau_perp']:.2f}")
    """
    # Validate inputs
    if len(tau) != 3:
        raise ValueError(f"tau must have 3 components, got {len(tau)}")
    if len(principal_axis) != 3:
        raise ValueError(f"principal_axis must have 3 components")

    # Normalize principal axis
    p = np.array(principal_axis) / np.linalg.norm(principal_axis)

    # Parallel component: τ_∥ = (τ · p) p
    tau_parallel_scalar = np.dot(tau, p)
    tau_parallel = tau_parallel_scalar * p

    # Perpendicular component: τ_⊥ = τ - τ_∥
    tau_perp = tau - tau_parallel

    # Magnitudes
    tau_total_mag = np.linalg.norm(tau)
    tau_parallel_mag = np.abs(tau_parallel_scalar)
    tau_perp_mag = np.linalg.norm(tau_perp)

    magnitudes = {
        'tau_total': tau_total_mag,
        'tau_parallel': tau_parallel_mag,
        'tau_perp': tau_perp_mag
    }

    # Membrane-normal component
    if membrane_normal is not None:
        if len(membrane_normal) != 3:
            raise ValueError(f"membrane_normal must have 3 components")
        n = np.array(membrane_normal) / np.linalg.norm(membrane_normal)
        tau_z_scalar = np.dot(tau, n)
        tau_z = tau_z_scalar * n
        magnitudes['tau_z'] = np.abs(tau_z_scalar)
    else:
        tau_z = None

    return tau_parallel, tau_perp, magnitudes, tau_z


def compute_dL_dt(
    L_trajectory: np.ndarray,
    times: np.ndarray,
    method: str = 'central'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dL/dt from angular momentum trajectory.

    Args:
        L_trajectory: Angular momentum, shape (N_frames, 3) in amu·Å²/ps
        times: Time points, shape (N_frames,) in ps
        method: Finite difference method ('forward', 'backward', 'central')

    Returns:
        Tuple of:
            - dL_dt: Time derivative of L, shape (N_valid, 3) in amu·Å²/ps²
            - time_points: Corresponding time points, shape (N_valid,)

    Notes:
        - Central difference loses first and last frames
        - Via Euler's equation, dL/dt should equal τ

    Example:
        >>> L_traj = np.random.randn(1000, 3) * 100
        >>> times = np.arange(1000) * 0.001  # ps
        >>> dL_dt, t = compute_dL_dt(L_traj, times, method='central')
        >>> print(dL_dt.shape)
        (998, 3)
    """
    N = len(L_trajectory)
    if N != len(times):
        raise ValueError(f"L_trajectory and times must have same length")

    if N < 3 and method == 'central':
        raise ValueError("Need at least 3 frames for central difference")

    dL_dt_list = []
    time_list = []

    if method == 'central':
        for i in range(1, N-1):
            dt_backward = times[i] - times[i-1]
            dt_forward = times[i+1] - times[i]
            dt_avg = (dt_backward + dt_forward) / 2

            dL_dt = (L_trajectory[i+1] - L_trajectory[i-1]) / (2 * dt_avg)
            dL_dt_list.append(dL_dt)
            time_list.append(times[i])

    elif method == 'forward':
        for i in range(N-1):
            dt = times[i+1] - times[i]
            dL_dt = (L_trajectory[i+1] - L_trajectory[i]) / dt
            dL_dt_list.append(dL_dt)
            time_list.append(times[i])

    elif method == 'backward':
        for i in range(1, N):
            dt = times[i] - times[i-1]
            dL_dt = (L_trajectory[i] - L_trajectory[i-1]) / dt
            dL_dt_list.append(dL_dt)
            time_list.append(times[i])

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.array(dL_dt_list), np.array(time_list)


def validate_euler_equation(
    tau: np.ndarray,
    dL_dt: np.ndarray,
    rtol: float = 0.1
) -> Dict[str, any]:
    """
    Validate Euler's equation: dL/dt = τ.

    Args:
        tau: Torque from forces, shape (3,) in kcal·Å/mol
        dL_dt: Time derivative of angular momentum, shape (3,) in amu·Å²/ps²
        rtol: Relative tolerance for validation

    Returns:
        Dictionary with:
            - 'tau': Torque vector
            - 'dL_dt': Angular momentum derivative
            - 'residual': |dL/dt - τ|
            - 'relative_error': residual / |τ|
            - 'valid': Boolean (True if residual < rtol * |τ|)

    Notes:
        - Units must be consistent (conversion required)
        - 1 kcal·Å/mol = 69.4786 amu·Å²/ps²

    Example:
        >>> tau = np.array([10, 5, 20])  # kcal·Å/mol
        >>> dL_dt = tau * 69.4786  # Convert to amu·Å²/ps²
        >>> result = validate_euler_equation(tau, dL_dt)
        >>> assert result['valid']
    """
    # Unit conversion: kcal·Å/mol → amu·Å²/ps²
    CONVERSION = 69.4786  # 1 kcal·Å/mol = 69.4786 amu·Å²/ps²
    tau_converted = tau * CONVERSION

    residual_vec = dL_dt - tau_converted
    residual_mag = np.linalg.norm(residual_vec)
    tau_mag = np.linalg.norm(tau_converted)

    if tau_mag == 0:
        relative_error = 0 if residual_mag == 0 else np.inf
    else:
        relative_error = residual_mag / tau_mag

    valid = relative_error < rtol

    return {
        'tau': tau,
        'tau_converted': tau_converted,
        'dL_dt': dL_dt,
        'residual': residual_mag,
        'relative_error': relative_error,
        'valid': valid,
        'tolerance': rtol
    }


def compute_torque_trajectory(
    positions_traj: np.ndarray,
    forces_traj: np.ndarray,
    masses: np.ndarray,
    principal_axes_traj: np.ndarray,
    membrane_normal: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute torque decomposition for entire trajectory.

    Args:
        positions_traj: Positions, shape (N_frames, N_atoms, 3) in Å
        forces_traj: Forces, shape (N_frames, N_atoms, 3) in kcal/(mol·Å)
        masses: Atomic masses, shape (N_atoms,) in amu
        principal_axes_traj: Principal axes, shape (N_frames, 3)
        membrane_normal: Membrane normal n, shape (3,)

    Returns:
        Dictionary with:
            - 'tau_total': Total torque, shape (N_frames, 3)
            - 'tau_parallel': Spin torque, shape (N_frames, 3)
            - 'tau_perp': Nutation torque, shape (N_frames, 3)
            - 'tau_total_mag': |τ| magnitude, shape (N_frames,)
            - 'tau_parallel_mag': |τ_∥| magnitude, shape (N_frames,)
            - 'tau_perp_mag': |τ_⊥| magnitude, shape (N_frames,)
            - 'tau_z' (optional): Membrane-normal torque
            - 'tau_z_mag' (optional): |τ_z| magnitude

    Example:
        >>> positions = np.random.randn(1000, 100, 3)
        >>> forces = np.random.randn(1000, 100, 3) * 0.1
        >>> masses = np.ones(100)
        >>> p_axes = np.repeat([[0, 0, 1]], 1000, axis=0)
        >>> result = compute_torque_trajectory(positions, forces, masses, p_axes)
        >>> print(f"Mean |τ| = {np.mean(result['tau_total_mag']):.2f}")
    """
    N_frames = len(positions_traj)

    if len(forces_traj) != N_frames:
        raise ValueError("positions_traj and forces_traj must have same length")

    if len(principal_axes_traj) != N_frames:
        raise ValueError("principal_axes_traj must match trajectory length")

    tau_total_list = []
    tau_parallel_list = []
    tau_perp_list = []
    tau_total_mag_list = []
    tau_parallel_mag_list = []
    tau_perp_mag_list = []

    if membrane_normal is not None:
        tau_z_list = []
        tau_z_mag_list = []

    for i in range(N_frames):
        # Compute torque
        tau = compute_torque(
            positions_traj[i],
            forces_traj[i],
            masses
        )

        # Decompose
        tau_par, tau_perp, mags, tau_z = decompose_torque(
            tau,
            principal_axes_traj[i],
            membrane_normal
        )

        tau_total_list.append(tau)
        tau_parallel_list.append(tau_par)
        tau_perp_list.append(tau_perp)
        tau_total_mag_list.append(mags['tau_total'])
        tau_parallel_mag_list.append(mags['tau_parallel'])
        tau_perp_mag_list.append(mags['tau_perp'])

        if membrane_normal is not None:
            tau_z_list.append(tau_z)
            tau_z_mag_list.append(mags['tau_z'])

    result = {
        'tau_total': np.array(tau_total_list),
        'tau_parallel': np.array(tau_parallel_list),
        'tau_perp': np.array(tau_perp_list),
        'tau_total_mag': np.array(tau_total_mag_list),
        'tau_parallel_mag': np.array(tau_parallel_mag_list),
        'tau_perp_mag': np.array(tau_perp_mag_list)
    }

    if membrane_normal is not None:
        result['tau_z'] = np.array(tau_z_list)
        result['tau_z_mag'] = np.array(tau_z_mag_list)

    return result


if __name__ == '__main__':
    print("Torque Module - Example Usage\n")

    # Example 1: Single frame torque
    print("Example 1: Compute torque from forces")
    np.random.seed(42)

    N_atoms = 100
    positions = np.random.randn(N_atoms, 3) * 10  # Å
    forces = np.random.randn(N_atoms, 3) * 0.5  # kcal/(mol·Å)
    masses = np.ones(N_atoms) * 12.0  # amu

    tau = compute_torque(positions, forces, masses)
    print(f"Torque: {tau}")
    print(f"|τ| = {np.linalg.norm(tau):.2f} kcal·Å/mol")

    # Example 2: Decomposition
    print("\n" + "="*60)
    print("Example 2: Decompose torque")

    principal_axis = np.array([0, 0, 1])
    membrane_normal = np.array([1, 0, 0])

    tau_par, tau_perp, mags, tau_z = decompose_torque(
        tau, principal_axis, membrane_normal
    )

    print(f"\nTotal τ: {tau}")
    print(f"τ_parallel (spin): {tau_par}")
    print(f"τ_perp (nutation): {tau_perp}")

    print(f"\nMagnitudes:")
    print(f"  |τ| = {mags['tau_total']:.2f}")
    print(f"  |τ_∥| = {mags['tau_parallel']:.2f}")
    print(f"  |τ_⊥| = {mags['tau_perp']:.2f}")

    # Example 3: Euler's equation validation
    print("\n" + "="*60)
    print("Example 3: Validate Euler's equation dL/dt = τ")

    # Simulate consistent tau and dL/dt
    tau_test = np.array([10.0, 5.0, 20.0])  # kcal·Å/mol
    CONVERSION = 69.4786
    dL_dt_test = tau_test * CONVERSION  # amu·Å²/ps²

    result = validate_euler_equation(tau_test, dL_dt_test, rtol=0.1)

    print(f"τ = {result['tau']}")
    print(f"dL/dt = {result['dL_dt']}")
    print(f"Residual: {result['residual']:.2e}")
    print(f"Relative error: {result['relative_error']:.2e}")
    print(f"Valid: {result['valid']}")

    # Example 4: Trajectory analysis
    print("\n" + "="*60)
    print("Example 4: Torque trajectory")

    N_frames = 100
    positions_traj = np.random.randn(N_frames, N_atoms, 3) * 10
    forces_traj = np.random.randn(N_frames, N_atoms, 3) * 0.5
    principal_axes_traj = np.repeat([[0, 0, 1]], N_frames, axis=0)

    result_traj = compute_torque_trajectory(
        positions_traj,
        forces_traj,
        masses,
        principal_axes_traj,
        membrane_normal
    )

    print(f"Trajectory length: {N_frames} frames")
    print(f"\nMean values:")
    print(f"  <|τ|> = {np.mean(result_traj['tau_total_mag']):.2f} ± {np.std(result_traj['tau_total_mag']):.2f}")
    print(f"  <|τ_∥|> = {np.mean(result_traj['tau_parallel_mag']):.2f} ± {np.std(result_traj['tau_parallel_mag']):.2f}")
    print(f"  <|τ_⊥|> = {np.mean(result_traj['tau_perp_mag']):.2f} ± {np.std(result_traj['tau_perp_mag']):.2f}")

    # Example 5: dL/dt computation
    print("\n" + "="*60)
    print("Example 5: Compute dL/dt from L trajectory")

    # Generate synthetic L trajectory
    L_traj = np.random.randn(N_frames, 3) * 100  # amu·Å²/ps
    times = np.arange(N_frames) * 0.001  # ps

    dL_dt_traj, t_dL = compute_dL_dt(L_traj, times, method='central')

    print(f"L trajectory: {L_traj.shape}")
    print(f"dL/dt: {dL_dt_traj.shape}")
    print(f"Mean |dL/dt| = {np.mean(np.linalg.norm(dL_dt_traj, axis=1)):.2f} amu·Å²/ps²")
