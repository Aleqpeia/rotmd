#!/usr/bin/env python
"""
Friction Coefficient Extraction for Protein Orientation Dynamics

This module computes orientation-dependent friction coefficients γ(θ, ψ)
from angular velocity autocorrelation functions and validates against
rotational diffusion models.

Mathematical Background:
========================

Friction in Rotational Diffusion:
---------------------------------
For a particle undergoing rotational Brownian motion with friction γ:

    I dω/dt = -γ ω + ξ(t)

where:
- I: moment of inertia tensor
- ω: angular velocity
- γ: friction coefficient (tensor in general)
- ξ(t): random torque (white noise)

Overdamped Limit (high friction):
---------------------------------
When γ >> I/τ, the dynamics simplifies to:

    dθ/dt = (1/γ) τ + noise

where τ is the deterministic torque.

Angular Velocity Autocorrelation:
---------------------------------
In the overdamped limit:

    C_ω(t) = <ω(0) · ω(t)> / <ω²> = exp(-γt/I)

Friction extraction:
    γ = I / τ_c

where τ_c is the correlation time from C_ω(t).

Orientation-Dependent Friction:
-------------------------------
For membrane proteins, friction depends on orientation (θ, ψ):

    γ(θ, ψ) = γ_∥(θ, ψ) p⊗p + γ_⊥(θ, ψ) (I - p⊗p)

where:
- γ_∥: friction for spin (rotation around protein axis p)
- γ_⊥: friction for nutation (wobbling)
- p: principal axis

Binning Strategy:
-----------------
1. Divide (θ, ψ) space into bins
2. For each bin, collect ω(t) trajectory segments
3. Compute C_ω(t) for that bin
4. Extract γ from decay rate
5. Build γ(θ, ψ) map

Physical Interpretation:
------------------------

Well-bound transmembrane protein:
- High γ (slow relaxation)
- γ_⊥ >> γ_∥ (nutation highly damped)
- Weak dependence on θ, ψ (uniform binding)

Poorly-bound peripheral protein:
- Low γ (fast relaxation)
- γ_⊥ ≈ γ_∥ (isotropic-like)
- Strong dependence on θ, ψ (heterogeneous binding)

Green-Kubo Relations:
---------------------
Alternative friction calculation via force autocorrelation:

    γ_ij = ∫_0^∞ <τ_i(t) τ_j(0)> / (k_B T) dt

where τ is the random torque.

References:
-----------
- Kubo (1966). Rep. Prog. Phys. 29, 255.
- Zwanzig (2001). Nonequilibrium Statistical Mechanics.
- Doi & Edwards (1988). The Theory of Polymer Dynamics.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.optimize import curve_fit
import warnings


def extract_friction_from_acf(
    times: np.ndarray,
    acf: np.ndarray,
    moment_of_inertia: float,
    method: str = 'integral'
) -> Dict[str, float]:
    """
    Extract friction coefficient γ from angular velocity ACF.

    Args:
        times: Time points (ps), shape (N,)
        acf: Angular velocity autocorrelation C_ω(t), shape (N,)
        moment_of_inertia: Moment of inertia I in amu·Å²
        method: Extraction method:
            - 'exponential_fit': Fit C_ω(t) = exp(-γt/I), extract γ
            - 'initial_slope': γ = -I · dC/dt|_{t=0}
            - 'integral': Use Green-Kubo relation

    Returns:
        Dictionary with:
            - 'gamma': Friction coefficient γ in amu/ps
            - 'tau_c': Correlation time τ_c in ps
            - 'diffusion_coeff': Rotational diffusion coefficient D = k_B T / γ

    Mathematical Formula:
        C_ω(t) = exp(-γ t / I)
        Therefore: γ = I / τ_c

    Example:
        >>> times = np.arange(100) * 0.01  # ps
        >>> acf = np.exp(-times / 0.5)  # τ_c = 0.5 ps
        >>> I = 1000.0  # amu·Å²
        >>> result = extract_friction_from_acf(times, acf, I)
        >>> print(f"γ = {result['gamma']:.1f} amu/ps")
    """
    if len(times) != len(acf):
        raise ValueError("times and acf must have same length")

    gamma = None  # Initialize

    if method == 'exponential_fit':
        # Fit C_ω(t) = A exp(-γ t / I)
        def fit_func(t, gamma):
            return np.exp(-gamma * t / moment_of_inertia)

        try:
            # Initial guess: γ ~ I / <time>
            gamma_guess = moment_of_inertia / (times[-1] / 3)
            popt, _ = curve_fit(
                fit_func, times, acf,
                p0=[gamma_guess],
                bounds=([0], [np.inf]),
                maxfev=10000
            )
            gamma = popt[0]
        except RuntimeError:
            warnings.warn("Exponential fit failed. Using initial slope method.")
            # Fall back to initial slope
            dt = times[1] - times[0]
            dC_dt_0 = (acf[1] - acf[0]) / dt
            gamma = -moment_of_inertia * dC_dt_0

    elif method == 'initial_slope':
        # γ = -I · dC/dt|_{t=0}
        # Approximate derivative at t=0
        dt = times[1] - times[0]
        dC_dt_0 = (acf[1] - acf[0]) / dt
        gamma = -moment_of_inertia * dC_dt_0

    elif method == 'integral':
        # τ_c = ∫ C_ω(t) dt
        # γ = I / τ_c
        tau_c = np.trapz(acf, times)
        gamma = moment_of_inertia / tau_c if tau_c > 0 else np.inf

    else:
        raise ValueError(f"Unknown method: {method}")

    if gamma is None:
        gamma = 0.0

    # Correlation time
    tau_c = moment_of_inertia / gamma if gamma > 0 else np.inf

    # Rotational diffusion coefficient: D = k_B T / γ
    # k_B T at 310 K ≈ 0.615 kcal/mol
    k_B_T = 0.615  # kcal/mol
    diffusion_coeff = k_B_T / gamma if gamma > 0 else np.inf

    return {
        'gamma': gamma,
        'tau_c': tau_c,
        'diffusion_coeff': diffusion_coeff,
        'method': method
    }


def orientation_dependent_friction(
    theta_trajectory: np.ndarray,
    psi_trajectory: np.ndarray,
    omega_trajectory: np.ndarray,
    times: np.ndarray,
    moment_of_inertia: float,
    theta_bins: int = 10,
    psi_bins: int = 12,
    min_samples_per_bin: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute orientation-dependent friction γ(θ, ψ).

    Args:
        theta_trajectory: Tilt angle θ, shape (N_frames,) in degrees
        psi_trajectory: Spin angle ψ, shape (N_frames,) in degrees
        omega_trajectory: Angular velocity, shape (N_frames, 3) in rad/ps
        times: Time points, shape (N_frames,) in ps
        moment_of_inertia: Moment of inertia I in amu·Å²
        theta_bins: Number of bins for θ ∈ [0°, 90°]
        psi_bins: Number of bins for ψ ∈ [0°, 360°]
        min_samples_per_bin: Minimum frames per bin for valid statistics

    Returns:
        Dictionary with:
            - 'theta_edges': Bin edges for θ, shape (theta_bins+1,)
            - 'psi_edges': Bin edges for ψ, shape (psi_bins+1,)
            - 'gamma_map': Friction map γ(θ, ψ), shape (theta_bins, psi_bins)
            - 'counts': Number of samples per bin, shape (theta_bins, psi_bins)
            - 'tau_c_map': Correlation time map, shape (theta_bins, psi_bins)

    Physical Interpretation:
        - γ(θ, ψ) reveals how friction varies with orientation
        - High γ regions: Strong membrane coupling
        - Low γ regions: Weak coupling, free rotation

    Example:
        >>> theta = np.random.rand(10000) * 90  # degrees
        >>> psi = np.random.rand(10000) * 360
        >>> omega = np.random.randn(10000, 3) * 0.1
        >>> times = np.arange(10000) * 0.001
        >>> I = 1000.0
        >>> result = orientation_dependent_friction(
        ...     theta, psi, omega, times, I, theta_bins=5, psi_bins=6
        ... )
        >>> print(result['gamma_map'].shape)
        (5, 6)
    """
    N_frames = len(theta_trajectory)

    if (len(psi_trajectory) != N_frames or
        len(omega_trajectory) != N_frames or
        len(times) != N_frames):
        raise ValueError("All trajectories must have same length")

    # Create bins
    theta_edges = np.linspace(0, 90, theta_bins + 1)
    psi_edges = np.linspace(0, 360, psi_bins + 1)

    # Initialize maps
    gamma_map = np.full((theta_bins, psi_bins), np.nan)
    tau_c_map = np.full((theta_bins, psi_bins), np.nan)
    counts = np.zeros((theta_bins, psi_bins), dtype=int)

    # Digitize trajectories
    theta_idx = np.digitize(theta_trajectory, theta_edges) - 1
    psi_idx = np.digitize(psi_trajectory, psi_edges) - 1

    # Clip to valid range (handle edge cases)
    theta_idx = np.clip(theta_idx, 0, theta_bins - 1)
    psi_idx = np.clip(psi_idx, 0, psi_bins - 1)

    # Process each bin
    for i in range(theta_bins):
        for j in range(psi_bins):
            # Find frames in this bin
            mask = (theta_idx == i) & (psi_idx == j)
            n_samples = np.sum(mask)
            counts[i, j] = n_samples

            if n_samples < min_samples_per_bin:
                continue  # Skip bins with insufficient data

            # Extract omega for this bin
            omega_bin = omega_trajectory[mask]
            times_bin = times[mask]

            # Sort by time (in case mask is not sequential)
            sort_idx = np.argsort(times_bin)
            times_bin = times_bin[sort_idx]
            omega_bin = omega_bin[sort_idx]

            # Compute ACF for this bin
            try:
                from .correlations import autocorrelation_function

                # Use magnitude of omega
                omega_mag = np.linalg.norm(omega_bin, axis=1)

                max_lag = min(len(omega_mag) // 4, 200)  # Use 1/4 of data or 200 frames
                lags, acf = autocorrelation_function(
                    omega_mag, max_lag=max_lag, normalize=True
                )

                # Extract friction
                dt = np.mean(np.diff(times_bin)) if len(times_bin) > 1 else 0.001
                times_acf = lags * dt

                friction_result = extract_friction_from_acf(
                    times_acf, acf, moment_of_inertia, method='integral'
                )

                gamma_map[i, j] = friction_result['gamma']
                tau_c_map[i, j] = friction_result['tau_c']

            except Exception as e:
                warnings.warn(f"Failed to compute friction for bin ({i}, {j}): {e}")
                continue

    return {
        'theta_edges': theta_edges,
        'psi_edges': psi_edges,
        'gamma_map': gamma_map,
        'counts': counts,
        'tau_c_map': tau_c_map,
        'theta_centers': (theta_edges[:-1] + theta_edges[1:]) / 2,
        'psi_centers': (psi_edges[:-1] + psi_edges[1:]) / 2
    }


def anisotropic_friction_tensor(
    omega_parallel_acf: np.ndarray,
    omega_perp_acf: np.ndarray,
    times: np.ndarray,
    I_parallel: float,
    I_perp: float
) -> Dict[str, float]:
    """
    Extract anisotropic friction tensor components γ_∥ and γ_⊥.

    Args:
        omega_parallel_acf: ACF of spin angular velocity C_{ω_∥}(t)
        omega_perp_acf: ACF of nutation angular velocity C_{ω_⊥}(t)
        times: Time points (ps)
        I_parallel: Moment of inertia for spin (amu·Å²)
        I_perp: Moment of inertia for nutation (amu·Å²)

    Returns:
        Dictionary with:
            - 'gamma_parallel': Friction for spin γ_∥
            - 'gamma_perp': Friction for nutation γ_⊥
            - 'anisotropy': γ_⊥ / γ_∥ ratio

    Physical Interpretation:
        - γ_⊥ >> γ_∥: Nutation highly damped (well-bound)
        - γ_⊥ ≈ γ_∥: Isotropic friction (poorly-bound)

    Example:
        >>> times = np.arange(100) * 0.01
        >>> C_par = np.exp(-times / 1.0)
        >>> C_perp = np.exp(-times / 0.1)  # Faster decay
        >>> I_par = 1000.0
        >>> I_perp = 500.0
        >>> result = anisotropic_friction_tensor(C_par, C_perp, times, I_par, I_perp)
        >>> print(f"Anisotropy γ_⊥/γ_∥ = {result['anisotropy']:.1f}")
    """
    # Extract γ_∥
    result_par = extract_friction_from_acf(
        times, omega_parallel_acf, I_parallel, method='integral'
    )
    gamma_par = result_par['gamma']

    # Extract γ_⊥
    result_perp = extract_friction_from_acf(
        times, omega_perp_acf, I_perp, method='integral'
    )
    gamma_perp = result_perp['gamma']

    # Anisotropy ratio
    anisotropy = gamma_perp / gamma_par if gamma_par > 0 else np.inf

    return {
        'gamma_parallel': gamma_par,
        'gamma_perp': gamma_perp,
        'anisotropy': anisotropy,
        'tau_c_parallel': result_par['tau_c'],
        'tau_c_perp': result_perp['tau_c']
    }


if __name__ == '__main__':
    print("Friction Module - Example Usage\n")

    # Example 1: Extract friction from ACF
    print("Example 1: Friction from angular velocity ACF")
    np.random.seed(42)

    # Synthetic ACF with known decay rate
    times = np.arange(100) * 0.01  # ps
    gamma_true = 500.0  # amu/ps
    I = 1000.0  # amu·Å²
    tau_c_true = I / gamma_true  # = 2.0 ps
    acf = np.exp(-times / tau_c_true)

    result = extract_friction_from_acf(times, acf, I, method='exponential_fit')
    print(f"True γ: {gamma_true:.1f} amu/ps")
    print(f"Extracted γ: {result['gamma']:.1f} amu/ps")
    print(f"True τ_c: {tau_c_true:.2f} ps")
    print(f"Extracted τ_c: {result['tau_c']:.2f} ps")
    print(f"Diffusion coefficient D: {result['diffusion_coeff']:.4f}")

    # Example 2: Orientation-dependent friction
    print("\n" + "="*60)
    print("Example 2: Orientation-dependent friction γ(θ, ψ)")

    # Generate synthetic trajectory
    N_frames = 5000
    theta_traj = np.random.rand(N_frames) * 90  # 0-90 degrees
    psi_traj = np.random.rand(N_frames) * 360  # 0-360 degrees
    omega_traj = np.random.randn(N_frames, 3) * 0.1  # rad/ps
    times_traj = np.arange(N_frames) * 0.001  # ps

    # Compute γ(θ, ψ) map
    result_map = orientation_dependent_friction(
        theta_traj, psi_traj, omega_traj, times_traj,
        moment_of_inertia=1000.0,
        theta_bins=5,
        psi_bins=6,
        min_samples_per_bin=50
    )

    print(f"Friction map shape: {result_map['gamma_map'].shape}")
    print(f"Number of valid bins: {np.sum(~np.isnan(result_map['gamma_map']))}")
    print(f"Mean γ (valid bins): {np.nanmean(result_map['gamma_map']):.1f} amu/ps")
    print(f"Min samples per bin: {np.min(result_map['counts'])}")
    print(f"Max samples per bin: {np.max(result_map['counts'])}")

    # Example 3: Anisotropic friction
    print("\n" + "="*60)
    print("Example 3: Anisotropic friction tensor")

    # Spin (parallel) decays slowly
    C_par = np.exp(-times / 2.0)  # τ_c = 2.0 ps

    # Nutation (perpendicular) decays fast
    C_perp = np.exp(-times / 0.5)  # τ_c = 0.5 ps

    I_par = 1000.0
    I_perp = 500.0

    result_aniso = anisotropic_friction_tensor(C_par, C_perp, times, I_par, I_perp)

    print(f"γ_∥ (spin): {result_aniso['gamma_parallel']:.1f} amu/ps")
    print(f"γ_⊥ (nutation): {result_aniso['gamma_perp']:.1f} amu/ps")
    print(f"Anisotropy γ_⊥/γ_∥: {result_aniso['anisotropy']:.2f}")

    if result_aniso['anisotropy'] > 2:
        print("→ Nutation highly damped (well-bound protein)")
    elif result_aniso['anisotropy'] < 0.5:
        print("→ Spin highly damped")
    else:
        print("→ Isotropic-like friction")
