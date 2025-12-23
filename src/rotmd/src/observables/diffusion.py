"""
Rotational Diffusion Analysis

This module provides utilities for computing and analyzing the rotational
diffusion tensor from MD trajectories.

Key Features:
- Rotational diffusion tensor D_ij computation
- Einstein relation: <Δθ²(t)> = 2D_eff·t
- Anisotropic diffusion analysis
- Time-dependent diffusion coefficient
- Integration with Euler angle trajectories

Theoretical Background:
- For isotropic diffusion: D is scalar, <Δθ²> = 6Dt
- For anisotropic diffusion: D is tensor, different rates along principal axes
- Related to friction: D = kBT / ζ (Einstein-Stokes relation)

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
from numpy._typing import NDArray
from scipy.optimize import curve_fit


def mean_squared_angular_displacement(euler_angles: np.ndarray,
                                      max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean squared angular displacement (MSAD).

    MSAD(t) = <|Δθ(t)|²> where Δθ(t) is angular displacement after time t.

    Args:
        euler_angles: (n_frames, 3) array of Euler angles [phi, theta, psi]
        max_lag: Maximum lag time in frames (None = n_frames // 4)

    Returns:
        lags: (max_lag,) lag times in frame units
        msad: (max_lag,) mean squared angular displacement in rad²

    Notes:
        - Uses quaternion-based angular displacement for proper SO(3) metric
        - At short times: MSAD ~ 2D_eff·t (linear regime)
        - At long times: MSAD → plateau for confined diffusion
    """
    from ..core.orientation import compute_angular_displacement

    n_frames = len(euler_angles)
    if max_lag is None:
        max_lag = n_frames // 4

    lags = np.arange(max_lag)
    msad = np.zeros(max_lag)

    for lag in lags:
        squared_displacements = []
        for i in range(n_frames - lag):
            angle = compute_angular_displacement(euler_angles[i],
                                                euler_angles[i + lag])
            squared_displacements.append(angle**2)
        msad[lag] = np.mean(squared_displacements)

    return lags, msad


def extract_diffusion_coefficient(times: np.ndarray,
                                  msad: np.ndarray,
                                  fit_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """
    Extract rotational diffusion coefficient from MSAD.

    For 3D isotropic diffusion: MSAD(t) = 6Dt
    Therefore: D = MSAD(t) / (6t) from linear regime

    Args:
        times: (n_points,) time values in ps
        msad: (n_points,) mean squared angular displacement in rad²
        fit_range: Optional (t_min, t_max) range for linear fit in ps
                   If None, uses first quarter of data

    Returns:
        D: Diffusion coefficient in rad²/ps
        D_err: Standard error of diffusion coefficient

    Notes:
        - Only fits the initial linear regime (ballistic → diffusive crossover)
        - Assumes isotropic 3D diffusion
    """
    if fit_range is None:
        # Use first 25% of data for linear regime
        max_idx = len(times) // 4
        fit_range = (times[1], times[max_idx])  # Skip t=0

    # Select data in fit range
    mask = (times >= fit_range[0]) & (times <= fit_range[1])
    t_fit = times[mask]
    msad_fit = msad[mask]

    # Linear fit: MSAD = 6Dt
    def linear(t, D):
        return 6 * D * t

    try:
        popt, pcov = curve_fit(linear, t_fit, msad_fit)
        D = popt[0]
        D_err = np.sqrt(pcov[0, 0])
    except:
        # Fallback: simple slope
        D = np.mean(msad_fit / (6 * t_fit))
        D_err = np.std(msad_fit / (6 * t_fit)) / np.sqrt(len(t_fit))

    return D, D_err


def anisotropic_diffusion_tensor(euler_angles: np.ndarray,
                                 times: np.ndarray,
                                 max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute anisotropic rotational diffusion tensor.

    For anisotropic diffusion, different Euler angles diffuse at different rates.
    This function computes diffusion coefficients for each Euler angle separately.

    Args:
        euler_angles: (n_frames, 3) array of [phi, theta, psi]
        times: (n_frames,) timestamps in ps
        max_lag: Maximum lag for computing MSD (None = n_frames // 4)

    Returns:
        D_diag: (3,) diagonal diffusion coefficients [D_phi, D_theta, D_psi] in rad²/ps
        D_err: (3,) standard errors for each coefficient

    Notes:
        - Assumes principal diffusion axes align with Euler angle coordinates
        - For truly anisotropic systems, need full tensor with off-diagonal terms
        - Related to principal moments of inertia: D_i ~ 1/I_i
    """
    from ..core.orientation import unwrap_euler_angles

    n_frames = len(euler_angles)
    if max_lag is None:
        max_lag = n_frames // 4

    # Unwrap angles to handle 2π discontinuities
    unwrapped = unwrap_euler_angles(euler_angles)

    D_diag = np.zeros(3)
    D_err = np.zeros(3)

    for i in range(3):
        lags = np.arange(max_lag)
        msd = np.zeros(max_lag)

        # Skip lag 0 (it remains 0.0 as initialized)
        for lag in range(1, max_lag):
            diff = unwrapped[lag:, i] - unwrapped[:-lag, i]
            if diff.size > 0:
                msd[lag] = np.mean(diff**2)

        # Convert lags to time
        dt = times[1] - times[0]  # Assume uniform spacing
        lag_times = lags * dt

        # Fit linear regime: MSD = 2Dt (1D diffusion for each angle)
        # Use first quarter of data
        fit_max = max_lag // 4
        t_fit = lag_times[1:fit_max]  # Skip t=0
        msd_fit = msd[1:fit_max]

        if len(t_fit) > 3:
            def linear(t, D):
                return 2 * D * t

            try:
                popt, pcov = curve_fit(linear, t_fit, msd_fit)
                D_diag[i] = popt[0]
                D_err[i] = np.sqrt(pcov[0, 0])
            except:
                # Fallback
                D_diag[i] = np.mean(msd_fit / (2 * t_fit))
                D_err[i] = np.std(msd_fit / (2 * t_fit)) / np.sqrt(len(t_fit))
        else:
            D_diag[i] = 0.0
            D_err[i] = 0.0

    return D_diag, D_err


def time_dependent_diffusion(euler_angles: np.ndarray,
                            times: np.ndarray,
                            window_size: int = 100,
                            stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time-dependent diffusion coefficient.

    Useful for detecting conformational transitions or non-stationary dynamics.

    Args:
        euler_angles: (n_frames, 3) array of Euler angles
        times: (n_frames,) timestamps in ps
        window_size: Window size for local diffusion calculation (frames)
        stride: Stride between windows (frames)

    Returns:
        t_centers: (n_windows,) center times of windows in ps
        D_t: (n_windows,) diffusion coefficients in rad²/ps

    Notes:
        - Each window gives local diffusion coefficient
        - Variations indicate non-equilibrium dynamics
        - Requires stationary dynamics within each window
    """
    from ..core.orientation import compute_angular_displacement

    n_frames = len(euler_angles)
    n_windows = (n_frames - window_size) // stride + 1

    t_centers = np.zeros(n_windows)
    D_t = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size

        # Center time of window
        t_centers[i] = np.mean(times[start:end])

        # Compute MSAD within window
        window_angles = euler_angles[start:end]
        window_times = times[start:end]

        max_lag = min(window_size // 4, 50)
        lags, msad = mean_squared_angular_displacement(window_angles, max_lag)

        # Convert to time
        dt = window_times[1] - window_times[0]
        lag_times = lags * dt

        # Extract D from linear regime
        if len(lag_times) > 5:
            D, _ = extract_diffusion_coefficient(lag_times, msad)
            D_t[i] = D
        else:
            D_t[i] = 0.0

    return t_centers, D_t


def diffusion_from_velocity_acf(angular_velocities: np.ndarray,
                                times: np.ndarray,
                                max_lag: Optional[int] = None) -> float:
    """
    Compute diffusion coefficient from angular velocity autocorrelation.

    Uses Green-Kubo relation: D = ∫₀^∞ <ω(0)·ω(t)> dt

    Args:
        angular_velocities: (n_frames, 3) angular velocity vectors in rad/ps
        times: (n_frames,) timestamps in ps
        max_lag: Maximum lag for ACF integration (None = n_frames // 2)

    Returns:
        D: Diffusion coefficient in rad²/ps

    Notes:
        - Alternative to MSD-based diffusion
        - Requires velocity data from trajectory
        - Green-Kubo gives exact result in linear response theory
    """
    from ..analysis.correlations import autocorrelation_function

    n_frames = len(angular_velocities)
    if max_lag is None:
        max_lag = n_frames // 2

    # Compute ACF of angular velocity magnitude
    omega_mag = np.linalg.norm(angular_velocities, axis=1)
    lags, acf = autocorrelation_function(omega_mag, max_lag=max_lag)

    # Integrate ACF: D = (1/3) ∫ C_ω(t) dt
    # Factor 1/3 for 3D isotropic case
    dt = times[1] - times[0]
    D = (1.0 / 3.0) * np.trapz(acf, dx=dt)

    return D


def analyze_diffusion(angular_velocities: np.ndarray,
                     times: np.ndarray,
                     verbose: bool = True) -> Dict:
    """
    Comprehensive diffusion analysis.

    Args:
        euler_angles: (n_frames, 3) array of Euler angles
        times: (n_frames,) timestamps in ps
        angular_velocities: Optional (n_frames, 3) angular velocities in rad/ps
        verbose: Print results

    Returns:
        results: Dictionary containing:
            - D_msad: Diffusion from MSAD analysis
            - D_msad_err: Error estimate
            - D_aniso: (3,) anisotropic diffusion coefficients
            - D_aniso_err: (3,) error estimates
            - D_acf: Diffusion from velocity ACF (if velocities provided)
            - msad_times: Time values for MSAD curve
            - msad_values: MSAD values

    Example:
        >>> results = analyze_diffusion(euler, times, omega)
        >>> print(f"D = {results['D_msad']:.3f} ± {results['D_msad_err']:.3f} rad²/ps")
    """

    D_acf = diffusion_from_velocity_acf(angular_velocities, times)

    if verbose:
        print("Rotational Diffusion Analysis")
        print("=" * 50)
        # print(f"Isotropic diffusion (MSAD): D = {D:.4f} ± {D_err:.4f} rad²/ps")
        # print(f"Anisotropic diffusion:")
        # print(f"  D_φ   = {D_aniso[0]:.4f} ± {D_aniso_err[0]:.4f} rad²/ps")
        # print(f"  D_θ   = {D_aniso[1]:.4f} ± {D_aniso_err[1]:.4f} rad²/ps")
        # print(f"  D_ψ   = {D_aniso[2]:.4f} ± {D_aniso_err[2]:.4f} rad²/ps")
        print(f"Green-Kubo (ACF):           D = {D_acf:.4f} rad²/ps")

    return D_acf 


if __name__ == '__main__':
    # Example usage
    print("Rotational Diffusion Module")
    print("============================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.observables.diffusion import analyze_diffusion")
    print("from protein_orientation.core.orientation import extract_orientation_trajectory")
    print()
    print("# Extract Euler angles from trajectory")
    print("euler = extract_orientation_trajectory(positions, masses)")
    print()
    print("# Analyze diffusion")
    print("results = analyze_diffusion(euler, times)")
    print()
    print("print(f'D = {results[\"D_msad\"]:.4f} rad²/ps')")
