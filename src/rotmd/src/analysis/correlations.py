#!/usr/bin/env python
"""
Time Correlation Functions and Friction Extraction

This module computes autocorrelation functions (ACF) and cross-correlation functions
for dynamical observables, and extracts friction coefficients from decay rates.

Mathematical Background:
========================

Autocorrelation Function (ACF):
-------------------------------
For a time-dependent observable A(t):

    C_A(τ) = <A(t) · A(t+τ)> / <A(t) · A(t)>

Properties:
- C_A(0) = 1 (normalized)
- C_A(τ) → 0 as τ → ∞ (decorrelation)
- C_A(τ) ≥ 0 for most physical observables

Exponential Decay Model:
------------------------
For diffusive dynamics:

    C_A(τ) = exp(-τ/τ_c)

where τ_c is the correlation time (characteristic decay time).

Multi-Exponential Decay:
-------------------------
For complex systems:

    C_A(τ) = Σ_i a_i exp(-τ/τ_i)

with Σ_i a_i = 1.

Angular Velocity ACF and Friction:
----------------------------------
For rotational diffusion with friction γ:

    C_ω(τ) = <ω(t) · ω(t+τ)> / <ω²>

In overdamped limit:
    C_ω(τ) ≈ exp(-γ τ / I)

where I is the moment of inertia.

Friction coefficient:
    γ = -I / τ_c

Green-Kubo Relation:
--------------------
Friction can also be extracted via:

    γ = ∫_0^∞ <F(t) · F(0)> dt

where F(t) is the random force.

Angular Momentum ACF:
---------------------
For angular momentum L(t):

    C_L(τ) = <L(t) · L(t+τ)> / <L²>

This reveals:
- Correlation time τ_L
- Persistence of angular momentum
- Transition between ballistic (short τ) and diffusive (long τ) regimes

Physical Interpretation:
------------------------

Well-bound transmembrane protein:
- Long correlation time τ_c (slow decay)
- High friction γ (strong membrane coupling)
- C_ω(τ) decays slowly

Poorly-bound peripheral protein:
- Short correlation time τ_c (fast decay)
- Low friction γ (weak membrane coupling)
- C_ω(τ) decays rapidly

References:
-----------
- Kubo (1966). Rep. Prog. Phys. 29, 255.
- Hansen & McDonald (2013). Theory of Simple Liquids (4th ed.), Chapter 7.
- Zwanzig (2001). Nonequilibrium Statistical Mechanics, Chapter 9.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.optimize import curve_fit
from scipy import signal
import warnings


def autocorrelation_function(
    data: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True,
    method: str = 'fft'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function C(τ) = <A(t) · A(t+τ)>.

    Args:
        data: Time series data, shape (N_frames,) or (N_frames, N_components)
        max_lag: Maximum lag time (frames). If None, use N_frames // 2
        normalize: If True, normalize such that C(0) = 1
        method: Computation method:
            - 'fft': Fast Fourier Transform (O(N log N), recommended)
            - 'direct': Direct computation (O(N²), slow but exact)

    Returns:
        Tuple of:
            - lags: Lag times (frames), shape (max_lag+1,)
            - acf: Autocorrelation function, shape (max_lag+1,)

    Mathematical Formula:
        C(τ) = <A(t) · A(t+τ)> / <A(t)²>  (normalized)

    Example:
        >>> # Exponentially decaying process
        >>> t = np.arange(1000)
        >>> data = np.exp(-t/100) * np.random.randn(1000)
        >>> lags, acf = autocorrelation_function(data)
        >>> print(f"ACF(0) = {acf[0]:.2f}")  # Should be 1.0
        1.00
    """
    if data.ndim == 1:
        # 1D time series
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        raise ValueError(f"data must be 1D or 2D, got shape {data.shape}")

    N_frames, N_components = data.shape

    if max_lag is None:
        max_lag = N_frames // 2

    if max_lag >= N_frames:
        raise ValueError(f"max_lag ({max_lag}) must be less than N_frames ({N_frames})")

    lags = np.arange(max_lag + 1)
    acf = np.zeros(max_lag + 1)

    if method == 'fft':
        # FFT method (fast)
        # ACF = IFFT(|FFT(data)|²) / N
        for i in range(N_components):
            # Zero-pad to 2*N for proper correlation
            padded = np.concatenate([data[:, i], np.zeros(N_frames)])

            # FFT
            fft_data = np.fft.fft(padded)

            # Power spectrum
            power = np.abs(fft_data)**2

            # IFFT to get ACF
            acf_full = np.fft.ifft(power).real

            # Take only valid lags
            acf += acf_full[:max_lag+1]

        # Normalize by number of pairs at each lag
        norm = np.arange(N_frames, N_frames - max_lag - 1, -1)
        acf /= norm

    elif method == 'direct':
        # Direct computation (slow but exact)
        for lag in lags:
            if lag == 0:
                # C(0) = <A²>
                acf[lag] = np.mean(np.sum(data**2, axis=1))
            else:
                # C(τ) = <A(t) · A(t+τ)>
                dot_products = np.sum(data[:-lag] * data[lag:], axis=1)
                acf[lag] = np.mean(dot_products)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize such that C(0) = 1
    if normalize and acf[0] != 0:
        acf /= acf[0]

    return lags, acf


def cross_correlation_function(
    data1: np.ndarray,
    data2: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation function C_{AB}(τ) = <A(t) · B(t+τ)>.

    Args:
        data1: First time series A(t), shape (N_frames,) or (N_frames, N_comp)
        data2: Second time series B(t), same shape as data1
        max_lag: Maximum lag time (frames)
        normalize: If True, normalize by sqrt(<A²><B²>)

    Returns:
        Tuple of:
            - lags: Lag times (frames), shape (max_lag+1,)
            - ccf: Cross-correlation function, shape (max_lag+1,)

    Example:
        >>> A = np.sin(np.arange(1000) * 0.1)
        >>> B = np.sin(np.arange(1000) * 0.1 + 0.5)  # Phase-shifted
        >>> lags, ccf = cross_correlation_function(A, B)
        >>> print(f"CCF(0) = {ccf[0]:.2f}")
    """
    if data1.shape != data2.shape:
        raise ValueError(f"data1 and data2 must have same shape")

    if data1.ndim == 1:
        data1 = data1.reshape(-1, 1)
        data2 = data2.reshape(-1, 1)

    N_frames, N_components = data1.shape

    if max_lag is None:
        max_lag = N_frames // 2

    lags = np.arange(max_lag + 1)
    ccf = np.zeros(max_lag + 1)

    # Direct computation
    for lag in lags:
        if lag == 0:
            dot_products = np.sum(data1 * data2, axis=1)
        else:
            dot_products = np.sum(data1[:-lag] * data2[lag:], axis=1)
        ccf[lag] = np.mean(dot_products)

    # Normalize
    if normalize:
        norm1 = np.sqrt(np.mean(np.sum(data1**2, axis=1)))
        norm2 = np.sqrt(np.mean(np.sum(data2**2, axis=1)))
        if norm1 != 0 and norm2 != 0:
            ccf /= (norm1 * norm2)

    return lags, ccf


def fit_exponential_decay(
    lags: np.ndarray,
    acf: np.ndarray,
    n_exponentials: int = 1,
    dt: float = 1.0
) -> Dict[str, any]:
    """
    Fit autocorrelation function to exponential decay model.

    Args:
        lags: Lag times (frames), shape (N_lags,)
        acf: Autocorrelation function, shape (N_lags,)
        n_exponentials: Number of exponential components (1, 2, or 3)
        dt: Time step per frame in ps

    Returns:
        Dictionary with:
            - 'tau': Correlation times τ_i in ps, shape (n_exponentials,)
            - 'amplitudes': Amplitudes a_i, shape (n_exponentials,)
            - 'fit': Fitted ACF values, shape (N_lags,)
            - 'residuals': Residuals (acf - fit)
            - 'r_squared': Coefficient of determination R²

    Model:
        Single exponential: C(τ) = exp(-τ/τ_c)
        Multi-exponential: C(τ) = Σ_i a_i exp(-τ/τ_i)

    Example:
        >>> lags = np.arange(100)
        >>> acf = np.exp(-lags/20.0)
        >>> result = fit_exponential_decay(lags, acf, n_exponentials=1)
        >>> print(f"τ_c = {result['tau'][0]:.1f} frames")
        20.0
    """
    if len(lags) != len(acf):
        raise ValueError("lags and acf must have same length")

    if n_exponentials not in [1, 2, 3]:
        raise ValueError("n_exponentials must be 1, 2, or 3")

    # Convert lags to time
    times = lags * dt

    # Define fit function
    if n_exponentials == 1:
        def fit_func(t, tau):
            return np.exp(-t / tau)

        # Initial guess
        p0 = [times[-1] / 3]  # Guess τ as 1/3 of total time

        # Bounds: τ > 0
        bounds = ([0], [np.inf])

    elif n_exponentials == 2:
        def fit_func(t, a1, tau1, tau2):
            a2 = 1 - a1
            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

        p0 = [0.5, times[-1] / 5, times[-1] / 2]
        bounds = ([0, 0, 0], [1, np.inf, np.inf])

    elif n_exponentials == 3:
        def fit_func(t, a1, a2, tau1, tau2, tau3):
            a3 = 1 - a1 - a2
            return (a1 * np.exp(-t / tau1) +
                    a2 * np.exp(-t / tau2) +
                    a3 * np.exp(-t / tau3))

        p0 = [0.33, 0.33, times[-1] / 10, times[-1] / 3, times[-1]]
        bounds = ([0, 0, 0, 0, 0], [1, 1, np.inf, np.inf, np.inf])

    else:
        raise ValueError(f"n_exponentials must be 1, 2, or 3")

    # Fit
    try:
        popt, pcov = curve_fit(fit_func, times, acf, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        warnings.warn(f"Fit failed: {e}. Returning initial guess.")
        popt = np.array(p0)

    # Extract parameters
    if n_exponentials == 1:
        tau = np.array([popt[0]])
        amplitudes = np.array([1.0])
    elif n_exponentials == 2:
        a1 = popt[0]
        tau = np.array([popt[1], popt[2]])
        amplitudes = np.array([a1, 1 - a1])
    elif n_exponentials == 3:
        a1, a2 = popt[0], popt[1]
        tau = np.array([popt[2], popt[3], popt[4]])
        amplitudes = np.array([a1, a2, 1 - a1 - a2])

    # Compute fit and residuals
    fit = fit_func(times, *popt)
    residuals = acf - fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((acf - np.mean(acf))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        'tau': tau,
        'amplitudes': amplitudes,
        'fit': fit,
        'residuals': residuals,
        'r_squared': r_squared,
        'times': times
    }


def extract_correlation_time(
    lags: np.ndarray,
    acf: np.ndarray,
    dt: float = 1.0,
    method: str = 'integral'
) -> float:
    """
    Extract correlation time τ_c from autocorrelation function.

    Args:
        lags: Lag times (frames), shape (N_lags,)
        acf: Autocorrelation function, shape (N_lags,)
        dt: Time step per frame in ps
        method: Extraction method:
            - 'integral': τ_c = ∫ C(τ) dτ (integrated ACF)
            - 'exponential': Fit exp(-τ/τ_c) and extract τ_c
            - 'decay_to_e': Time for C(τ) to decay to 1/e

    Returns:
        Correlation time τ_c in ps

    Mathematical Formulas:
        Integral method: τ_c = ∫_0^∞ C(τ) dτ
        Exponential fit: C(τ) = exp(-τ/τ_c)
        Decay to 1/e: τ_c such that C(τ_c) = 1/e ≈ 0.368

    Example:
        >>> lags = np.arange(100)
        >>> acf = np.exp(-lags/20.0)
        >>> tau_c = extract_correlation_time(lags, acf, method='exponential')
        >>> print(f"τ_c = {tau_c:.1f}")
        20.0
    """
    times = lags * dt

    if method == 'integral':
        # τ_c = ∫ C(τ) dτ
        tau_c = np.trapz(acf, times)

    elif method == 'exponential':
        # Fit exp(-τ/τ_c)
        fit_result = fit_exponential_decay(lags, acf, n_exponentials=1, dt=dt)
        tau_c = fit_result['tau'][0]

    elif method == 'decay_to_e':
        # Find τ where C(τ) = 1/e
        target = 1.0 / np.e
        idx = np.argmin(np.abs(acf - target))
        tau_c = times[idx]

    else:
        raise ValueError(f"Unknown method: {method}")

    return tau_c


def angular_velocity_acf(
    omega_trajectory: np.ndarray,
    times: np.ndarray,
    max_lag: Optional[int] = None,
    component: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compute angular velocity autocorrelation function.

    Args:
        omega_trajectory: Angular velocity, shape (N_frames, 3) in rad/ps
        times: Time points, shape (N_frames,) in ps
        max_lag: Maximum lag (frames). If None, use N_frames // 2
        component: Which component to analyze:
            - None: Total |ω|²
            - 'x', 'y', 'z': Individual components
            - 'parallel', 'perp': Parallel/perpendicular (requires decomposition)

    Returns:
        Dictionary with:
            - 'lags': Lag times (frames)
            - 'acf': Autocorrelation function C_ω(τ)
            - 'tau_c': Correlation time (ps)
            - 'dt': Time step (ps)

    Physical Interpretation:
        - Long τ_c: Slow rotational relaxation (well-bound)
        - Short τ_c: Fast rotational relaxation (poorly-bound)

    Example:
        >>> omega = np.random.randn(1000, 3) * 0.1  # rad/ps
        >>> times = np.arange(1000) * 0.001  # ps
        >>> result = angular_velocity_acf(omega, times)
        >>> print(f"τ_c = {result['tau_c']:.3f} ps")
    """
    N_frames = len(omega_trajectory)
    dt = np.mean(np.diff(times))

    if component is None:
        # Total magnitude squared
        data = np.linalg.norm(omega_trajectory, axis=1)
    elif component in ['x', 'y', 'z']:
        idx = {'x': 0, 'y': 1, 'z': 2}[component]
        data = omega_trajectory[:, idx]
    else:
        raise ValueError(f"Unknown component: {component}")

    # Compute ACF
    lags, acf = autocorrelation_function(data, max_lag=max_lag, normalize=True)

    # Extract correlation time
    tau_c = extract_correlation_time(lags, acf, dt=dt, method='integral')

    return {
        'lags': lags,
        'acf': acf,
        'tau_c': tau_c,
        'dt': dt,
        'times': lags * dt
    }


def angular_momentum_acf(
    L_trajectory: np.ndarray,
    times: np.ndarray,
    max_lag: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute angular momentum autocorrelation function.

    Args:
        L_trajectory: Angular momentum, shape (N_frames, 3) in amu·Å²/ps
        times: Time points, shape (N_frames,) in ps
        max_lag: Maximum lag (frames)

    Returns:
        Dictionary with:
            - 'lags': Lag times (frames)
            - 'acf': Autocorrelation function C_L(τ)
            - 'tau_c': Correlation time (ps)

    Example:
        >>> L = np.random.randn(1000, 3) * 100
        >>> times = np.arange(1000) * 0.001
        >>> result = angular_momentum_acf(L, times)
        >>> print(f"L correlation time: {result['tau_c']:.3f} ps")
    """
    dt = np.mean(np.diff(times))

    # Compute ACF for total |L|
    L_magnitude = np.linalg.norm(L_trajectory, axis=1)
    lags, acf = autocorrelation_function(L_magnitude, max_lag=max_lag, normalize=True)

    # Extract correlation time
    tau_c = extract_correlation_time(lags, acf, dt=dt, method='integral')

    return {
        'lags': lags,
        'acf': acf,
        'tau_c': tau_c,
        'dt': dt,
        'times': lags * dt
    }


if __name__ == '__main__':
    print("Correlations Module - Example Usage\n")

    # Example 1: Exponential decay ACF
    print("Example 1: Exponential decay autocorrelation")
    np.random.seed(42)

    # Generate exponentially decaying process
    N = 1000
    tau_true = 20.0  # frames
    lags_true = np.arange(N)
    acf_true = np.exp(-lags_true / tau_true)

    # Add noise
    acf_noisy = acf_true + np.random.randn(N) * 0.05

    # Fit
    fit_result = fit_exponential_decay(lags_true, acf_noisy, n_exponentials=1, dt=1.0)
    print(f"True τ: {tau_true:.1f} frames")
    print(f"Fitted τ: {fit_result['tau'][0]:.1f} frames")
    print(f"R²: {fit_result['r_squared']:.4f}")

    # Example 2: Angular velocity ACF
    print("\n" + "="*60)
    print("Example 2: Angular velocity autocorrelation")

    N_frames = 1000
    omega_traj = np.random.randn(N_frames, 3) * 0.1  # rad/ps
    times = np.arange(N_frames) * 0.001  # ps

    result_omega = angular_velocity_acf(omega_traj, times, max_lag=200)
    print(f"Angular velocity correlation time: {result_omega['tau_c']:.3f} ps")
    print(f"ACF(0) = {result_omega['acf'][0]:.2f}")
    print(f"ACF decays to {result_omega['acf'][-1]:.2f} at lag {result_omega['lags'][-1]}")

    # Example 3: Multi-exponential fit
    print("\n" + "="*60)
    print("Example 3: Multi-exponential decay")

    # Generate two-component decay
    tau1, tau2 = 5.0, 50.0
    a1, a2 = 0.3, 0.7
    lags_multi = np.arange(200)
    acf_multi = a1 * np.exp(-lags_multi / tau1) + a2 * np.exp(-lags_multi / tau2)

    fit_multi = fit_exponential_decay(lags_multi, acf_multi, n_exponentials=2, dt=1.0)
    print(f"True τ: [{tau1:.1f}, {tau2:.1f}]")
    print(f"Fitted τ: {fit_multi['tau']}")
    print(f"True amplitudes: [{a1:.2f}, {a2:.2f}]")
    print(f"Fitted amplitudes: {fit_multi['amplitudes']}")
    print(f"R²: {fit_multi['r_squared']:.4f}")

    # Example 4: Extract correlation time (different methods)
    print("\n" + "="*60)
    print("Example 4: Correlation time extraction methods")

    lags_test = np.arange(100)
    acf_test = np.exp(-lags_test / 25.0)

    tau_integral = extract_correlation_time(lags_test, acf_test, method='integral')
    tau_exp = extract_correlation_time(lags_test, acf_test, method='exponential')
    tau_decay = extract_correlation_time(lags_test, acf_test, method='decay_to_e')

    print(f"True τ: 25.0")
    print(f"Integral method: {tau_integral:.1f}")
    print(f"Exponential fit: {tau_exp:.1f}")
    print(f"Decay to 1/e: {tau_decay:.1f}")
