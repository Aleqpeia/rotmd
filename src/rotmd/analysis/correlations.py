"""Time correlation functions and correlation-time extraction for angular dynamics.

Friction coefficients and rotational relaxation times (see :mod:`rotmd.analysis.friction`)
are read off the *decay rate* of an autocorrelation function, not measured
directly, so getting the ACF right is the first link in that chain: a
well-bound, strongly membrane-coupled protein shows a slowly-decaying
:math:`C_\\omega(\\tau)` and a long correlation time; a poorly-bound one decays
fast. Everything here exists to turn a raw angular-velocity or -momentum
trajectory into that single number, :math:`\\tau_c`, robustly.

The FFT method is the default autocorrelation estimator because the direct
:math:`O(N^2)` sum becomes the bottleneck past a few thousand frames; 'direct'
is kept only to sanity-check the FFT path on short series.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


def autocorrelation_function(
    data: np.ndarray,
    max_lag: int | None = None,
    normalize: bool = True,
    method: str = "fft",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute :math:`C(\\tau) = \\langle A(t)\\cdot A(t+\\tau)\\rangle`, normalized so ``C(0) = 1``.

    ``data`` is ``(n_frames,)`` or ``(n_frames, n_components)`` (component
    dot-products are summed, e.g. for a vector observable like :math:`\\omega`).
    ``method='fft'`` computes the ACF as ``ifft(|fft(data)|**2)`` in
    :math:`O(N\\log N)`; ``method='direct'`` is the exact :math:`O(N^2)` sum,
    useful only as a check.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        raise ValueError(f"data must be 1D or 2D, got shape {data.shape}")

    n_frames, n_components = data.shape

    if max_lag is None:
        max_lag = n_frames // 2

    if max_lag >= n_frames:
        raise ValueError(f"max_lag ({max_lag}) must be less than n_frames ({n_frames})")

    lags = np.arange(max_lag + 1)
    acf = np.zeros(max_lag + 1)

    if method == "fft":
        for i in range(n_components):
            # Zero-pad to 2N so the circular FFT convolution doesn't wrap
            # around and mix the end of the series into the start.
            padded = np.concatenate([data[:, i], np.zeros(n_frames)])
            power = np.abs(np.fft.fft(padded)) ** 2
            acf_full = np.fft.ifft(power).real
            acf += acf_full[: max_lag + 1]

        # Each lag averages over fewer pairs than lag 0; divide by the
        # count of pairs actually summed, not by n_frames uniformly.
        norm = np.arange(n_frames, n_frames - max_lag - 1, -1)
        acf /= norm

    elif method == "direct":
        for lag in lags:
            if lag == 0:
                acf[lag] = np.mean(np.sum(data**2, axis=1))
            else:
                dot_products = np.sum(data[:-lag] * data[lag:], axis=1)
                acf[lag] = np.mean(dot_products)

    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize and acf[0] != 0:
        acf /= acf[0]

    return lags, acf


def cross_correlation_function(
    data1: np.ndarray,
    data2: np.ndarray,
    max_lag: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute :math:`C_{AB}(\\tau) = \\langle A(t)\\cdot B(t+\\tau)\\rangle`, normalized by :math:`\\sqrt{\\langle A^2\\rangle\\langle B^2\\rangle}`.

    ``data1``/``data2`` must share shape ``(n_frames,)`` or ``(n_frames, n_components)``.
    """
    if data1.shape != data2.shape:
        raise ValueError("data1 and data2 must have same shape")

    if data1.ndim == 1:
        data1 = data1.reshape(-1, 1)
        data2 = data2.reshape(-1, 1)

    n_frames, _ = data1.shape

    if max_lag is None:
        max_lag = n_frames // 2

    lags = np.arange(max_lag + 1)
    ccf = np.zeros(max_lag + 1)

    for lag in lags:
        if lag == 0:
            dot_products = np.sum(data1 * data2, axis=1)
        else:
            dot_products = np.sum(data1[:-lag] * data2[lag:], axis=1)
        ccf[lag] = np.mean(dot_products)

    if normalize:
        norm1 = np.sqrt(np.mean(np.sum(data1**2, axis=1)))
        norm2 = np.sqrt(np.mean(np.sum(data2**2, axis=1)))
        if norm1 != 0 and norm2 != 0:
            ccf /= norm1 * norm2

    return lags, ccf


def fit_exponential_decay(
    lags: np.ndarray,
    acf: np.ndarray,
    n_exponentials: int = 1,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Fit the ACF to a sum of ``n_exponentials`` decaying exponentials, ``sum_i a_i exp(-t/tau_i)``.

    A single exponential is the diffusive-limit model; 2 or 3 components let a
    fast (local) and slow (global reorientation) relaxation be told apart when
    a single tau_c would average them into a meaningless number. Amplitudes are
    constrained to sum to 1 and ``tau_i > 0``.
    """
    if len(lags) != len(acf):
        raise ValueError("lags and acf must have same length")

    if n_exponentials not in (1, 2, 3):
        raise ValueError("n_exponentials must be 1, 2, or 3")

    times = lags * dt

    if n_exponentials == 1:

        def fit_func(t, tau):
            return np.exp(-t / tau)

        p0 = [times[-1] / 3]
        bounds = ([0], [np.inf])

    elif n_exponentials == 2:

        def fit_func(t, a1, tau1, tau2):
            a2 = 1 - a1
            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

        p0 = [0.5, times[-1] / 5, times[-1] / 2]
        bounds = ([0, 0, 0], [1, np.inf, np.inf])

    else:

        def fit_func(t, a1, a2, tau1, tau2, tau3):
            a3 = 1 - a1 - a2
            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + a3 * np.exp(-t / tau3)

        p0 = [0.33, 0.33, times[-1] / 10, times[-1] / 3, times[-1]]
        bounds = ([0, 0, 0, 0, 0], [1, 1, np.inf, np.inf, np.inf])

    try:
        popt, _ = curve_fit(fit_func, times, acf, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        warnings.warn(f"Fit failed: {e}. Returning initial guess.")
        popt = np.array(p0)

    if n_exponentials == 1:
        tau = np.array([popt[0]])
        amplitudes = np.array([1.0])
    elif n_exponentials == 2:
        a1 = popt[0]
        tau = np.array([popt[1], popt[2]])
        amplitudes = np.array([a1, 1 - a1])
    else:
        a1, a2 = popt[0], popt[1]
        tau = np.array([popt[2], popt[3], popt[4]])
        amplitudes = np.array([a1, a2, 1 - a1 - a2])

    fit = fit_func(times, *popt)
    residuals = acf - fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((acf - np.mean(acf)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "tau": tau,
        "amplitudes": amplitudes,
        "fit": fit,
        "residuals": residuals,
        "r_squared": r_squared,
        "times": times,
    }


def extract_correlation_time(
    lags: np.ndarray,
    acf: np.ndarray,
    dt: float = 1.0,
    method: str = "integral",
) -> float:
    """Extract the correlation time tau_c from an ACF.

    ``method='integral'`` integrates the ACF (exact for any decay shape,
    including multi-exponential); ``'exponential'`` fits a single exponential
    and returns its tau; ``'decay_to_e'`` reads off the lag where ``C(tau) = 1/e``,
    the cheapest and noisiest of the three.
    """
    times = lags * dt

    if method == "integral":
        # tau_c = integral_0^T C(tau) dtau, truncated at the first non-positive
        # lag. The truncation is not optional. Beyond a few correlation times
        # the ACF is pure sampling noise fluctuating about zero, and
        # integrating all N/2 lags accumulates it: on an OU series with tau=25
        # and 20000 samples the untruncated integral returns 88.5 (tail mean
        # 0.0099 over 10^4 lags contributes ~99), and merely reseeding gives
        # -0.1. Cutting at the first zero crossing is the standard remedy and
        # is what `rotmd.analysis.equilibration.statistical_inefficiency`
        # already does.
        acf = np.asarray(acf, dtype=np.float64)
        nonpos = np.flatnonzero(acf <= 0.0)
        cut = int(nonpos[0]) if nonpos.size else acf.size
        tau_c = np.trapezoid(acf[:cut], times[:cut]) if cut > 1 else 0.0

    elif method == "exponential":
        fit_result = fit_exponential_decay(lags, acf, n_exponentials=1, dt=dt)
        tau_c = fit_result["tau"][0]

    elif method == "decay_to_e":
        target = 1.0 / np.e
        idx = np.argmin(np.abs(acf - target))
        tau_c = times[idx]

    else:
        raise ValueError(f"Unknown method: {method}")

    return tau_c


def angular_velocity_acf(
    omega_trajectory: np.ndarray,
    times: np.ndarray,
    max_lag: int | None = None,
    component: str | None = None,
) -> dict[str, np.ndarray]:
    """ACF and correlation time of the angular velocity, ``omega_trajectory`` ``(n_frames, 3)`` rad/ps.

    ``component=None`` uses ``|omega|``; ``'x'``/``'y'``/``'z'`` isolates one
    lab-frame axis. A long tau_c here is the rotational-diffusion signature of
    strong membrane coupling; a short one, weak coupling and near-free rotation.
    """
    dt = np.mean(np.diff(times))

    if component is None:
        data = np.linalg.norm(omega_trajectory, axis=1)
    elif component in ("x", "y", "z"):
        idx = {"x": 0, "y": 1, "z": 2}[component]
        data = omega_trajectory[:, idx]
    else:
        raise ValueError(f"Unknown component: {component}")

    lags, acf = autocorrelation_function(data, max_lag=max_lag, normalize=True)
    tau_c = extract_correlation_time(lags, acf, dt=dt, method="integral")

    return {
        "lags": lags,
        "acf": acf,
        "tau_c": tau_c,
        "dt": dt,
        "times": lags * dt,
    }


def angular_momentum_acf(
    l_trajectory: np.ndarray,
    times: np.ndarray,
    max_lag: int | None = None,
) -> dict[str, np.ndarray]:
    """ACF and correlation time of ``|L|`` for the angular momentum trajectory, ``(n_frames, 3)`` amu*A^2/ps."""
    dt = np.mean(np.diff(times))

    l_magnitude = np.linalg.norm(l_trajectory, axis=1)
    lags, acf = autocorrelation_function(l_magnitude, max_lag=max_lag, normalize=True)
    tau_c = extract_correlation_time(lags, acf, dt=dt, method="integral")

    return {
        "lags": lags,
        "acf": acf,
        "tau_c": tau_c,
        "dt": dt,
        "times": lags * dt,
    }


if __name__ == "__main__":
    print("Correlations Module - Example Usage\n")

    print("Example 1: Exponential decay autocorrelation")
    np.random.seed(42)

    n = 1000
    tau_true = 20.0
    lags_true = np.arange(n)
    acf_true = np.exp(-lags_true / tau_true)
    acf_noisy = acf_true + np.random.randn(n) * 0.05

    fit_result = fit_exponential_decay(lags_true, acf_noisy, n_exponentials=1, dt=1.0)
    print(f"True tau: {tau_true:.1f} frames")
    print(f"Fitted tau: {fit_result['tau'][0]:.1f} frames")
    print(f"R^2: {fit_result['r_squared']:.4f}")

    print("\n" + "=" * 60)
    print("Example 2: Angular velocity autocorrelation")

    n_frames = 1000
    omega_traj = np.random.randn(n_frames, 3) * 0.1
    times = np.arange(n_frames) * 0.001

    result_omega = angular_velocity_acf(omega_traj, times, max_lag=200)
    print(f"Angular velocity correlation time: {result_omega['tau_c']:.3f} ps")
    print(f"ACF(0) = {result_omega['acf'][0]:.2f}")
    print(f"ACF decays to {result_omega['acf'][-1]:.2f} at lag {result_omega['lags'][-1]}")

    print("\n" + "=" * 60)
    print("Example 3: Multi-exponential decay")

    tau1, tau2 = 5.0, 50.0
    a1, a2 = 0.3, 0.7
    lags_multi = np.arange(200)
    acf_multi = a1 * np.exp(-lags_multi / tau1) + a2 * np.exp(-lags_multi / tau2)

    fit_multi = fit_exponential_decay(lags_multi, acf_multi, n_exponentials=2, dt=1.0)
    print(f"True tau: [{tau1:.1f}, {tau2:.1f}]")
    print(f"Fitted tau: {fit_multi['tau']}")
    print(f"True amplitudes: [{a1:.2f}, {a2:.2f}]")
    print(f"Fitted amplitudes: {fit_multi['amplitudes']}")
    print(f"R^2: {fit_multi['r_squared']:.4f}")

    print("\n" + "=" * 60)
    print("Example 4: Correlation time extraction methods")

    lags_test = np.arange(100)
    acf_test = np.exp(-lags_test / 25.0)

    tau_integral = extract_correlation_time(lags_test, acf_test, method="integral")
    tau_exp = extract_correlation_time(lags_test, acf_test, method="exponential")
    tau_decay = extract_correlation_time(lags_test, acf_test, method="decay_to_e")

    print("True tau: 25.0")
    print(f"Integral method: {tau_integral:.1f}")
    print(f"Exponential fit: {tau_exp:.1f}")
    print(f"Decay to 1/e: {tau_decay:.1f}")
