"""
Response Theory for Protein Orientation Dynamics

Implements linear and non-linear response theory for rotational diffusion.
Based on fluctuation-dissipation theorem and Kubo formalism.

References:
- McQuarrie, Statistical Mechanics (Ch. 21)
- Kubo, Statistical Mechanics (1965)
- Berne & Pecora, Dynamic Light Scattering (1976)

Author: Mykyta Bobylyow
Date: 2025-12-21
"""

import numpy as np
from typing import Tuple, Optional
from scipy.integrate import simps
from scipy.fft import fft, fftfreq


def rotational_correlation_function(
    theta: np.ndarray,
    times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientational correlation function C(t) = ⟨cos θ(t) cos θ(0)⟩.

    For rotational diffusion: C(t) = (1/3)exp(-2D_R·t)

    Parameters
    ----------
    theta : ndarray
        Tilt angle trajectory (rad)
    times : ndarray
        Time points (ps)

    Returns
    -------
    lags : ndarray
        Time lags (ps)
    C : ndarray
        Correlation function

    Notes
    -----
    Problem II.2 from homework:
    - For a rigid rotator with diffusion constant D_R
    - C(t) decays exponentially with rate 2D_R
    - Extracting D_R: fit C(t) = C₀·exp(-2D_R·t)
    """
    cos_theta = np.cos(theta)

    # Autocorrelation of cos(θ)
    n = len(cos_theta)
    mean = np.mean(cos_theta)
    variance = np.var(cos_theta)

    C = np.correlate(cos_theta - mean, cos_theta - mean, mode='full')
    C = C[n-1:] / (variance * n)  # Normalize

    # Time lags
    dt = np.mean(np.diff(times))
    lags = np.arange(len(C)) * dt

    return lags, C


def extract_rotational_diffusion(
    lags: np.ndarray,
    C: np.ndarray,
    max_lag: Optional[float] = None
) -> Tuple[float, float]:
    """
    Extract rotational diffusion constant from C(t) = (1/3)exp(-2D_R·t).

    Parameters
    ----------
    lags : ndarray
        Time lags (ps)
    C : ndarray
        Correlation function
    max_lag : float, optional
        Maximum lag for fitting (ps)

    Returns
    -------
    D_R : float
        Rotational diffusion constant (rad²/ps)
    tau_R : float
        Rotational correlation time = 1/(2D_R) (ps)

    Notes
    -----
    Problem II.2: Average orientation decays as u̇ = -2D_R·u
    Solution: u(t) = u₀·exp(-2D_R·t)
    """
    if max_lag is None:
        max_lag = lags[-1] / 2

    mask = lags <= max_lag
    lags_fit = lags[mask]
    C_fit = C[mask]

    # Fit exponential: C(t) = C₀·exp(-t/τ)
    # Take log: ln(C) = ln(C₀) - t/τ
    C_fit_positive = np.abs(C_fit) + 1e-10
    log_C = np.log(C_fit_positive)

    # Linear fit
    coeffs = np.polyfit(lags_fit, log_C, deg=1)
    decay_rate = -coeffs[0]  # 1/τ

    # Decay rate = 2D_R → D_R = decay_rate/2
    D_R = decay_rate / 2
    tau_R = 1 / decay_rate

    return D_R, tau_R


def response_function_from_correlation(
    C: np.ndarray,
    dt: float,
    temperature: float = 310.0
) -> np.ndarray:
    """
    Compute response function K(t) = -βĊ(t) from correlation function.

    Fluctuation-dissipation theorem (Problem III.6):
    K(t) = -(1/kT) · dC/dt

    Parameters
    ----------
    C : ndarray
        Correlation function
    dt : float
        Time step (ps)
    temperature : float
        Temperature (K)

    Returns
    -------
    K : ndarray
        Response function

    Notes
    -----
    Classical limit of quantum result (Problem IV.4):
    K(t) = -βĊ(t) where β = 1/(k_B·T)

    Physical meaning:
    - K(t) describes response to perturbation at time 0
    - Related to friction via K̃(ω) = -iω·χ″(ω)
    """
    kB = 0.001987  # kcal/(mol·K)
    beta = 1 / (kB * temperature)

    # Numerical derivative
    C_dot = np.gradient(C, dt)

    # K(t) = -β·Ċ(t)
    K = -beta * C_dot

    return K


def susceptibility_from_correlation(
    C: np.ndarray,
    dt: float,
    temperature: float = 310.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frequency-dependent susceptibility χ(ω) from C(t).

    Fourier transform of fluctuation-dissipation theorem:
    χ″(ω) = (βω/2) · C̃(ω)

    Parameters
    ----------
    C : ndarray
        Correlation function
    dt : float
        Time step (ps)
    temperature : float
        Temperature (K)

    Returns
    -------
    omega : ndarray
        Angular frequencies (rad/ps)
    chi_real : ndarray
        Real part of susceptibility (storage modulus)
    chi_imag : ndarray
        Imaginary part (loss modulus)

    Notes
    -----
    Problem III.5: Verify χ″(ω) = (βω/2)·C̃(ω)

    Kramers-Kronig relations (Problem I.6):
    - χ'(ω) and χ″(ω) are Hilbert transforms
    - Causality constraint on response
    """
    kB = 0.001987  # kcal/(mol·K)
    beta = 1 / (kB * temperature)

    # Fourier transform of C(t)
    n = len(C)
    C_fft = fft(C)
    omega = 2 * np.pi * fftfreq(n, dt)

    # Imaginary part: χ″(ω) = (βω/2)·C̃(ω)
    chi_imag = (beta * omega / 2) * C_fft.real

    # Real part via Kramers-Kronig (Hilbert transform)
    # For simplicity, use approximate relation
    chi_real = -(beta / 2) * C_fft.imag / (omega + 1e-10)

    # Return positive frequencies only
    pos_freq = omega >= 0
    omega = omega[pos_freq]
    chi_real = chi_real[pos_freq]
    chi_imag = chi_imag[pos_freq]

    return omega, chi_real, chi_imag


def absorption_rate_monochromatic(
    omega_drive: float,
    F0: float,
    chi_imag: np.ndarray,
    omega: np.ndarray
) -> float:
    """
    Calculate average absorption rate for monochromatic force F(t) = F₀·cos(ω₀t).

    Power absorption: P = (1/2)·F₀²·χ″(ω₀)

    Parameters
    ----------
    omega_drive : float
        Driving frequency (rad/ps)
    F0 : float
        Force amplitude
    chi_imag : ndarray
        Imaginary susceptibility
    omega : ndarray
        Frequency array

    Returns
    -------
    P : float
        Average power absorption

    Notes
    -----
    Problem II.5: Calculate response to monochromatic force
    H = -F(t)·cos θ(t), F(t) = F₀·cos(ω₀t)

    Result: ⟨Ė⟩ = (1/2)·F₀²·χ″(ω₀)
    """
    # Interpolate χ″ at driving frequency
    chi_imag_interp = np.interp(omega_drive, omega, chi_imag)

    # Power = (1/2)·F₀²·χ″(ω₀)
    P = 0.5 * F0**2 * chi_imag_interp

    return P


def underdamped_correlation_function(
    times: np.ndarray,
    gamma: float,
    omega0: float,
    temperature: float = 310.0,
    mass: float = 1.0
) -> np.ndarray:
    """
    Analytical correlation function for underdamped harmonic oscillator.

    C(t) = C₀·exp(-γt/2)·cos(Ωt)
    where Ω = √(ω₀² - γ²/4)

    Parameters
    ----------
    times : ndarray
        Time points (ps)
    gamma : float
        Damping coefficient (1/ps)
    omega0 : float
        Natural frequency (rad/ps)
    temperature : float
        Temperature (K)
    mass : float
        Effective mass (amu)

    Returns
    -------
    C : ndarray
        Position autocorrelation function

    Notes
    -----
    Problem III.2: Solve damped oscillator equation
    mC̈ + mγĊ + mω₀²C = 0
    C(0) = (βmω₀²)⁻¹  # Equipartition

    Three regimes:
    1. Underdamped (γ < 2ω₀): Oscillatory decay → LIMIT CYCLE
    2. Critically damped (γ = 2ω₀): Fastest decay without oscillation
    3. Overdamped (γ > 2ω₀): Slow exponential decay
    """
    kB = 0.001987  # kcal/(mol·K)
    beta = 1 / (kB * temperature)

    # Initial condition: C(0) = 1/(β·m·ω₀²)
    C0 = 1 / (beta * mass * omega0**2)

    # Discriminant
    discriminant = omega0**2 - (gamma/2)**2

    if discriminant > 0:
        # Underdamped: oscillatory
        Omega = np.sqrt(discriminant)
        C = C0 * np.exp(-gamma * times / 2) * np.cos(Omega * times)

    elif discriminant == 0:
        # Critically damped
        C = C0 * (1 + gamma * times / 2) * np.exp(-gamma * times / 2)

    else:
        # Overdamped
        lambda1 = -gamma/2 + np.sqrt((gamma/2)**2 - omega0**2)
        lambda2 = -gamma/2 - np.sqrt((gamma/2)**2 - omega0**2)
        C = C0 * ((lambda2 * np.exp(lambda1 * times) - lambda1 * np.exp(lambda2 * times))
                  / (lambda2 - lambda1))

    return C


def classify_oscillator_regime(gamma: float, omega0: float) -> str:
    """
    Classify damping regime.

    Parameters
    ----------
    gamma : float
        Damping coefficient
    omega0 : float
        Natural frequency

    Returns
    -------
    regime : str
        'underdamped', 'critically_damped', or 'overdamped'

    Notes
    -----
    Underdamped (γ < 2ω₀): Oscillations decay → possible limit cycle
    Critically damped (γ = 2ω₀): Boundary, fastest non-oscillatory decay
    Overdamped (γ > 2ω₀): Pure exponential decay, no oscillations

    Connection to Hopf bifurcation:
    - Hopf occurs when stable node → focus (γ crosses 2ω₀)
    - N75K shows relaxation oscillation → near critical damping
    """
    if gamma < 2 * omega0:
        return 'underdamped'
    elif gamma == 2 * omega0:
        return 'critically_damped'
    else:
        return 'overdamped'


# Export for use in analysis
__all__ = [
    'rotational_correlation_function',
    'extract_rotational_diffusion',
    'response_function_from_correlation',
    'susceptibility_from_correlation',
    'absorption_rate_monochromatic',
    'underdamped_correlation_function',
    'classify_oscillator_regime'
]
