"""Orientation-dependent friction :math:`\\gamma(\\theta, \\psi)` from angular velocity ACFs.

In the overdamped limit of rotational Brownian motion, ``I domega/dt = -gamma
omega + xi(t)`` collapses to :math:`C_\\omega(t) = \\exp(-\\gamma t/I)`, so a
friction coefficient is read off a correlation time rather than measured
directly: :math:`\\gamma = I/\\tau_c` (see :mod:`rotmd.analysis.correlations`
for how :math:`\\tau_c` itself is extracted). Binning that estimate over the
protein's tilt/spin angles turns a single number into a map, which is the
point: a well-bound transmembrane protein should show high, roughly
orientation-independent gamma with :math:`\\gamma_\\perp \\gg \\gamma_\\parallel`
(nutation strongly damped, spin free); a poorly-bound peripheral one shows
low gamma with strong (theta, psi) dependence and closer to isotropic damping.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

# k_B*T at 310 K, in kcal/mol.
_K_B_T = 0.615


def extract_friction_from_acf(
    times: np.ndarray,
    acf: np.ndarray,
    moment_of_inertia: float,
    method: str = "integral",
) -> dict[str, Any]:
    """Extract gamma (amu/ps) from an angular-velocity ACF, via :math:`C_\\omega(t) = \\exp(-\\gamma t/I)`.

    ``method='exponential_fit'`` fits that model directly (falls back to
    ``'initial_slope'`` if the fit doesn't converge); ``'initial_slope'`` uses
    :math:`\\gamma = -I\\,\\dot C(0)`, cheap but sensitive to noise in the first
    lag; ``'integral'`` uses the Green-Kubo form :math:`\\gamma = I/\\int C_\\omega\\,dt`
    and is the most robust to noise in the tail.
    """
    if len(times) != len(acf):
        raise ValueError("times and acf must have same length")

    gamma = None

    if method == "exponential_fit":

        def fit_func(t, gamma):
            return np.exp(-gamma * t / moment_of_inertia)

        try:
            gamma_guess = moment_of_inertia / (times[-1] / 3)
            popt, _ = curve_fit(
                fit_func, times, acf, p0=[gamma_guess], bounds=([0], [np.inf]), maxfev=10000
            )
            gamma = popt[0]
        except RuntimeError:
            warnings.warn("Exponential fit failed. Using initial slope method.")
            dt = times[1] - times[0]
            dc_dt_0 = (acf[1] - acf[0]) / dt
            gamma = -moment_of_inertia * dc_dt_0

    elif method == "initial_slope":
        dt = times[1] - times[0]
        dc_dt_0 = (acf[1] - acf[0]) / dt
        gamma = -moment_of_inertia * dc_dt_0

    elif method == "integral":
        tau_c = np.trapezoid(acf, times)
        gamma = moment_of_inertia / tau_c if tau_c > 0 else np.inf

    else:
        raise ValueError(f"Unknown method: {method}")

    if gamma is None:
        gamma = 0.0

    tau_c = moment_of_inertia / gamma if gamma > 0 else np.inf
    diffusion_coeff = _K_B_T / gamma if gamma > 0 else np.inf

    return {
        "gamma": gamma,
        "tau_c": tau_c,
        "diffusion_coeff": diffusion_coeff,
        "method": method,
    }


def orientation_dependent_friction(
    theta_trajectory: np.ndarray,
    psi_trajectory: np.ndarray,
    omega_trajectory: np.ndarray,
    times: np.ndarray,
    moment_of_inertia: float,
    theta_bins: int = 10,
    psi_bins: int = 12,
    min_samples_per_bin: int = 50,
) -> dict[str, np.ndarray]:
    """Bin the trajectory by (theta in [0, 90] deg, psi in [0, 360] deg) and fit gamma within each bin.

    Bins with fewer than ``min_samples_per_bin`` frames are left ``NaN`` rather
    than fit: an ACF from a handful of frames gives a correlation time that is
    mostly sampling noise, and a spuriously-precise gamma in a sparse bin would
    be indistinguishable from a real feature once plotted as a heatmap.
    """
    n_frames = len(theta_trajectory)

    if len(psi_trajectory) != n_frames or len(omega_trajectory) != n_frames or len(times) != n_frames:
        raise ValueError("All trajectories must have same length")

    theta_edges = np.linspace(0, 90, theta_bins + 1)
    psi_edges = np.linspace(0, 360, psi_bins + 1)

    gamma_map = np.full((theta_bins, psi_bins), np.nan)
    tau_c_map = np.full((theta_bins, psi_bins), np.nan)
    counts = np.zeros((theta_bins, psi_bins), dtype=int)

    theta_idx = np.digitize(theta_trajectory, theta_edges) - 1
    psi_idx = np.digitize(psi_trajectory, psi_edges) - 1
    theta_idx = np.clip(theta_idx, 0, theta_bins - 1)
    psi_idx = np.clip(psi_idx, 0, psi_bins - 1)

    for i in range(theta_bins):
        for j in range(psi_bins):
            mask = (theta_idx == i) & (psi_idx == j)
            n_samples = np.sum(mask)
            counts[i, j] = n_samples

            if n_samples < min_samples_per_bin:
                continue

            omega_bin = omega_trajectory[mask]
            times_bin = times[mask]

            # mask need not be time-ordered, but the ACF assumes it is.
            sort_idx = np.argsort(times_bin)
            times_bin = times_bin[sort_idx]
            omega_bin = omega_bin[sort_idx]

            try:
                from .correlations import autocorrelation_function

                omega_mag = np.linalg.norm(omega_bin, axis=1)
                max_lag = min(len(omega_mag) // 4, 200)
                lags, acf = autocorrelation_function(omega_mag, max_lag=max_lag, normalize=True)

                dt = np.mean(np.diff(times_bin)) if len(times_bin) > 1 else 0.001
                times_acf = lags * dt

                friction_result = extract_friction_from_acf(
                    times_acf, acf, moment_of_inertia, method="integral"
                )

                gamma_map[i, j] = friction_result["gamma"]
                tau_c_map[i, j] = friction_result["tau_c"]

            except Exception as e:
                warnings.warn(f"Failed to compute friction for bin ({i}, {j}): {e}")
                continue

    return {
        "theta_edges": theta_edges,
        "psi_edges": psi_edges,
        "gamma_map": gamma_map,
        "counts": counts,
        "tau_c_map": tau_c_map,
        "theta_centers": (theta_edges[:-1] + theta_edges[1:]) / 2,
        "psi_centers": (psi_edges[:-1] + psi_edges[1:]) / 2,
    }


def anisotropic_friction_tensor(
    omega_parallel_acf: np.ndarray,
    omega_perp_acf: np.ndarray,
    times: np.ndarray,
    i_parallel: float,
    i_perp: float,
) -> dict[str, float]:
    """Fit spin (parallel) and nutation (perpendicular) friction separately and return their ratio.

    ``anisotropy = gamma_perp / gamma_parallel`` is the single number that
    distinguishes a well-bound protein (nutation strongly damped, ratio >> 1)
    from a poorly-bound one (near-isotropic damping, ratio ~ 1).
    """
    result_par = extract_friction_from_acf(times, omega_parallel_acf, i_parallel, method="integral")
    gamma_par = result_par["gamma"]

    result_perp = extract_friction_from_acf(times, omega_perp_acf, i_perp, method="integral")
    gamma_perp = result_perp["gamma"]

    anisotropy = gamma_perp / gamma_par if gamma_par > 0 else np.inf

    return {
        "gamma_parallel": gamma_par,
        "gamma_perp": gamma_perp,
        "anisotropy": anisotropy,
        "tau_c_parallel": result_par["tau_c"],
        "tau_c_perp": result_perp["tau_c"],
    }


if __name__ == "__main__":
    print("Friction Module - Example Usage\n")

    print("Example 1: Friction from angular velocity ACF")
    np.random.seed(42)

    times = np.arange(100) * 0.01
    gamma_true = 500.0
    moment_of_inertia = 1000.0
    tau_c_true = moment_of_inertia / gamma_true
    acf = np.exp(-times / tau_c_true)

    result = extract_friction_from_acf(times, acf, moment_of_inertia, method="exponential_fit")
    print(f"True gamma: {gamma_true:.1f} amu/ps")
    print(f"Extracted gamma: {result['gamma']:.1f} amu/ps")
    print(f"True tau_c: {tau_c_true:.2f} ps")
    print(f"Extracted tau_c: {result['tau_c']:.2f} ps")
    print(f"Diffusion coefficient D: {result['diffusion_coeff']:.4f}")

    print("\n" + "=" * 60)
    print("Example 2: Orientation-dependent friction gamma(theta, psi)")

    n_frames = 5000
    theta_traj = np.random.rand(n_frames) * 90
    psi_traj = np.random.rand(n_frames) * 360
    omega_traj = np.random.randn(n_frames, 3) * 0.1
    times_traj = np.arange(n_frames) * 0.001

    result_map = orientation_dependent_friction(
        theta_traj,
        psi_traj,
        omega_traj,
        times_traj,
        moment_of_inertia=1000.0,
        theta_bins=5,
        psi_bins=6,
        min_samples_per_bin=50,
    )

    print(f"Friction map shape: {result_map['gamma_map'].shape}")
    print(f"Number of valid bins: {np.sum(~np.isnan(result_map['gamma_map']))}")
    print(f"Mean gamma (valid bins): {np.nanmean(result_map['gamma_map']):.1f} amu/ps")
    print(f"Min samples per bin: {np.min(result_map['counts'])}")
    print(f"Max samples per bin: {np.max(result_map['counts'])}")

    print("\n" + "=" * 60)
    print("Example 3: Anisotropic friction tensor")

    c_par = np.exp(-times / 2.0)
    c_perp = np.exp(-times / 0.5)

    i_par = 1000.0
    i_perp = 500.0

    result_aniso = anisotropic_friction_tensor(c_par, c_perp, times, i_par, i_perp)

    print(f"gamma_parallel (spin): {result_aniso['gamma_parallel']:.1f} amu/ps")
    print(f"gamma_perp (nutation): {result_aniso['gamma_perp']:.1f} amu/ps")
    print(f"Anisotropy gamma_perp/gamma_parallel: {result_aniso['anisotropy']:.2f}")

    if result_aniso["anisotropy"] > 2:
        print("-> Nutation highly damped (well-bound protein)")
    elif result_aniso["anisotropy"] < 0.5:
        print("-> Spin highly damped")
    else:
        print("-> Isotropic-like friction")
