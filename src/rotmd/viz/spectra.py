"""Autocorrelation, power spectra and friction-extraction figures.

Restored from the pre-slimming ``visualization/spectra.py``. Two things
changed beyond the shared API:

* The correlation-time fit is no longer done inside the plot. It calls
  :func:`rotmd.analysis.correlations.extract_correlation_time`, so the figure
  and any number in the text come from the same estimator — the legacy plot
  carried its own private ``curve_fit`` and could disagree with the analysis.
* Frequency axes are labelled in the units actually computed. ``np.fft.fftfreq``
  returns ordinary frequency in cycles/ps; the legacy spectral-density plot
  labelled that axis "ω (rad/ps)", off by 2π.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import figure


@figure(figsize=(9, 6), name="acf")
def plot_autocorrelation(
    ax: Any,
    times: np.ndarray,
    acf: np.ndarray,
    *,
    label: str = "C(t)",
    tau: float | None = None,
    fit: bool = True,
) -> Any:
    """An ACF with an optional fitted exponential envelope.

    Args:
        times: Lag times in ps.
        acf: Autocorrelation at each lag.
        tau: Correlation time in ps. If ``None`` and ``fit`` is set, it is
            estimated with the package's exponential estimator.
        fit: Draw ``exp(-t/τ)`` over the data and annotate τ.

    Notes:
        The estimator used is ``method='exponential'`` rather than the
        ``'integral'`` default. Both are correct, but the integral is the
        quantity the eye cannot check against this figure: it is an area,
        dominated by the tail, and the curve drawn here is a decay envelope.
        Fitting what is plotted keeps the annotation falsifiable by looking.
    """
    times = np.asarray(times, dtype=np.float64)
    acf = np.asarray(acf, dtype=np.float64)

    ax.plot(times, acf, lw=1.8, color="tab:blue", label=label)

    if fit and tau is None and times.size > 10:
        from rotmd.analysis.correlations import extract_correlation_time

        tau = extract_correlation_time(times, acf, method="exponential")

    if fit and tau is not None and np.isfinite(tau) and tau > 0:
        ax.plot(
            times, np.exp(-times / tau),
            "--", lw=1.8, color="crimson", label=f"exp(−t/{tau:.1f} ps)",
        )

    ax.axhline(0, color="0.4", ls="--", lw=1, alpha=0.6)
    ax.set(xlabel="lag time (ps)", ylabel="autocorrelation", title="Autocorrelation function")
    ax.grid(True)
    ax.legend()
    return ax


@figure(figsize=(9, 6), name="acf-multi")
def plot_multiple_acfs(
    ax: Any,
    times: np.ndarray,
    acfs: dict[str, np.ndarray],
    *,
    cmap: str = "tab10",
) -> Any:
    """Several ACFs on shared axes, for comparing components or systems."""
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=np.float64)
    colours = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(acfs), 2)))

    for (label, acf), colour in zip(acfs.items(), colours):
        ax.plot(times, np.asarray(acf, dtype=np.float64), lw=1.8, label=label, color=colour)

    ax.axhline(0, color="0.4", ls="--", lw=1, alpha=0.6)
    ax.set(xlabel="lag time (ps)", ylabel="autocorrelation", title="Autocorrelation functions")
    ax.grid(True)
    ax.legend(ncols=min(3, max(1, len(acfs))))
    return ax


def power_spectrum(
    times: np.ndarray, signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of ``signal``, mean removed.

    Returns:
        ``(freqs, power)`` with freqs in cycles/ps, strictly positive.
    """
    times = np.asarray(times, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    dt = float(times[1] - times[0])

    spectrum = np.fft.rfft(signal - signal.mean())
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(signal.size, dt)

    positive = freqs > 0  # drop the DC bin: it is zero by construction
    return freqs[positive], power[positive]


@figure(figsize=(9, 6), name="power-spectrum")
def plot_power_spectrum(
    ax: Any,
    times: np.ndarray,
    signal: np.ndarray,
    *,
    max_freq: float | None = None,
    log_scale: bool = True,
) -> Any:
    """Power spectrum of a time series.

    Args:
        times: Timestamps in ps, uniformly spaced.
        signal: Values at those times.
        max_freq: Upper limit of the frequency axis, in cycles/ps.
        log_scale: Log the power axis, which is almost always what is wanted —
            spectra span decades.
    """
    freqs, power = power_spectrum(times, signal)
    if max_freq is not None:
        keep = freqs <= max_freq
        freqs, power = freqs[keep], power[keep]

    plot = ax.semilogy if log_scale else ax.plot
    plot(freqs, power, lw=1.4, color="tab:blue")

    ax.set(
        xlabel="frequency (1/ps)",
        ylabel="power" + (" (log)" if log_scale else ""),
        title="Power spectrum",
    )
    ax.grid(True)
    return ax


def spectral_density(
    times: np.ndarray, acf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral density from a one-sided ACF, by cosine transform.

    ``S(f) = 2 ∫_0^∞ C(t) cos(2πft) dt`` — the Wiener-Khinchin transform of a
    real, even correlation function, written over the half-axis that is
    actually stored.

    The legacy implementation took ``np.real(np.fft.fft(acf)) * dt`` of the
    one-sided array. That drops the factor of 2 from folding the negative
    lags, so every value came out half of what the title claimed, and it
    reported the mirrored upper half of the FFT as if it were signal.

    The endpoints are half-weighted, making this a trapezoid rule rather than a
    rectangle rule. Without it every value carries a constant ``+C(0)·dt``
    offset — C(0) is the largest sample in the array, so at a coarse lag
    spacing that bias is not small.

    Returns:
        ``(freqs, S)`` with freqs in cycles/ps.
    """
    times = np.asarray(times, dtype=np.float64)
    acf = np.array(acf, dtype=np.float64)  # copied: the endpoints are reweighted
    dt = float(times[1] - times[0])

    acf[0] *= 0.5
    acf[-1] *= 0.5

    freqs = np.fft.rfftfreq(acf.size, dt)
    density = 2.0 * np.real(np.fft.rfft(acf)) * dt
    return freqs, density


@figure(figsize=(9, 6), name="spectral-density")
def plot_spectral_density(
    ax: Any,
    times: np.ndarray,
    acf: np.ndarray,
    *,
    max_freq: float | None = None,
    angular: bool = False,
) -> Any:
    """Spectral density S(f) obtained from the ACF.

    Args:
        max_freq: Axis limit, in whichever unit ``angular`` selects.
        angular: Plot against ω = 2πf in rad/ps instead of f in cycles/ps.
    """
    freqs, density = spectral_density(times, acf)
    x = 2 * np.pi * freqs if angular else freqs
    xlabel = "angular frequency ω (rad/ps)" if angular else "frequency f (1/ps)"

    if max_freq is not None:
        keep = x <= max_freq
        x, density = x[keep], density[keep]

    ax.plot(x, density, lw=1.8, color="tab:blue")
    ax.set(xlabel=xlabel, ylabel="spectral density S", title="Spectral density")
    ax.grid(True)
    return ax


@figure(nrows=1, ncols=2, figsize=(12, 5), name="friction-extraction")
def plot_friction_extraction(
    axes: Any,
    times: np.ndarray,
    acf: np.ndarray,
    friction: float,
) -> Any:
    """The ACF beside its running integral, with the reported γ marked.

    This is the diagnostic for a Green-Kubo friction coefficient, and it is
    worth reading carefully: γ is the *plateau* of the right-hand curve, not
    its endpoint. Once the ACF has decayed into noise the integral stops
    converging and starts performing a random walk, so a curve that never
    flattens — or that flattens and then drifts — means the estimate is being
    read from noise and the integration limit has to be cut back.

    The first-non-positive-lag truncation in
    :func:`rotmd.analysis.correlations.extract_correlation_time` exists for the
    same reason; the drift is drawn here so it can be seen rather than assumed
    absent.
    """
    times = np.asarray(times, dtype=np.float64)
    acf = np.asarray(acf, dtype=np.float64)
    left, right = axes

    left.plot(times, acf, lw=1.8, color="tab:blue", label="C(t)")
    left.axhline(0, color="0.4", ls="--", lw=1, alpha=0.6)
    left.set(xlabel="lag time (ps)", ylabel="autocorrelation", title="Autocorrelation function")
    left.grid(True)
    left.legend()

    # Cumulative trapezoid, not cumsum*dt: the legacy rectangle rule biases the
    # integral by half a step of C(0), which is the largest value in the array.
    cumulative = np.concatenate([[0.0], np.cumsum((acf[1:] + acf[:-1]) / 2.0) * np.diff(times)])

    right.plot(times, cumulative, lw=1.8, color="crimson", label="∫₀ᵗ C(t′) dt′")
    right.axhline(friction, color="tab:green", ls="--", lw=1.8, label=f"γ = {friction:.3g}")

    first_zero = np.flatnonzero(acf <= 0.0)
    if first_zero.size:
        right.axvline(
            times[first_zero[0]], color="0.4", ls=":", lw=1.4,
            label="first zero crossing",
        )
    right.set(
        xlabel="integration limit (ps)", ylabel="cumulative integral",
        title="Friction extraction",
    )
    right.grid(True)
    right.legend()
    return axes


@figure(figsize=(9, 6), name="acf-comparison")
def plot_correlation_comparison(
    ax: Any,
    times: np.ndarray,
    reference: np.ndarray,
    model: np.ndarray,
    *,
    labels: tuple[str, str] = ("MD", "model"),
) -> Any:
    """Reference and model ACFs together, annotated with their RMSE."""
    times = np.asarray(times, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)

    ax.plot(times, reference, "-", lw=1.8, alpha=0.85, color="tab:blue", label=labels[0])
    ax.plot(times, model, "--", lw=1.8, alpha=0.85, color="crimson", label=labels[1])
    ax.axhline(0, color="0.4", ls="--", lw=1, alpha=0.6)

    rmse = float(np.sqrt(np.mean((reference - model) ** 2)))
    ax.text(
        0.98, 0.96, f"RMSE = {rmse:.4f}", transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.6},
    )

    ax.set(xlabel="lag time (ps)", ylabel="autocorrelation", title="ACF: reference vs model")
    ax.grid(True)
    ax.legend()
    return ax
