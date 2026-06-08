"""
Spectra and Correlation Visualization

This module provides utilities for visualizing power spectra, autocorrelation
functions, and frequency-domain analysis.

Key Features:
- Autocorrelation function plots
- Power spectra (FFT)
- Spectral density
- Relaxation time extraction visualization
- Multi-component comparisons

Author: Mykyta Bobylyow
Date: 2025
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_autocorrelation(times: np.ndarray,
                        acf: np.ndarray,
                        label: str = 'ACF',
                        fit_exponential: bool = True,
                        figsize: Tuple[float, float] = (10, 6),
                        ax: Optional[plt.Axes] = None,
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot autocorrelation function.

    Args:
        times: (n_lags,) lag times in ps
        acf: (n_lags,) autocorrelation values
        label: Label for plot
        fit_exponential: Fit exponential decay (drawn on the plot)
        figsize: Figure size (used only when ``ax`` is None)
        ax: Optional existing Axes to draw on; a new figure is created if None.
        save_path: If given, the figure is written there.

    Returns:
        ``(fig, ax)`` so the caller can further customise, show, or save.

    Example:
        >>> fig, ax = plot_autocorrelation(lag_times, acf, label='C_L(t)')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(times, acf, 'b-', linewidth=2, label=label)

    tau = None
    if fit_exponential and len(times) > 10:
        # Fit exponential: C(t) = exp(-t/τ)
        from scipy.optimize import curve_fit

        def exp_decay(t, tau):
            return np.exp(-t / tau)

        # Only fit positive values
        positive_mask = acf > 0
        if np.sum(positive_mask) > 5:
            try:
                popt, _ = curve_fit(exp_decay,
                                   times[positive_mask],
                                   acf[positive_mask],
                                   p0=[times[-1]/3])
                tau = popt[0]

                # Plot fit
                ax.plot(times, exp_decay(times, tau),
                       'r--', linewidth=2,
                       label=f'Fit: exp(-t/{tau:.1f} ps)')

            except:
                warnings.warn("Exponential fit failed")

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Lag Time (ps)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Autocorrelation Function', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_multiple_acfs(times: np.ndarray,
                      acfs: Dict[str, np.ndarray],
                      figsize: Tuple[float, float] = (10, 6),
                      ax: Optional[plt.Axes] = None,
                      save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple autocorrelation functions for comparison.

    Args:
        times: (n_lags,) lag times
        acfs: Dictionary of {label: acf_array}
        figsize: Figure size (used only when ``ax`` is None)
        ax: Optional existing Axes to draw on; a new figure is created if None.
        save_path: If given, the figure is written there.

    Returns:
        ``(fig, ax)``.

    Example:
        >>> acfs = {
        ...     'C_Lx': acf_x,
        ...     'C_Ly': acf_y,
        ...     'C_Lz': acf_z
        ... }
        >>> fig, ax = plot_multiple_acfs(lag_times, acfs)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, len(acfs)))

    for (label, acf), color in zip(acfs.items(), colors):
        ax.plot(times, acf, linewidth=2, label=label, color=color)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Lag Time (ps)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Autocorrelation Functions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_power_spectrum(times: np.ndarray,
                       signal: np.ndarray,
                       max_freq: Optional[float] = None,
                       log_scale: bool = True,
                       figsize: Tuple[float, float] = (10, 6),
                       ax: Optional[plt.Axes] = None,
                       save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot power spectrum (FFT) of time series.

    Args:
        times: (n_frames,) time values in ps
        signal: (n_frames,) signal values
        max_freq: Maximum frequency to plot (1/ps)
        log_scale: Use log scale for power
        figsize: Figure size (used only when ``ax`` is None)
        ax: Optional existing Axes to draw on; a new figure is created if None.
        save_path: If given, the figure is written there.

    Returns:
        ``(fig, ax)``.

    Example:
        >>> fig, ax = plot_power_spectrum(times, L_mag, max_freq=1.0)
    """
    # Compute FFT
    dt = times[1] - times[0]
    n = len(signal)

    # Detrend signal
    signal_detrended = signal - np.mean(signal)

    # FFT
    fft = np.fft.fft(signal_detrended)
    power = np.abs(fft)**2
    freqs = np.fft.fftfreq(n, dt)

    # Only positive frequencies
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    power = power[positive_mask]

    # Limit to max_freq
    if max_freq is not None:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        power = power[freq_mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if log_scale:
        ax.semilogy(freqs, power, 'b-', linewidth=1.5)
        ax.set_ylabel('Power (log scale)', fontsize=12)
    else:
        ax.plot(freqs, power, 'b-', linewidth=1.5)
        ax.set_ylabel('Power', fontsize=12)

    ax.set_xlabel('Frequency (1/ps)', fontsize=12)
    ax.set_title('Power Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_spectral_density(times: np.ndarray,
                         acf: np.ndarray,
                         max_freq: Optional[float] = None,
                         figsize: Tuple[float, float] = (10, 6),
                         ax: Optional[plt.Axes] = None,
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spectral density from autocorrelation function.

    S(ω) = ∫ C(t) exp(-iωt) dt

    Args:
        times: (n_lags,) lag times
        acf: (n_lags,) autocorrelation
        max_freq: Maximum frequency to plot
        figsize: Figure size (used only when ``ax`` is None)
        ax: Optional existing Axes to draw on; a new figure is created if None.
        save_path: If given, the figure is written there.

    Returns:
        ``(fig, ax)``.

    Example:
        >>> fig, ax = plot_spectral_density(lag_times, acf)
    """
    # Fourier transform of ACF
    dt = times[1] - times[0]
    n = len(acf)

    fft = np.fft.fft(acf)
    spectral_density = np.real(fft) * dt
    freqs = np.fft.fftfreq(n, dt)

    # Positive frequencies only
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    spectral_density = spectral_density[positive_mask]

    if max_freq is not None:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        spectral_density = spectral_density[freq_mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(freqs, spectral_density, 'b-', linewidth=2)

    ax.set_xlabel('Frequency ω (rad/ps)', fontsize=12)
    ax.set_ylabel('Spectral Density S(ω)', fontsize=12)
    ax.set_title('Spectral Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_friction_extraction(times: np.ndarray,
                            acf: np.ndarray,
                            friction: float,
                            figsize: Tuple[float, float] = (12, 5),
                            save_path: Optional[str] = None) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Visualize friction coefficient extraction from ACF.

    Shows ACF and its integral (cumulative friction). This is a two-panel
    figure, so it always creates its own figure.

    Args:
        times: (n_lags,) lag times
        acf: (n_lags,) autocorrelation
        friction: Extracted friction coefficient
        figsize: Figure size
        save_path: If given, the figure is written there.

    Returns:
        ``(fig, (ax_acf, ax_integral))``.

    Example:
        >>> fig, (ax1, ax2) = plot_friction_extraction(lag_times, acf, gamma)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: ACF
    ax1.plot(times, acf, 'b-', linewidth=2, label='C(t)')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Lag Time (ps)', fontsize=11)
    ax1.set_ylabel('Autocorrelation', fontsize=11)
    ax1.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Cumulative integral
    cumulative = np.cumsum(acf) * (times[1] - times[0])

    ax2.plot(times, cumulative, 'r-', linewidth=2, label='∫C(t)dt')
    ax2.axhline(friction, color='green', linestyle='--', linewidth=2,
               label=f'γ = {friction:.2f}')
    ax2.set_xlabel('Integration Limit (ps)', fontsize=11)
    ax2.set_ylabel('Cumulative Integral', fontsize=11)
    ax2.set_title('Friction Extraction', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, (ax1, ax2)


def plot_correlation_comparison(lag_times: np.ndarray,
                                md_acf: np.ndarray,
                                model_acf: np.ndarray,
                                labels: Tuple[str, str] = ('MD', 'Model'),
                                figsize: Tuple[float, float] = (10, 6),
                                ax: Optional[plt.Axes] = None,
                                save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare autocorrelation functions from MD and model.

    Args:
        lag_times: (n_lags,) lag times
        md_acf: (n_lags,) MD autocorrelation
        model_acf: (n_lags,) model autocorrelation
        labels: (md_label, model_label)
        figsize: Figure size (used only when ``ax`` is None)
        ax: Optional existing Axes to draw on; a new figure is created if None.
        save_path: If given, the figure is written there.

    Returns:
        ``(fig, ax)``.

    Example:
        >>> fig, ax = plot_correlation_comparison(lag_times, acf_md, acf_langevin)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(lag_times, md_acf, 'b-', linewidth=2, label=labels[0], alpha=0.8)
    ax.plot(lag_times, model_acf, 'r--', linewidth=2, label=labels[1], alpha=0.8)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Lag Time (ps)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('ACF Comparison: MD vs Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Compute and display RMSE
    rmse = np.sqrt(np.mean((md_acf - model_acf)**2))
    ax.text(0.98, 0.97, f'RMSE = {rmse:.4f}',
           transform=ax.transAxes,
           ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


if __name__ == '__main__':
    print("Spectra Visualization Module")
    print("============================")
    print()
    print("Example usage:")
    print()
    print("from rotmd.visualization.spectra import plot_autocorrelation, plot_power_spectrum")
    print("from rotmd.analysis.correlations import autocorrelation_function")
    print()
    print("# Compute ACF")
    print("acf = autocorrelation_function(L_mag, max_lag=1000)")
    print("lag_times = times[:len(acf)]")
    print()
    print("# Visualize")
    print("fig, ax = plot_autocorrelation(lag_times, acf, fit_exponential=True)")
    print("plot_power_spectrum(times, L_mag)")
