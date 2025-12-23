"""
Utility Functions

This module provides general-purpose utilities for numerical analysis,
binning, bootstrapping, and statistical calculations.

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import stats


def bootstrap_confidence_interval(data: np.ndarray,
                                  statistic: Callable = np.mean,
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for statistic.

    Args:
        data: (n_samples,) data array
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        stat_value: Point estimate of statistic
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound

    Example:
        >>> mean, ci_low, ci_high = bootstrap_confidence_interval(data)
        >>> print(f"Mean: {mean:.2f} [{ci_low:.2f}, {ci_high:.2f}]")
    """
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    # Compute percentiles
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    stat_value = statistic(data)

    return stat_value, ci_lower, ci_upper


def block_average(data: np.ndarray,
                 block_size: int) -> Tuple[float, float]:
    """
    Compute block average and standard error.

    Useful for correlated time series data.

    Args:
        data: (n_samples,) time series
        block_size: Size of blocks for averaging

    Returns:
        mean: Block-averaged mean
        sem: Standard error of the mean

    Notes:
        - Reduces correlation effects in error estimation
        - Block size should be > correlation time
    """
    n = len(data)
    n_blocks = n // block_size

    if n_blocks < 2:
        raise ValueError("Need at least 2 blocks. Reduce block_size or use more data.")

    # Reshape into blocks
    blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)

    # Block averages
    block_means = np.mean(blocks, axis=1)

    # Overall mean and SEM
    mean = np.mean(block_means)
    sem = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

    return mean, sem


def running_average(data: np.ndarray,
                   window_size: int) -> np.ndarray:
    """
    Compute running average with fixed window.

    Args:
        data: (n_samples,) input data
        window_size: Window size for averaging

    Returns:
        smoothed: (n_samples,) smoothed data

    Example:
        >>> smoothed = running_average(noisy_signal, window_size=10)
    """
    if window_size < 1:
        return data

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')

    return smoothed


def exponential_moving_average(data: np.ndarray,
                               alpha: float = 0.1) -> np.ndarray:
    """
    Compute exponential moving average.

    EMA(t) = α·x(t) + (1-α)·EMA(t-1)

    Args:
        data: (n_samples,) input data
        alpha: Smoothing factor (0 < α < 1)
               Smaller = more smoothing

    Returns:
        ema: (n_samples,) exponentially smoothed data

    Example:
        >>> ema = exponential_moving_average(data, alpha=0.05)
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

    return ema


def find_plateaus(data: np.ndarray,
                 threshold: float = 0.01,
                 min_length: int = 10) -> list:
    """
    Find plateau regions in data.

    Useful for detecting equilibrated regions, metastable states, etc.

    Args:
        data: (n_samples,) time series
        threshold: Maximum allowed variation in plateau
        min_length: Minimum plateau length

    Returns:
        plateaus: List of (start, end, mean_value) tuples

    Example:
        >>> plateaus = find_plateaus(energy_trajectory)
        >>> for start, end, value in plateaus:
        ...     print(f"Plateau at {value:.2f} from frame {start} to {end}")
    """
    plateaus = []

    i = 0
    while i < len(data):
        # Start potential plateau
        plateau_start = i
        plateau_values = [data[i]]

        # Extend plateau while variation is small
        j = i + 1
        while j < len(data):
            plateau_values.append(data[j])
            variation = np.std(plateau_values)

            if variation > threshold:
                # Exceeded threshold, end plateau
                break

            j += 1

        plateau_length = j - i

        # Check if plateau is long enough
        if plateau_length >= min_length:
            plateau_mean = np.mean(plateau_values)
            plateaus.append((plateau_start, j, plateau_mean))

        i = j if j > i else i + 1

    return plateaus


def histogram_with_errors(data: np.ndarray,
                         bins: int = 50,
                         weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute histogram with Poisson errors.

    Args:
        data: (n_samples,) data to histogram
        bins: Number of bins or bin edges
        weights: Optional sample weights

    Returns:
        counts: Histogram counts
        bin_edges: Bin edges
        errors: Poisson errors (√counts)

    Example:
        >>> counts, edges, errors = histogram_with_errors(observable)
    """
    counts, bin_edges = np.histogram(data, bins=bins, weights=weights)

    # Poisson errors
    errors = np.sqrt(counts)
    errors[counts == 0] = 1.0  # Avoid zero errors

    return counts, bin_edges, errors


def reweight_histogram(observable: np.ndarray,
                      weights_old: np.ndarray,
                      weights_new: np.ndarray,
                      bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reweight histogram using importance sampling.

    P_new(x) = P_old(x) · w_new(x) / w_old(x)

    Args:
        observable: (n_samples,) observable values
        weights_old: (n_samples,) old weights (e.g., from simulation)
        weights_new: (n_samples,) new weights (e.g., reweighting potential)
        bins: Number of histogram bins

    Returns:
        hist_reweighted: Reweighted histogram
        bin_edges: Bin edges

    Example:
        >>> # Reweight to different temperature
        >>> beta_old = 1 / (kB * T_old)
        >>> beta_new = 1 / (kB * T_new)
        >>> w_new = np.exp((beta_old - beta_new) * energies)
        >>> hist_new, edges = reweight_histogram(observable, w_old, w_new)
    """
    reweight_factors = weights_new / (weights_old + 1e-10)  # Avoid division by zero

    hist, bin_edges = np.histogram(observable, bins=bins, weights=reweight_factors)

    # Normalize
    hist = hist / np.sum(hist)

    return hist, bin_edges


def circular_mean(angles: np.ndarray) -> float:
    """
    Compute circular mean for angular data.

    Args:
        angles: (n_samples,) angles in radians

    Returns:
        mean_angle: Circular mean in radians [-π, π]

    Example:
        >>> mean_phi = circular_mean(phi_angles)
    """
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))

    mean_angle = np.arctan2(sin_mean, cos_mean)

    return mean_angle


def circular_std(angles: np.ndarray) -> float:
    """
    Compute circular standard deviation.

    Args:
        angles: (n_samples,) angles in radians

    Returns:
        std: Circular standard deviation in radians

    Example:
        >>> std_phi = circular_std(phi_angles)
    """
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    std = np.sqrt(-2 * np.log(R))

    return std


def autocorrelation_time(data: np.ndarray,
                        max_lag: Optional[int] = None) -> float:
    """
    Estimate autocorrelation time τ.

    τ = ∫ C(t) dt where C(t) is normalized ACF.

    Args:
        data: (n_samples,) time series
        max_lag: Maximum lag to integrate (default: n_samples // 4)

    Returns:
        tau: Autocorrelation time in sample units

    Example:
        >>> tau = autocorrelation_time(observable)
        >>> effective_samples = len(observable) / tau
    """
    from .analysis.correlations import autocorrelation_function

    if max_lag is None:
        max_lag = len(data) // 4

    acf = autocorrelation_function(data, max_lag=max_lag)

    # Integrate ACF
    tau = np.sum(acf[acf > 0])  # Only integrate positive part

    return tau


def gaussian_kernel_density(data: np.ndarray,
                            x_eval: np.ndarray,
                            bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Kernel density estimation with Gaussian kernel.

    Args:
        data: (n_samples,) data points
        x_eval: (n_points,) points to evaluate density
        bandwidth: Kernel bandwidth (default: Scott's rule)

    Returns:
        density: (n_points,) density estimates

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> density = gaussian_kernel_density(samples, x)
    """
    from scipy.stats import gaussian_kde

    if bandwidth is None:
        kde = gaussian_kde(data)
    else:
        kde = gaussian_kde(data, bw_method=bandwidth)

    density = kde(x_eval)

    return density


def numerical_derivative(y: np.ndarray,
                        x: Optional[np.ndarray] = None,
                        order: int = 2) -> np.ndarray:
    """
    Compute numerical derivative using finite differences.

    Args:
        y: (n_points,) function values
        x: Optional (n_points,) x coordinates (default: unit spacing)
        order: Derivative order (1 or 2)

    Returns:
        dy: Numerical derivative

    Example:
        >>> velocity = numerical_derivative(position, times)
    """
    if x is None:
        x = np.arange(len(y))

    if order == 1:
        dy = np.gradient(y, x)
    elif order == 2:
        dy = np.gradient(np.gradient(y, x), x)
    else:
        raise ValueError("Only order 1 and 2 supported")

    return dy


def numerical_integral(y: np.ndarray,
                      x: Optional[np.ndarray] = None,
                      method: str = 'trapz') -> np.ndarray:
    """
    Compute numerical integral (cumulative).

    Args:
        y: (n_points,) integrand values
        x: Optional (n_points,) x coordinates
        method: Integration method ('trapz' or 'simps')

    Returns:
        integral: (n_points,) cumulative integral

    Example:
        >>> displacement = numerical_integral(velocity, times)
    """
    if x is None:
        x = np.arange(len(y))

    if method == 'trapz':
        from scipy.integrate import cumulative_trapezoid
        integral = cumulative_trapezoid(y, x, initial=0)
    elif method == 'simps':
        # Simpson's rule (requires scipy)
        from scipy.integrate import simpson
        integral = np.zeros_like(y)
        for i in range(1, len(y)):
            integral[i] = simpson(y[:i+1], x[:i+1])
    else:
        raise ValueError(f"Unknown method: {method}")

    return integral


if __name__ == '__main__':
    print("Utility Functions Module")
    print("========================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.utils import bootstrap_confidence_interval")
    print()
    print("# Compute confidence interval for mean")
    print("mean, ci_low, ci_high = bootstrap_confidence_interval(data)")
    print("print(f'Mean: {mean:.2f} [95% CI: {ci_low:.2f}, {ci_high:.2f}]')")
