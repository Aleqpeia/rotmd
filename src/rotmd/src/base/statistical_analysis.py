#!/usr/bin/python
"""
Statistical Analysis Module for Protein Orientation

This module provides advanced statistical analysis for comparing protein orientations
between different systems (e.g., wild-type vs mutant), using ergodicity assumptions
to perform bootstrapping, block averaging, and Bayesian inference.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalComparison:
    """Results from statistical comparison of two distributions."""
    metric_name: str
    group1_mean: float
    group1_std: float
    group1_sem: float
    group1_ci_lower: float
    group1_ci_upper: float
    group2_mean: float
    group2_std: float
    group2_sem: float
    group2_ci_lower: float
    group2_ci_upper: float
    delta_mean: float
    delta_ci_lower: float
    delta_ci_upper: float
    t_statistic: float
    p_value: float
    effect_size_cohens_d: float
    bayesian_probability_different: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric': self.metric_name,
            'group1': {
                'mean': float(self.group1_mean),
                'std': float(self.group1_std),
                'sem': float(self.group1_sem),
                'ci_95': [float(self.group1_ci_lower), float(self.group1_ci_upper)]
            },
            'group2': {
                'mean': float(self.group2_mean),
                'std': float(self.group2_std),
                'sem': float(self.group2_sem),
                'ci_95': [float(self.group2_ci_lower), float(self.group2_ci_upper)]
            },
            'difference': {
                'delta_mean': float(self.delta_mean),
                'ci_95': [float(self.delta_ci_lower), float(self.delta_ci_upper)]
            },
            'hypothesis_test': {
                't_statistic': float(self.t_statistic),
                'p_value': float(self.p_value),
                'significance': 'p < 0.001' if self.p_value < 0.001 else
                               'p < 0.01' if self.p_value < 0.01 else
                               'p < 0.05' if self.p_value < 0.05 else 'ns'
            },
            'effect_size': {
                'cohens_d': float(self.effect_size_cohens_d),
                'magnitude': 'large' if abs(self.effect_size_cohens_d) > 0.8 else
                            'medium' if abs(self.effect_size_cohens_d) > 0.5 else
                            'small' if abs(self.effect_size_cohens_d) > 0.2 else 'negligible'
            },
            'bayesian': {
                'probability_different': float(self.bayesian_probability_different)
            }
        }


def calculate_autocorrelation_time(
    data: np.ndarray,
    max_lag: Optional[int] = None,
    method: str = 'sokal'
) -> Tuple[float, int]:
    """
    Calculate integrated autocorrelation time with automatic windowing.

    Uses Sokal's automatic windowing procedure (Sokal 1989) to determine
    optimal cutoff for autocorrelation sum, avoiding bias from noise at
    long lags.

    Args:
        data: Time series data
        max_lag: Maximum lag to consider (default: len(data)//2)
        method: 'sokal' for automatic windowing, 'simple' for fixed cutoff

    Returns:
        Tuple of (autocorrelation_time, effective_sample_size)
    """
    n = len(data)

    if max_lag is None:
        max_lag = n // 2

    # Normalize data
    data_norm = data - np.mean(data)

    # Calculate autocorrelation function
    c0 = np.dot(data_norm, data_norm) / n

    if c0 == 0:
        return 1.0, n

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf[0] = 1.0
        else:
            c_lag = np.dot(data_norm[:-lag], data_norm[lag:]) / (n - lag)
            acf[lag] = c_lag / c0

    if method == 'sokal':
        # Sokal's automatic windowing procedure
        # Integrate ACF until it becomes unreliable
        # Reference: Sokal (1989), "Monte Carlo Methods in Statistical Mechanics"
        tau_int = 0.5  # τ = 0.5 + 2*Σρ(t) for t=1,2,...

        # Find cutoff where statistical error in τ becomes too large
        # Use criterion: M = c * τ_int(M), where c ≈ 6
        cutoff = max_lag
        for m in range(1, max_lag):
            if acf[m] > 0:
                tau_int += 2 * acf[m]

            # Automatic windowing: stop when we've gone far enough
            # that noise dominates signal (M > 6*τ)
            if m >= 6 * tau_int:
                cutoff = m
                break

            # Also stop if ACF becomes strongly negative
            if acf[m] < -0.1:
                cutoff = m
                break

        # Recalculate with determined cutoff
        tau_int = 0.5
        for m in range(1, cutoff):
            if acf[m] > 0:
                tau_int += 2 * acf[m]

        tau = max(1.0, tau_int)

    else:  # Simple method (legacy)
        tau = 1.0
        for rho in acf[1:]:
            if rho < 0.05:
                break
            tau += 2 * rho
        tau = max(1.0, tau)

    # Effective sample size
    n_eff = int(n / tau)

    return float(tau), n_eff


def block_average(data: np.ndarray, block_size: int) -> Tuple[float, float]:
    """
    Perform block averaging to estimate uncertainty accounting for correlations.

    Under ergodicity, dividing trajectory into blocks provides independent
    samples for error estimation.

    Args:
        data: Time series data
        block_size: Size of each block in frames

    Returns:
        Tuple of (mean, standard_error)
    """
    n_blocks = len(data) // block_size

    if n_blocks < 2:
        return np.mean(data), np.std(data) / np.sqrt(len(data))

    # Calculate block means
    block_means = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_means.append(np.mean(data[start:end]))

    block_means = np.array(block_means)

    # Mean and SEM from block averages
    mean = np.mean(block_means)
    sem = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

    return mean, sem


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    statistic_func: callable = np.mean,
    block_size: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence intervals using ergodic resampling.

    Supports both standard bootstrap (independent) and block bootstrap
    (for correlated data). Block bootstrap maintains temporal structure
    by resampling blocks rather than individual points.

    Args:
        data: Time series data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        statistic_func: Function to compute statistic (default: mean)
        block_size: Block size for block bootstrap (None = standard bootstrap)
                   Recommended: block_size ≈ 2*τ_int for correlated data

    Returns:
        Tuple of (statistic, lower_ci, upper_ci)
    """
    bootstrap_stats = []
    n = len(data)

    if block_size is None or block_size >= n:
        # Standard bootstrap (assumes independence)
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))

    else:
        # Block bootstrap (for correlated data)
        # Moving block bootstrap: resample overlapping blocks
        n_blocks_needed = int(np.ceil(n / block_size))

        for _ in range(n_bootstrap):
            # Resample blocks with replacement
            sample = []
            for _ in range(n_blocks_needed):
                # Random starting position for block
                start_idx = np.random.randint(0, n - block_size + 1)
                block = data[start_idx:start_idx + block_size]
                sample.extend(block)

            # Trim to original length
            sample = np.array(sample[:n])
            bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate statistic and confidence interval
    stat = statistic_func(data)
    alpha = 1 - confidence
    lower_ci = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return stat, lower_ci, upper_ci


def bootstrap_difference_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    block_size1: Optional[int] = None,
    block_size2: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for difference between two groups.

    Supports block bootstrap for correlated data.

    Args:
        data1: First group data
        data2: Second group data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        block_size1: Block size for group 1 (None = standard bootstrap)
        block_size2: Block size for group 2 (None = standard bootstrap)

    Returns:
        Tuple of (delta_mean, lower_ci, upper_ci)
    """
    delta_bootstraps = []
    n1, n2 = len(data1), len(data2)

    for _ in range(n_bootstrap):
        # Resample group 1
        if block_size1 is None or block_size1 >= n1:
            sample1 = np.random.choice(data1, size=n1, replace=True)
        else:
            sample1_list = []
            n_blocks_needed = int(np.ceil(n1 / block_size1))
            for _ in range(n_blocks_needed):
                start_idx = np.random.randint(0, n1 - block_size1 + 1)
                sample1_list.extend(data1[start_idx:start_idx + block_size1])
            sample1 = np.array(sample1_list[:n1])

        # Resample group 2
        if block_size2 is None or block_size2 >= n2:
            sample2 = np.random.choice(data2, size=n2, replace=True)
        else:
            sample2_list = []
            n_blocks_needed = int(np.ceil(n2 / block_size2))
            for _ in range(n_blocks_needed):
                start_idx = np.random.randint(0, n2 - block_size2 + 1)
                sample2_list.extend(data2[start_idx:start_idx + block_size2])
            sample2 = np.array(sample2_list[:n2])

        delta_bootstraps.append(np.mean(sample2) - np.mean(sample1))

    delta_bootstraps = np.array(delta_bootstraps)

    delta_mean = np.mean(data2) - np.mean(data1)
    alpha = 1 - confidence
    lower_ci = np.percentile(delta_bootstraps, 100 * alpha / 2)
    upper_ci = np.percentile(delta_bootstraps, 100 * (1 - alpha / 2))

    return delta_mean, lower_ci, upper_ci


def bayesian_probability_different(
    data1: np.ndarray,
    data2: np.ndarray,
    n_samples: int = 10000,
    rope_fraction: float = 0.01,
    n_eff1: Optional[int] = None,
    n_eff2: Optional[int] = None
) -> float:
    """
    Bayesian estimate of probability that two distributions are different.

    Uses Student's t-distribution for posterior sampling to account for
    uncertainty with finite (effective) sample sizes. Accounts for
    autocorrelation via effective sample size.

    Args:
        data1: First group data
        data2: Second group data
        n_samples: Number of posterior samples
        rope_fraction: Region of practical equivalence (fraction of pooled std)
        n_eff1: Effective sample size for group 1 (None = len(data1))
        n_eff2: Effective sample size for group 2 (None = len(data2))

    Returns:
        Probability that distributions are practically different (0-1)
    """
    # Estimate parameters for group 1
    mu1 = np.mean(data1)
    s1 = np.std(data1, ddof=1)
    n1_actual = len(data1)
    n1 = n_eff1 if n_eff1 is not None else n1_actual

    # Estimate parameters for group 2
    mu2 = np.mean(data2)
    s2 = np.std(data2, ddof=1)
    n2_actual = len(data2)
    n2 = n_eff2 if n_eff2 is not None else n2_actual

    # Use Student's t-distribution for robust posterior sampling
    # Degrees of freedom from effective sample size
    df1 = max(1, n1 - 1)
    df2 = max(1, n2 - 1)

    # Sample from posterior using t-distribution (heavier tails than normal)
    # Scale by sqrt(df/(df-2)) to get correct variance for t-distribution
    if df1 > 2:
        scale1 = s1 / np.sqrt(n1) * np.sqrt(df1 / (df1 - 2))
        posterior_mu1 = mu1 + stats.t.rvs(df=df1, size=n_samples) * scale1
    else:
        # For very small df, use normal approximation
        posterior_mu1 = np.random.normal(mu1, s1 / np.sqrt(n1), n_samples)

    if df2 > 2:
        scale2 = s2 / np.sqrt(n2) * np.sqrt(df2 / (df2 - 2))
        posterior_mu2 = mu2 + stats.t.rvs(df=df2, size=n_samples) * scale2
    else:
        posterior_mu2 = np.random.normal(mu2, s2 / np.sqrt(n2), n_samples)

    # Calculate differences
    posterior_delta = posterior_mu2 - posterior_mu1

    # Define region of practical equivalence (ROPE)
    # Use pooled std for more robust estimate
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    rope = rope_fraction * pooled_std

    # Probability that difference is outside ROPE
    prob_different = np.mean(np.abs(posterior_delta) > rope)

    return prob_different


def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        data1: First group data
        data2: Second group data

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(data2) - np.mean(data1)) / pooled_std


def compare_distributions(
    data1: np.ndarray,
    data2: np.ndarray,
    metric_name: str,
    n_bootstrap: int = 10000,
    account_for_autocorr: bool = True
) -> StatisticalComparison:
    """
    Comprehensive statistical comparison of two distributions.

    Performs multiple analyses leveraging ergodic hypothesis:
    - Autocorrelation time and effective sample size
    - Block bootstrap confidence intervals (if correlated)
    - Two-sample t-test
    - Effect size estimation
    - Bayesian probability estimation (accounting for N_eff)

    Args:
        data1: First group data (e.g., wild-type)
        data2: Second group data (e.g., mutant)
        metric_name: Name of the metric being compared
        n_bootstrap: Number of bootstrap samples
        account_for_autocorr: Account for autocorrelation in statistics

    Returns:
        StatisticalComparison object with all results
    """
    # Calculate autocorrelation and effective sample size
    if account_for_autocorr:
        tau1, n_eff1 = calculate_autocorrelation_time(data1)
        tau2, n_eff2 = calculate_autocorrelation_time(data2)

        # Block size for bootstrap (approximately 2*tau)
        block_size1 = max(1, int(2 * tau1))
        block_size2 = max(1, int(2 * tau2))
    else:
        n_eff1 = len(data1)
        n_eff2 = len(data2)
        block_size1 = None
        block_size2 = None

    # Basic statistics
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

    # SEM accounting for effective sample size
    sem1 = std1 / np.sqrt(n_eff1)
    sem2 = std2 / np.sqrt(n_eff2)

    # Block bootstrap confidence intervals
    _, ci1_lower, ci1_upper = bootstrap_confidence_interval(
        data1, n_bootstrap=n_bootstrap, block_size=block_size1
    )
    _, ci2_lower, ci2_upper = bootstrap_confidence_interval(
        data2, n_bootstrap=n_bootstrap, block_size=block_size2
    )

    # Block bootstrap difference CI
    delta_mean, delta_ci_lower, delta_ci_upper = bootstrap_difference_ci(
        data1, data2, n_bootstrap=n_bootstrap,
        block_size1=block_size1, block_size2=block_size2
    )

    # Hypothesis testing
    t_stat, p_val = stats.ttest_ind(data1, data2)

    # Effect size
    effect_size = cohens_d(data1, data2)

    # Bayesian probability (accounting for effective sample size)
    prob_different = bayesian_probability_different(
        data1, data2, n_eff1=n_eff1, n_eff2=n_eff2
    )

    return StatisticalComparison(
        metric_name=metric_name,
        group1_mean=mean1,
        group1_std=std1,
        group1_sem=sem1,
        group1_ci_lower=ci1_lower,
        group1_ci_upper=ci1_upper,
        group2_mean=mean2,
        group2_std=std2,
        group2_sem=sem2,
        group2_ci_lower=ci2_lower,
        group2_ci_upper=ci2_upper,
        delta_mean=delta_mean,
        delta_ci_lower=delta_ci_lower,
        delta_ci_upper=delta_ci_upper,
        t_statistic=t_stat,
        p_value=p_val,
        effect_size_cohens_d=effect_size,
        bayesian_probability_different=prob_different
    )


def geweke_diagnostic(
    data: np.ndarray,
    first_fraction: float = 0.1,
    last_fraction: float = 0.5,
    intervals: int = 20
) -> Dict[str, Any]:
    """
    Geweke convergence diagnostic for MCMC/trajectory equilibration.

    Compares means from early portion (first 10%) vs multiple intervals
    in later portion (last 50%). Z-scores should be ~N(0,1) if converged.

    Args:
        data: Time series data
        first_fraction: Fraction of data for early window (default: 0.1)
        last_fraction: Fraction of data for late windows (default: 0.5)
        intervals: Number of intervals in late portion

    Returns:
        Dictionary with Geweke z-scores and convergence assessment
    """
    n = len(data)
    first_n = int(n * first_fraction)
    last_start = int(n * (1 - last_fraction))

    # Early window
    early_data = data[:first_n]
    early_mean = np.mean(early_data)
    early_var = np.var(early_data, ddof=1)

    # Spectral density at zero for early window (accounts for autocorrelation)
    if len(early_data) > 10:
        tau_early, _ = calculate_autocorrelation_time(early_data)
        # Spectral density S(0) = σ² * 2τ (for continuous process)
        spectral_density_early = early_var * 2 * tau_early
    else:
        spectral_density_early = early_var
    se_early = np.sqrt(spectral_density_early / first_n)

    # Late windows
    late_interval_size = (n - last_start) // intervals
    z_scores = []
    late_means = []

    for i in range(intervals):
        start = last_start + i * late_interval_size
        end = start + late_interval_size
        if end > n:
            break

        late_data = data[start:end]
        late_mean = np.mean(late_data)
        late_var = np.var(late_data, ddof=1)

        # Spectral density for late window
        if len(late_data) > 10:
            tau_late, _ = calculate_autocorrelation_time(late_data)
            spectral_density_late = late_var * 2 * tau_late
        else:
            spectral_density_late = late_var
        se_late = np.sqrt(spectral_density_late / len(late_data))

        # Geweke z-score
        z = (late_mean - early_mean) / np.sqrt(se_early**2 + se_late**2)
        z_scores.append(z)
        late_means.append(late_mean)

    z_scores = np.array(z_scores)
    late_means = np.array(late_means)

    # Convergence criterion: |z| < 2 for 95% confidence
    # Allow some outliers since we're doing multiple comparisons
    # Use a more lenient threshold: 20% outliers allowed
    n_outliers = np.sum(np.abs(z_scores) > 2.0)
    converged = n_outliers <= intervals * 0.2  # Allow 30% outliers (6 out of 20)

    return {
        'z_scores': z_scores.tolist(),
        'z_mean': float(np.mean(z_scores)),
        'z_std': float(np.std(z_scores)),
        'z_max_abs': float(np.max(np.abs(z_scores))),
        'n_outliers': int(n_outliers),
        'converged': bool(converged),
        'early_mean': float(early_mean),
        'late_means': late_means.tolist()
    }


def analyze_trajectory_convergence(
    data: np.ndarray,
    window_size: int = 100
) -> Dict[str, Any]:
    """
    Analyze trajectory convergence and ergodic behavior.

    Combines multiple diagnostics:
    - Autocorrelation time and effective sample size
    - Block averaging
    - Window-based t-test
    - Geweke diagnostic for equilibration

    Args:
        data: Time series data
        window_size: Size of sliding window for convergence analysis

    Returns:
        Dictionary with convergence statistics
    """
    # Calculate autocorrelation time
    tau, n_effective = calculate_autocorrelation_time(data)

    # Block averaging with optimal block size
    block_size = max(1, int(2 * tau))
    block_mean, block_sem = block_average(data, block_size)

    # Sliding window analysis for convergence
    n_windows = (len(data) - window_size) // window_size + 1
    window_means = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        if end <= len(data):
            window_means.append(np.mean(data[start:end]))

    window_means = np.array(window_means)

    # Check if later windows are converged (low variance)
    if len(window_means) > 2:
        early_windows = window_means[:len(window_means)//2]
        late_windows = window_means[len(window_means)//2:]

        convergence_ttest_t, convergence_ttest_p = stats.ttest_ind(early_windows, late_windows)
        converged_windows = convergence_ttest_p > 0.05  # Not significantly different = converged
    else:
        convergence_ttest_t, convergence_ttest_p = 0.0, 1.0
        converged_windows = True

    # Geweke diagnostic
    geweke_result = geweke_diagnostic(data)

    # Overall convergence: both tests should pass
    converged_overall = converged_windows and geweke_result['converged']

    return {
        'autocorrelation_time': float(tau),
        'effective_sample_size': int(n_effective),
        'optimal_block_size': int(block_size),
        'block_mean': float(block_mean),
        'block_sem': float(block_sem),
        'window_test': {
            't_statistic': float(convergence_ttest_t),
            'p_value': float(convergence_ttest_p),
            'converged': bool(converged_windows)
        },
        'geweke_diagnostic': geweke_result,
        'overall_converged': bool(converged_overall),
        'window_statistics': {
            'mean': float(np.mean(window_means)),
            'std': float(np.std(window_means, ddof=1))
        }
    }
