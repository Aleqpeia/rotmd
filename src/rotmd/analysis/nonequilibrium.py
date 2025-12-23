"""
Non-equilibrium Thermodynamics Analysis

This module provides utilities for analyzing non-equilibrium aspects
of protein rotational dynamics.

Key Features:
- Detailed balance tests
- Entropy production rate
- Time-reversal symmetry breaking
- Irreversibility measures
- Fluctuation theorems

Theoretical Background:
- Equilibrium: detailed balance holds, σ = 0
- Non-equilibrium: net currents, σ > 0 (entropy production)
- Fluctuation theorems: P(σ)/P(-σ) = exp(σ·t)

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.stats import ks_2samp


def test_detailed_balance(state_trajectory: np.ndarray,
                         n_states: int,
                         verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    Test detailed balance condition for state transitions.

    Detailed balance: k_ij · P_i = k_ji · P_j for all i,j
    where k_ij is transition rate from state i to j, P_i is population of state i

    Args:
        state_trajectory: (n_frames,) state indices
        n_states: Number of discrete states
        verbose: Print results

    Returns:
        balance_matrix: (n_states, n_states) detailed balance ratios
                       balance_matrix[i,j] = (k_ij * P_i) / (k_ji * P_j)
                       Should be 1.0 at equilibrium
        max_violation: Maximum deviation from 1.0

    Notes:
        - Requires sufficient statistics for all state pairs
        - Values > 1: forward flux dominates
        - Values < 1: backward flux dominates
    """
    # Count transitions
    transition_counts = np.zeros((n_states, n_states))

    for i in range(len(state_trajectory) - 1):
        from_state = state_trajectory[i]
        to_state = state_trajectory[i + 1]

        if 0 <= from_state < n_states and 0 <= to_state < n_states:
            transition_counts[from_state, to_state] += 1

    # State populations
    populations = np.zeros(n_states)
    for state in range(n_states):
        populations[state] = np.sum(state_trajectory == state)

    # Normalize populations
    populations = populations / np.sum(populations)

    # Compute detailed balance ratios
    balance_matrix = np.ones((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                numerator = transition_counts[i, j] * populations[i]
                denominator = transition_counts[j, i] * populations[j]

                if denominator > 0:
                    balance_matrix[i, j] = numerator / denominator
                else:
                    balance_matrix[i, j] = np.nan

    # Maximum violation
    finite_values = balance_matrix[np.isfinite(balance_matrix)]
    if len(finite_values) > 0:
        max_violation = np.max(np.abs(np.log(finite_values)))
    else:
        max_violation = np.nan

    if verbose:
        print("Detailed Balance Test")
        print("=" * 50)
        print("Balance matrix (should be 1.0 at equilibrium):")
        print(balance_matrix)
        print(f"\nMaximum violation: {max_violation:.3f}")

        if max_violation < 0.1:
            print("  → System appears to be at equilibrium")
        elif max_violation < 0.5:
            print("  → Weak violation of detailed balance")
        else:
            print("  → Strong violation - non-equilibrium dynamics")

    return balance_matrix, max_violation


def entropy_production_rate(fluxes: np.ndarray,
                           forces: np.ndarray) -> float:
    """
    Compute entropy production rate from thermodynamic fluxes and forces.

    σ = Σ_i J_i · X_i
    where J_i are fluxes and X_i are thermodynamic forces

    Args:
        fluxes: (n_observables,) thermodynamic fluxes
        forces: (n_observables,) conjugate forces

    Returns:
        sigma: Entropy production rate (non-negative)

    Notes:
        - σ = 0 at equilibrium (detailed balance)
        - σ > 0 for non-equilibrium steady states
        - Related to irreversibility
    """
    sigma = np.sum(fluxes * forces)
    return max(0.0, sigma)  # Non-negative by second law


def time_reversal_asymmetry(observable: np.ndarray,
                           lag: int = 1) -> float:
    """
    Quantify time-reversal asymmetry of observable.

    Measures asymmetry: A(τ) = <O(t+τ) - O(t-τ)>²
    Zero for time-reversible processes.

    Args:
        observable: (n_frames,) time series
        lag: Time lag in frames

    Returns:
        asymmetry: Time-reversal asymmetry measure

    Notes:
        - Zero at equilibrium (microscopic reversibility)
        - Non-zero indicates broken time-reversal symmetry
        - Requires stationary statistics
    """
    n_frames = len(observable)

    # Forward differences
    forward_diffs = observable[lag:] - observable[:-lag]

    # Backward differences (time-reversed)
    backward_diffs = observable[:-lag] - observable[lag:]

    # Asymmetry = mean squared difference
    asymmetry = np.mean((forward_diffs - backward_diffs)**2)

    return asymmetry


def fluctuation_dissipation_test(observable: np.ndarray,
                                response_function: np.ndarray,
                                temperature: float = 300.0,
                                verbose: bool = True) -> float:
    """
    Test fluctuation-dissipation theorem.

    FDT: C(t) = (kT) · χ(t)
    where C(t) is autocorrelation, χ(t) is response function

    Args:
        observable: (n_frames,) equilibrium fluctuations
        response_function: (n_frames,) response to perturbation
        temperature: Temperature in Kelvin
        verbose: Print results

    Returns:
        fdt_ratio: Ratio of LHS/RHS (should be ~1 at equilibrium)

    Notes:
        - Assumes linear response regime
        - Violations indicate non-equilibrium conditions
        - Requires independent measurement of C(t) and χ(t)
    """
    from ..analysis.correlations import autocorrelation_function

    kB = 0.001987204  # kcal/(mol·K)
    kT = kB * temperature

    # Compute autocorrelation
    acf = autocorrelation_function(observable, max_lag=len(response_function))

    # Normalize both
    acf_norm = acf / acf[0]
    response_norm = response_function / response_function[0]

    # FDT ratio: C(t) / (kT · χ(t))
    # After normalization: C(t)/C(0) vs χ(t)/χ(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = acf_norm / response_norm

    # Average ratio (excluding infinities/NaNs)
    finite_mask = np.isfinite(ratio)
    if np.sum(finite_mask) > 0:
        fdt_ratio = np.mean(ratio[finite_mask])
    else:
        fdt_ratio = np.nan

    if verbose:
        print("Fluctuation-Dissipation Test")
        print("=" * 50)
        print(f"FDT ratio: {fdt_ratio:.3f} (should be ~1.0)")

        if np.abs(fdt_ratio - 1.0) < 0.1:
            print("  → FDT satisfied - equilibrium")
        else:
            print("  → FDT violated - non-equilibrium or nonlinear response")

    return fdt_ratio


def phase_space_compressibility(positions: np.ndarray,
                                velocities: np.ndarray,
                                masses: np.ndarray) -> float:
    """
    Compute phase space compressibility (Liouville theorem test).

    At equilibrium: ∇·(ρv) = 0 (incompressible phase space flow)
    Non-zero indicates non-equilibrium driving.

    Args:
        positions: (n_atoms, 3) atomic positions
        velocities: (n_atoms, 3) atomic velocities
        masses: (n_atoms,) atomic masses

    Returns:
        compressibility: Phase space compression rate

    Notes:
        - Zero for Hamiltonian dynamics
        - Non-zero for dissipative/driven systems
        - Related to entropy production
    """
    # Simplified estimate: divergence of velocity field
    # ∇·v ≈ (∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z)

    # Compute spatial gradients of velocity
    div_v = 0.0

    for dim in range(3):
        # Finite differences
        if len(positions) > 1:
            dv = np.gradient(velocities[:, dim])
            dx = np.gradient(positions[:, dim])

            with np.errstate(divide='ignore', invalid='ignore'):
                grad = dv / dx

            finite_mask = np.isfinite(grad)
            if np.sum(finite_mask) > 0:
                div_v += np.mean(grad[finite_mask])

    compressibility = div_v

    return compressibility


def irreversibility_index(trajectory_forward: np.ndarray,
                        trajectory_backward: np.ndarray) -> Tuple[float, float]:
    """
    Quantify irreversibility by comparing forward and time-reversed trajectories.

    Args:
        trajectory_forward: (n_frames, n_features) forward trajectory
        trajectory_backward: (n_frames, n_features) time-reversed trajectory

    Returns:
        kl_divergence: Kullback-Leibler divergence D(P_forward || P_backward)
        ks_statistic: Kolmogorov-Smirnov test statistic

    Notes:
        - Both measures are zero for time-reversible processes
        - Requires matched forward/backward trajectory pairs
        - Can use from independent simulations or time-reversed analysis
    """
    # Flatten trajectories
    forward_flat = trajectory_forward.flatten()
    backward_flat = trajectory_backward.flatten()

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = ks_2samp(forward_flat, backward_flat)

    # Estimate KL divergence via histograms
    bins = 50
    hist_forward, bin_edges = np.histogram(forward_flat, bins=bins, density=True)
    hist_backward, _ = np.histogram(backward_flat, bins=bin_edges, density=True)

    # Add small constant to avoid log(0)
    eps = 1e-10
    hist_forward = hist_forward + eps
    hist_backward = hist_backward + eps

    # Normalize
    hist_forward = hist_forward / np.sum(hist_forward)
    hist_backward = hist_backward / np.sum(hist_backward)

    # KL divergence: D(P||Q) = Σ P(i) log(P(i)/Q(i))
    kl_divergence = np.sum(hist_forward * np.log(hist_forward / hist_backward))

    return kl_divergence, ks_stat


def crooks_fluctuation_theorem(work_forward: np.ndarray,
                               work_backward: np.ndarray,
                               temperature: float = 300.0,
                               verbose: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Test Crooks fluctuation theorem.

    Crooks: P_F(W) / P_R(-W) = exp[(W - ΔF) / kT]

    Args:
        work_forward: (n_samples,) work values from forward process
        work_backward: (n_samples,) work values from reverse process
        temperature: Temperature in Kelvin
        verbose: Print results

    Returns:
        DeltaF: Free energy difference (kcal/mol)
        work_bins: Histogram bins for work
        crooks_ratio: P_F(W) / P_R(-W) for each bin

    Notes:
        - Exact for arbitrary non-equilibrium processes
        - Allows free energy calculation from non-equilibrium work
        - Requires many samples for accurate histograms
    """
    kB = 0.001987204  # kcal/(mol·K)
    kT = kB * temperature

    # Create histograms
    n_bins = 30
    work_min = min(work_forward.min(), -work_backward.max())
    work_max = max(work_forward.max(), -work_backward.min())

    bins = np.linspace(work_min, work_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Forward distribution P_F(W)
    hist_forward, _ = np.histogram(work_forward, bins=bins, density=True)

    # Reverse distribution P_R(-W)
    hist_backward, _ = np.histogram(-work_backward, bins=bins, density=True)

    # Crooks ratio
    eps = 1e-10
    crooks_ratio = (hist_forward + eps) / (hist_backward + eps)

    # Extract ΔF from intersection point (where ratio = 1)
    # At W = ΔF: P_F(ΔF) = P_R(-ΔF)
    crossing_indices = np.where(np.diff(np.sign(np.log(crooks_ratio))))[0]

    if len(crossing_indices) > 0:
        crossing_idx = crossing_indices[0]
        DeltaF = bin_centers[crossing_idx]
    else:
        # Fallback: use Jarzynski equality
        DeltaF = -kT * np.log(np.mean(np.exp(-work_forward / kT)))

    if verbose:
        print("Crooks Fluctuation Theorem Test")
        print("=" * 50)
        print(f"Free energy difference: ΔF = {DeltaF:.3f} kcal/mol")
        print(f"Mean forward work: <W_F> = {np.mean(work_forward):.3f} kcal/mol")
        print(f"Mean reverse work: <W_R> = {np.mean(work_backward):.3f} kcal/mol")

        dissipation = np.mean(work_forward) - DeltaF
        print(f"Dissipated work: {dissipation:.3f} kcal/mol")

    return DeltaF, bin_centers, crooks_ratio


if __name__ == '__main__':
    # Example usage
    print("Non-equilibrium Thermodynamics Module")
    print("======================================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.analysis.nonequilibrium import test_detailed_balance")
    print()
    print("# Test if system is at equilibrium")
    print("balance_matrix, violation = test_detailed_balance(state_traj, n_states=3)")
    print()
    print("if violation < 0.1:")
    print("    print('System at equilibrium')")
    print("else:")
    print("    print('Non-equilibrium dynamics detected')")
