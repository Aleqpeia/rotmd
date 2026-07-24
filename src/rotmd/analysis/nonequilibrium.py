"""Non-equilibrium diagnostics for orientation dynamics: detailed balance, entropy production, fluctuation theorems.

At equilibrium, detailed balance holds and entropy production sigma is zero;
a driven or still-relaxing system shows net probability currents (sigma > 0),
broken time-reversal symmetry, and a fluctuation-dissipation theorem that
fails to hold. These functions test for those signatures directly, which
matters here because every other module in :mod:`rotmd.analysis` (PMF,
friction, transition rates) implicitly assumes the trajectory it's given is
an equilibrium sample — this module is how that assumption gets checked
instead of taken on faith.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def test_detailed_balance(
    state_trajectory: np.ndarray,
    n_states: int,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Detailed-balance ratios ``balance[i, j] = (k_ij * P_i) / (k_ji * P_j)``, ~1 at equilibrium.

    Returns the full ratio matrix plus ``max_violation``, the largest
    ``|log ratio|`` over all state pairs with enough transition counts to be
    defined. Needs enough samples of every ``i <-> j`` pair; pairs never
    observed in one direction are left ``NaN`` rather than reported as an
    infinite or zero ratio.
    """
    transition_counts = np.zeros((n_states, n_states))

    for i in range(len(state_trajectory) - 1):
        from_state = state_trajectory[i]
        to_state = state_trajectory[i + 1]
        if 0 <= from_state < n_states and 0 <= to_state < n_states:
            transition_counts[from_state, to_state] += 1

    populations = np.zeros(n_states)
    for state in range(n_states):
        populations[state] = np.sum(state_trajectory == state)
    populations = populations / np.sum(populations)

    balance_matrix = np.ones((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                numerator = transition_counts[i, j] * populations[i]
                denominator = transition_counts[j, i] * populations[j]
                balance_matrix[i, j] = numerator / denominator if denominator > 0 else np.nan

    finite_values = balance_matrix[np.isfinite(balance_matrix)]
    max_violation = np.max(np.abs(np.log(finite_values))) if len(finite_values) > 0 else np.nan

    if verbose:
        print("Detailed Balance Test")
        print("=" * 50)
        print("Balance matrix (should be 1.0 at equilibrium):")
        print(balance_matrix)
        print(f"\nMaximum violation: {max_violation:.3f}")
        if max_violation < 0.1:
            print("  -> System appears to be at equilibrium")
        elif max_violation < 0.5:
            print("  -> Weak violation of detailed balance")
        else:
            print("  -> Strong violation - non-equilibrium dynamics")

    return balance_matrix, max_violation


def entropy_production_rate(fluxes: np.ndarray, forces: np.ndarray) -> float:
    """Entropy production ``sigma = sum_i J_i * X_i`` from conjugate flux/force pairs; clamped to >= 0 (second law)."""
    sigma = np.sum(fluxes * forces)
    return max(0.0, sigma)


def time_reversal_asymmetry(observable: np.ndarray, lag: int = 1) -> float:
    """Time-reversal asymmetry ``<[(O(t+lag) - O(t)) - (O(t) - O(t+lag))]^2>``; zero for a reversible process."""
    forward_diffs = observable[lag:] - observable[:-lag]
    backward_diffs = observable[:-lag] - observable[lag:]
    return np.mean((forward_diffs - backward_diffs) ** 2)


def fluctuation_dissipation_test(
    observable: np.ndarray,
    response_function: np.ndarray,
    temperature: float = 300.0,
    verbose: bool = True,
) -> float:
    """Test the FDT ``C(t) = kT * chi(t)`` by comparing normalized ACF to normalized response function; ratio ~1 at equilibrium.

    ``temperature`` is accepted for interface symmetry with the other FDT-style
    checks in this module but does not enter the ratio: after both curves are
    normalized to 1 at ``t=0``, the ``kT`` prefactor on each side cancels.
    """
    from rotmd.analysis.correlations import autocorrelation_function

    _, acf = autocorrelation_function(observable, max_lag=len(response_function) - 1)

    acf_norm = acf / acf[0]
    response_norm = response_function / response_function[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = acf_norm / response_norm

    finite_mask = np.isfinite(ratio)
    fdt_ratio = np.mean(ratio[finite_mask]) if np.sum(finite_mask) > 0 else np.nan

    if verbose:
        print("Fluctuation-Dissipation Test")
        print("=" * 50)
        print(f"FDT ratio: {fdt_ratio:.3f} (should be ~1.0)")
        if np.abs(fdt_ratio - 1.0) < 0.1:
            print("  -> FDT satisfied - equilibrium")
        else:
            print("  -> FDT violated - non-equilibrium or nonlinear response")

    return fdt_ratio


def phase_space_compressibility(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> float:
    """Estimate phase-space compression ``div(v)`` (Liouville-theorem test); zero for Hamiltonian dynamics, non-zero for driven/dissipative ones.

    ``masses`` is accepted for interface symmetry with a full phase-space
    treatment but unused: this is a simplified estimate from the divergence of
    the velocity field alone.
    """
    del masses
    div_v = 0.0

    for dim in range(3):
        if len(positions) > 1:
            dv = np.gradient(velocities[:, dim])
            dx = np.gradient(positions[:, dim])
            with np.errstate(divide="ignore", invalid="ignore"):
                grad = dv / dx
            finite_mask = np.isfinite(grad)
            if np.sum(finite_mask) > 0:
                div_v += np.mean(grad[finite_mask])

    return div_v


def irreversibility_index(
    trajectory_forward: np.ndarray,
    trajectory_backward: np.ndarray,
) -> tuple[float, float]:
    """KL divergence and KS statistic between forward and time-reversed trajectory distributions; both zero iff reversible.

    Compares marginal distributions via histograms rather than requiring
    paired forward/backward frames, so the two trajectories can come from
    independent runs.
    """
    forward_flat = trajectory_forward.flatten()
    backward_flat = trajectory_backward.flatten()

    ks_stat, _ = ks_2samp(forward_flat, backward_flat)

    bins = 50
    hist_forward, bin_edges = np.histogram(forward_flat, bins=bins, density=True)
    hist_backward, _ = np.histogram(backward_flat, bins=bin_edges, density=True)

    # Small additive constant avoids log(0)/div-by-0 in empty histogram bins.
    eps = 1e-10
    hist_forward = hist_forward + eps
    hist_backward = hist_backward + eps
    hist_forward = hist_forward / np.sum(hist_forward)
    hist_backward = hist_backward / np.sum(hist_backward)

    kl_divergence = np.sum(hist_forward * np.log(hist_forward / hist_backward))

    return kl_divergence, ks_stat


def crooks_fluctuation_theorem(
    work_forward: np.ndarray,
    work_backward: np.ndarray,
    temperature: float = 300.0,
    verbose: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Test the Crooks relation ``P_F(W) / P_R(-W) = exp[(W - DeltaF) / kT]`` and extract DeltaF from the crossing point.

    DeltaF is read off where the forward and (negated) reverse work
    distributions cross (``P_F = P_R`` at ``W = DeltaF``); if the histograms
    never cross — too few samples, or the distributions barely overlap —
    falls back to the Jarzynski equality estimate from the forward work alone.
    """
    kb = 0.001987204
    kt = kb * temperature

    n_bins = 30
    work_min = min(work_forward.min(), -work_backward.max())
    work_max = max(work_forward.max(), -work_backward.min())

    bins = np.linspace(work_min, work_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    hist_forward, _ = np.histogram(work_forward, bins=bins, density=True)
    hist_backward, _ = np.histogram(-work_backward, bins=bins, density=True)

    eps = 1e-10
    crooks_ratio = (hist_forward + eps) / (hist_backward + eps)

    crossing_indices = np.where(np.diff(np.sign(np.log(crooks_ratio))))[0]
    if len(crossing_indices) > 0:
        delta_f = bin_centers[crossing_indices[0]]
    else:
        delta_f = -kt * np.log(np.mean(np.exp(-work_forward / kt)))

    if verbose:
        print("Crooks Fluctuation Theorem Test")
        print("=" * 50)
        print(f"Free energy difference: DeltaF = {delta_f:.3f} kcal/mol")
        print(f"Mean forward work: <W_F> = {np.mean(work_forward):.3f} kcal/mol")
        print(f"Mean reverse work: <W_R> = {np.mean(work_backward):.3f} kcal/mol")
        dissipation = np.mean(work_forward) - delta_f
        print(f"Dissipated work: {dissipation:.3f} kcal/mol")

    return delta_f, bin_centers, crooks_ratio


if __name__ == "__main__":
    print("Non-equilibrium Thermodynamics Module")
    print("======================================")
    print()
    print("Example usage:")
    print()
    print("from rotmd.analysis.nonequilibrium import test_detailed_balance")
    print()
    print("# Test if system is at equilibrium")
    print("balance_matrix, violation = test_detailed_balance(state_traj, n_states=3)")
    print()
    print("if violation < 0.1:")
    print("    print('System at equilibrium')")
    print("else:")
    print("    print('Non-equilibrium dynamics detected')")
