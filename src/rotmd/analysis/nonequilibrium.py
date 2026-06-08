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

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import numpy as np
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


# ==============================================================================
# Broken detailed balance in a 2D collective-variable plane
# ==============================================================================
#
# `test_detailed_balance` above works on a 1D sequence of *discrete* states. For
# collective behaviour of a bilayer the informative coordinates are continuous
# and coupled -- e.g. (area-per-lipid, thickness), (tilt, splay), or a pair of
# leaflet order parameters. Even when the full system is at equilibrium, the
# dynamics *projected* onto two such slow coordinates is non-Markovian and can
# exhibit a non-zero steady-state probability current that circulates in closed
# loops. Persistent circulation is a direct, model-free signature of broken
# detailed balance in the projected space (Battle et al., Science 2016).
#
# We estimate the current field J(r) = p(r) v(r) on a grid, then isolate its
# rotational part via the discrete curl (scalar vorticity in 2D). At true
# equilibrium J = 0 up to finite-sampling noise; a significance test against a
# detailed-balance null tells circulation apart from that noise floor.


@dataclass
class ProbabilityCurrentField:
    """Steady-state probability current of a 2D collective-variable trajectory.

    All grids are indexed ``[ix, iy]`` (first axis = ``x`` coordinate). The
    current components ``jx``/``jy`` are probability currents ``p * v`` (units
    of ``1 / time``); ``curl`` is the scalar vorticity ``dJy/dx - dJx/dy`` whose
    area integral is the net circulation.
    """

    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray
    jx: np.ndarray
    jy: np.ndarray
    curl: np.ndarray
    net_circulation: float
    cycling_intensity: float
    current_magnitude: float
    n_transitions: int
    dt: float

    def to_dict(self) -> Dict[str, float | int]:
        """Scalar summary (the grids are omitted; they are for plotting)."""
        return {
            "net_circulation": float(self.net_circulation),
            "cycling_intensity": float(self.cycling_intensity),
            "current_magnitude": float(self.current_magnitude),
            "n_transitions": int(self.n_transitions),
            "dt": float(self.dt),
        }


def _bin_edges(
    x: np.ndarray,
    y: np.ndarray,
    bins: int | Tuple[int, int],
    extent: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build grid edges from a bin count and an optional explicit extent."""
    nbx, nby = (bins, bins) if isinstance(bins, int) else bins
    if nbx < 2 or nby < 2:
        raise ValueError("need at least 2 bins per axis to take a discrete curl")
    if extent is None:
        xr = (float(np.min(x)), float(np.max(x)))
        yr = (float(np.min(y)), float(np.max(y)))
    else:
        xr, yr = ((float(extent[0][0]), float(extent[0][1])),
                  (float(extent[1][0]), float(extent[1][1])))
    if xr[0] == xr[1] or yr[0] == yr[1]:
        raise ValueError("a collective variable has zero range; cannot grid it")
    return (
        np.linspace(xr[0], xr[1], nbx + 1),
        np.linspace(yr[0], yr[1], nby + 1),
    )


def _current_from_signed_displacements(
    flat_bins: np.ndarray,
    dx_disp: np.ndarray,
    dy_disp: np.ndarray,
    shape: Tuple[int, int],
    n_transitions: int,
    dt: float,
    dxw: float,
    dyw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Accumulate the current field and its curl from binned displacements.

    Factored out so the significance test can recompute the current cheaply for
    each resample: the (expensive) bin assignment in ``flat_bins`` is fixed and
    only the displacement *signs* change between resamples.
    """
    nbx, nby = shape
    size = nbx * nby
    # J = (1 / (N dt)) * sum of displacements whose midpoint falls in the cell.
    # This is exactly p(cell) * <v|cell>, the probability current.
    sum_dx = np.bincount(flat_bins, weights=dx_disp, minlength=size).reshape(shape)
    sum_dy = np.bincount(flat_bins, weights=dy_disp, minlength=size).reshape(shape)
    jx = sum_dx / (n_transitions * dt)
    jy = sum_dy / (n_transitions * dt)

    # Scalar vorticity dJy/dx - dJx/dy. Axis 0 is x (spacing dxw), axis 1 is y.
    curl = np.gradient(jy, dxw, axis=0) - np.gradient(jx, dyw, axis=1)
    cycling_intensity = float(np.sum(np.abs(curl)) * dxw * dyw)
    return jx, jy, curl, cycling_intensity


def probability_current_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int | Tuple[int, int] = 20,
    dt: float = 1.0,
    extent: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> ProbabilityCurrentField:
    """
    Estimate the steady-state probability current of a 2D CV trajectory.

    Each consecutive frame pair contributes a displacement assigned to the grid
    cell containing its midpoint; averaging these gives the current field
    ``J = p * v``. The rotational part (``curl``) is the signature of interest:
    closed-loop circulation means probability is being pumped around a cycle,
    which is impossible under detailed balance.

    Args:
        x: (n_frames,) first collective variable (e.g. area per lipid).
        y: (n_frames,) second collective variable (e.g. bilayer thickness).
        bins: Grid resolution, an int (square grid) or ``(nx, ny)``.
        dt: Time between consecutive frames (sets the current's time units).
        extent: Optional ``((xmin, xmax), (ymin, ymax))`` to fix the grid;
            defaults to the data range.

    Returns:
        A :class:`ProbabilityCurrentField`. Use ``net_circulation`` (signed) or
        ``cycling_intensity`` (rotational activity, robust to multiple loops) as
        scalar summaries; the grids support quiver/stream plots.

    Raises:
        ValueError: If the series are too short, mismatched, or degenerate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same length")
    if x.size < 3:
        raise ValueError("need at least 3 frames to form transitions")

    x_edges, y_edges = _bin_edges(x, y, bins, extent)
    nbx, nby = x_edges.size - 1, y_edges.size - 1
    dxw = float(x_edges[1] - x_edges[0])
    dyw = float(y_edges[1] - y_edges[0])

    # Transitions live at the midpoint of each consecutive frame pair.
    mx = 0.5 * (x[:-1] + x[1:])
    my = 0.5 * (y[:-1] + y[1:])
    dx_disp = x[1:] - x[:-1]
    dy_disp = y[1:] - y[:-1]
    n_transitions = mx.size

    # Clip so midpoints on the upper edge land in the last cell, not out of range.
    ix = np.clip(np.digitize(mx, x_edges) - 1, 0, nbx - 1)
    iy = np.clip(np.digitize(my, y_edges) - 1, 0, nby - 1)
    flat = ix * nby + iy

    counts = np.bincount(flat, minlength=nbx * nby).reshape((nbx, nby))
    density = counts / n_transitions

    jx, jy, curl, cycling_intensity = _current_from_signed_displacements(
        flat, dx_disp, dy_disp, (nbx, nby), n_transitions, dt, dxw, dyw
    )
    net_circulation = float(np.sum(curl) * dxw * dyw)
    current_magnitude = float(np.sum(np.hypot(jx, jy)))

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return ProbabilityCurrentField(
        x_centers=x_centers,
        y_centers=y_centers,
        density=density,
        jx=jx,
        jy=jy,
        curl=curl,
        net_circulation=net_circulation,
        cycling_intensity=cycling_intensity,
        current_magnitude=current_magnitude,
        n_transitions=n_transitions,
        dt=dt,
    )


@dataclass
class DetailedBalanceTest:
    """Result of testing broken detailed balance in a 2D CV plane."""

    field: ProbabilityCurrentField
    statistic: float
    p_value: float
    null_mean: float
    null_q95: float
    n_resamples: int
    block_size: int

    @property
    def broken(self) -> bool:
        """Whether detailed balance is rejected at the 5% level."""
        return self.p_value < 0.05

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "null_mean": float(self.null_mean),
            "null_q95": float(self.null_q95),
            "n_resamples": int(self.n_resamples),
            "block_size": int(self.block_size),
            "broken": bool(self.broken),
        }


def broken_detailed_balance_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int | Tuple[int, int] = 20,
    dt: float = 1.0,
    extent: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    n_resamples: int = 500,
    block_size: int = 1,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> DetailedBalanceTest:
    """
    Test for broken detailed balance via circulation in a 2D CV plane.

    The observed statistic is the current field's ``cycling_intensity`` (the
    integrated absolute curl), which targets *rotational* flow specifically --
    the only part of the current that can persist in a steady state and the
    fingerprint of broken detailed balance.

    Null model -- ``J = 0`` (the necessary condition for detailed balance): we
    keep every displacement's magnitude and spatial location but randomise its
    *direction* by flipping the sign of each transition. This reproduces the
    finite-sampling noise floor of the curl under "no net flow", so the p-value
    is the fraction of null resamples whose cycling matches or exceeds the data.

    Because consecutive transitions are autocorrelated, signs are flipped in
    contiguous *blocks* of length ``block_size``; set this near the trajectory's
    correlation time (in frames) to avoid an anti-conservative p-value.

    Args:
        x, y: The two collective-variable series (n_frames,).
        bins: Grid resolution (int or ``(nx, ny)``).
        dt: Time between frames.
        extent: Optional fixed grid extent ``((xmin, xmax), (ymin, ymax))``.
        n_resamples: Number of sign-flip resamples for the null distribution.
        block_size: Block length (in transitions) for the sign flips.
        random_state: Seed for reproducibility.
        verbose: Print a short verdict.

    Returns:
        A :class:`DetailedBalanceTest` carrying the estimated field, the
        statistic, and the permutation p-value.
    """
    field = probability_current_2d(x, y, bins=bins, dt=dt, extent=extent)
    rng = np.random.default_rng(random_state)

    # Recompute the binning once so the null only has to re-weight by signs.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_edges, y_edges = _bin_edges(x, y, bins, extent)
    nbx, nby = x_edges.size - 1, y_edges.size - 1
    dxw = float(x_edges[1] - x_edges[0])
    dyw = float(y_edges[1] - y_edges[0])
    mx = 0.5 * (x[:-1] + x[1:])
    my = 0.5 * (y[:-1] + y[1:])
    dx_disp = x[1:] - x[:-1]
    dy_disp = y[1:] - y[:-1]
    ix = np.clip(np.digitize(mx, x_edges) - 1, 0, nbx - 1)
    iy = np.clip(np.digitize(my, y_edges) - 1, 0, nby - 1)
    flat = ix * nby + iy
    n_transitions = mx.size

    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    n_blocks = int(np.ceil(n_transitions / block_size))

    null_stats = np.empty(n_resamples)
    for r in range(n_resamples):
        block_signs = rng.choice((-1.0, 1.0), size=n_blocks)
        signs = np.repeat(block_signs, block_size)[:n_transitions]
        _, _, _, cycling = _current_from_signed_displacements(
            flat, signs * dx_disp, signs * dy_disp,
            (nbx, nby), n_transitions, dt, dxw, dyw,
        )
        null_stats[r] = cycling

    statistic = field.cycling_intensity
    # +1 in numerator and denominator: the observed data is itself a valid
    # arrangement under exchangeability (standard permutation-test correction).
    p_value = float((1 + np.sum(null_stats >= statistic)) / (1 + n_resamples))

    result = DetailedBalanceTest(
        field=field,
        statistic=float(statistic),
        p_value=p_value,
        null_mean=float(np.mean(null_stats)),
        null_q95=float(np.quantile(null_stats, 0.95)),
        n_resamples=int(n_resamples),
        block_size=int(block_size),
    )

    if verbose:
        print("Broken Detailed Balance Test (2D probability current)")
        print("=" * 50)
        print(f"cycling intensity : {statistic:.4g}")
        print(f"null mean / q95   : {result.null_mean:.4g} / {result.null_q95:.4g}")
        print(f"p-value           : {p_value:.4g}")
        if result.broken:
            print("  -> circulation exceeds the equilibrium noise floor:")
            print("     detailed balance is broken in these coordinates")
        else:
            print("  -> circulation consistent with equilibrium (no net cycling)")

    return result


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
