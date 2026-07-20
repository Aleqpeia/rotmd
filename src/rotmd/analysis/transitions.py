"""
Transition State Analysis

This module provides utilities for analyzing transitions between
conformational states in protein orientation space.

Key Features:
- Reactive flux calculation
- Transmission coefficient κ
- Committor probability p_B(x)
- Transition path sampling
- Free energy barriers from transition rates

Theoretical Background:
- Transition State Theory: k = κ · k_TST
- k_TST = (kT/h) exp(-ΔF‡/kT): TST rate
- κ: transmission coefficient (0 < κ ≤ 1)
- Accounts for recrossings of dividing surface

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from scipy.ndimage import label


def identify_states(observable: np.ndarray,
                   thresholds: List[Tuple[float, float]],
                   min_duration: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Identify discrete states from continuous observable.

    Args:
        observable: (n_frames,) time series of observable (e.g., θ angle)
        thresholds: List of (min, max) ranges defining each state
                   Example: [(0, 30), (60, 90)] for two states
        min_duration: Minimum frames to count as stable state

    Returns:
        state_trajectory: (n_frames,) integer array with state indices
                         -1 = transition region, 0,1,... = stable states
        state_info: Dictionary with state populations and lifetimes

    Notes:
        - Filters out short-lived fluctuations using min_duration
        - Useful for defining metastable basins in free energy landscape
    """
    n_frames = len(observable)
    state_trajectory = -np.ones(n_frames, dtype=int)  # -1 = transition

    # Assign frames to states based on thresholds
    for state_idx, (min_val, max_val) in enumerate(thresholds):
        mask = (observable >= min_val) & (observable <= max_val)
        state_trajectory[mask] = state_idx

    # Filter short-lived states
    for state_idx in range(len(thresholds)):
        # Find connected regions of this state
        mask = (state_trajectory == state_idx)
        labeled, n_regions = label(mask)

        for region_idx in range(1, n_regions + 1):
            region_mask = (labeled == region_idx)
            region_size = np.sum(region_mask)

            # Mark as transition if too short
            if region_size < min_duration:
                state_trajectory[region_mask] = -1

    # Compute state statistics
    state_info = {}
    for state_idx in range(len(thresholds)):
        mask = (state_trajectory == state_idx)
        population = np.sum(mask) / n_frames
        state_info[f'state_{state_idx}_population'] = population

    transition_fraction = np.sum(state_trajectory == -1) / n_frames
    state_info['transition_fraction'] = transition_fraction

    return state_trajectory, state_info


def detect_transitions(state_trajectory: np.ndarray,
                       from_state: int,
                       to_state: int) -> List[Tuple[int, int]]:
    """
    Detect all transitions from state A to state B.

    Args:
        state_trajectory: (n_frames,) state indices from identify_states()
        from_state: Initial state index
        to_state: Final state index

    Returns:
        transitions: List of (start_frame, end_frame) tuples for each transition

    Notes:
        - Start frame: last frame in from_state before transition
        - End frame: first frame in to_state after transition
        - Includes intermediate transition region
    """
    transitions = []

    in_from_state = False
    start_frame = None

    for i, state in enumerate(state_trajectory):
        if state == from_state:
            in_from_state = True
            start_frame = i

        elif state == to_state and in_from_state:
            # Transition completed
            transitions.append((start_frame, i))
            in_from_state = False

        elif state != -1 and state != from_state:
            # Entered different stable state without reaching to_state
            in_from_state = False

    return transitions


def compute_reactive_flux(positions: np.ndarray,
                         velocities: np.ndarray,
                         dividing_surface: Callable[[np.ndarray], float],
                         threshold: float = 0.0) -> float:
    """
    Compute reactive flux through dividing surface.

    Reactive flux: J = <δ(ξ - ξ‡) · ξ̇ · H(ξ̇)>
    where ξ is reaction coordinate, ξ‡ is transition state value

    Args:
        positions: (n_frames, n_atoms, 3) atomic positions
        velocities: (n_frames, n_atoms, 3) atomic velocities
        dividing_surface: Function that computes ξ(positions)
        threshold: ξ‡ value for dividing surface

    Returns:
        flux: Reactive flux (forward crossings per unit time)

    Notes:
        - Only counts forward crossings (ξ̇ > 0 at ξ = ξ‡)
        - Related to rate: k = J / P_A (flux / reactant population)
    """
    n_frames = len(positions)

    # Compute reaction coordinate time series
    xi = np.array([dividing_surface(positions[i]) for i in range(n_frames)])

    # Compute time derivative (numerical)
    xi_dot = np.gradient(xi)

    # Count forward crossings
    crossings = []
    for i in range(1, n_frames):
        # Check if crossed threshold
        if xi[i-1] < threshold <= xi[i] and xi_dot[i] > 0:
            crossings.append(i)

    flux = len(crossings) / n_frames

    return flux


def transmission_coefficient(state_trajectory: np.ndarray,
                            from_state: int,
                            to_state: int,
                            transition_region: int = -1,
                            verbose: bool = True) -> Tuple[float, Dict]:
    """
    Compute transmission coefficient κ.

    κ = (actual transition rate) / (TST rate)
    Accounts for recrossings of dividing surface.

    Args:
        state_trajectory: (n_frames,) state indices
        from_state: Reactant state index
        to_state: Product state index
        transition_region: Index for transition region (default -1)
        verbose: Print analysis details

    Returns:
        kappa: Transmission coefficient (0 < κ ≤ 1)
        info: Dictionary with crossing statistics

    Notes:
        - κ = 1: no recrossings (TST exact)
        - κ < 1: recrossings reduce rate below TST prediction
        - Requires good statistics (many transitions)
    """
    # Detect all transitions
    AB_transitions = detect_transitions(state_trajectory, from_state, to_state)
    BA_transitions = detect_transitions(state_trajectory, to_state, from_state)

    n_AB = len(AB_transitions)
    n_BA = len(BA_transitions)

    # Count barrier crossings (entries into transition region from either state)
    barrier_crossings = 0

    for i in range(1, len(state_trajectory)):
        prev_state = state_trajectory[i-1]
        curr_state = state_trajectory[i]

        # Count crossing if entering transition region from stable state
        if (prev_state == from_state or prev_state == to_state) and \
           curr_state == transition_region:
            barrier_crossings += 1

    # Transmission coefficient
    # κ = (successful transitions) / (total barrier crossings)
    if barrier_crossings > 0:
        kappa = (n_AB + n_BA) / barrier_crossings
    else:
        kappa = 0.0

    info = {
        'n_AB_transitions': n_AB,
        'n_BA_transitions': n_BA,
        'barrier_crossings': barrier_crossings,
        'kappa': kappa
    }

    if verbose:
        print("Transmission Coefficient Analysis")
        print("=" * 50)
        print(f"State {from_state} → {to_state} transitions: {n_AB}")
        print(f"State {to_state} → {from_state} transitions: {n_BA}")
        print(f"Total barrier crossings: {barrier_crossings}")
        print(f"Transmission coefficient: κ = {kappa:.3f}")

        if kappa < 0.5:
            print("  → Significant recrossings detected")
        elif kappa > 0.9:
            print("  → Near-TST behavior (few recrossings)")

    return kappa, info


def committor_probability(state_trajectory: np.ndarray,
                         from_state: int,
                         to_state: int,
                         n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute committor probability p_B(ξ).

    p_B(ξ) = probability of reaching state B before state A,
             starting from configuration with reaction coordinate ξ

    Args:
        state_trajectory: (n_frames,) state indices
        from_state: State A index
        to_state: State B index
        n_bins: Number of bins for discretizing transition region

    Returns:
        bin_centers: (n_bins,) bin centers for reaction coordinate
        p_B: (n_bins,) committor probability

    Notes:
        - p_B = 0 in state A, p_B = 1 in state B
        - p_B = 0.5 defines optimal transition state
        - Requires many transitions for good statistics
    """
    # Find all frames in transition region
    transition_mask = (state_trajectory != from_state) & \
                     (state_trajectory != to_state) & \
                     (state_trajectory >= 0)  # Exclude unassigned

    transition_frames = np.where(transition_mask)[0]

    # For each transition frame, determine final destination
    committor_samples = []

    for frame in transition_frames:
        # Look forward to see which state is reached first
        for future_frame in range(frame + 1, len(state_trajectory)):
            future_state = state_trajectory[future_frame]

            if future_state == to_state:
                # Committed to B
                committor_samples.append((frame, 1))
                break
            elif future_state == from_state:
                # Returned to A
                committor_samples.append((frame, 0))
                break

    if len(committor_samples) == 0:
        # No statistics available
        return np.array([]), np.array([])

    # Bin by frame index (proxy for reaction coordinate)
    frames, commitments = zip(*committor_samples)
    frames = np.array(frames)
    commitments = np.array(commitments)

    bins = np.linspace(frames.min(), frames.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p_B = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (frames >= bins[i]) & (frames < bins[i+1])
        if np.sum(mask) > 0:
            p_B[i] = np.mean(commitments[mask])

    return bin_centers, p_B


def transition_path_ensemble(state_trajectory: np.ndarray,
                            from_state: int,
                            to_state: int,
                            include_data: Optional[np.ndarray] = None) -> Dict:
    """
    Extract ensemble of transition paths.

    Args:
        state_trajectory: (n_frames,) state indices
        from_state: Initial state
        to_state: Final state
        include_data: Optional (n_frames, ...) data to extract along paths

    Returns:
        ensemble: Dictionary containing:
            - paths: List of transition paths (state trajectories)
            - lengths: Distribution of transition path lengths
            - data_along_paths: If include_data provided

    Notes:
        - Useful for characterizing transition mechanisms
        - Can compute averages over transition path ensemble
    """
    transitions = detect_transitions(state_trajectory, from_state, to_state)

    paths = []
    lengths = []
    data_along_paths = [] if include_data is not None else None

    for start, end in transitions:
        path = state_trajectory[start:end+1]
        paths.append(path)
        lengths.append(len(path))

        if include_data is not None:
            data_along_paths.append(include_data[start:end+1])

    ensemble = {
        'paths': paths,
        'lengths': np.array(lengths),
        'n_transitions': len(transitions)
    }

    if data_along_paths is not None:
        ensemble['data_along_paths'] = data_along_paths

    return ensemble


def free_energy_barrier_from_rate(k_AB: float,
                                  k_BA: float,
                                  temperature: float = 300.0) -> Tuple[float, float]:
    """
    Estimate free energy barriers from transition rates.

    Uses Arrhenius relation: k = A exp(-ΔF‡/kT)

    Args:
        k_AB: Forward rate (1/ps)
        k_BA: Backward rate (1/ps)
        temperature: Temperature in Kelvin

    Returns:
        DeltaF_AB: Forward barrier (kcal/mol)
        DeltaF_BA: Backward barrier (kcal/mol)

    Notes:
        - Assumes Arrhenius form (valid at high barriers)
        - Prefactor A typically ~kT/h ≈ 6 ps⁻¹ at 300K
    """
    kB = 0.001987204  # kcal/(mol·K)
    h = 1.5836e-4  # ps·kcal/mol (reduced Planck constant)

    kT = kB * temperature
    A = kT / h  # TST prefactor

    # ΔF‡ = -kT ln(k/A)
    DeltaF_AB = -kT * np.log(k_AB / A) if k_AB > 0 else np.inf
    DeltaF_BA = -kT * np.log(k_BA / A) if k_BA > 0 else np.inf

    return DeltaF_AB, DeltaF_BA


if __name__ == '__main__':
    # Example usage
    print("Transition State Analysis Module")
    print("=================================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.analysis.transitions import identify_states, transmission_coefficient")
    print()
    print("# Identify metastable states from nutation angle")
    print("states, info = identify_states(theta, thresholds=[(0, 30), (60, 90)])")
    print()
    print("# Compute transmission coefficient")
    print("kappa, stats = transmission_coefficient(states, from_state=0, to_state=1)")
    print()
    print(f"print(f'κ = {kappa:.3f}')  # Accounts for recrossings")
