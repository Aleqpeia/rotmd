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
        state_info: Dictionary with, per state ``i``,
            ``state_{i}_population`` (occupancy fraction),
            ``state_{i}_n_episodes`` (number of surviving residence episodes),
            and ``state_{i}_mean_lifetime`` (mean episode length **in frames**;
            multiply by the frame timestep for a physical lifetime), plus the
            overall ``transition_fraction``.

    Notes:
        - Filters out short-lived fluctuations using min_duration.
        - Useful for defining metastable basins in a free energy landscape.

    .. note::
        The threshold ranges are expected to be **non-overlapping**. If they
        overlap, a frame in the overlap is assigned to the last matching state
        (later thresholds win), because assignment is sequential.
    """
    observable = np.asarray(observable, dtype=float)
    n_frames = observable.size
    state_trajectory = -np.ones(n_frames, dtype=int)  # -1 = transition

    if n_frames == 0:
        return state_trajectory, {'transition_fraction': 0.0}

    # Assign frames to states based on thresholds. Sequential assignment means
    # overlapping ranges resolve in favour of the later threshold (see note).
    for state_idx, (min_val, max_val) in enumerate(thresholds):
        mask = (observable >= min_val) & (observable <= max_val)
        state_trajectory[mask] = state_idx

    # Filter short-lived states: any contiguous residence shorter than
    # min_duration is demoted to the transition region (-1).
    for state_idx in range(len(thresholds)):
        mask = (state_trajectory == state_idx)
        labeled, n_regions = label(mask)
        for region_idx in range(1, n_regions + 1):
            region_mask = (labeled == region_idx)
            if np.sum(region_mask) < min_duration:
                state_trajectory[region_mask] = -1

    # Compute per-state statistics on the *filtered* trajectory, including the
    # residence-time (lifetime) distribution promised by the API.
    state_info = {}
    for state_idx in range(len(thresholds)):
        mask = (state_trajectory == state_idx)
        state_info[f'state_{state_idx}_population'] = float(np.sum(mask) / n_frames)

        labeled, n_regions = label(mask)
        if n_regions > 0:
            # Episode lengths are the sizes of the surviving connected regions.
            sizes = np.bincount(labeled.ravel())[1:]  # drop the background (0) bin
            mean_lifetime = float(np.mean(sizes))
        else:
            mean_lifetime = 0.0
        state_info[f'state_{state_idx}_n_episodes'] = int(n_regions)
        state_info[f'state_{state_idx}_mean_lifetime'] = mean_lifetime

    state_info['transition_fraction'] = float(np.sum(state_trajectory == -1) / n_frames)

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
                         dividing_surface: Callable[[np.ndarray], float],
                         threshold: float = 0.0,
                         dt: float = 1.0) -> float:
    r"""
    Forward reactive flux through the dividing surface :math:`\xi = \xi^\ddagger`.

    Counts upward crossings of the reaction coordinate :math:`\xi(\text{positions})`
    past ``threshold`` and divides by the total observation time, giving forward
    crossings per unit time.

    Args:
        positions: (n_frames, n_atoms, 3) atomic positions.
        dividing_surface: Function computing :math:`\xi` from one frame's
            positions.
        threshold: :math:`\xi^\ddagger`, the dividing-surface value.
        dt: Time between consecutive frames (sets the flux's time units).

    Returns:
        flux: Forward crossings per unit time.

    Notes:
        - The crossing test ``xi[i-1] < threshold <= xi[i]`` already selects
          *forward* (upward) crossings, so no separate :math:`\dot\xi > 0` test
          is needed. Velocities are intentionally **not** taken as an argument:
          the true :math:`\dot\xi = \nabla\xi \cdot v` is unavailable from a
          positions-only ``dividing_surface``, and finite-differencing
          :math:`\xi` (as a previous version did) double-counts the direction
          already encoded in the crossing test.
        - This overlaps with :func:`rotmd.analysis.crossings.count_level_crossings`;
          prefer that for pure crossing counts without a time normalisation.
        - Related to the rate via ``k = flux / P_A`` (flux per reactant
          population).
    """
    positions = np.asarray(positions)
    n_frames = len(positions)
    if n_frames < 2:
        return 0.0

    xi = np.array([dividing_surface(positions[i]) for i in range(n_frames)])
    forward_crossings = int(np.sum((xi[:-1] < threshold) & (xi[1:] >= threshold)))

    total_time = (n_frames - 1) * dt
    return forward_crossings / total_time if total_time > 0 else 0.0


def transmission_coefficient(state_trajectory: np.ndarray,
                            from_state: int,
                            to_state: int,
                            transition_region: int = -1,
                            verbose: bool = True) -> Tuple[float, Dict]:
    r"""
    Estimate a recrossing (transmission) factor from a discrete state sequence.

    We measure the **reactive fraction of barrier excursions**: each time the
    system leaves one basin and next arrives at a basin, that excursion is
    *reactive* if it reached the opposite basin and a *recrossing* if it fell
    back to the basin it came from. The factor is

    .. math::

        \hat{\kappa} = \frac{\text{reactive excursions}}
                            {\text{reactive} + \text{recrossing excursions}}
        \qquad 0 \le \hat{\kappa} \le 1 .

    :math:`\hat{\kappa} = 1` means every barrier attempt succeeds (no
    recrossings); smaller values mean the system often recrosses and returns,
    which in transition-state theory is what lowers the true rate below the TST
    estimate.

    Args:
        state_trajectory: (n_frames,) state indices from :func:`identify_states`.
        from_state: Reactant basin index.
        to_state: Product basin index.
        transition_region: Label for the barrier region (``-1`` by default).
        verbose: Print analysis details.

    Returns:
        kappa: Reactive fraction in ``[0, 1]``.
        info: Dictionary with the reactive / recrossing excursion counts.

    Notes:
        This is a *discrete-trajectory recrossing diagnostic*, **not** the
        Bennett--Chandler transmission coefficient (the plateau of the
        reactive-flux time-correlation function). It needs no dividing surface
        or velocities, only the coarse-grained state sequence, and its value
        depends on how :func:`identify_states` carves the basins. Treat it as a
        qualitative recrossing measure, not as a multiplicative correction to a
        TST rate. Only the two basins ``from_state``/``to_state`` are followed;
        any other stable state is treated like the barrier region.
    """
    states = np.asarray(state_trajectory)

    n_reactive = 0
    n_recrossing = 0
    origin = None        # basin the current excursion departed from
    left_basin = False   # has the system left `origin` since last arrival?

    # Walk the sequence as a basin -> barrier -> basin state machine. A change
    # of basin (directly, or after a barrier excursion) is reactive; a return
    # to the same basin after leaving it is a recrossing.
    for s in states:
        if s == from_state or s == to_state:
            if origin is None:
                origin = int(s)
                left_basin = False
            elif s != origin:
                n_reactive += 1
                origin = int(s)
                left_basin = False
            elif left_basin:  # s == origin and we had left: a recrossing
                n_recrossing += 1
                left_basin = False
            # else: s == origin and still resident; nothing to count
        else:
            # Barrier region (or any non-basin state): the system has left.
            if origin is not None:
                left_basin = True

    n_excursions = n_reactive + n_recrossing
    kappa = n_reactive / n_excursions if n_excursions > 0 else 0.0

    info = {
        'n_reactive': n_reactive,
        'n_recrossing': n_recrossing,
        'n_excursions': n_excursions,
        'kappa': kappa,
    }

    if verbose:
        print("Recrossing (transmission) factor")
        print("=" * 50)
        print(f"Reactive excursions   : {n_reactive}")
        print(f"Recrossing excursions : {n_recrossing}")
        print(f"Reactive fraction     : kappa_hat = {kappa:.3f}")

        if n_excursions == 0:
            print("  → No barrier excursions observed (insufficient statistics)")
        elif kappa < 0.5:
            print("  → Significant recrossings detected")
        elif kappa > 0.9:
            print("  → Near-TST behavior (few recrossings)")

    return kappa, info


def committor_probability(state_trajectory: np.ndarray,
                         observable: np.ndarray,
                         from_state: int,
                         to_state: int,
                         transition_region: int = -1,
                         n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Estimate the committor probability :math:`p_B(\xi)` empirically.

    :math:`p_B(\xi)` is the probability of reaching state B before state A,
    starting from configurations whose reaction-coordinate value is
    :math:`\xi`. We estimate it by "shooting forward": for every frame in the
    transition region we follow the trajectory until it first reaches A or B,
    scoring 1 if B is reached first and 0 if A is. These outcomes are then
    averaged **within bins of the reaction coordinate** ``observable`` --- not
    by frame index, since the committor is a function of *configuration*, not
    of wall-clock time.

    Args:
        state_trajectory: (n_frames,) state indices from :func:`identify_states`.
        observable: (n_frames,) the reaction-coordinate value at each frame
            (the same series passed to :func:`identify_states`). This is what
            the committor is resolved against.
        from_state: State A index.
        to_state: State B index.
        transition_region: Label marking the barrier/transition region
            (``-1`` by default, matching :func:`identify_states`).
        n_bins: Number of reaction-coordinate bins.

    Returns:
        bin_centers: (n_bins,) reaction-coordinate bin centers.
        p_B: (n_bins,) committor estimate per bin; ``np.nan`` for empty bins.

    Notes:
        - :math:`p_B = 0` in state A, :math:`p_B = 1` in state B; the
          :math:`p_B = 0.5` isocommittor locates the true transition state.
        - Frames whose forward trajectory ends in the barrier (the run finishes
          before reaching either basin) are dropped, not scored.
        - Requires many independent transitions for stable statistics.
    """
    state_trajectory = np.asarray(state_trajectory)
    observable = np.asarray(observable, dtype=float)
    if observable.shape != state_trajectory.shape:
        raise ValueError("observable and state_trajectory must have equal length")

    n_frames = state_trajectory.size

    # First basin (A or B) reached at or after each frame, found in one backward
    # pass so the per-frame look-ahead is O(n) rather than O(n^2). Sentinel -2
    # marks "no basin reached before the trajectory ends".
    NONE = -2
    next_basin = np.full(n_frames, NONE, dtype=int)
    carry = NONE
    for i in range(n_frames - 1, -1, -1):
        s = state_trajectory[i]
        if s == from_state or s == to_state:
            carry = int(s)
        next_basin[i] = carry

    # Score each transition-region frame by the basin its future first hits.
    in_barrier = np.where(state_trajectory == transition_region)[0]
    xi_samples = []
    outcomes = []
    for frame in in_barrier:
        destination = next_basin[frame]
        if destination == to_state:
            outcomes.append(1.0)
            xi_samples.append(observable[frame])
        elif destination == from_state:
            outcomes.append(0.0)
            xi_samples.append(observable[frame])
        # destination == NONE: trajectory ended in the barrier; unscored.

    if not outcomes:
        return np.array([]), np.array([])

    xi = np.asarray(xi_samples, dtype=float)
    scored = np.asarray(outcomes, dtype=float)

    bins = np.linspace(xi.min(), xi.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    p_B = np.full(n_bins, np.nan)
    for i in range(n_bins):
        # Last bin is closed on the right so the maximum xi is not dropped.
        upper = xi <= bins[i + 1] if i == n_bins - 1 else xi < bins[i + 1]
        mask = (xi >= bins[i]) & upper
        if np.any(mask):
            p_B[i] = float(np.mean(scored[mask]))

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
        - Assumes Arrhenius / Eyring form (valid at high barriers).
        - The prefactor is the Eyring value A = kT/h ≈ 6.25 ps⁻¹ at 300 K.
    """
    kB = 0.001987204  # kcal/(mol·K)
    # Planck constant in kcal·ps/mol (the Eyring/TST prefactor is kT/h, using the
    # *full* h, not ħ). Conversion of h = 6.62607015e-34 J·s:
    #   h · N_A / 4184 (J -> kcal/mol) · 1e12 (s -> ps) ≈ 0.095371 kcal·ps/mol,
    # which gives kT/h ≈ 6.25 ps⁻¹ at 300 K. The previous value (1.5836e-4)
    # was ~600x too small and inflated every barrier by ~kT·ln(600) ≈ 3.8 kcal/mol.
    h = 0.095371  # kcal·ps/mol

    kT = kB * temperature
    A = kT / h  # Eyring prefactor (~6.25 ps⁻¹ at 300 K)

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
    print("from rotmd.analysis.transitions import identify_states, transmission_coefficient")
    print()
    print("# Identify metastable states from nutation angle")
    print("states, info = identify_states(theta, thresholds=[(0, 30), (60, 90)])")
    print()
    print("# Compute transmission coefficient")
    print("kappa, stats = transmission_coefficient(states, from_state=0, to_state=1)")
    print()
    print("print(f'kappa_hat = {kappa:.3f}')  # reactive fraction (recrossings)")
