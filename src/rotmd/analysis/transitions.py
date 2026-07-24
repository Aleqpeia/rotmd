"""Transition-state analysis between metastable orientation states.

Turns a continuous observable (e.g. tilt angle theta) into discrete states,
then quantifies how the protein moves between them: the transmission
coefficient kappa (how much recrossing inflates the naive TST rate above the
true one), the committor p_B (which side of the barrier a configuration is
really on), and the transition path ensemble itself. Transition-state theory
gives the rate as ``k = kappa * k_TST`` with ``k_TST = (kT/h) exp(-DeltaF‡/kT)``;
``kappa`` is 1 only when the dividing surface has no recrossings, which is
rarely true for a real, structured barrier like a membrane protein
reorienting past a bound state.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import label

# k_B in kcal/(mol*K); h in ps*kcal/mol (both used in free_energy_barrier_from_rate).
_K_B = 0.001987204
_H = 1.5836e-4


def identify_states(
    observable: np.ndarray,
    thresholds: list[tuple[float, float]],
    min_duration: int = 10,
) -> tuple[np.ndarray, dict]:
    """Discretize ``observable`` into states defined by ``thresholds`` (e.g. ``[(0, 30), (60, 90)]``).

    Frames are ``-1`` (transition region) unless they fall in one of the
    ranges *and* the contiguous run is at least ``min_duration`` frames long —
    short excursions are folded back into "transition" rather than counted as
    a visited state, since a metastable basin implies dwelling, not just
    passing through.
    """
    n_frames = len(observable)
    state_trajectory = -np.ones(n_frames, dtype=int)

    for state_idx, (min_val, max_val) in enumerate(thresholds):
        mask = (observable >= min_val) & (observable <= max_val)
        state_trajectory[mask] = state_idx

    for state_idx in range(len(thresholds)):
        mask = state_trajectory == state_idx
        labeled, n_regions = label(mask)

        for region_idx in range(1, n_regions + 1):
            region_mask = labeled == region_idx
            region_size = np.sum(region_mask)
            if region_size < min_duration:
                state_trajectory[region_mask] = -1

    state_info = {}
    for state_idx in range(len(thresholds)):
        mask = state_trajectory == state_idx
        state_info[f"state_{state_idx}_population"] = np.sum(mask) / n_frames

    state_info["transition_fraction"] = np.sum(state_trajectory == -1) / n_frames

    return state_trajectory, state_info


def detect_transitions(
    state_trajectory: np.ndarray,
    from_state: int,
    to_state: int,
) -> list[tuple[int, int]]:
    """``(start, end)`` frame pairs for each ``from_state -> to_state`` transition, from :func:`identify_states` output.

    ``start`` is the last frame in ``from_state`` before leaving it; ``end``
    is the first frame reaching ``to_state``. A run that leaves ``from_state``
    and settles into some other stable state without ever reaching
    ``to_state`` does not count.
    """
    transitions = []
    in_from_state = False
    start_frame = None

    for i, state in enumerate(state_trajectory):
        if state == from_state:
            in_from_state = True
            start_frame = i
        elif state == to_state and in_from_state:
            transitions.append((start_frame, i))
            in_from_state = False
        elif state != -1 and state != from_state:
            in_from_state = False

    return transitions


def compute_reactive_flux(
    positions: np.ndarray,
    velocities: np.ndarray,
    dividing_surface: Callable[[np.ndarray], float],
    threshold: float = 0.0,
) -> float:
    """Forward-crossing flux through ``dividing_surface(positions) == threshold``, per frame.

    ``J = <delta(xi - xi‡) xi_dot H(xi_dot)>``: only crossings with positive
    velocity (reactant -> product direction) are counted, since flux in TST is
    directional by definition. Related to a rate via ``k = J / P_reactant``.
    """
    n_frames = len(positions)

    xi = np.array([dividing_surface(positions[i]) for i in range(n_frames)])
    xi_dot = np.gradient(xi)

    crossings = [
        i
        for i in range(1, n_frames)
        if xi[i - 1] < threshold <= xi[i] and xi_dot[i] > 0
    ]

    return len(crossings) / n_frames


def transmission_coefficient(
    state_trajectory: np.ndarray,
    from_state: int,
    to_state: int,
    transition_region: int = -1,
    verbose: bool = True,
) -> tuple[float, dict]:
    """Transmission coefficient kappa = (successful transitions) / (barrier crossings), 0 < kappa <= 1.

    kappa = 1 means every crossing into the transition region commits to the
    other state (TST is exact); kappa << 1 means most crossings recross back
    to where they started, and the true rate is much slower than the naive
    TST estimate.
    """
    ab_transitions = detect_transitions(state_trajectory, from_state, to_state)
    ba_transitions = detect_transitions(state_trajectory, to_state, from_state)

    n_ab = len(ab_transitions)
    n_ba = len(ba_transitions)

    barrier_crossings = 0
    for i in range(1, len(state_trajectory)):
        prev_state = state_trajectory[i - 1]
        curr_state = state_trajectory[i]
        if (prev_state == from_state or prev_state == to_state) and curr_state == transition_region:
            barrier_crossings += 1

    kappa = (n_ab + n_ba) / barrier_crossings if barrier_crossings > 0 else 0.0

    info = {
        "n_AB_transitions": n_ab,
        "n_BA_transitions": n_ba,
        "barrier_crossings": barrier_crossings,
        "kappa": kappa,
    }

    if verbose:
        print("Transmission Coefficient Analysis")
        print("=" * 50)
        print(f"State {from_state} -> {to_state} transitions: {n_ab}")
        print(f"State {to_state} -> {from_state} transitions: {n_ba}")
        print(f"Total barrier crossings: {barrier_crossings}")
        print(f"Transmission coefficient: kappa = {kappa:.3f}")

        if kappa < 0.5:
            print("  -> Significant recrossings detected")
        elif kappa > 0.9:
            print("  -> Near-TST behavior (few recrossings)")

    return kappa, info


def committor_probability(
    state_trajectory: np.ndarray,
    from_state: int,
    to_state: int,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Committor p_B: probability a transition-region frame reaches ``to_state`` before ``from_state``.

    Estimated by following each transition-region frame forward until it
    first hits either stable state, then averaging the 0/1 outcome within
    bins along the trajectory. ``p_B = 0.5`` marks the operational transition
    state — the point of no preference between reactant and product.
    """
    transition_mask = (
        (state_trajectory != from_state) & (state_trajectory != to_state) & (state_trajectory >= 0)
    )
    transition_frames = np.where(transition_mask)[0]

    committor_samples = []
    for frame in transition_frames:
        for future_frame in range(frame + 1, len(state_trajectory)):
            future_state = state_trajectory[future_frame]
            if future_state == to_state:
                committor_samples.append((frame, 1))
                break
            if future_state == from_state:
                committor_samples.append((frame, 0))
                break

    if len(committor_samples) == 0:
        return np.array([]), np.array([])

    frames, commitments = zip(*committor_samples)
    frames = np.array(frames)
    commitments = np.array(commitments)

    bins = np.linspace(frames.min(), frames.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p_b = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (frames >= bins[i]) & (frames < bins[i + 1])
        if np.sum(mask) > 0:
            p_b[i] = np.mean(commitments[mask])

    return bin_centers, p_b


def transition_path_ensemble(
    state_trajectory: np.ndarray,
    from_state: int,
    to_state: int,
    include_data: np.ndarray | None = None,
) -> dict:
    """Collect every ``from_state -> to_state`` transition path, optionally with a data array sliced alongside.

    Useful for characterizing *how* the protein crosses, not just how often:
    averaging ``include_data`` over the ensemble (e.g. per-residue RMSF along
    the path) reveals the transition mechanism.
    """
    transitions = detect_transitions(state_trajectory, from_state, to_state)

    paths = []
    lengths = []
    data_along_paths = [] if include_data is not None else None

    for start, end in transitions:
        path = state_trajectory[start : end + 1]
        paths.append(path)
        lengths.append(len(path))
        if include_data is not None:
            data_along_paths.append(include_data[start : end + 1])

    ensemble = {
        "paths": paths,
        "lengths": np.array(lengths),
        "n_transitions": len(transitions),
    }
    if data_along_paths is not None:
        ensemble["data_along_paths"] = data_along_paths

    return ensemble


def free_energy_barrier_from_rate(
    k_ab: float,
    k_ba: float,
    temperature: float = 300.0,
) -> tuple[float, float]:
    """Forward/backward free-energy barriers (kcal/mol) from rates (1/ps) via the Arrhenius/TST relation ``k = (kT/h) exp(-DeltaF‡/kT)``.

    Only valid at high barriers, where TST's assumption of a single rare
    crossing event (rather than diffusive barrier-top dynamics) holds.
    """
    kt = _K_B * temperature
    a = kt / _H

    delta_f_ab = -kt * np.log(k_ab / a) if k_ab > 0 else np.inf
    delta_f_ba = -kt * np.log(k_ba / a) if k_ba > 0 else np.inf

    return delta_f_ab, delta_f_ba


if __name__ == "__main__":
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
    print("print(f'kappa = {kappa:.3f}')  # Accounts for recrossings")
