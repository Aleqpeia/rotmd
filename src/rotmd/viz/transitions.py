"""Metastable-state trajectories and committor curves.

New rather than restored — the legacy ``visualization/`` package predates
:mod:`rotmd.analysis.transitions`, so there is no prior version of these to
recover defects from. Built on the same :mod:`rotmd.viz.core` API as the
restored PMF/spectra/phase-space plots for consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import figure


@figure(figsize=(9, 3.5), name="state-trajectory")
def plot_state_trajectory(
    ax: Any,
    times: np.ndarray,
    states: np.ndarray,
    *,
    cmap: str = "tab10",
) -> Any:
    """Discrete state index over time, from :func:`rotmd.analysis.transitions.identify_states`.

    Args:
        times: Timestamps, any unit (labelled generically since the caller
            may pass ps or ns).
        states: Integer state index per frame; ``-1`` marks the transition
            region (frames assigned to no stable state).
    """
    times = np.asarray(times, dtype=np.float64)
    states = np.asarray(states)
    n_states = int(states.max()) + 1 if states.max() >= 0 else 0

    import matplotlib.pyplot as plt

    palette = plt.get_cmap(cmap)
    for state in range(n_states):
        mask = states == state
        ax.scatter(times[mask], states[mask], s=3, color=palette(state % 10), label=f"state {state}")

    transition = states == -1
    if transition.any():
        ax.scatter(times[transition], states[transition], s=3, color="0.75", label="transition")

    ax.set(xlabel="time", ylabel="state", title="State trajectory")
    if n_states > 0:
        ax.set_yticks(range(-1, n_states))
    ax.legend(fontsize=8, ncols=min(4, n_states + 1), markerscale=3, loc="upper right")
    return ax


@figure(figsize=(7, 5.5), name="committor")
def plot_committor(
    ax: Any,
    bin_centers: np.ndarray,
    p_b: np.ndarray,
    *,
    kappa: float | None = None,
    labels: tuple[str, str] = ("A", "B"),
) -> Any:
    """Committor p_B against a reaction-coordinate proxy, from :func:`rotmd.analysis.transitions.committor_probability`.

    Args:
        bin_centers: Binning variable ``committor_probability`` used (frame
            index by default — see that function's docstring).
        p_b: Committor value in each bin, in ``[0, 1]``.
        kappa: Transmission coefficient to annotate, if available.
        labels: Names for state A (p_B = 0) and state B (p_B = 1).
    """
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)

    ax.plot(bin_centers, p_b, marker="o", lw=1.4, ms=4, color="tab:blue")
    ax.axhline(0.5, color="0.5", ls="--", lw=1.0)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.axhline(1.0, color="0.8", lw=0.8)

    title = f"Committor: {labels[0]} -> {labels[1]}"
    if kappa is not None:
        title += f"  (kappa = {kappa:.2f})"
    ax.set(xlabel="frame index", ylabel=f"p_B ({labels[1]})", title=title, ylim=(-0.05, 1.05))
    ax.grid(True)
    return ax
