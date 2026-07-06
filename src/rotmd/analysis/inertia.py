"""Inertia-tensor analysis helpers (analyze worktree).

Derived quantities that build on the principal moments / inertia tensors emitted
by the extract pipeline but are not needed by the extract CLI itself.
"""

from __future__ import annotations

import numpy as np


def parallel_axis_theorem(
        I_com: np.ndarray, total_mass: float, displacement: np.ndarray
) -> np.ndarray:
    """
    Apply parallel axis theorem to shift inertia tensor to a new origin.

    The parallel axis theorem states:
        I_new = I_com + M [(d · d) I_3 - d ⊗ d]
    """
    d = np.asarray(displacement)
    d_squared = np.dot(d, d)
    # Steiner term: M [(d · d) I_3 - d ⊗ d]
    steiner = total_mass * (d_squared * np.eye(3) - np.outer(d, d))

    return I_com + steiner


def asymmetry_parameter(moments: np.ndarray) -> float:
    """
    Compute Ray's asymmetry parameter κ.

    The asymmetry parameter quantifies deviation from a symmetric top:
        κ = (2I_b - I_a - I_c) / (I_c - I_a)
    """
    I_a, I_b, I_c = moments

    if np.isclose(I_c, I_a):
        # Spherical top: κ undefined, return 0
        return 0.0

    kappa = (2 * I_b - I_a - I_c) / (I_c - I_a)
    return kappa
