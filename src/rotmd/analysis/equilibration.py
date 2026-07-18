"""Equilibration detection (T3) — where does production sampling start?

This is the join point of the whole analysis plan. Every later stage slices
``time_ps >= t0_ps`` from the ``window.json`` this module writes, which is what
makes "which part of the trajectory did you analyse?" an answerable, recorded,
reproducible question instead of an undocumented choice.

Method
------
``t0`` is chosen by Chodera's rule (J. Chem. Theory Comput. 2016, 12, 1799):
for every candidate start ``t`` compute the statistical inefficiency ``g`` of
the remaining series and keep the ``t`` that maximises the number of
*effectively uncorrelated* samples ``N_eff = (T - t) / g``. Discarding more data
lowers ``T - t`` but also lowers ``g`` as the transient leaves the series, so the
maximum is a genuine trade-off rather than a threshold anyone tuned.

``g = 1 + 2 Σ_t C(t) (1 - t/N)`` is accumulated until the autocorrelation first
goes non-positive, the standard truncation that keeps the noisy long-lag tail
out of the sum.

Dependencies
------------
pymbar is used when it is installed, matching the plan. It is *not* required:
the same estimator is implemented here in numpy, because rotmd has to install
from wheels into an air-gapped cluster where adding a dependency is expensive
and pymbar buys ~60 lines. ``method="native"`` forces the built-in path so the
two can be compared directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def statistical_inefficiency(series: np.ndarray, mintime: int = 3) -> float:
    """``g = 1 + 2τ``, the number of samples per independent observation.

    Error bars on a correlated series must be inflated by ``sqrt(g)``; block
    analyses use ``g`` as the block size. Returns ``>= 1`` always.

    The autocorrelation is evaluated by FFT (O(N log N)) rather than by the
    direct double loop, because this is called once per candidate ``t0`` and the
    trajectories here run to 10^5 frames.
    """
    a = np.asarray(series, dtype=np.float64)
    n = a.size
    if n < 2:
        return 1.0

    da = a - a.mean()
    sigma2 = float(np.dot(da, da) / n)
    if sigma2 <= 0.0:  # constant series — every sample is "independent"
        return 1.0

    # Linear (non-circular) autocorrelation via zero-padded FFT.
    n_pad = 1 << int(2 * n - 1).bit_length()
    spec = np.fft.rfft(da, n_pad)
    acf = np.fft.irfft(spec * np.conjugate(spec), n_pad)[:n]
    # Unbiased: lag t has (n - t) contributing pairs.
    acf /= (n - np.arange(n)) * sigma2

    # Truncate at the first non-positive lag past `mintime`, then sum what is
    # left in one shot (a Python loop here would dominate the runtime).
    lags = np.arange(1, n)
    tail = acf[1:]
    # `nonpos[0]` is the index of the first non-positive lag; summing up to
    # (not including) it is the standard truncation.
    nonpos = np.flatnonzero((tail <= 0.0) & (lags > mintime))
    cut = int(nonpos[0]) if nonpos.size else n - 1

    g = 1.0 + 2.0 * float(np.sum(tail[:cut] * (1.0 - lags[:cut] / n)))
    return max(g, 1.0)


def _detect_native(series: np.ndarray, nskip: int) -> tuple[int, float, float]:
    n = series.size
    # Seed with -inf, never a fabricated (t=0, g=1) candidate: an assumed g of 1
    # implies N_eff = n, which no real candidate on a correlated series can beat,
    # so the scan would always return t0 = 0 and silently keep the transient.
    best_t, best_g, best_n_eff = 0, 1.0, -np.inf
    for t in range(0, n - 1, nskip):
        g = statistical_inefficiency(series[t:])
        n_eff = (n - t) / g
        if n_eff > best_n_eff:
            best_t, best_g, best_n_eff = t, g, n_eff
    return best_t, best_g, best_n_eff


def _detect_pymbar(series: np.ndarray, nskip: int) -> tuple[int, float, float]:
    from pymbar import timeseries

    # fast=False, not pymbar's default: `fast=True` uses an increment-doubling
    # approximation to the autocorrelation sum that overestimates g (on AR(1)
    # phi=0.9 it returns 25.4 where the exact sum gives 22.6). The native path
    # here always does the exact sum, so without this the reported g would jump
    # by ~10% purely because pymbar happened to be installed.
    t0, g, n_eff = timeseries.detect_equilibration(
        np.asarray(series, dtype=np.float64), nskip=nskip, fast=False
    )
    return int(t0), float(g), float(n_eff)


def detect_equilibration(
    series: np.ndarray, nskip: int | None = None, method: str = "auto"
) -> dict[str, Any]:
    """Locate the start of the equilibrated region of ``series``.

    Args:
        series: (n_frames,) scalar order parameter, e.g. backbone RMSD.
        nskip: Stride over candidate ``t0`` values. ``None`` targets ~200
            candidates, which bounds the cost on long trajectories while
            keeping ``t0`` resolved far more finely than any window matters.
        method: ``"auto"`` (pymbar when installed), ``"pymbar"``, or ``"native"``.

    Returns:
        ``{t0_index, g, n_eff, method, nskip, n_frames}``.
    """
    a = np.asarray(series, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {a.shape}")
    if a.size < 3:
        raise ValueError(f"series too short to detect equilibration: {a.size} frames")

    if nskip is None:
        nskip = max(1, a.size // 200)

    resolved = method
    if method == "auto":
        from importlib.util import find_spec

        resolved = "pymbar" if find_spec("pymbar") is not None else "native"

    if resolved == "pymbar":
        t0, g, n_eff = _detect_pymbar(a, nskip)
    elif resolved == "native":
        t0, g, n_eff = _detect_native(a, nskip)
    else:
        raise ValueError(f"unknown method {method!r} (expected auto/pymbar/native)")

    return {
        "t0_index": int(t0),
        "g": float(g),
        "n_eff": float(n_eff),
        "method": resolved,
        "nskip": int(nskip),
        "n_frames": int(a.size),
    }


def block_average_t0(series: np.ndarray, n_blocks: int = 50, n_sigma: float = 2.0) -> int:
    """Independent cross-check on ``t0`` via the block-average plateau.

    The series is cut into ``n_blocks`` equal blocks; the equilibrated plateau
    is characterised by the mean and scatter of the *later* half of the block
    means, and ``t0`` is the start of the first block whose mean falls inside
    ``n_sigma`` of that plateau.

    Deliberately a different family of estimator from
    :func:`detect_equilibration` (a threshold on block means, not a maximisation
    of ``N_eff``), so agreement between the two is evidence the window is real
    rather than an artefact of one estimator's assumptions.

    The plan offers reverse cumulative averaging as the alternative here; block
    averaging is used instead because the reverse cumulative mean is a smooth,
    strongly autocorrelated function of ``t`` whose tail is estimated from
    vanishingly few samples. That combination makes its threshold crossings
    fire on stationary noise, whereas block means decorrelate once the block is
    longer than the correlation time and behave.
    """
    a = np.asarray(series, dtype=np.float64)
    n = a.size
    if n < 2 * n_blocks:  # too short to block meaningfully
        return 0

    block = n // n_blocks
    means = a[: block * n_blocks].reshape(n_blocks, block).mean(axis=1)

    plateau = means[n_blocks // 2:]
    mu = float(plateau.mean())
    sd = float(plateau.std(ddof=1))
    if sd == 0.0:
        return 0

    inside = np.abs(means - mu) <= n_sigma * sd
    if not inside.any():
        return 0
    return int(np.argmax(inside)) * block


def build_window(
    time_ps: np.ndarray,
    series: np.ndarray,
    column: str,
    nskip: int | None = None,
    method: str = "auto",
    disagree_tol: float = 0.20,
) -> dict[str, Any]:
    """Full T3 result for one replica: detection + cross-check + provenance."""
    time_ps = np.asarray(time_ps, dtype=np.float64)
    if time_ps.shape != np.shape(series):
        raise ValueError(
            f"time_ps {time_ps.shape} and series {np.shape(series)} must have the same length"
        )

    det = detect_equilibration(series, nskip=nskip, method=method)
    t0_idx = det["t0_index"]
    t0_ps = float(time_ps[t0_idx])

    xc_idx = block_average_t0(series)
    xc_ps = float(time_ps[xc_idx])

    span = float(time_ps[-1] - time_ps[0]) or 1.0
    rel_diff = abs(t0_ps - xc_ps) / span
    agree = rel_diff <= disagree_tol

    n_frames = len(time_ps)
    return {
        "t0_ps": t0_ps,
        "t0_index": t0_idx,
        "g": det["g"],
        "n_eff": det["n_eff"],
        "column": column,
        "method": det["method"],
        "nskip": det["nskip"],
        "n_frames": n_frames,
        "n_frames_used": n_frames - t0_idx,
        "frac_discarded": t0_idx / n_frames,
        "t_start_ps": float(time_ps[0]),
        "t_end_ps": float(time_ps[-1]),
        "cross_check": {
            "method": "block_average_plateau",
            "t0_ps": xc_ps,
            "t0_index": xc_idx,
            "rel_diff": rel_diff,
            "tolerance": disagree_tol,
            "agree": bool(agree),
        },
        "pooled": False,
    }


def pool_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Conservative common window: the latest ``t0`` across replicas.

    Systems being compared must be sampled over equivalent windows, otherwise a
    ΔDCCM partly reflects one system having been given more equilibration than
    the other. Taking the max ``t0`` costs data but removes that confound.
    """
    if not windows:
        raise ValueError("at least one window required to pool")

    worst = max(windows, key=lambda w: w["t0_ps"])
    return {
        "t0_ps": worst["t0_ps"],
        "g": max(w["g"] for w in windows),
        "n_eff": min(w["n_eff"] for w in windows),
        "column": worst["column"],
        "method": worst["method"],
        "pooled": True,
        "n_replicas": len(windows),
        "members": [
            {"t0_ps": w["t0_ps"], "g": w["g"], "n_eff": w["n_eff"], "source": w.get("source_npz")}
            for w in windows
        ],
        "all_cross_checks_agree": all(w["cross_check"]["agree"] for w in windows if "cross_check" in w),
    }


def apply_window(time_ps: np.ndarray, window: dict[str, Any]) -> np.ndarray:
    """Boolean mask selecting the production frames of ``time_ps``.

    The single place any analysis is allowed to turn a window into frames —
    grep for hard-coded frame ranges and you should find none.
    """
    t0 = float(window["t0_ps"])
    mask = np.asarray(time_ps, dtype=np.float64) >= t0
    if not mask.any():
        raise ValueError(
            f"window t0_ps={t0} excludes every frame "
            f"(trajectory spans {float(np.min(time_ps))}-{float(np.max(time_ps))} ps)"
        )
    return mask
