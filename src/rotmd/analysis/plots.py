"""Equilibration figures (T2).

RMSD, Rg and per-domain RMSD against time, with a running mean and the
detected ``t0`` drawn on top. These are the figures that make the
equilibration argument visible rather than asserted — the reviewer asked to
see them, and a reader can check the window was placed sensibly by eye.

Always renders through the Agg backend: this runs on a compute/login node with
no display, where importing pyplot against an interactive backend fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _pyplot():
    """Import pyplot with a headless backend pinned first."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean, same length as ``values``.

    Uses a cumulative sum with edge shrinking rather than ``np.convolve`` so
    the ends stay honest averages of the frames that exist instead of decaying
    toward zero, which would fake a trend exactly where the eye looks for one.
    """
    a = np.asarray(values, dtype=np.float64)
    n = a.size
    window = int(max(1, min(window, n)))
    if window == 1:
        return a.copy()

    csum = np.concatenate(([0.0], np.cumsum(a)))
    half = window // 2
    lo = np.maximum(np.arange(n) - half, 0)
    hi = np.minimum(np.arange(n) + half + 1, n)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _series_panels(data: dict[str, np.ndarray], domain_names: list[str]) -> list[tuple]:
    """(title, ylabel, [(label, series)]) for each panel that has data."""
    panels: list[tuple] = []
    if "rmsd" in data:
        panels.append(("Backbone RMSD", "RMSD (Å)", [("rmsd", data["rmsd"])]))
    if "rg" in data:
        panels.append(("Radius of gyration", "Rg (Å)", [("rg", data["rg"])]))

    dom = data.get("rmsd_dom")
    if dom is not None and dom.ndim == 2 and dom.shape[1] > 0:
        names = domain_names or [f"domain {i}" for i in range(dom.shape[1])]
        panels.append((
            "Per-domain RMSD",
            "RMSD (Å)",
            [(names[i], dom[:, i]) for i in range(dom.shape[1])],
        ))
    return panels


def plot_equilibration(
    data: dict[str, np.ndarray],
    output: str | Path,
    window: dict[str, Any] | None = None,
    domain_names: list[str] | None = None,
    title: str = "",
    smooth_frac: float = 0.02,
) -> Path:
    """One figure per replica: each observable with running mean and ``t0``."""
    plt = _pyplot()

    time_ns = np.asarray(data["time_ps"], dtype=np.float64) / 1000.0
    panels = _series_panels(data, domain_names or [])
    if not panels:
        raise ValueError("nothing to plot: expected at least one of rmsd/rg/rmsd_dom")

    smooth = max(1, int(len(time_ns) * smooth_frac))
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9, 2.6 * len(panels)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, (panel_title, ylabel, series) in zip(axes, panels, strict=True):
        for label, values in series:
            line, = ax.plot(time_ns, values, lw=0.6, alpha=0.45, label=label)
            # The running mean is what the eye should follow; match its colour
            # to the raw trace so multi-line panels stay readable.
            ax.plot(time_ns, running_mean(values, smooth), lw=1.6, color=line.get_color())
        ax.set(title=panel_title, ylabel=ylabel)
        if len(series) > 1:
            ax.legend(fontsize=7, ncols=min(4, len(series)))

        if window is not None:
            t0_ns = float(window["t0_ps"]) / 1000.0
            ax.axvline(t0_ns, color="crimson", ls="--", lw=1.2)
            ax.axvspan(time_ns[0], t0_ns, color="crimson", alpha=0.07)

    if window is not None:
        axes[0].text(
            0.99, 0.94,
            f"t0 = {float(window['t0_ps']) / 1000.0:.1f} ns  "
            f"(g = {window.get('g', float('nan')):.1f}, "
            f"N_eff = {window.get('n_eff', float('nan')):.0f})",
            transform=axes[0].transAxes, ha="right", va="top", fontsize=8, color="crimson",
        )

    axes[-1].set_xlabel("time (ns)")
    if title:
        fig.suptitle(title, fontsize=11)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_ddccm(
    result: dict[str, np.ndarray],
    output: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "",
) -> Path:
    """Pooled maps, their difference, and the significant/distal difference.

    Correlation maps use a diverging colormap symmetric about zero — the field
    convention, and the only honest choice here: an asymmetric or sequential
    scale would make +0.2 and -0.2 look like different magnitudes.
    """
    plt = _pyplot()

    resids = np.asarray(result["resids"])
    extent = [resids[0], resids[-1], resids[-1], resids[0]]
    delta = result["ddccm"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

    for ax, matrix, panel_title in [
        (axes[0, 0], result["dcc_a"], f"DCCM — {label_a}"),
        (axes[0, 1], result["dcc_b"], f"DCCM — {label_b}"),
    ]:
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent, origin="upper")
        ax.set(title=panel_title, xlabel="residue", ylabel="residue")
        fig.colorbar(im, ax=ax, fraction=0.046, label="correlation")

    # Symmetric limits from the data so the difference panels share a scale.
    span = float(np.abs(delta).max()) or 1.0
    im = axes[1, 0].imshow(
        delta, cmap="RdBu_r", vmin=-span, vmax=span, extent=extent, origin="upper"
    )
    axes[1, 0].set(title=f"Δ DCCM ({label_b} − {label_a})", xlabel="residue", ylabel="residue")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, label="Δ correlation")

    # Final panel: only cells that survived the bootstrap and sit away from the
    # mutation site — the long-range rewiring, with everything expected removed.
    shown = np.array(delta, dtype=np.float64)
    kept = np.ones_like(shown, dtype=bool)
    caption = []
    if "significant" in result:
        kept &= np.asarray(result["significant"], dtype=bool)
        caption.append(f"{int((1 - float(result.get('alpha', 0.05))) * 100)}% CI excludes 0")
    if "distal_mask" in result:
        kept &= np.asarray(result["distal_mask"], dtype=bool)
        caption.append(f"|i − {int(result['site'])}| ≥ {int(result['exclude_within'])}")
    shown[~kept] = np.nan

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.92")  # masked cells read as "no claim", not as zero
    im = axes[1, 1].imshow(
        shown, cmap=cmap, vmin=-span, vmax=span, extent=extent, origin="upper"
    )
    axes[1, 1].set(
        title="Δ DCCM — " + (" & ".join(caption) if caption else "unmasked"),
        xlabel="residue", ylabel="residue",
    )
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, label="Δ correlation")

    if title:
        fig.suptitle(title, fontsize=12)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_dssp_delta(
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    resids: np.ndarray,
    output: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    site: int | None = None,
    codes: tuple[str, ...] = ("-", "H", "E"),
    title: str = "",
) -> Path:
    """Per-residue secondary-structure occupancy and its change."""
    plt = _pyplot()

    resids = np.asarray(resids)
    occ_a = np.asarray(occ_a, dtype=np.float64)
    occ_b = np.asarray(occ_b, dtype=np.float64)
    names = {"-": "coil", "H": "helix", "E": "sheet"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)

    helix = codes.index("H") if "H" in codes else 1
    axes[0].plot(resids, occ_a[:, helix], lw=1.4, label=f"{label_a} helix")
    axes[0].plot(resids, occ_b[:, helix], lw=1.4, label=f"{label_b} helix")
    axes[0].set(ylabel="occupancy", title="Helix occupancy per residue", ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=8)

    delta = occ_b - occ_a
    width = 0.8
    for index, code in enumerate(codes):
        if code == "-":
            continue  # coil is the complement of the others; plotting it triples the ink
        axes[1].bar(resids, delta[:, index], width=width, alpha=0.75, label=names.get(code, code))
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set(
        xlabel="residue", ylabel=f"Δ occupancy ({label_b} − {label_a})",
        title="Change in secondary-structure occupancy",
    )
    axes[1].legend(fontsize=8)

    if site is not None:
        for ax in axes:
            ax.axvline(site, color="crimson", ls="--", lw=1.2)
        axes[0].annotate(
            f"site {site}", xy=(site, 1.0), xytext=(4, -8),
            textcoords="offset points", color="crimson", fontsize=8,
        )

    if title:
        fig.suptitle(title, fontsize=11)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_replica_overlay(
    datasets: list[tuple[str, dict[str, np.ndarray]]],
    output: str | Path,
    window: dict[str, Any] | None = None,
    title: str = "",
    smooth_frac: float = 0.02,
) -> Path:
    """All replicas of one system on shared axes.

    Replicas that disagree about where they settle are the strongest signal
    that a single-replica window is not safe to trust, so they belong on one
    pair of axes rather than in separate figures.
    """
    plt = _pyplot()

    keys = [k for k in ("rmsd", "rg") if all(k in d for _, d in datasets)]
    if not keys:
        raise ValueError("no observable common to every replica (need rmsd and/or rg)")

    fig, axes = plt.subplots(
        len(keys), 1, figsize=(9, 2.8 * len(keys)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    labels = {"rmsd": ("Backbone RMSD", "RMSD (Å)"), "rg": ("Radius of gyration", "Rg (Å)")}

    for ax, key in zip(axes, keys, strict=True):
        for label, data in datasets:
            time_ns = np.asarray(data["time_ps"], dtype=np.float64) / 1000.0
            smooth = max(1, int(len(time_ns) * smooth_frac))
            ax.plot(time_ns, running_mean(data[key], smooth), lw=1.2, label=label)
        panel_title, ylabel = labels[key]
        ax.set(title=f"{panel_title} — running mean per replica", ylabel=ylabel)
        ax.legend(fontsize=8, ncols=min(4, len(datasets)))
        if window is not None:
            ax.axvline(float(window["t0_ps"]) / 1000.0, color="crimson", ls="--", lw=1.2)

    axes[-1].set_xlabel("time (ns)")
    if title:
        fig.suptitle(title, fontsize=11)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
