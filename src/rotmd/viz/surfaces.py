"""PMF surfaces and vector fields over orientation space.

Restored from the pre-slimming ``visualization/surfaces.py``. The plots are the
same ones; what changed is that they draw on axes handed to them (see
:mod:`rotmd.viz.core`) and that the radian/degree handling now goes through
:func:`~rotmd.viz.core.angle_grid_degrees`. The original mixed the two — see
that function's docstring for what it produced.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .core import (
    angle_grid_degrees,
    figure,
    finite_percentile,
    label_orientation_axes,
)


def local_minima(values: np.ndarray, threshold: float = 2.0) -> list[tuple[int, int]]:
    """One representative index per basin, within ``threshold`` of the global minimum.

    These are *regional* minima: connected regions no cell of which has a
    strictly lower neighbour. The distinction is what the legacy version got
    wrong in both directions. It tested ``values == minimum_filter(values,
    size=3)``, which is true for every cell of any flat region — so a broad,
    undersampled basin came out sprayed with markers along its plateau, while a
    sloping shelf could be marked despite draining downhill. Tightening that to
    a strict inequality fixes the spray but then reports nothing at all when the
    basin floor is an exact tie between adjacent bins, which symmetry makes
    common. Grouping the plateau and testing its border resolves both: one
    marker per basin, and only for basins.

    Args:
        values: 2-D grid; non-finite entries (unvisited bins) are ignored.
        threshold: Only report minima within this much of the global minimum,
            in the units of ``values`` (kcal/mol).

    Returns:
        ``(i, j)`` index pairs, one per basin, ordered by depth.
    """
    from scipy.ndimage import binary_dilation, label, minimum_filter

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return []

    # inf in unvisited bins would swallow real minima through the filter; NaN
    # would propagate. Fill with +inf so empty neighbours never win.
    filled = np.where(finite, values, np.inf)
    connectivity = np.ones((3, 3), dtype=bool)

    centre_excluded = connectivity.copy()
    centre_excluded[1, 1] = False
    neighbourhood = minimum_filter(filled, footprint=centre_excluded, mode="nearest")

    # Non-strict, so a tied plateau survives as a whole region rather than
    # vanishing; the border test below is what decides whether it is a basin.
    candidates = finite & (filled <= neighbourhood)
    regions, n_regions = label(candidates, structure=connectivity)

    floor = float(filled[finite].min())
    found: list[tuple[float, int, int]] = []

    for k in range(1, n_regions + 1):
        region = regions == k
        depth = float(filled[region].min())
        if depth >= floor + threshold:
            continue

        border = binary_dilation(region, structure=connectivity) & ~region
        if border.any() and float(filled[border].min()) <= depth:
            continue  # drains to a lower neighbour: a shelf, not a basin

        # Representative: the deepest cell, breaking ties toward the centre of
        # the plateau so a marker sits in the middle of the basin it names.
        cells = np.argwhere(region & (filled <= depth))
        centre = cells.mean(axis=0)
        i, j = cells[np.argmin(((cells - centre) ** 2).sum(axis=1))]
        found.append((depth, int(i), int(j)))

    return [(i, j) for _depth, i, j in sorted(found)]


@figure(figsize=(9, 7), name="pmf-heatmap")
def plot_pmf_heatmap(
    ax: Any,
    pmf: np.ndarray,
    theta_bins: np.ndarray,
    psi_bins: np.ndarray,
    *,
    vmax: float | None = None,
    mark_minima: bool = True,
    cmap: str = "viridis",
    angle_kind: str = "theta",
) -> Any:
    """PMF over (ψ, θ) as a filled mesh, with basin minima marked.

    Args:
        pmf: ``(n_theta, n_psi)`` free energy in kcal/mol.
        theta_bins, psi_bins: Bin centres in **radians**.
        vmax: Colour ceiling; defaults to the 95th percentile of the finite
            values, so a few near-infinite empty bins cannot flatten the scale.
        mark_minima: Star the strict local minima.
        angle_kind: ``"theta"`` (default) labels the y-axis over its full
            ``[0, 180]`` degree range. ``"tilt"`` labels it ``[0, 90]``
            instead, for PMFs computed with
            ``rotmd.analysis.pmf.compute_pmf_2d(..., angle_kind="tilt")`` —
            using the wrong one doesn't misplace any data, it just leaves half
            the panel empty (``"theta"`` on a tilt PMF) or clips the axis
            short of an out-of-range point (``"tilt"`` on a theta PMF).
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if vmax is None:
        vmax = finite_percentile(pmf, 95)

    psi_grid, theta_grid = angle_grid_degrees(theta_bins, psi_bins)
    mesh = ax.pcolormesh(
        psi_grid, theta_grid, np.clip(pmf, 0, vmax),
        cmap=cmap, shading="auto", vmin=0, vmax=vmax,
    )
    ax.figure.colorbar(mesh, ax=ax, label="PMF (kcal/mol)")

    if mark_minima:
        for i, j in local_minima(pmf):
            # Degrees, from the same converter as the mesh. The legacy version
            # drew the mesh in radians and the markers in degrees.
            ax.plot(
                np.degrees(psi_bins[j]), np.degrees(theta_bins[i]),
                "r*", markersize=14, markeredgecolor="white", markeredgewidth=1,
            )

    label_orientation_axes(ax, limits=(angle_kind == "theta"))
    if angle_kind == "tilt":
        ax.set_xlim(0, 360)
        ax.set_ylabel("tilt (degrees)")
        ax.set_ylim(0, 90)
    ax.set_title("Potential of mean force")
    return mesh


@figure(figsize=(9, 7), name="friction-map")
def plot_friction_map(
    ax: Any,
    gamma_map: np.ndarray,
    theta_centers_deg: np.ndarray,
    psi_centers_deg: np.ndarray,
    *,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Any:
    """Orientation-dependent friction gamma(theta, psi) as a filled mesh.

    Args:
        gamma_map: ``(n_theta, n_psi)`` friction coefficient in amu/ps; NaN in
            bins with too few samples to fit (see
            :func:`rotmd.analysis.friction.orientation_dependent_friction`).
        theta_centers_deg, psi_centers_deg: Bin centres, already in
            **degrees** — unlike the PMF bins above, friction bins natively in
            degrees (``orientation_dependent_friction`` hard-codes its ranges
            as ``[0, 90]``/``[0, 360]``), so this does not go through
            :func:`~rotmd.viz.core.angle_grid_degrees`, which expects radians.
        vmax: Colour ceiling; defaults to the 95th percentile of the finite
            (fitted) bins, so sparse-bin NaNs cannot flatten the scale.
    """
    gamma_map = np.asarray(gamma_map, dtype=np.float64)
    if vmax is None:
        vmax = finite_percentile(gamma_map, 95)

    psi_grid, theta_grid = np.meshgrid(
        np.asarray(psi_centers_deg, dtype=np.float64),
        np.asarray(theta_centers_deg, dtype=np.float64),
    )
    mesh = ax.pcolormesh(
        psi_grid, theta_grid, np.clip(gamma_map, 0, vmax),
        cmap=cmap, shading="auto", vmin=0, vmax=vmax,
    )
    ax.figure.colorbar(mesh, ax=ax, label="gamma (amu/ps)")

    label_orientation_axes(ax, limits=False)
    ax.set_xlim(psi_grid.min(), psi_grid.max())
    ax.set_ylim(0, 90)  # friction bins only span [0, 90] deg, unlike full theta
    ax.set_title("Orientation-dependent friction")
    return mesh


@figure(figsize=(9, 7), name="pmf-contour")
def plot_pmf_contour(
    ax: Any,
    pmf: np.ndarray,
    theta_bins: np.ndarray,
    psi_bins: np.ndarray,
    *,
    n_levels: int = 15,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    angle_kind: str = "theta",
) -> Any:
    """PMF as filled contours with labelled iso-energy lines.

    Args:
        angle_kind: See :func:`plot_pmf_heatmap` — ``"tilt"`` labels the
            y-axis ``[0, 90]`` degrees instead of the full ``[0, 180]``.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if vmax is None:
        vmax = finite_percentile(pmf, 95)

    psi_grid, theta_grid = angle_grid_degrees(theta_bins, psi_bins)
    levels = np.linspace(0, vmax, n_levels)

    filled = ax.contourf(psi_grid, theta_grid, pmf, levels=levels, cmap=cmap, extend="max")
    lines = ax.contour(
        psi_grid, theta_grid, pmf, levels=levels, colors="black", linewidths=0.5, alpha=0.3
    )
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.1f")
    ax.figure.colorbar(filled, ax=ax, label="PMF (kcal/mol)")

    label_orientation_axes(ax, limits=(angle_kind == "theta"))
    if angle_kind == "tilt":
        ax.set_xlim(0, 360)
        ax.set_ylabel("tilt (degrees)")
        ax.set_ylim(0, 90)
    ax.set_title("PMF contours")
    return filled


@figure(figsize=(10, 8), projection="3d", name="pmf-surface")
def plot_pmf_3d_surface(
    ax: Any,
    pmf: np.ndarray,
    theta_bins: np.ndarray,
    psi_bins: np.ndarray,
    *,
    vmax: float | None = None,
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = 45,
) -> Any:
    """PMF as a 3-D surface. Barriers above ``vmax`` are clipped flat."""
    pmf = np.asarray(pmf, dtype=np.float64)
    if vmax is None:
        vmax = finite_percentile(pmf, 95)

    psi_grid, theta_grid = angle_grid_degrees(theta_bins, psi_bins)
    surface = ax.plot_surface(
        psi_grid, theta_grid, np.clip(pmf, 0, vmax), cmap=cmap, edgecolor="none", alpha=0.9
    )
    ax.figure.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="PMF (kcal/mol)")

    ax.set_xlabel("ψ (degrees)")
    ax.set_ylabel("θ (degrees)")
    ax.set_zlabel("PMF (kcal/mol)")
    ax.set_title("PMF surface")
    ax.view_init(elev=elev, azim=azim)
    return surface


@figure(figsize=(9, 7), name="torque-field")
def plot_torque_vector_field(
    ax: Any,
    torque_field: Callable[[float, float], tuple[float, float]],
    *,
    theta_range: tuple[float, float] = (0.0, np.pi),
    psi_range: tuple[float, float] = (0.0, 2 * np.pi),
    n_grid: int = 16,
) -> Any:
    """Torque as a direction field on the (ψ, θ) plane.

    Args:
        torque_field: ``f(theta, psi) -> (tau_theta, tau_psi)``, angles in
            radians. Called on a grid; it need not be vectorised.
        theta_range, psi_range: Sampled span in radians.
        n_grid: Points per axis.

    Notes:
        Arrows are unit-normalised and coloured by magnitude. The legacy
        version passed raw torques with ``scale=100, scale_units='xy'``, which
        ties arrow length to the data units of the axes — with θ in radians on
        one axis and torque in kcal/mol in the arrows, that scale was
        arbitrary, and a change of angle units silently rescaled every arrow.
        Direction and magnitude are separated here so neither depends on the
        axis units.
    """
    theta = np.linspace(*theta_range, n_grid)
    psi = np.linspace(*psi_range, n_grid)
    theta_grid, psi_grid = np.meshgrid(theta, psi, indexing="ij")

    tau_theta = np.empty_like(theta_grid)
    tau_psi = np.empty_like(psi_grid)
    for i in range(n_grid):
        for j in range(n_grid):
            tau_theta[i, j], tau_psi[i, j] = torque_field(theta_grid[i, j], psi_grid[i, j])

    magnitude = np.hypot(tau_theta, tau_psi)
    scale = np.where(magnitude > 0, magnitude, 1.0)
    arrows = ax.quiver(
        np.degrees(psi_grid), np.degrees(theta_grid),
        tau_psi / scale, tau_theta / scale,
        magnitude, cmap="magma", pivot="mid", alpha=0.85,
    )
    ax.figure.colorbar(arrows, ax=ax, label="|τ| (kcal/mol)")

    label_orientation_axes(ax)
    ax.set_title("Torque field")
    ax.grid(True, alpha=0.15)
    return arrows


@figure(figsize=(10, 8), name="free-energy-landscape")
def plot_free_energy_landscape(
    ax: Any,
    pmf: np.ndarray,
    theta_bins: np.ndarray,
    psi_bins: np.ndarray,
    *,
    trajectory: tuple[np.ndarray, np.ndarray] | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Any:
    """PMF with a trajectory drawn over it, start and end marked.

    Args:
        trajectory: ``(theta, psi)`` in radians, the path to overlay.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if vmax is None:
        vmax = finite_percentile(pmf, 95)

    psi_grid, theta_grid = angle_grid_degrees(theta_bins, psi_bins)
    mesh = ax.pcolormesh(
        psi_grid, theta_grid, np.clip(pmf, 0, vmax),
        cmap=cmap, shading="auto", vmin=0, vmax=vmax, alpha=0.85,
    )
    ax.figure.colorbar(mesh, ax=ax, label="PMF (kcal/mol)")

    if trajectory is not None:
        theta_traj, psi_traj = (np.degrees(np.asarray(a, dtype=np.float64)) for a in trajectory)
        ax.plot(psi_traj, theta_traj, "-", color="crimson", lw=0.5, alpha=0.35, label="trajectory")
        ax.plot(psi_traj[0], theta_traj[0], "o", color="lime", ms=8,
                markeredgecolor="black", label="start")
        ax.plot(psi_traj[-1], theta_traj[-1], "s", color="red", ms=8,
                markeredgecolor="black", label="end")
        ax.legend(loc="upper right")

    for i, j in local_minima(pmf):
        ax.plot(
            np.degrees(psi_bins[j]), np.degrees(theta_bins[i]),
            "w*", markersize=14, markeredgecolor="black", markeredgewidth=1.2,
        )

    label_orientation_axes(ax)
    ax.set_title("Free energy landscape")
    return mesh
