"""Phase-space figures for orientational dynamics.

Restored from the pre-slimming ``visualization/phase_space.py``, which carried
the most defects of the three visualization modules. What changed, beyond the
shared API:

* ``plot_energy_phase_space`` drew a radian-valued PMF behind a degree-valued
  scatter on axes fixed to 0-360 and 0-180, so the landscape collapsed into an
  invisible sliver in the corner of every figure it produced. It also indexed
  ``energy[1:]`` against full-length angle arrays, shifting the colour of every
  point by one frame, and dereferenced ``pmf`` unconditionally despite
  documenting it as optional.
* ``plot_poincare_section_improved`` sliced ``euler_angles[1:-1]`` but
  ``angular_velocities[:]``, then used a mask built from the shorter array to
  index the longer one — an ``IndexError`` on any real input. Its "crossings"
  were also a proximity test, which counts a slow approach many times and
  misses a fast transit entirely; it is a true interpolated crossing now.
* ``plot_phase_portrait_with_vector_field`` set ``dω/dt`` to a single global
  mean torque, making the field constant everywhere — it could not show the
  structure it existed to show. It now bins the measured torque over the
  (angle, ω) plane, and applies the unit conversion the legacy comment
  described but never performed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import figure, label_orientation_axes, time_coloured_path

#: 1 kcal/mol expressed in amu·Å²/ps², the package's mechanical unit system.
#:
#: Needed to turn a torque in kcal/mol/rad and a moment of inertia in amu·Å²
#: into an angular acceleration in rad/ps². Without it ``τ/I`` is off by this
#: factor and the phase-space flow is unreadable against ω.
KCAL_PER_MOL_IN_AMU_A2_PS2 = 418.4


def poincare_crossings(
    phi: np.ndarray,
    target: float,
    *,
    direction: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate crossings of ``phi`` through ``target``, to sub-frame precision.

    Args:
        phi: Angle series in radians. Wrapping at ±π is handled.
        target: Section plane, in radians.
        direction: ``+1`` increasing crossings only, ``-1`` decreasing only,
            ``0`` both. A Poincaré section conventionally takes one direction;
            mixing them superimposes two different sections.

    Returns:
        ``(index, frac)`` — for each crossing, the frame before it and the
        fraction of the way to the next frame. Interpolate any co-recorded
        observable with ``x[index] + frac * (x[index + 1] - x[index])``.

    Notes:
        The signed distance is taken on the circle, so a crossing at 0 is found
        whether ``phi`` approaches from 0.01 or from 6.27. Steps that jump more
        than π are branch-cut artefacts of the wrap, not crossings, and are
        excluded — the legacy proximity test could not tell the two apart.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.size < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float64)

    # Signed distance to the plane, in (-pi, pi].
    delta = np.angle(np.exp(1j * (phi - target)))
    before, after = delta[:-1], delta[1:]

    crossed = np.signbit(before) != np.signbit(after)
    crossed &= np.abs(after - before) < np.pi  # reject wrap-around steps
    if direction > 0:
        crossed &= after > before
    elif direction < 0:
        crossed &= after < before

    index = np.flatnonzero(crossed)
    span = after[index] - before[index]
    frac = np.where(span != 0, -before[index] / np.where(span != 0, span, 1.0), 0.0)
    return index, frac


def _interpolate(values: np.ndarray, index: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``values`` at ``index + frac``."""
    values = np.asarray(values, dtype=np.float64)
    return values[index] + frac * (values[index + 1] - values[index])


@figure(figsize=(9, 7), name="phase-portrait")
def plot_phase_portrait_2d(
    ax: Any,
    angle: np.ndarray,
    angular_velocity: np.ndarray,
    *,
    angle_label: str = "θ",
    times: np.ndarray | None = None,
    gridsize: int = 30,
    show_trajectory: bool = True,
    trajectory_alpha: float = 0.35,
) -> Any:
    """Density of states in the (angle, ω) plane, with the path over it.

    Hexbin rather than scatter because the question this figure answers is
    where the trajectory *spends time*; at trajectory lengths this package
    produces, a scatter saturates and every region looks equally occupied.

    Args:
        angle: Angle series in radians.
        angular_velocity: Conjugate ω in rad/ps.
        angle_label: Symbol for the axis labels.
        times: Timestamps in ps; colours the path by time when given.
        gridsize: Hexbin resolution.
    """
    angle = np.asarray(angle, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity, dtype=np.float64)

    density = ax.hexbin(
        angle, angular_velocity, gridsize=gridsize, cmap="YlOrRd", mincnt=1, linewidths=0.2
    )
    ax.figure.colorbar(density, ax=ax, label="density (frames)")

    if show_trajectory:
        if times is not None:
            path = time_coloured_path(
                ax, angle, angular_velocity, np.asarray(times, dtype=np.float64),
                cmap="viridis", linewidth=1.0, alpha=trajectory_alpha,
            )
            ax.figure.colorbar(path, ax=ax, label="time (ps)")
        else:
            ax.plot(angle, angular_velocity, "-", color="tab:blue",
                    alpha=trajectory_alpha, lw=0.8)

    sub = angle_label[0]
    ax.set(
        xlabel=f"{angle_label} (rad)",
        ylabel=f"ω_{sub} (rad/ps)",
        title=f"Phase portrait: ({angle_label}, ω_{sub})",
    )
    _radian_ticks(ax, angle)
    ax.grid(True)
    return density


def _radian_ticks(ax: Any, angle: np.ndarray) -> None:
    """Label the x axis in multiples of π/2, over the range the data occupies.

    The legacy version hard-coded ticks from 0 to 2π on every portrait. For θ,
    which is a polar angle confined to [0, π], that stretched the axis to twice
    the occupied range and pushed the data into the left half of the panel.
    """
    lo, hi = float(np.min(angle)), float(np.max(angle))
    names = {0: "0", 1: "π/2", 2: "π", 3: "3π/2", 4: "2π", -1: "−π/2", -2: "−π"}
    ticks = [k * np.pi / 2 for k in names if lo - 0.05 <= k * np.pi / 2 <= hi + 0.05]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([names[int(round(t / (np.pi / 2)))] for t in ticks])


@figure(figsize=(10, 8), name="energy-phase-space")
def plot_energy_phase_space(
    ax: Any,
    theta: np.ndarray,
    psi: np.ndarray,
    *,
    energy: np.ndarray | None = None,
    times: np.ndarray | None = None,
    pmf: np.ndarray | None = None,
    theta_bins: np.ndarray | None = None,
    psi_bins: np.ndarray | None = None,
    n_contours: int = 8,
) -> Any:
    """The (ψ, θ) path, optionally over PMF contours, coloured by time or energy.

    Args:
        theta, psi: Angle series in radians.
        energy: Per-frame energy in kcal/mol; colours the path if ``times`` is
            not given. Must be the same length as the angles.
        times: Timestamps in ps; takes precedence for colouring.
        pmf: ``(n_theta, n_psi)`` background. Requires both bin arrays; all
            three are ignored together if any is missing.
        n_contours: Contour levels for the background.
    """
    theta = np.degrees(np.asarray(theta, dtype=np.float64))
    psi = np.degrees(np.asarray(psi, dtype=np.float64))

    if pmf is not None and theta_bins is not None and psi_bins is not None:
        # Both grids in degrees, matching the scatter and the axis limits. The
        # legacy version passed radians here and degrees below.
        from .core import angle_grid_degrees

        psi_grid, theta_grid = angle_grid_degrees(theta_bins, psi_bins)
        ax.contourf(psi_grid, theta_grid, np.asarray(pmf, dtype=np.float64),
                    levels=n_contours, cmap="bone")
        ax.contour(psi_grid, theta_grid, np.asarray(pmf, dtype=np.float64),
                   levels=n_contours, colors="black", alpha=0.5, linewidths=1.0)

    colour, label = (times, "time (ps)") if times is not None else (energy, "energy (kcal/mol)")
    if colour is not None:
        colour = np.asarray(colour, dtype=np.float64)
        if colour.size != theta.size:
            raise ValueError(
                f"colour series has {colour.size} values but the trajectory has "
                f"{theta.size} frames; slice both the same way before plotting"
            )
        points = ax.scatter(psi, theta, c=colour, cmap="viridis", s=10, alpha=0.65,
                            linewidths=0)
        ax.figure.colorbar(points, ax=ax, label=label)
    else:
        points = ax.scatter(psi, theta, s=10, alpha=0.5, color="tab:blue", linewidths=0)

    label_orientation_axes(ax)
    ax.set_title("Orientation trajectory on the energy landscape")
    ax.grid(True, alpha=0.2)
    return points


@figure(figsize=(9, 7), name="L-phase-space")
def plot_L_phase_space(
    ax: Any,
    L_parallel: np.ndarray,
    L_perp: np.ndarray,
    *,
    times: np.ndarray | None = None,
    energy: np.ndarray | None = None,
    n_arrows: int = 50,
) -> Any:
    """Angular-momentum flow: L_∥ against L_⊥ with the velocity field on it.

    Arrows are ``(dL_∥/dt, dL_⊥/dt)`` at trajectory midpoints, subsampled to
    roughly ``n_arrows``. Dashed rays mark constant L_∥/L_⊥, which separate
    spin-dominated from nutation-dominated motion.

    Args:
        L_parallel, L_perp: Magnitudes in amu·Å²/ps.
        times: Timestamps in ps; sets the arrow colour and the time derivative.
        energy: Alternative arrow colouring when ``times`` is absent.
        n_arrows: Approximate arrow count.
    """
    L_parallel = np.asarray(L_parallel, dtype=np.float64)
    L_perp = np.asarray(L_perp, dtype=np.float64)

    if L_parallel.size > 1:
        dt = np.diff(np.asarray(times, dtype=np.float64)) if times is not None \
            else np.ones(L_parallel.size - 1)
        v_par = np.diff(L_parallel) / dt
        v_perp = np.diff(L_perp) / dt
        mid_par = (L_parallel[:-1] + L_parallel[1:]) / 2
        mid_perp = (L_perp[:-1] + L_perp[1:]) / 2

        stride = max(1, mid_par.size // max(n_arrows, 1))
        sl = slice(None, None, stride)
        quiver_kw = {
            "angles": "xy", "scale_units": "xy", "alpha": 0.75,
            "width": 0.004, "headwidth": 4, "headlength": 5,
        }

        colour, cbar_label, cmap = None, None, None
        if times is not None:
            colour, cbar_label, cmap = np.asarray(times)[:-1][sl], "time (ps)", "viridis"
        elif energy is not None:
            colour, cbar_label, cmap = np.asarray(energy)[:-1][sl], "energy (kcal/mol)", "coolwarm"

        if colour is not None:
            arrows = ax.quiver(mid_par[sl], mid_perp[sl], v_par[sl], v_perp[sl],
                               colour, cmap=cmap, **quiver_kw)
            ax.figure.colorbar(arrows, ax=ax, label=cbar_label)
        else:
            ax.quiver(mid_par[sl], mid_perp[sl], v_par[sl], v_perp[sl],
                      color="tab:blue", **quiver_kw)

        ax.plot(L_parallel, L_perp, "k-", alpha=0.15, lw=0.8, zorder=1)

    span = float(max(L_parallel.max(), L_perp.max()))
    for ratio in (0.5, 1.0, 2.0):
        line = np.linspace(0, span, 2)
        ax.plot(ratio * line, line, "k--", alpha=0.25, lw=0.8, zorder=0)

    ax.set(
        xlabel="L∥ (spin) [amu·Å²/ps]",
        ylabel="L⊥ (nutation) [amu·Å²/ps]",
        title="Angular-momentum phase space",
        xlim=(0, span * 1.05),
        ylim=(0, span * 1.05),
    )
    ax.grid(True)
    return ax


@figure(figsize=(10, 8), name="phase-portrait-field")
def plot_phase_portrait_with_vector_field(
    ax: Any,
    angle: np.ndarray,
    angular_velocity: np.ndarray,
    torque: np.ndarray,
    moment_of_inertia: float,
    *,
    angle_label: str = "θ",
    n_arrows: int = 20,
    min_samples: int = 3,
) -> Any:
    """Phase portrait with the empirical flow field ``(ω, τ/I)`` over it.

    The field is built by binning the measured torque over the (angle, ω)
    plane, so each arrow reports the mean angular acceleration actually
    observed in that cell. Cells with fewer than ``min_samples`` frames are
    left blank rather than drawn from one or two points.

    Args:
        angle: Angle series in radians.
        angular_velocity: ω in rad/ps.
        torque: Conjugate torque component per frame, in kcal/mol/rad.
        moment_of_inertia: I about the same axis, in amu·Å².
        n_arrows: Grid resolution per axis.
        min_samples: Frames required in a cell before its arrow is drawn.

    Notes:
        ``dθ/dt = ω`` is exact by definition, so only the vertical component
        is estimated. τ/I is converted through
        :data:`KCAL_PER_MOL_IN_AMU_A2_PS2` to give rad/ps², which is what the
        vertical axis of this plane is measured in — the legacy version divided
        the raw kcal/mol number by I and plotted the result as if it were
        commensurate with ω.
    """
    from scipy.stats import binned_statistic_2d

    angle = np.asarray(angle, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
    torque = np.asarray(torque, dtype=np.float64)

    density = ax.hexbin(angle, angular_velocity, gridsize=n_arrows + 5,
                        cmap="Blues", mincnt=1, alpha=0.45)
    ax.figure.colorbar(density, ax=ax, label="density (frames)")

    alpha = torque * KCAL_PER_MOL_IN_AMU_A2_PS2 / float(moment_of_inertia)
    edges = (
        np.linspace(angle.min(), angle.max(), n_arrows + 1),
        np.linspace(angular_velocity.min(), angular_velocity.max(), n_arrows + 1),
    )
    mean_alpha, x_edges, y_edges, _ = binned_statistic_2d(
        angle, angular_velocity, alpha, statistic="mean", bins=edges
    )
    counts, *_ = binned_statistic_2d(
        angle, angular_velocity, alpha, statistic="count", bins=edges
    )

    centres_x = (x_edges[:-1] + x_edges[1:]) / 2
    centres_y = (y_edges[:-1] + y_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(centres_x, centres_y, indexing="ij")

    occupied = counts >= min_samples
    u = grid_y[occupied]                      # dθ/dt = ω
    v = mean_alpha[occupied]                  # dω/dt = τ/I
    magnitude = np.hypot(u, v)
    scale = np.where(magnitude > 0, magnitude, 1.0)

    arrows = ax.quiver(
        grid_x[occupied], grid_y[occupied], u / scale, v / scale, magnitude,
        cmap="magma", pivot="mid", alpha=0.9, width=0.004,
    )
    ax.figure.colorbar(arrows, ax=ax, label="|flow| (rad/ps, rad/ps²)")

    sub = angle_label[0]
    ax.set(
        xlabel=f"{angle_label} (rad)",
        ylabel=f"ω_{sub} (rad/ps)",
        title=f"Flow field: ({angle_label}, ω_{sub})",
    )
    ax.grid(True)
    return arrows


@figure(nrows=2, ncols=2, figsize=(13, 10), name="poincare")
def plot_poincare_sections(
    axes: Any,
    euler_angles: np.ndarray,
    angular_velocities: np.ndarray,
    *,
    section_angles: tuple[float, ...] = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2),
    direction: int = 1,
    gridsize: int = 30,
) -> Any:
    """Poincaré sections in (θ, ω_θ), taken where φ crosses each plane.

    Args:
        euler_angles: ``(n_frames, 3)`` ZYZ angles (φ, θ, ψ) in radians.
        angular_velocities: ``(n_frames, 3)`` in rad/ps.
        section_angles: φ planes to section at; the first four are used.
        direction: Crossing sense passed to :func:`poincare_crossings`.
        gridsize: Hexbin resolution.
    """
    euler_angles = np.asarray(euler_angles, dtype=np.float64)
    angular_velocities = np.asarray(angular_velocities, dtype=np.float64)
    if euler_angles.shape[0] != angular_velocities.shape[0]:
        raise ValueError(
            f"euler_angles has {euler_angles.shape[0]} frames but "
            f"angular_velocities has {angular_velocities.shape[0]}; they must "
            f"describe the same frames for a section to pair them correctly"
        )

    phi, theta = euler_angles[:, 0], euler_angles[:, 1]
    omega_theta = angular_velocities[:, 1]
    flat = axes.reshape(-1)

    for ax, plane in zip(flat, section_angles[: flat.size]):
        index, frac = poincare_crossings(phi, plane, direction=direction)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("ω_θ (rad/ps)")

        if index.size < 10:
            ax.text(0.5, 0.5, f"insufficient crossings\n(n = {index.size})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"φ = {plane:.2f} rad")
            continue

        section = ax.hexbin(
            _interpolate(theta, index, frac), _interpolate(omega_theta, index, frac),
            gridsize=gridsize, cmap="YlOrRd", mincnt=1,
        )
        ax.figure.colorbar(section, ax=ax, label="count")
        ax.set_title(f"φ = {plane:.2f} rad  (n = {index.size})")
        ax.grid(True)

    return axes


@figure(pass_figure=True, figsize=(15, 11), name="phase-space-summary")
def plot_multi_panel_summary(
    fig: Any,
    theta: np.ndarray,
    psi: np.ndarray,
    omega_theta: np.ndarray,
    omega_psi: np.ndarray,
    energy: np.ndarray,
    times: np.ndarray,
    *,
    gridsize: int = 25,
) -> Any:
    """Four-panel overview: both phase portraits, the path, and energy in time.

    Builds its own gridspec, so it takes the figure rather than axes — and
    composes the single-panel plots above rather than reimplementing them,
    which is the point of the ``ax=`` calling mode.
    """
    times = np.asarray(times, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    gs = fig.add_gridspec(2, 2)

    plot_phase_portrait_2d(
        theta, omega_theta, angle_label="θ", gridsize=gridsize, show_trajectory=False,
        ax=fig.add_subplot(gs[0, 0]),
    )
    plot_phase_portrait_2d(
        psi, omega_psi, angle_label="ψ", gridsize=gridsize, show_trajectory=False,
        ax=fig.add_subplot(gs[0, 1]),
    )
    plot_energy_phase_space(theta, psi, times=times, ax=fig.add_subplot(gs[1, 0]))

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(times, energy, lw=1.0, alpha=0.8, color="darkblue")
    ax.fill_between(times, energy, alpha=0.15, color="tab:blue")
    mean, sd = float(energy.mean()), float(energy.std(ddof=1))
    ax.axhline(mean, color="crimson", ls="--", lw=1.5,
               label=f"mean {mean:.2f} ± {sd:.2f} kcal/mol")
    ax.set(xlabel="time (ps)", ylabel="energy (kcal/mol)", title="Energy evolution")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Phase-space summary")
    return fig
