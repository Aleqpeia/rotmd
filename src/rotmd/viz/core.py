"""The plotting API: one decorator that turns a drawing function into a figure function.

Every plot in the legacy ``visualization/`` package repeated the same eleven
lines — build a figure, draw, ``tight_layout``, ``savefig(dpi=300,
bbox_inches='tight')``, ``print("✓ Saved")``, ``show()`` — and each copy
diverged slightly (one saved at dpi=900, some closed the figure and some
leaked it, all of them called ``show()``, which does nothing on a headless
node). That boilerplate is what this module removes.

A plot is written as a function that draws on axes it is handed::

    @figure(figsize=(10, 8), name="pmf-heatmap")
    def plot_pmf_heatmap(ax, pmf, theta_bins, psi_bins, *, vmax=None):
        ...draw on ax...

and the decorator gives it three calling modes for free:

    ``plot_pmf_heatmap(pmf, tb, pb, output="pmf.png")``  -> writes it, returns ``Path``
    ``plot_pmf_heatmap(pmf, tb, pb)``                    -> returns the ``Figure``
    ``plot_pmf_heatmap(pmf, tb, pb, ax=ax)``             -> draws into *your* axes

The third mode is the one that matters for reuse: because the drawing code
never creates or closes a figure, any plot here can be dropped into a panel of
a larger figure without being rewritten. That was impossible before — every
legacy function owned its figure and closed it.

Rendering always goes through the Agg backend. This runs on compute and login
nodes with no display, where importing pyplot against an interactive backend
fails outright.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DPI = 150

#: House style, applied around every figure built through :func:`figure`.
#:
#: These were the values repeated inline in every legacy plot (bold titles,
#: 12-point axis labels, a faint dashed grid). Setting them once means a plot
#: function contains only the parts that differ between plots.
#:
#: ``axes.grid`` stays off: grids belong on phase portraits and spectra, not on
#: the correlation-matrix images in :mod:`rotmd.analysis.plots`. Individual
#: plots opt in with ``ax.grid(True)``, which picks up the alpha and style set
#: here.
RC: dict[str, Any] = {
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
}

_REGISTRY: dict[str, Callable[..., Any]] = {}

_UNSET: Any = object()


def pyplot():
    """Import ``pyplot`` with a headless backend pinned first.

    ``force=True`` because a notebook kernel may already have selected an
    interactive backend; on a compute node that backend cannot draw, and the
    failure surfaces deep inside ``savefig`` rather than here.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


@contextmanager
def style(**overrides: Any) -> Iterator[Any]:
    """``rc_context`` over :data:`RC`, yielding the pyplot module.

    Use directly only when building a figure by hand; :func:`figure` already
    wraps the drawing call in this.
    """
    plt = pyplot()
    with plt.rc_context({**RC, **overrides}):
        yield plt


def save(fig: Any, output: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """Write ``fig`` to ``output``, creating parent directories."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    return output


def figure(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    projection: str | None = None,
    pass_figure: bool = False,
    layout: str | None = "constrained",
    dpi: int = DEFAULT_DPI,
    rc: dict[str, Any] | None = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Turn a drawing function into a figure function.

    The wrapped function is called as ``draw(target, *args, **kwargs)`` where
    ``target`` is a single ``Axes`` (the 1x1 case), an ``ndarray`` of axes, or
    the ``Figure`` itself when ``pass_figure=True`` — the escape hatch for
    plots that need their own gridspec.

    The wrapper adds these keyword arguments to whatever the drawing function
    already takes:

    ``output``
        Path to write to. Returns the ``Path``; the figure is closed.
    ``ax``
        Draw into existing axes instead of creating a figure. Nothing is saved
        or closed — the caller owns the figure. Returns the drawing
        function's own return value.
    ``title``
        Overrides the title: ``set_title`` on a single-axes figure,
        ``suptitle`` otherwise.
    ``figsize``, ``dpi``
        Per-call overrides of the decorator's defaults.

    With neither ``output`` nor ``ax``, the ``Figure`` is returned open, for a
    notebook to display. Closing it is then the caller's business — the same
    contract as ``plt.subplots``.

    Args:
        nrows, ncols: Subplot grid handed to the drawing function.
        figsize: Default size; ``None`` scales with the grid.
        projection: e.g. ``"3d"``, applied to every subplot.
        pass_figure: Hand over the ``Figure`` instead of axes. Incompatible
            with ``ax=``, since there is no figure to own.
        layout: Matplotlib layout engine. ``"constrained"`` by default because
            ``tight_layout`` (what the legacy plots used) fights with the
            colorbars and divider axes several of them create.
        rc: Style overrides for this plot only, on top of :data:`RC`.
        name: Register under this name for :func:`render` and :func:`available`.

    Returns:
        A decorator.
    """

    def decorate(draw: Callable[..., Any]) -> Callable[..., Any]:
        # Captured here because the wrapper exposes parameters of the same
        # names, which shadow the decorator's inside its body.
        default_dpi = dpi
        default_figsize = figsize if figsize is not None else _default_figsize(nrows, ncols)

        @functools.wraps(draw)
        def wrapper(
            *args: Any,
            output: str | Path | None = None,
            ax: Any = None,
            title: str | None = None,
            figsize: tuple[float, float] | None = _UNSET,
            dpi: int = _UNSET,
            **kwargs: Any,
        ) -> Any:
            size = default_figsize if figsize is _UNSET else figsize
            resolution = default_dpi if dpi is _UNSET else dpi

            if ax is not None:
                if pass_figure:
                    raise TypeError(
                        f"{draw.__name__} builds its own multi-panel layout and "
                        f"cannot draw into a single supplied ax"
                    )
                if output is not None:
                    raise TypeError(
                        "pass either ax= or output=, not both: with ax= the "
                        "figure belongs to the caller, who decides when to save it"
                    )
                with style(**(rc or {})):
                    result = draw(ax, *args, **kwargs)
                if title is not None:
                    ax.set_title(title)
                return result

            with style(**(rc or {})) as plt:
                if pass_figure:
                    fig = plt.figure(figsize=size, layout=layout)
                    target: Any = fig
                else:
                    fig, target = plt.subplots(
                        nrows,
                        ncols,
                        figsize=size,
                        sharex=sharex,
                        sharey=sharey,
                        layout=layout,
                        subplot_kw=({"projection": projection} if projection else None),
                        squeeze=False,
                    )
                    # squeeze=False above, then reshaped explicitly: a single
                    # panel arrives as a bare Axes, any strip of panels as a
                    # flat array, and a true grid keeps its 2-D shape. Letting
                    # matplotlib squeeze would make the 1xN and Nx1 cases
                    # disagree, so `axes[i]` would break on a grid transpose.
                    if nrows == ncols == 1:
                        target = target[0, 0]
                    elif nrows == 1 or ncols == 1:
                        target = target.reshape(-1)
                try:
                    draw(target, *args, **kwargs)
                    if title is not None:
                        if nrows == ncols == 1 and not pass_figure:
                            target.set_title(title)
                        else:
                            fig.suptitle(title)
                    if output is None:
                        return fig
                    path = save(fig, output, resolution)
                except BaseException:
                    plt.close(fig)
                    raise
                plt.close(fig)
                return path

        wrapper.draw = draw  # type: ignore[attr-defined]
        if name is not None:
            if name in _REGISTRY:
                raise ValueError(f"plot name {name!r} is already registered")
            _REGISTRY[name] = wrapper
        return wrapper

    return decorate


def _default_figsize(nrows: int, ncols: int) -> tuple[float, float]:
    """Grow the canvas with the grid so panels keep a usable aspect."""
    return (5.5 * ncols + 1.0, 4.2 * nrows + 0.6)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def available() -> list[str]:
    """Registered plot names, sorted."""
    return sorted(_REGISTRY)


def render(name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a registered plot by name.

    Lets a caller that only has a string — a CLI ``--figure`` flag, a config
    file, a notebook loop over several plots — reach any plot in the package
    without importing its module.
    """
    try:
        plot = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown plot {name!r}; available: {', '.join(available())}"
        ) from None
    return plot(*args, **kwargs)


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def angle_grid_degrees(
    theta_bins: np.ndarray, psi_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(psi, theta)`` meshgrids in **degrees**, ordered for pcolormesh/contour.

    Angles are radians everywhere inside the package and degrees on every
    axis, and mixing the two was the single most common defect in the legacy
    plots: ``plot_pmf_heatmap`` drew its mesh in radians while placing its
    minima markers in degrees (a factor of 57.3 apart, so the markers landed
    off the map), and ``plot_energy_phase_space`` drew a radian-valued PMF
    behind a degree-valued scatter on axes fixed to 0-360, collapsing the
    background into an invisible sliver in the corner.

    Going through one converter makes that mistake unrepresentable: the
    returned grids are always degrees, in the ``(x, y)`` order the matplotlib
    2-D functions expect.
    """
    theta_grid, psi_grid = np.meshgrid(
        np.degrees(np.asarray(theta_bins, dtype=np.float64)),
        np.degrees(np.asarray(psi_bins, dtype=np.float64)),
        indexing="ij",
    )
    return psi_grid, theta_grid


def label_orientation_axes(ax: Any, *, limits: bool = True) -> None:
    """Label a (psi, theta) panel in degrees, with the physical ranges."""
    ax.set_xlabel("ψ (degrees)")
    ax.set_ylabel("θ (degrees)")
    if limits:
        # ZYZ Euler: psi is a full turn, theta only a half — the polar angle
        # is measured from the axis, not around it.
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)


def time_coloured_path(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    *,
    cmap: str = "viridis",
    linewidth: float = 1.0,
    alpha: float = 0.6,
) -> Any:
    """Draw a path whose colour follows ``c``, as one artist.

    The legacy plots did this with ``for i in range(len(x) - 1): ax.plot(...)``,
    which creates one ``Line2D`` per frame. At the 25 000-frame trajectories
    this package produces that is 25 000 artists in the figure: rendering takes
    minutes and a vector export is unopenable. A ``LineCollection`` is one
    artist regardless of length, and is the reason these functions are usable
    on real trajectory lengths at all.

    Returns:
        The ``LineCollection``, ready to pass to ``fig.colorbar``.
    """
    from matplotlib.collections import LineCollection

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    collection = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=alpha)
    # One colour per segment, so c is averaged across each segment's endpoints
    # rather than silently truncated -- the legacy code indexed c[i] for the
    # segment from i to i+1, biasing every segment toward its start.
    collection.set_array((c[:-1] + c[1:]) / 2.0)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def finite_percentile(values: np.ndarray, q: float) -> float:
    """``percentile`` over the finite entries, or 1.0 if there are none.

    PMF grids carry ``inf`` in unvisited bins, and every legacy caller wrote
    ``np.percentile(v[np.isfinite(v)], 95)`` inline — which raises on an
    all-empty grid instead of producing an empty figure with a sane scale.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(finite, q))
