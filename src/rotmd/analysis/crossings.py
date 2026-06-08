"""
Angular Section-Crossing Counts for Orientation Stability
=========================================================

This module turns a continuous orientation trajectory into *count data*
suitable for Poisson / Negative-Binomial regression. The physical idea is
simple:

    A protein that sits stably in an orientational well rarely crosses a
    fixed angular gridline; a protein that wobbles crosses many gridlines.

So the number of times an orientation coordinate crosses an evenly spaced
grid of "sections" (e.g. every 15 degrees) is a direct, interpretable proxy
for *orientational mobility*. Fewer crossings means a more stable
orientation. We expose three complementary count observables:

1. ``grid crossings``      -- how often a coordinate sweeps past gridlines,
2. ``direction changes``   -- how often the coordinate reverses (turning
                              points), a grid-placement-independent mobility
                              measure,
3. ``plane crossings``     -- the full-SO(3) generalisation, where a "section"
                              is a plane through the origin and we count how
                              often the protein body axis passes through it.

The plane formulation is the most robust for rotational motion because it is
free of the gimbal-lock pathologies of Euler angles: instead of binning an
angle, we track the body axis as a unit vector on the sphere and count sign
changes of its projection onto each section normal.

Author: rotmd contributors
Date: 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ==============================================================================
# Result container
# ==============================================================================
#
# One ``CrossingCounts`` describes the count observables extracted from a single
# scalar/axis trajectory of one replica. ``exposure`` is the observation time
# (used as the GLM offset so that we model a *rate*, not raw counts that merely
# grow with trajectory length).


@dataclass(frozen=True)
class CrossingCounts:
    """Counts of section crossings for one coordinate of one trajectory."""

    coordinate: str
    n_crossings: int
    n_up: int
    n_down: int
    n_direction_changes: int
    n_frames: int
    exposure: float
    spacing_deg: float = field(default=float("nan"))

    def to_dict(self) -> dict[str, float | int | str]:
        """Flat dict for tabular aggregation / JSON export."""
        return {
            "coordinate": self.coordinate,
            "n_crossings": int(self.n_crossings),
            "n_up": int(self.n_up),
            "n_down": int(self.n_down),
            "n_direction_changes": int(self.n_direction_changes),
            "n_frames": int(self.n_frames),
            "exposure": float(self.exposure),
            "spacing_deg": float(self.spacing_deg),
        }


# ==============================================================================
# Primitive 1: level / grid crossings of a scalar series
# ==============================================================================


def count_level_crossings(
    signal: np.ndarray,
    level: float,
) -> tuple[int, int, int]:
    """
    Count how often ``signal`` crosses a single horizontal ``level``.

    A crossing between consecutive frames ``i-1`` and ``i`` occurs when
    ``signal - level`` changes sign. Frames sitting exactly on the level are
    carried forward to the previous sign so that touching the level without
    passing through it is not double counted.

    Args:
        signal: (n_frames,) coordinate time series (any units).
        level: The horizontal level to test against.

    Returns:
        ``(n_total, n_up, n_down)`` crossing counts, where "up" means the
        signal increased through the level.
    """
    x = np.asarray(signal, dtype=float) - float(level)

    # Carry the previous nonzero sign forward across exact-zero samples. This
    # treats a trajectory that grazes the level (== level for a few frames)
    # and returns to the same side as *no* crossing.
    sign = np.sign(x)
    nonzero = sign != 0
    if not nonzero.any():
        return 0, 0, 0
    idx = np.where(nonzero, np.arange(sign.size), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = sign[idx]

    changes = np.diff(filled)
    n_up = int(np.count_nonzero(changes > 0))
    n_down = int(np.count_nonzero(changes < 0))
    return n_up + n_down, n_up, n_down


def make_grid_levels(low: float, high: float, spacing: float) -> np.ndarray:
    """Return the interior gridlines in ``(low, high)`` spaced by ``spacing``."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    start = np.ceil(low / spacing) * spacing
    levels = np.arange(start, high + 0.5 * spacing, spacing)
    # Keep strictly interior lines; endpoints are not crossable boundaries.
    return levels[(levels > low) & (levels < high)]


def count_grid_crossings(
    signal: np.ndarray,
    spacing: float,
    *,
    periodic: bool = False,
    period: float = 360.0,
) -> tuple[int, int, int]:
    """
    Count crossings of an evenly spaced grid of sections.

    Rather than iterating over every gridline, we map each sample to its grid
    cell index ``floor(signal / spacing)`` and sum the absolute index changes
    between consecutive frames. This counts *every* gridline swept in a single
    step (a fast move that jumps three cells counts as three crossings), which
    is exactly the total-variation interpretation of mobility.

    Args:
        signal: (n_frames,) coordinate series. For periodic coordinates such
            as the azimuth, pass the raw wrapped angle together with
            ``periodic=True``; the shortest-arc step is then used so the
            wraparound at the period boundary is not counted as a huge jump.
        spacing: Grid spacing in the same units as ``signal``.
        periodic: Whether the coordinate wraps at ``period``.
        period: Wrap period (default 360 for degrees).

    Returns:
        ``(n_total, n_up, n_down)`` crossing counts.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    x = np.asarray(signal, dtype=float)
    if x.size < 2:
        return 0, 0, 0

    if periodic:
        # Reduce each step to its shortest signed arc in (-period/2, period/2],
        # then accumulate so a genuine multi-turn rotation still counts fully
        # while a single wrap at the boundary does not masquerade as motion.
        steps = np.diff(x)
        steps = (steps + period / 2.0) % period - period / 2.0
        x = np.concatenate([[x[0]], x[0] + np.cumsum(steps)])

    cells = np.floor(x / spacing).astype(np.int64)
    dcells = np.diff(cells)
    n_up = int(np.sum(dcells[dcells > 0]))
    n_down = int(-np.sum(dcells[dcells < 0]))
    return n_up + n_down, n_up, n_down


# ==============================================================================
# Primitive 2: direction changes (turning points)
# ==============================================================================


def count_direction_changes(
    signal: np.ndarray,
    *,
    min_step: float = 0.0,
) -> int:
    """
    Count turning points: sign reversals of the first difference.

    This is a grid-placement-independent mobility measure -- it asks how often
    the coordinate reverses course, regardless of where you draw section
    boundaries. ``min_step`` ignores reversals smaller than a threshold, a
    cheap deadband that suppresses numerical jitter / thermal chatter so that
    only genuine reversals are counted.

    Args:
        signal: (n_frames,) coordinate series.
        min_step: Reversals whose preceding move had magnitude below this are
            ignored. Use 0 to count every reversal.

    Returns:
        Number of direction reversals.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        return 0

    steps = np.diff(x)
    if min_step > 0:
        steps = steps[np.abs(steps) >= min_step]
    sign = np.sign(steps)
    sign = sign[sign != 0]
    if sign.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(sign) != 0))


# ==============================================================================
# Primitive 3: plane (section) crossings on SO(3) / the sphere
# ==============================================================================
#
# For the full-SO(3) view we represent orientation by where a chosen *body
# axis* points in the lab frame: a unit vector u(t) tracing a path on the
# sphere S^2. A "section defined by a plane" is a plane through the origin with
# normal n; the body axis crosses that section whenever u(t).n changes sign.
# This is the rigorous, gimbal-lock-free analogue of crossing an angular
# gridline.


def body_axis_vectors(
    rotation_matrices: np.ndarray,
    body_axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Map a sequence of rotation matrices to the lab-frame direction of a body axis.

    Args:
        rotation_matrices: (n_frames, 3, 3) rotations (columns are the body
            principal axes, as produced by ``extract_orientation_trajectory``).
        body_axis: Body-frame axis to track (default the 3rd principal axis).

    Returns:
        (n_frames, 3) unit vectors u(t) = R(t) @ body_axis.
    """
    R = np.asarray(rotation_matrices, dtype=float)
    e = np.asarray(body_axis, dtype=float)
    e = e / np.linalg.norm(e)
    u = np.einsum("nij,j->ni", R, e)
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.where(norms == 0, 1.0, norms)


def count_plane_crossings(
    unit_vectors: np.ndarray,
    plane_normal: np.ndarray | tuple[float, float, float],
) -> tuple[int, int, int]:
    """
    Count how often a unit-vector path crosses a section plane through the origin.

    Args:
        unit_vectors: (n_frames, 3) body-axis directions u(t).
        plane_normal: Normal n of the section plane; the body axis crosses the
            plane when ``u.n`` changes sign.

    Returns:
        ``(n_total, n_up, n_down)`` where "up" is a crossing in the +n direction.
    """
    u = np.asarray(unit_vectors, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)
    projection = u @ n
    return count_level_crossings(projection, 0.0)


def geodesic_angle_series(
    rotation_matrices: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """
    Geodesic angle (radians) of each frame from a reference orientation on SO(3).

    The geodesic distance between rotations ``R`` and ``R_ref`` is
    ``arccos((trace(R R_ref^T) - 1) / 2)`` -- the single rotation angle needed
    to align them. Counting gridline crossings of this scalar gives a
    rotation-magnitude mobility measure that ignores the *axis* of rotation.

    Args:
        rotation_matrices: (n_frames, 3, 3) rotations.
        reference: (3, 3) reference rotation; defaults to the first frame.

    Returns:
        (n_frames,) geodesic angles in radians, in [0, pi].
    """
    R = np.asarray(rotation_matrices, dtype=float)
    R_ref = R[0] if reference is None else np.asarray(reference, dtype=float)
    rel = np.einsum("nij,kj->nik", R, R_ref)  # R @ R_ref^T
    trace = np.trace(rel, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


# ==============================================================================
# High-level: orientation coordinates -> crossing counts
# ==============================================================================
#
# These wrappers compute the physically meaningful coordinates (tilt of a body
# axis vs a reference, azimuth around it, geodesic magnitude) and reduce each to
# a CrossingCounts. The reference axis is normally the membrane normal, so
# "tilt" is the protein insertion angle and "azimuth" is in-plane rotation.


def _exposure(times: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return 0.0
    return float(times[-1] - times[0])


def tilt_azimuth_from_axis(
    unit_vectors: np.ndarray,
    reference_axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a body-axis path into tilt and azimuth relative to a reference.

    ``tilt`` is the polar angle from the reference axis (the membrane normal by
    convention) and lives in [0, 180] degrees with no wraparound. ``azimuth``
    is the rotation about the reference axis in [-180, 180) degrees and *is*
    periodic.

    Args:
        unit_vectors: (n_frames, 3) body-axis directions.
        reference_axis: Reference direction (membrane normal by default).

    Returns:
        ``(tilt_deg, azimuth_deg)`` arrays.
    """
    u = np.asarray(unit_vectors, dtype=float)
    r = np.asarray(reference_axis, dtype=float)
    r = r / np.linalg.norm(r)

    tilt = np.degrees(np.arccos(np.clip(u @ r, -1.0, 1.0)))

    # Build an orthonormal in-plane basis (e1, e2) perpendicular to r so that
    # azimuth = atan2(u.e2, u.e1) is well defined and continuous.
    helper = np.array([1.0, 0.0, 0.0]) if abs(r[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - (helper @ r) * r
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(r, e1)
    azimuth = np.degrees(np.arctan2(u @ e2, u @ e1))
    return tilt, azimuth


def count_orientation_crossings(
    rotation_matrices: np.ndarray,
    times: np.ndarray,
    *,
    coordinate: str = "tilt",
    spacing_deg: float = 15.0,
    body_axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    reference_axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_step_deg: float = 0.0,
) -> CrossingCounts:
    """
    Compute section-crossing counts for one orientation coordinate.

    Args:
        rotation_matrices: (n_frames, 3, 3) orientation trajectory.
        times: (n_frames,) timestamps (ps); used for the exposure offset.
        coordinate: One of ``"tilt"``, ``"azimuth"`` or ``"geodesic"``.
        spacing_deg: Section spacing in degrees.
        body_axis: Body-frame axis whose orientation defines the coordinate.
        reference_axis: Lab-frame reference (membrane normal) for tilt/azimuth.
        min_step_deg: Deadband for direction-change counting (degrees).

    Returns:
        A ``CrossingCounts`` for the requested coordinate.

    Notes:
        ``"geodesic"`` ignores ``reference_axis`` and instead measures the
        rotation magnitude from the first frame; it has no meaningful up/down
        split (the angle is non-negative), so down counts will be the boundary
        reflections only.
    """
    R = np.asarray(rotation_matrices, dtype=float)
    n_frames = R.shape[0]

    if coordinate == "geodesic":
        series = np.degrees(geodesic_angle_series(R))
        n_total, n_up, n_down = count_grid_crossings(series, spacing_deg)
        periodic = False
    else:
        u = body_axis_vectors(R, body_axis)
        tilt, azimuth = tilt_azimuth_from_axis(u, reference_axis)
        if coordinate == "tilt":
            series, periodic = tilt, False
        elif coordinate == "azimuth":
            series, periodic = azimuth, True
        else:
            raise ValueError(
                f"unknown coordinate {coordinate!r}; "
                "expected 'tilt', 'azimuth' or 'geodesic'"
            )
        n_total, n_up, n_down = count_grid_crossings(
            series, spacing_deg, periodic=periodic, period=360.0
        )

    n_dir = count_direction_changes(series, min_step=min_step_deg)
    return CrossingCounts(
        coordinate=coordinate,
        n_crossings=n_total,
        n_up=n_up,
        n_down=n_down,
        n_direction_changes=n_dir,
        n_frames=n_frames,
        exposure=_exposure(times),
        spacing_deg=spacing_deg,
    )


def count_angle_series_crossings(
    angle_deg: np.ndarray,
    times: np.ndarray,
    *,
    coordinate: str,
    spacing_deg: float = 15.0,
    periodic: bool = False,
    min_step_deg: float = 0.0,
) -> CrossingCounts:
    """
    Crossing counts for a pre-computed angle series (degrees).

    Convenience wrapper for the common case where you already extracted a
    scalar angle (e.g. ``theta`` from an NPZ) and just want its counts without
    going through rotation matrices.

    Args:
        angle_deg: (n_frames,) angle time series in degrees.
        times: (n_frames,) timestamps.
        coordinate: Label stored on the result.
        spacing_deg: Section spacing in degrees.
        periodic: Whether the angle wraps (azimuth/spin: True; tilt: False).
        min_step_deg: Deadband for direction-change counting.

    Returns:
        A ``CrossingCounts``.
    """
    series = np.asarray(angle_deg, dtype=float)
    n_total, n_up, n_down = count_grid_crossings(
        series, spacing_deg, periodic=periodic, period=360.0
    )
    return CrossingCounts(
        coordinate=coordinate,
        n_crossings=n_total,
        n_up=n_up,
        n_down=n_down,
        n_direction_changes=count_direction_changes(series, min_step=min_step_deg),
        n_frames=series.size,
        exposure=_exposure(times),
        spacing_deg=spacing_deg,
    )
