"""Potential of mean force over Euler angles, with the SO(3) Jacobian correction applied.

The configuration space of an orientation is SO(3), parametrized here by Euler
angles ``(theta, psi, phi)`` in ZYZ convention, and its volume element is
**not** uniform in those angles: ``dmu(R) = sin(theta) dtheta dpsi dphi``. A
free energy computed as ``F = -kT ln P(theta, psi)`` without dividing out that
``sin(theta)`` factor is biased by several kT near ``theta = 0`` simply
because less *solid angle* lives there, not because the protein prefers it
less — this is the single most common way to get a PMF over Euler angles
wrong. Every function below that bins ``theta`` applies the correction;
functions over ``psi`` (uniform on ``SO(2)``) or derived scalars like energy
or ``|omega|`` do not need one.

All angles in this module are in **radians** — the same units
``rotmd.core.orientation.rotation_matrix_to_euler_zyz`` writes to the extract
schema (``theta`` in ``[0, pi]``, ``psi`` in ``[0, 2*pi]``), so callers can
pass the extracted arrays straight through with no conversion.
"""

from __future__ import annotations

import warnings

import numpy as np

# k_B in kcal/(mol*K).
_K_B = 0.001987
_DEFAULT_TEMPERATURE_K = 310.15


def jacobian_euler_angles(theta: np.ndarray) -> np.ndarray:
    """SO(3) volume-element Jacobian ``sin(theta)`` (theta in radians); vanishes at the poles (gimbal lock)."""
    return np.sin(theta)


#: ``(range, jacobian)`` for each supported polar-angle convention.
#:
#: ``theta`` is the raw ZYZ nutation angle over its full ``[0, pi]`` domain,
#: with the ``sin(theta)`` SO(3) measure.
#:
#: ``tilt`` is ``rotmd.core.orientation.membrane_tilt_angle(theta)`` —
#: ``min(theta, pi - theta)``, folding ``theta`` and ``pi - theta`` onto
#: ``[0, pi/2]``, because the principal axis extract reports is a headless
#: line (its eigenvector sign is arbitrary from frame to frame), so those two
#: raw-theta values are the same physical orientation, not two different
#: ones. Folding two branches of a ``sin(theta) dtheta`` measure onto one
#: coordinate sums their contributions: at fixed tilt ``t``, the preimages
#: are ``theta = t`` and ``theta = pi - t``, each covering ``dtheta = dt``,
#: and ``sin(t) + sin(pi - t) = 2 sin(t)`` — so the folded measure is
#: ``sin(tilt)``, the same functional form as ``sin(theta)`` restricted to
#: the acute domain. The factor of 2 is an additive constant in
#: ``-kT ln(...)`` and drops out when the PMF is shifted so its populated
#: minimum is zero.
_ANGLE_CONVENTIONS: dict[str, tuple[tuple[float, float], "np.ufunc"]] = {
    "theta": ((0.0, np.pi), np.sin),
    "tilt": ((0.0, np.pi / 2), np.sin),
}


def compute_pmf_2d(
    theta: np.ndarray,
    psi: np.ndarray,
    theta_bins: int = 30,
    psi_bins: int = 36,
    temperature: float = _DEFAULT_TEMPERATURE_K,
    jacobian_correction: bool = True,
    angle_kind: str = "theta",
) -> dict[str, np.ndarray]:
    """2D PMF F(theta, psi) in kcal/mol from a ``(theta, psi)`` trajectory (radians).

    ``F = -kT ln[P(theta, psi) / J(theta)]``, shifted so the populated minimum
    is zero; empty bins are ``NaN`` rather than an arbitrarily large finite
    number. Set ``jacobian_correction=False`` only to inspect the raw,
    uncorrected histogram-derived PMF — never to report a result.

    ``angle_kind='theta'`` (default) bins the raw ZYZ nutation angle over
    ``[0, pi]`` with the ``sin(theta)`` measure; ``angle_kind='tilt'`` bins
    :func:`rotmd.core.orientation.membrane_tilt_angle` over ``[0, pi/2]``
    with the folded ``sin(tilt)`` measure (see :data:`_ANGLE_CONVENTIONS`) —
    use this when ``theta``'s sign ambiguity would otherwise split one
    physical orientation into two mirrored PMF basins.
    """
    if len(theta) != len(psi):
        raise ValueError("theta and psi must have same length")
    if angle_kind not in _ANGLE_CONVENTIONS:
        raise ValueError(f"angle_kind must be one of {sorted(_ANGLE_CONVENTIONS)}, got {angle_kind!r}")

    theta_range, jacobian_fn = _ANGLE_CONVENTIONS[angle_kind]
    theta_edges = np.linspace(*theta_range, theta_bins + 1)
    psi_edges = np.linspace(0, 2 * np.pi, psi_bins + 1)

    counts, theta_bins_edges, psi_bins_edges = np.histogram2d(
        theta, psi, bins=[theta_edges, psi_edges]
    )

    total_counts = np.sum(counts)
    if total_counts == 0:
        raise ValueError("No data in bins")

    prob = counts / total_counts

    theta_centers = (theta_bins_edges[:-1] + theta_bins_edges[1:]) / 2
    psi_centers = (psi_bins_edges[:-1] + psi_bins_edges[1:]) / 2

    if jacobian_correction:
        jacobian = jacobian_fn(theta_centers)[:, np.newaxis]
        # Avoid division by zero where the Jacobian vanishes (theta=0/pi, or
        # tilt=pi/2); that bin's PMF is meaningless regardless and gets
        # masked to NaN below via the counts check.
        jacobian = np.maximum(jacobian, 1e-10)
    else:
        jacobian = 1.0

    prob_corrected = prob / jacobian

    beta_kt = _K_B * temperature
    prob_corrected = np.maximum(prob_corrected, 1e-30)
    pmf = -beta_kt * np.log(prob_corrected)

    pmf_min = np.nanmin(pmf[counts > 0]) if np.sum(counts > 0) > 0 else 0
    pmf = pmf - pmf_min
    pmf[counts == 0] = np.nan

    return {
        "theta_edges": theta_edges,
        "psi_edges": psi_edges,
        "theta_centers": theta_centers,
        "psi_centers": psi_centers,
        "pmf": pmf,
        "counts": counts,
        "probability": prob,
        "jacobian": jacobian if isinstance(jacobian, np.ndarray) else np.ones_like(prob),
    }


def compute_pmf_1d(
    coordinate: np.ndarray,
    bins: int = 50,
    temperature: float = _DEFAULT_TEMPERATURE_K,
    coordinate_type: str = "theta",
    range_bounds: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """1D PMF F(q) in kcal/mol, applying the SO(3) angle correction for ``coordinate_type in ('theta', 'tilt')``.

    ``coordinate_type`` also picks the default bin range when
    ``range_bounds`` is not given: ``[0, pi]`` radians for ``'theta'``,
    ``[0, pi/2]`` for ``'tilt'`` (see :data:`_ANGLE_CONVENTIONS` for the
    folded-domain derivation), ``[0, 2*pi]`` for ``'psi'``, and the data's
    own min/max otherwise (e.g. ``'energy'``, ``'angular_momentum'``).
    """
    if len(coordinate) == 0:
        raise ValueError("coordinate array is empty")

    if range_bounds is None:
        if coordinate_type in _ANGLE_CONVENTIONS:
            range_bounds = _ANGLE_CONVENTIONS[coordinate_type][0]
        elif coordinate_type == "psi":
            range_bounds = (0, 2 * np.pi)
        else:
            range_bounds = (np.min(coordinate), np.max(coordinate))

    counts, edges = np.histogram(coordinate, bins=bins, range=range_bounds)
    centers = (edges[:-1] + edges[1:]) / 2

    total_counts = np.sum(counts)
    if total_counts == 0:
        raise ValueError("No data in bins")

    prob = counts / total_counts

    if coordinate_type in _ANGLE_CONVENTIONS:
        jacobian = _ANGLE_CONVENTIONS[coordinate_type][1](centers)
        jacobian = np.maximum(jacobian, 1e-10)
    else:
        jacobian = 1.0

    prob_corrected = prob / jacobian

    prob_corrected = np.maximum(prob_corrected, 1e-30)
    pmf = -_K_B * temperature * np.log(prob_corrected)

    pmf_min = np.nanmin(pmf[counts > 0]) if np.sum(counts > 0) > 0 else 0
    pmf = pmf - pmf_min
    pmf[counts == 0] = np.nan

    return {
        "edges": edges,
        "centers": centers,
        "pmf": pmf,
        "counts": counts,
        "probability": prob,
    }


def compute_pmf_6d_projection(
    theta: np.ndarray,
    psi: np.ndarray,
    omega: np.ndarray,
    theta_bins: int = 15,
    psi_bins: int = 18,
    omega_bins: int = 10,
    temperature: float = _DEFAULT_TEMPERATURE_K,
) -> dict[str, np.ndarray]:
    """PMF projections from the full ``(theta, psi, omega)`` phase space onto cheaper-to-visualize subspaces.

    A full 6D histogram is prohibitively sparse for typical trajectory
    lengths, so this computes the marginals actually useful for interpretation
    instead: ``F(theta, psi)`` (configuration space, Jacobian-corrected),
    ``F(theta, |omega|)`` (correlating orientation with dynamics, Jacobian
    applied only to theta since :math:`|\\omega|` is a derived quantity), and
    the 1D ``F(theta)``/``F(|omega|)``.
    """
    n_frames = len(theta)

    if len(psi) != n_frames or len(omega) != n_frames:
        raise ValueError("All inputs must have same length")

    omega_mag = np.linalg.norm(omega, axis=1)

    pmf_config = compute_pmf_2d(
        theta, psi, theta_bins=theta_bins, psi_bins=psi_bins,
        temperature=temperature, jacobian_correction=True,
    )

    theta_edges = np.linspace(0, np.pi, theta_bins + 1)
    omega_edges = np.linspace(0, np.max(omega_mag) * 1.1, omega_bins + 1)

    counts_theta_omega, _, _ = np.histogram2d(theta, omega_mag, bins=[theta_edges, omega_edges])
    prob_theta_omega = counts_theta_omega / np.sum(counts_theta_omega)

    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    jacobian_theta = np.sin(theta_centers)[:, np.newaxis]
    jacobian_theta = np.maximum(jacobian_theta, 1e-10)

    prob_corrected = prob_theta_omega / jacobian_theta
    prob_corrected = np.maximum(prob_corrected, 1e-30)
    pmf_theta_omega = -_K_B * temperature * np.log(prob_corrected)
    pmf_theta_omega -= np.nanmin(pmf_theta_omega[counts_theta_omega > 0])
    pmf_theta_omega[counts_theta_omega == 0] = np.nan

    pmf_theta = compute_pmf_1d(theta, bins=theta_bins, coordinate_type="theta", temperature=temperature)
    pmf_omega = compute_pmf_1d(
        omega_mag, bins=omega_bins, coordinate_type="angular_momentum", temperature=temperature
    )

    return {
        "pmf_2d_config": pmf_config,
        "pmf_2d_theta_omega": {
            "pmf": pmf_theta_omega,
            "theta_edges": theta_edges,
            "omega_edges": omega_edges,
            "counts": counts_theta_omega,
        },
        "pmf_1d_theta": pmf_theta,
        "pmf_1d_omega": pmf_omega,
    }


def free_energy_difference(
    pmf: np.ndarray,
    region1_mask: np.ndarray,
    region2_mask: np.ndarray,
) -> float:
    """``F(region2) - F(region1)`` in kcal/mol, combining each region's bins via log-sum-exp.

    Averaging PMF *values* over a region would be wrong: what should be
    Boltzmann-averaged is the underlying population, i.e.
    ``F_region = -kT ln[sum_i exp(-F_i/kT)]`` over the region's bins,
    excluding ``NaN`` (unpopulated) bins. Returns ``NaN`` with a warning if
    either region has no valid bins, rather than silently comparing empty sets.
    """
    t = _DEFAULT_TEMPERATURE_K
    kt = _K_B * t

    values1 = pmf[region1_mask & ~np.isnan(pmf)]
    values2 = pmf[region2_mask & ~np.isnan(pmf)]

    if len(values1) == 0 or len(values2) == 0:
        warnings.warn("One or both regions have no valid data")
        return np.nan

    def log_sum_exp(values):
        v_min = np.min(values)
        return v_min - kt * np.log(np.sum(np.exp(-(values - v_min) / kt)))

    f1 = log_sum_exp(values1)
    f2 = log_sum_exp(values2)

    return f2 - f1


if __name__ == "__main__":
    print("PMF Module - Example Usage\n")
    print("(synthetic angles are generated in degrees for readability, then")
    print(" converted to radians before calling into this module)\n")

    print("Example 1: 1D PMF F(theta) with Jacobian correction")
    np.random.seed(42)

    theta_wt = np.radians(np.abs(np.random.randn(5000) * 15))

    pmf_theta = compute_pmf_1d(theta_wt, bins=30, coordinate_type="theta", temperature=310.15)

    print(
        f"theta range: {np.degrees(pmf_theta['centers'][0]):.1f} - "
        f"{np.degrees(pmf_theta['centers'][-1]):.1f} deg"
    )
    print(f"PMF minimum: {np.nanmin(pmf_theta['pmf']):.2f} kcal/mol")
    print(f"PMF maximum: {np.nanmax(pmf_theta['pmf']):.2f} kcal/mol")
    print(f"PMF at theta=0deg: {pmf_theta['pmf'][0]:.2f} kcal/mol")
    idx_45 = int(np.argmin(np.abs(np.degrees(pmf_theta["centers"]) - 45)))
    print(f"PMF at theta=45deg: {pmf_theta['pmf'][idx_45]:.2f} kcal/mol")

    print("\n" + "=" * 60)
    print("Example 2: 2D PMF F(theta, psi) with sin(theta) correction")

    n = 10000
    theta_2d = np.radians(np.abs(np.random.randn(n) * 20))
    psi_2d = np.radians(np.random.rand(n) * 360)

    pmf_2d = compute_pmf_2d(
        theta_2d, psi_2d, theta_bins=15, psi_bins=18, temperature=310.15, jacobian_correction=True
    )

    print(f"PMF shape: {pmf_2d['pmf'].shape}")
    print(f"Number of populated bins: {np.sum(pmf_2d['counts'] > 0)}")
    print(f"PMF range: {np.nanmin(pmf_2d['pmf']):.2f} - {np.nanmax(pmf_2d['pmf']):.2f} kcal/mol")

    pmf_2d_no_jac = compute_pmf_2d(
        theta_2d, psi_2d, theta_bins=15, psi_bins=18, jacobian_correction=False
    )

    diff = np.nanmean(np.abs(pmf_2d["pmf"] - pmf_2d_no_jac["pmf"]))
    print(f"Mean PMF difference (with vs without Jacobian): {diff:.2f} kcal/mol")

    print("\n" + "=" * 60)
    print("Example 3: Free energy difference between regions")

    region_perpendicular = pmf_2d["pmf"] < 2.0
    region_tilted = pmf_2d["pmf"] > 4.0

    d_f = free_energy_difference(pmf_2d["pmf"], region_perpendicular, region_tilted)
    print(f"Delta F (tilted - perpendicular): {d_f:.2f} kcal/mol")

    print("\n" + "=" * 60)
    print("Example 4: 6D projection")

    theta_6d = np.radians(np.abs(np.random.randn(5000) * 20))
    psi_6d = np.radians(np.random.rand(5000) * 360)
    omega_6d = np.random.randn(5000, 3) * 0.1

    result_6d = compute_pmf_6d_projection(
        theta_6d, psi_6d, omega_6d, theta_bins=10, psi_bins=12, omega_bins=8, temperature=310.15
    )

    print(f"F(theta, psi) shape: {result_6d['pmf_2d_config']['pmf'].shape}")
    print(f"F(theta, |omega|) shape: {result_6d['pmf_2d_theta_omega']['pmf'].shape}")
    print(f"F(theta) bins: {len(result_6d['pmf_1d_theta']['centers'])}")
    print(f"F(|omega|) bins: {len(result_6d['pmf_1d_omega']['centers'])}")
    print(f"Min F(theta): {np.nanmin(result_6d['pmf_1d_theta']['pmf']):.2f} kcal/mol")
    print(f"Max F(theta): {np.nanmax(result_6d['pmf_1d_theta']['pmf']):.2f} kcal/mol")
