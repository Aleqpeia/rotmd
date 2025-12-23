#!/usr/bin/env python
"""
Potential of Mean Force (PMF) with Jacobian Corrections

This module computes PMF in the full 6D phase space (θ, ψ, φ, ω_θ, ω_ψ, ω_φ)
with proper Jacobian corrections for volume elements on SO(3).

Mathematical Background:
========================

Phase Space on SO(3):
---------------------
The configuration space is SO(3) with coordinates (θ, ψ, φ).
The full phase space is T*SO(3), the cotangent bundle.

For Euler angles (ZYZ convention):
    R(φ, θ, ψ) = R_z(φ) · R_y(θ) · R_z(ψ)

Volume Element (Jacobian):
--------------------------
The volume element on SO(3) is NOT uniform in Euler angles:

    dμ(R) = sin(θ) dθ dψ dφ

This sin(θ) factor is the Jacobian correction.

For phase space (configuration + momentum):
    dΓ = sin(θ) dθ dψ dφ dω_θ dω_ψ dω_φ

Potential of Mean Force:
------------------------
The PMF F(q) is the free energy as a function of coordinates q:

    F(q) = -k_B T ln[ρ(q) / J(q)] + const

where:
- ρ(q): observed probability density
- J(q): Jacobian (volume element correction)
- k_B T: thermal energy

For Euler angles:
    F(θ, ψ) = -k_B T ln[P(θ, ψ) / sin(θ)] + const

The sin(θ) factor is crucial! Without it, PMF is biased.

6D Phase Space PMF:
-------------------
For full phase space F(θ, ψ, ω):

    F(θ, ψ, ω) = -k_B T ln[P(θ, ψ, ω) / J(θ, ψ, ω)] + const

where J(θ, ψ, ω) = sin(θ) for coordinates (momenta have J=1).

Energy-Dynamics PMF:
--------------------
For F(E, L) landscapes:

    F(E, L) = -k_B T ln[P(E, L)] + const

No Jacobian correction needed (E and L are derived quantities, not coordinates).

Marginalization:
----------------
To get F(θ, ψ) from F(θ, ψ, ω):

    F(θ, ψ) = -k_B T ln[∫ exp(-F(θ, ψ, ω)/k_B T) dω] + const

Physical Interpretation:
------------------------

Well-bound transmembrane protein:
- F(θ=0°, ψ) ≈ 0 (low free energy at perpendicular)
- F(θ=90°, ψ) >> 0 (high barrier to parallel orientation)
- Narrow well in θ

Poorly-bound peripheral protein:
- F(θ, ψ) relatively flat (weak orientation preference)
- Multiple minima possible (multiple binding modes)
- Broad distribution

Common Pitfall:
---------------
❌ WRONG: F = -k_B T ln P(θ, ψ)
✅ CORRECT: F = -k_B T ln[P(θ, ψ) / sin(θ)]

The sin(θ) correction can be several k_B T!

References:
-----------
- Fixman (1978). Proc. Natl. Acad. Sci. USA 75, 4428.
- Amadei et al. (1996). Proteins 17, 412.
- Hnizdo et al. (2007). J. Comput. Chem. 28, 655.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


def jacobian_euler_angles(theta: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian for Euler angles (ZYZ convention).

    Args:
        theta: Tilt angle θ in radians, shape (...,)

    Returns:
        Jacobian J = sin(θ), shape (...,)

    Notes:
        - For θ near 0 or π, J → 0 (gimbal lock)
        - Volume element: dV = sin(θ) dθ dψ dφ

    Example:
        >>> theta = np.array([0, np.pi/4, np.pi/2])  # 0°, 45°, 90°
        >>> J = jacobian_euler_angles(theta)
        >>> print(J)
        [0.         0.70710678 1.        ]
    """
    return np.sin(theta)


def compute_pmf_2d(
    theta: np.ndarray,
    psi: np.ndarray,
    theta_bins: int = 36,
    psi_bins: int = 52,
    temperature: float = 310.15,
    jacobian_correction: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute 2D PMF F(θ, ψ) with Jacobian correction.

    Args:
        theta: Tilt angle trajectory (degrees), shape (N_frames,)
        psi: Spin angle trajectory (degrees), shape (N_frames,)
        theta_bins: Number of bins for θ ∈ [0°, 90°]
        psi_bins: Number of bins for ψ ∈ [0°, 360°]
        temperature: Temperature in K
        jacobian_correction: Apply sin(θ) correction

    Returns:
        Dictionary with:
            - 'theta_edges': Bin edges for θ (degrees)
            - 'psi_edges': Bin edges for ψ (degrees)
            - 'pmf': PMF F(θ, ψ) in kcal/mol, shape (theta_bins, psi_bins)
            - 'counts': Histogram counts
            - 'probability': P(θ, ψ) before correction

    Formula:
        F(θ, ψ) = -k_B T ln[P(θ, ψ) / sin(θ)] + const

    Example:
        >>> theta = np.random.rand(10000) * 90  # degrees
        >>> psi = np.random.rand(10000) * 360
        >>> result = compute_pmf_2d(theta, psi, theta_bins=10, psi_bins=12)
        >>> print(result['pmf'].shape)
        (10, 12)
    """
    if len(theta) != len(psi):
        raise ValueError("theta and psi must have same length")

    print(theta, psi)
    # Create bins
    theta_edges = np.linspace(0, np.pi, theta_bins + 1)
    psi_edges = np.linspace(0, 2 * np.pi, psi_bins + 1)

    # Compute 2D histogram
    counts, theta_bins_edges, psi_bins_edges= np.histogram2d(
        theta, psi,
        bins=[theta_edges, psi_edges]
    )

    # Probability density
    total_counts = np.sum(counts)
    if total_counts == 0:
        raise ValueError("No data in bins")

    prob = counts / total_counts

    # Compute Jacobian at bin centers
    theta_centers = (theta_bins_edges[:-1] + theta_bins_edges[1:]) / 2
    psi_centers = (psi_bins_edges[:-1] + psi_bins_edges[1:]) / 2

    if jacobian_correction:
        # J(θ) = sin(θ)
        jacobian = np.sin(theta_centers)[:, np.newaxis]  # Shape (theta_bins, 1)

        # Avoid division by zero at θ=0
        jacobian = np.maximum(jacobian, 1e-10)
    else:
        jacobian = 1.0

    # Corrected probability: P_corrected = P / J
    prob_corrected = prob / jacobian

    # PMF: F = -k_B T ln(P_corrected)
    k_B = 0.001987  # kcal/(mol·K)
    beta = 1.0 / (k_B * temperature)

    # Avoid log(0)
    prob_corrected = np.maximum(prob_corrected, 1e-28)

    pmf = -k_B * temperature * np.log(prob_corrected)

    # Shift minimum to zero
    pmf_min = np.nanmin(pmf[counts > 0]) if np.sum(counts > 0) > 0 else 0
    pmf = pmf - pmf_min

    # Set empty bins to NaN
    pmf[counts == 0] = np.nan

    return {
        'theta_edges': theta_edges,
        'psi_edges': psi_edges,
        'theta_centers': theta_centers,
        'psi_centers': psi_centers,
        'pmf': pmf,
        'counts': counts,
        'probability': prob,
        'jacobian': jacobian if isinstance(jacobian, np.ndarray) else np.ones_like(prob)
    }


def compute_pmf_1d(
    coordinate: np.ndarray,
    bins: int = 50,
    temperature: float = 310.15,
    coordinate_type: str = 'theta',
    range_bounds: Optional[Tuple[float, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute 1D PMF F(q) with appropriate Jacobian correction.

    Args:
        coordinate: Coordinate trajectory, shape (N_frames,)
        bins: Number of bins
        temperature: Temperature in K
        coordinate_type: Type of coordinate:
            - 'theta': Tilt angle (apply sin(θ) correction)
            - 'psi': Spin angle (uniform, no correction)
            - 'energy': Energy (no correction)
            - 'angular_momentum': L (no correction)
        range_bounds: (min, max) for binning. If None, use data range.

    Returns:
        Dictionary with:
            - 'edges': Bin edges
            - 'centers': Bin centers
            - 'pmf': PMF F(q) in kcal/mol
            - 'counts': Histogram counts

    Example:
        >>> theta = np.random.randn(10000) * 20 + 45  # degrees
        >>> result = compute_pmf_1d(theta, bins=30, coordinate_type='theta')
        >>> print(f"PMF min: {np.nanmin(result['pmf']):.2f} kcal/mol")
        0.00
    """
    if len(coordinate) == 0:
        raise ValueError("coordinate array is empty")

    # Determine range
    if range_bounds is None:
        if coordinate_type == 'theta':
            range_bounds = (0, 90)
        elif coordinate_type == 'psi':
            range_bounds = (0, 360)
        else:
            range_bounds = (np.min(coordinate), np.max(coordinate))

    # Compute histogram
    counts, edges = np.histogram(coordinate, bins=bins, range=range_bounds)
    centers = (edges[:-1] + edges[1:]) / 2

    # Probability
    total_counts = np.sum(counts)
    if total_counts == 0:
        raise ValueError("No data in bins")

    prob = counts / total_counts

    # Jacobian correction
    if coordinate_type == 'theta':
        # Convert to radians for sin(θ)
        centers_rad = np.deg2rad(centers)
        jacobian = np.sin(centers_rad)
        jacobian = np.maximum(jacobian, 1e-10)
    else:
        # No correction for other coordinates
        jacobian = 1.0

    # Corrected probability
    prob_corrected = prob / jacobian

    # PMF
    k_B = 0.001987  # kcal/(mol·K)
    prob_corrected = np.maximum(prob_corrected, 1e-30)
    pmf = -k_B * temperature * np.log(prob_corrected)

    # Shift minimum to zero
    pmf_min = np.nanmin(pmf[counts > 0]) if np.sum(counts > 0) > 0 else 0
    pmf = pmf - pmf_min

    # Set empty bins to NaN
    pmf[counts == 0] = np.nan

    return {
        'edges': edges,
        'centers': centers,
        'pmf': pmf,
        'counts': counts,
        'probability': prob
    }


def compute_pmf_6d_projection(
    theta: np.ndarray,
    psi: np.ndarray,
    omega: np.ndarray,
    theta_bins: int = 15,
    psi_bins: int = 18,
    omega_bins: int = 10,
    temperature: float = 310.15
) -> Dict[str, np.ndarray]:
    """
    Compute 6D phase space PMF and project to various subspaces.

    Args:
        theta: Tilt angle (degrees), shape (N_frames,)
        psi: Spin angle (degrees), shape (N_frames,)
        omega: Angular velocity (rad/ps), shape (N_frames, 3)
        theta_bins: Bins for θ
        psi_bins: Bins for ψ
        omega_bins: Bins for |ω|
        temperature: Temperature in K

    Returns:
        Dictionary with:
            - 'pmf_2d_config': F(θ, ψ) marginalized over ω
            - 'pmf_2d_theta_omega': F(θ, |ω|)
            - 'pmf_1d_theta': F(θ)
            - 'pmf_1d_omega': F(|ω|)

    Notes:
        - Full 6D histogram is expensive for large datasets
        - Use subsampling if N_frames > 100k

    Example:
        >>> theta = np.random.rand(5000) * 90
        >>> psi = np.random.rand(5000) * 360
        >>> omega = np.random.randn(5000, 3) * 0.1
        >>> result = compute_pmf_6d_projection(theta, psi, omega)
        >>> print(result['pmf_2d_config'].shape)
    """
    N_frames = len(theta)

    if len(psi) != N_frames or len(omega) != N_frames:
        raise ValueError("All inputs must have same length")

    # Compute |ω|
    omega_mag = np.linalg.norm(omega, axis=1)

    # 1. Configuration space PMF: F(θ, ψ)
    pmf_config = compute_pmf_2d(
        theta, psi,
        theta_bins=theta_bins,
        psi_bins=psi_bins,
        temperature=temperature,
        jacobian_correction=True
    )

    # 2. Mixed space PMF: F(θ, |ω|)
    # Compute 2D histogram
    theta_edges = np.linspace(0, 90, theta_bins + 1)
    omega_edges = np.linspace(0, np.max(omega_mag) * 1.1, omega_bins + 1)

    counts_theta_omega, _, _ = np.histogram2d(
        theta, omega_mag,
        bins=[theta_edges, omega_edges]
    )

    # Probability
    prob_theta_omega = counts_theta_omega / np.sum(counts_theta_omega)

    # Jacobian correction (only for θ)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    theta_centers_rad = np.deg2rad(theta_centers)
    jacobian_theta = np.sin(theta_centers_rad)[:, np.newaxis]
    jacobian_theta = np.maximum(jacobian_theta, 1e-10)

    prob_corrected = prob_theta_omega / jacobian_theta

    # PMF
    k_B = 0.001987
    prob_corrected = np.maximum(prob_corrected, 1e-30)
    pmf_theta_omega = -k_B * temperature * np.log(prob_corrected)
    pmf_theta_omega -= np.nanmin(pmf_theta_omega[counts_theta_omega > 0])
    pmf_theta_omega[counts_theta_omega == 0] = np.nan

    # 3. 1D PMFs
    pmf_theta = compute_pmf_1d(theta, bins=theta_bins, coordinate_type='theta', temperature=temperature)
    pmf_omega = compute_pmf_1d(omega_mag, bins=omega_bins, coordinate_type='angular_momentum', temperature=temperature)

    return {
        'pmf_2d_config': pmf_config,
        'pmf_2d_theta_omega': {
            'pmf': pmf_theta_omega,
            'theta_edges': theta_edges,
            'omega_edges': omega_edges,
            'counts': counts_theta_omega
        },
        'pmf_1d_theta': pmf_theta,
        'pmf_1d_omega': pmf_omega
    }


def free_energy_difference(
    pmf: np.ndarray,
    region1_mask: np.ndarray,
    region2_mask: np.ndarray
) -> float:
    """
    Compute free energy difference ΔF between two regions.

    Args:
        pmf: PMF array (kcal/mol), any shape
        region1_mask: Boolean mask for region 1
        region2_mask: Boolean mask for region 2

    Returns:
        ΔF = F_region2 - F_region1 in kcal/mol

    Formula:
        F_region = -k_B T ln[Σ_{i∈region} exp(-F_i / k_B T)]

    Example:
        >>> pmf = np.random.rand(10, 10) * 5  # kcal/mol
        >>> region1 = pmf < 1.0  # Low energy region
        >>> region2 = pmf > 3.0  # High energy region
        >>> dF = free_energy_difference(pmf, region1, region2)
        >>> print(f"ΔF = {dF:.2f} kcal/mol")
    """
    k_B = 0.001987  # kcal/(mol·K)
    T = 310.15  # K

    # Extract values in each region (excluding NaN)
    values1 = pmf[region1_mask & ~np.isnan(pmf)]
    values2 = pmf[region2_mask & ~np.isnan(pmf)]

    if len(values1) == 0 or len(values2) == 0:
        warnings.warn("One or both regions have no valid data")
        return np.nan

    # Compute free energy via log-sum-exp trick for numerical stability
    # F = -k_B T ln[Σ exp(-F_i / k_B T)]

    def log_sum_exp(values):
        # Subtract minimum for numerical stability
        v_min = np.min(values)
        return v_min - k_B * T * np.log(np.sum(np.exp(-(values - v_min) / (k_B * T))))

    F1 = log_sum_exp(values1)
    F2 = log_sum_exp(values2)

    return F2 - F1


if __name__ == '__main__':
    print("PMF Module - Example Usage\n")

    # Example 1: 1D PMF for theta
    print("Example 1: 1D PMF F(θ) with Jacobian correction")
    np.random.seed(42)

    # Simulate well-bound protein (θ peaked at 0°)
    theta_wt = np.abs(np.random.randn(5000) * 15)  # degrees, peaked at 0°

    pmf_theta = compute_pmf_1d(
        theta_wt, bins=30,
        coordinate_type='theta',
        temperature=310.15
    )

    print(f"θ range: {pmf_theta['centers'][0]:.1f}° - {pmf_theta['centers'][-1]:.1f}°")
    print(f"PMF minimum: {np.nanmin(pmf_theta['pmf']):.2f} kcal/mol")
    print(f"PMF maximum: {np.nanmax(pmf_theta['pmf']):.2f} kcal/mol")
    print(f"PMF at θ=0°: {pmf_theta['pmf'][0]:.2f} kcal/mol")
    print(f"PMF at θ=45°: {pmf_theta['pmf'][15]:.2f} kcal/mol")

    # Example 2: 2D PMF F(θ, ψ)
    print("\n" + "="*60)
    print("Example 2: 2D PMF F(θ, ψ) with sin(θ) correction")

    # Simulate trajectory
    N = 10000
    theta_2d = np.abs(np.random.randn(N) * 20)  # degrees
    psi_2d = np.random.rand(N) * 360  # degrees (uniform)

    pmf_2d = compute_pmf_2d(
        theta_2d, psi_2d,
        theta_bins=15,
        psi_bins=18,
        temperature=310.15,
        jacobian_correction=True
    )

    print(f"PMF shape: {pmf_2d['pmf'].shape}")
    print(f"Number of populated bins: {np.sum(pmf_2d['counts'] > 0)}")
    print(f"PMF range: {np.nanmin(pmf_2d['pmf']):.2f} - {np.nanmax(pmf_2d['pmf']):.2f} kcal/mol")

    # Comparison with/without Jacobian
    pmf_2d_no_jac = compute_pmf_2d(
        theta_2d, psi_2d,
        theta_bins=15,
        psi_bins=18,
        jacobian_correction=False
    )

    diff = np.nanmean(np.abs(pmf_2d['pmf'] - pmf_2d_no_jac['pmf']))
    print(f"Mean PMF difference (with vs without Jacobian): {diff:.2f} kcal/mol")

    # Example 3: Free energy difference
    print("\n" + "="*60)
    print("Example 3: Free energy difference between regions")

    # Define regions
    region_perpendicular = pmf_2d['pmf'] < 2.0  # Low F region (θ ≈ 0°)
    region_tilted = pmf_2d['pmf'] > 4.0  # High F region (θ large)

    dF = free_energy_difference(pmf_2d['pmf'], region_perpendicular, region_tilted)
    print(f"ΔF (tilted - perpendicular): {dF:.2f} kcal/mol")

    # Example 4: 6D projection
    print("\n" + "="*60)
    print("Example 4: 6D phase space projection")

    theta_6d = np.abs(np.random.randn(5000) * 20)
    psi_6d = np.random.rand(5000) * 360
    omega_6d = np.random.randn(5000, 3) * 0.1  # rad/ps

    result_6d = compute_pmf_6d_projection(
        theta_6d, psi_6d, omega_6d,
        theta_bins=10,
        psi_bins=12,
        omega_bins=8,
        temperature=310.15
    )

    print(f"F(θ, ψ) shape: {result_6d['pmf_2d_config']['pmf'].shape}")
    print(f"F(θ, |ω|) shape: {result_6d['pmf_2d_theta_omega']['pmf'].shape}")
    print(f"F(θ) bins: {len(result_6d['pmf_1d_theta']['centers'])}")
    print(f"F(|ω|) bins: {len(result_6d['pmf_1d_omega']['centers'])}")
    print(f"Min F(θ): {np.nanmin(result_6d['pmf_1d_theta']['pmf']):.2f} kcal/mol")
    print(f"Max F(θ): {np.nanmax(result_6d['pmf_1d_theta']['pmf']):.2f} kcal/mol")
