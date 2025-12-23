"""
Structural Parameters

This module provides utilities for computing structural descriptors of proteins:
RMSD, radius of gyration, end-to-end distance, etc.

Key Features:
- RMSD calculation with optimal alignment
- Radius of gyration (total and per-axis)
- End-to-end distance
- Asphericity and shape parameters
- Solvent accessible surface area (if available)

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict


def compute_rmsd(
    positions1: np.ndarray,
    positions2: np.ndarray,
    masses: Optional[np.ndarray] = None,
    align: bool = True
) -> float:
    """
    Compute RMSD between two structures.

    Args:
        positions1: (n_atoms, 3) first structure in Å
        positions2: (n_atoms, 3) second structure in Å
        masses: Optional (n_atoms,) atomic masses for weighted RMSD
        align: If True, optimally align structures before computing RMSD

    Returns:
        rmsd: Root mean square deviation in Å

    Notes:
        - If align=True, uses Kabsch algorithm for optimal rotation
        - If masses provided, computes mass-weighted RMSD
        - Both structures should be pre-centered at origin for best results

    Example:
        >>> ref = positions[0]  # First frame as reference
        >>> rmsd = compute_rmsd(positions[100], ref, masses=masses)
        >>> print(f"RMSD from reference: {rmsd:.2f} Å")
    """
    if positions1.shape != positions2.shape:
        raise ValueError("Structures must have same shape")

    if masses is None:
        weights = np.ones(len(positions1))
    else:
        weights = masses

    # Center both structures
    com1 = np.average(positions1, weights=weights, axis=0)
    com2 = np.average(positions2, weights=weights, axis=0)

    pos1_centered = positions1 - com1
    pos2_centered = positions2 - com2

    if align:
        # Kabsch algorithm for optimal rotation
        W = weights[:, np.newaxis]
        H = (pos1_centered * W).T @ pos2_centered

        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T

        # Rotate structure 1 to align with structure 2
        pos1_aligned = pos1_centered @ R.T
    else:
        pos1_aligned = pos1_centered

    # Compute RMSD
    diff = pos1_aligned - pos2_centered
    if masses is not None:
        rmsd = np.sqrt(np.sum(weights * np.sum(diff**2, axis=1)) / np.sum(weights))
    else:
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


def compute_rmsd_trajectory(
    positions: np.ndarray,
    reference: np.ndarray,
    masses: Optional[np.ndarray] = None,
    align: bool = True
) -> np.ndarray:
    """
    Compute RMSD for each frame relative to reference.

    Args:
        positions: (n_frames, n_atoms, 3) trajectory
        reference: (n_atoms, 3) reference structure
        masses: Optional (n_atoms,) atomic masses
        align: Optimally align each frame

    Returns:
        rmsd: (n_frames,) RMSD values in Å

    Example:
        >>> ref = positions[0]
        >>> rmsd_traj = compute_rmsd_trajectory(positions, ref, masses)
        >>> print(f"Mean RMSD: {np.mean(rmsd_traj):.2f} Å")
    """
    n_frames = len(positions)
    rmsd = np.zeros(n_frames)

    for i in range(n_frames):
        rmsd[i] = compute_rmsd(positions[i], reference, masses, align)

    return rmsd


def radius_of_gyration(
    positions: np.ndarray,
    masses: np.ndarray,
    center_of_mass: Optional[np.ndarray] = None
) -> float:
    """
    Compute radius of gyration.

    R_g = sqrt(Σ m_i |r_i - r_com|² / Σ m_i)

    Args:
        positions: (n_atoms, 3) atomic positions in Å
        masses: (n_atoms,) atomic masses in amu
        center_of_mass: Optional pre-computed COM

    Returns:
        Rg: Radius of gyration in Å

    Notes:
        - Measures spatial extent of protein
        - Related to moment of inertia: I = M * R_g²
        - Sensitive to unfolding/compaction

    Example:
        >>> Rg = radius_of_gyration(positions[0], masses)
        >>> print(f"Radius of gyration: {Rg:.2f} Å")
    """
    if center_of_mass is None:
        center_of_mass = np.average(positions, weights=masses, axis=0)

    r = positions - center_of_mass
    r_squared = np.sum(r**2, axis=1)

    Rg = np.sqrt(np.sum(masses * r_squared) / np.sum(masses))

    return Rg


def radius_of_gyration_components(
    positions: np.ndarray,
    masses: np.ndarray,
    center_of_mass: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Compute radius of gyration and its components along x, y, z axes.

    Args:
        positions: (n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses
        center_of_mass: Optional pre-computed COM

    Returns:
        Rg_total: Total radius of gyration
        Rg_components: (3,) components [Rg_x, Rg_y, Rg_z]

    Notes:
        - Rg_total² = Rg_x² + Rg_y² + Rg_z²
        - Useful for detecting anisotropic shapes

    Example:
        >>> Rg, Rg_xyz = radius_of_gyration_components(pos, masses)
        >>> print(f"Rg: {Rg:.2f} Å  (x:{Rg_xyz[0]:.2f}, y:{Rg_xyz[1]:.2f}, z:{Rg_xyz[2]:.2f})")
    """
    if center_of_mass is None:
        center_of_mass = np.average(positions, weights=masses, axis=0)

    r = positions - center_of_mass

    # Total Rg
    Rg_total = np.sqrt(np.sum(masses * np.sum(r**2, axis=1)) / np.sum(masses))

    # Components
    Rg_components = np.zeros(3)
    for i in range(3):
        Rg_components[i] = np.sqrt(np.sum(masses * r[:, i]**2) / np.sum(masses))

    return Rg_total, Rg_components


def asphericity(
    positions: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Compute asphericity parameter.

    Measures deviation from spherical shape.

    Args:
        positions: (n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses

    Returns:
        b: Asphericity (0 = sphere, 1 = rod/disk)

    Formula:
        b = (3/2) * (λ₁² + λ₂² + λ₃²) / (λ₁ + λ₂ + λ₃)² - 1/2

    where λᵢ are principal moments of the gyration tensor.

    Example:
        >>> asph = asphericity(positions, masses)
        >>> if asph < 0.1:
        ...     print("Nearly spherical")
        >>> elif asph > 0.5:
        ...     print("Highly elongated")
    """
    from ..core.inertia import inertia_tensor, principal_axes

    # Gyration tensor G is proportional to inertia tensor
    i = inertia_tensor(positions, masses)
    moments, _ = principal_axes(i)

    # Normalize by total mass for gyration tensor
    M = np.sum(masses)
    lambda_vals = moments / M

    # Asphericity
    lambda_sum = np.sum(lambda_vals)
    lambda_sq_sum = np.sum(lambda_vals**2)

    if lambda_sum > 0:
        b = 1.5 * lambda_sq_sum / lambda_sum**2 - 0.5
    else:
        b = 0.0

    return b


def acylindricity(
    positions: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Compute acylindricity parameter.

    Measures deviation from cylindrical shape.

    Args:
        positions: (n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses

    Returns:
        c: Acylindricity (0 = cylinder, 1 = sphere or disk)

    Formula:
        c = (λ₂ - λ₁) / (λ₁ + λ₂ + λ₃)

    Example:
        >>> c = acylindricity(positions, masses)
        >>> if c < 0.1:
        ...     print("Nearly cylindrical (e.g., α-helix)")
    """
    from ..core.inertia import inertia_tensor, principal_axes

    I = inertia_tensor(positions, masses)
    moments, _ = principal_axes(I)

    M = np.sum(masses)
    lambda_vals = moments / M

    lambda_sum = np.sum(lambda_vals)
    if lambda_sum > 0:
        c = (lambda_vals[1] - lambda_vals[0]) / lambda_sum
    else:
        c = 0.0

    return c


def end_to_end_distance(
    positions: np.ndarray,
    atom_indices: Optional[Tuple[int, int]] = None
) -> float:
    """
    Compute end-to-end distance.

    Args:
        positions: (n_atoms, 3) atomic positions
        atom_indices: Optional (i, j) indices of terminal atoms
                     If None, uses first and last atom

    Returns:
        distance: End-to-end distance in Å

    Example:
        >>> # Distance between first and last Cα atom
        >>> distance = end_to_end_distance(ca_positions)
        >>> print(f"End-to-end: {distance:.2f} Å")
    """
    if atom_indices is None:
        i, j = 0, len(positions) - 1
    else:
        i, j = atom_indices

    distance = np.linalg.norm(positions[j] - positions[i])

    return distance


def compute_structural_parameters(
    positions: np.ndarray,
    masses: np.ndarray,
    reference: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all structural parameters for a single frame.

    Args:
        positions: (n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses
        reference: Optional reference structure for RMSD

    Returns:
        params: Dictionary with:
            - 'rmsd': RMSD from reference (if provided)
            - 'rg': Radius of gyration
            - 'rg_x', 'rg_y', 'rg_z': Rg components
            - 'asphericity': Shape parameter
            - 'acylindricity': Shape parameter
            - 'end_to_end': Terminal distance
            - 'com': (3,) center of mass

    Example:
        >>> params = compute_structural_parameters(positions[0], masses, reference=positions[0])
        >>> print(f"Rg = {params['rg']:.2f} Å")
        >>> print(f"Asphericity = {params['asphericity']:.3f}")
    """
    params = {}

    # Center of mass
    com = np.average(positions, weights=masses, axis=0)
    params['com'] = com

    # RMSD if reference provided
    if reference is not None:
        params['rmsd'] = compute_rmsd(positions, reference, masses, align=True)

    # Radius of gyration
    Rg, Rg_components = radius_of_gyration_components(positions, masses, com)
    params['rg'] = Rg
    params['rg_x'] = Rg_components[0]
    params['rg_y'] = Rg_components[1]
    params['rg_z'] = Rg_components[2]

    # Shape parameters
    params['asphericity'] = asphericity(positions, masses)
    params['acylindricity'] = acylindricity(positions, masses)

    # End-to-end distance
    params['end_to_end'] = end_to_end_distance(positions)

    return params


def compute_structural_trajectory(
    positions: np.ndarray,
    masses: np.ndarray,
    reference: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute structural parameters for entire trajectory.

    Args:
        positions: (n_frames, n_atoms, 3) trajectory
        masses: (n_atoms,) atomic masses
        reference: Optional reference structure (e.g., first frame)
        verbose: Print progress

    Returns:
        results: Dictionary with arrays:
            - 'rmsd': (n_frames,) RMSD values
            - 'rg': (n_frames,) radius of gyration
            - 'rg_components': (n_frames, 3) Rg components
            - 'asphericity': (n_frames,) shape parameter
            - 'acylindricity': (n_frames,) shape parameter
            - 'end_to_end': (n_frames,) terminal distance

    Example:
        >>> results = compute_structural_trajectory(positions, masses, reference=positions[0])
        >>>
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(times, results['rmsd'])
        >>> plt.xlabel('Time (ps)')
        >>> plt.ylabel('RMSD (Å)')
        >>> plt.show()
    """
    n_frames = len(positions)

    if verbose:
        print(f"Computing structural parameters for {n_frames} frames...")

    results = {
        'rg': np.zeros(n_frames),
        'rg_components': np.zeros((n_frames, 3)),
        'asphericity': np.zeros(n_frames),
        'acylindricity': np.zeros(n_frames),
        'end_to_end': np.zeros(n_frames),
    }

    if reference is not None:
        results['rmsd'] = np.zeros(n_frames)

    for i in range(n_frames):
        if verbose and i % max(1, n_frames // 10) == 0:
            print(f"  Frame {i}/{n_frames}")

        params = compute_structural_parameters(positions[i], masses, reference)

        if reference is not None:
            results['rmsd'][i] = params['rmsd']

        results['rg'][i] = params['rg']
        results['rg_components'][i] = [params['rg_x'], params['rg_y'], params['rg_z']]
        results['asphericity'][i] = params['asphericity']
        results['acylindricity'][i] = params['acylindricity']
        results['end_to_end'][i] = params['end_to_end']

    if verbose:
        print(f"  ✓ Complete")
        print(f"\nSummary:")
        if reference is not None:
            print(f"  RMSD:        {np.mean(results['rmsd']):.2f} ± {np.std(results['rmsd']):.2f} Å")
        print(f"  Rg:          {np.mean(results['rg']):.2f} ± {np.std(results['rg']):.2f} Å")
        print(f"  Asphericity: {np.mean(results['asphericity']):.3f} ± {np.std(results['asphericity']):.3f}")
        print(f"  End-to-end:  {np.mean(results['end_to_end']):.2f} ± {np.std(results['end_to_end']):.2f} Å")

    return results


if __name__ == '__main__':
    # Example usage
    print("Structural Parameters Module")
    print("============================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.observables.structural import compute_structural_trajectory")
    print("from protein_orientation.io.gromacs import load_gromacs_trajectory")
    print()
    print("# Load trajectory")
    print("traj = load_gromacs_trajectory('system.gro', 'traj.trr')")
    print()
    print("# Compute structural parameters")
    print("results = compute_structural_trajectory(")
    print("    traj['positions'],")
    print("    traj['masses'],")
    print("    reference=traj['positions'][0],  # First frame as reference")
    print("    verbose=True")
    print(")")
    print()
    print("# Access results")
    print("print(f'Mean RMSD: {np.mean(results[\"rmsd\"]):.2f} Å')")
    print("print(f'Mean Rg: {np.mean(results[\"rg\"]):.2f} Å')")
