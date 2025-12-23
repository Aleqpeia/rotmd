"""
Energetic and Kinetic Parameters

This module provides utilities for computing energy-related observables
from MD trajectories with velocities and forces.

Key Features:
- Kinetic energy (total, translational, rotational)
- Potential energy from forces
- Total energy and conservation
- Virial and pressure
- Temperature from kinetic energy

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict


def kinetic_energy_translational(
    velocities: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Compute translational kinetic energy.

    K_trans = (1/2) Σ m_i v_i²

    Args:
        velocities: (n_atoms, 3) velocities in Å/ps
        masses: (n_atoms,) atomic masses in amu

    Returns:
        K_trans: Kinetic energy in kcal/mol

    Notes:
        - Conversion: 1 amu·Å²/ps² = 0.000143933 kcal/mol
    """
    # Kinetic energy in amu·Å²/ps²
    v_squared = np.sum(velocities**2, axis=1)
    K = 0.5 * np.sum(masses * v_squared)

    # Convert to kcal/mol
    K_kcal = K * 0.000143933

    return K_kcal


def kinetic_energy_rotational(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Compute rotational kinetic energy.

    K_rot = (1/2) ω^T · I · ω

    where ω is angular velocity and I is inertia tensor.

    Args:
        positions: (n_atoms, 3) positions in Å
        velocities: (n_atoms, 3) velocities in Å/ps
        masses: (n_atoms,) atomic masses in amu

    Returns:
        K_rot: Rotational kinetic energy in kcal/mol

    Notes:
        - Computed from angular momentum: K_rot = L²/(2I)
        - Removes COM translation first
    """
    from ..core.inertia import inertia_tensor, principal_axes
    from .angular_momentum import compute_angular_momentum

    # Remove COM translation
    com_vel = np.average(velocities, weights=masses, axis=0)
    vel_relative = velocities - com_vel

    # Compute angular momentum
    L = compute_angular_momentum(positions, vel_relative, masses)

    # Compute inertia tensor
    I = inertia_tensor(positions, masses)
    moments, axes = principal_axes(I)

    # Transform L to principal frame
    L_principal = axes.T @ L

    # K_rot = L²/(2I) for each principal axis
    if np.any(moments > 0):
        K_rot = 0.5 * np.sum(L_principal**2 / (moments + 1e-10))
    else:
        K_rot = 0.0

    # Convert to kcal/mol
    K_rot_kcal = K_rot * 0.000143933

    return K_rot_kcal


def kinetic_energy_total(
    velocities: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Compute total kinetic energy (translational + internal).

    Args:
        velocities: (n_atoms, 3) velocities
        masses: (n_atoms,) masses

    Returns:
        K_total: Total kinetic energy in kcal/mol
    """
    return kinetic_energy_translational(velocities, masses)


def instantaneous_temperature(
    velocities: np.ndarray,
    masses: np.ndarray,
    n_constraints: int = 0
) -> float:
    """
    Compute instantaneous temperature from kinetic energy.

    T = 2K / (N_dof · k_B)

    where N_dof = 3N - N_constraints - 6 (translation + rotation removed).

    Args:
        velocities: (n_atoms, 3) velocities in Å/ps
        masses: (n_atoms,) masses in amu
        n_constraints: Number of constraints (bonds, etc.)

    Returns:
        T: Temperature in Kelvin

    Example:
        >>> T = instantaneous_temperature(velocities, masses)
        >>> print(f"Instantaneous T: {T:.1f} K")
    """
    K = kinetic_energy_translational(velocities, masses)  # kcal/mol

    n_atoms = len(masses)
    N_dof = 3 * n_atoms - 6 - n_constraints  # Remove COM translation/rotation

    if N_dof <= 0:
        return 0.0

    # k_B = 0.001987204 kcal/(mol·K)
    # T = 2K / (N_dof · k_B)
    kB = 0.001987204
    T = 2 * K / (N_dof * kB)

    return T


def potential_energy_from_forces(
    positions: np.ndarray,
    forces: np.ndarray,
    reference_positions: np.ndarray
) -> float:
    """
    Estimate potential energy change from forces.

    ΔU ≈ -∫ F · dr ≈ -F · Δr

    Args:
        positions: (n_atoms, 3) current positions
        forces: (n_atoms, 3) forces in kJ/(mol·nm) or kcal/(mol·Å)
        reference_positions: (n_atoms, 3) reference positions

    Returns:
        U: Potential energy change in kcal/mol

    Notes:
        - This is approximate; exact potential requires force field
        - Useful for relative energy differences
    """
    displacement = positions - reference_positions
    work = -np.sum(forces * displacement)

    return work


def virial_tensor(
    positions: np.ndarray,
    forces: np.ndarray
) -> np.ndarray:
    """
    Compute virial tensor.

    W = -Σ r_i ⊗ F_i

    Args:
        positions: (n_atoms, 3) positions in Å
        forces: (n_atoms, 3) forces in kcal/(mol·Å)

    Returns:
        W: (3, 3) virial tensor

    Notes:
        - Related to pressure: P = (NkT - W)/V
        - Diagonal elements give pressure components
    """
    W = -np.einsum('ij,ik->jk', positions, forces)
    return W


def compute_energetics(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: Optional[np.ndarray],
    masses: np.ndarray,
    reference_positions: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all energetic parameters for a single frame.

    Args:
        positions: (n_atoms, 3) positions
        velocities: (n_atoms, 3) velocities
        forces: Optional (n_atoms, 3) forces
        masses: (n_atoms,) masses
        reference_positions: Optional reference for potential energy

    Returns:
        energetics: Dictionary with:
            - 'K_trans': Translational kinetic energy
            - 'K_rot': Rotational kinetic energy
            - 'K_total': Total kinetic energy
            - 'U': Potential energy (if forces and reference provided)
            - 'E_total': Total energy
            - 'T': Instantaneous temperature
            - 'virial': Virial tensor (if forces provided)

    Example:
        >>> energetics = compute_energetics(pos, vel, forces, masses)
        >>> print(f"T = {energetics['T']:.1f} K")
        >>> print(f"K = {energetics['K_total']:.2f} kcal/mol")
    """
    results = {}

    # Kinetic energies
    results['K_trans'] = kinetic_energy_translational(velocities, masses)
    results['K_rot'] = kinetic_energy_rotational(positions, velocities, masses)
    results['K_total'] = results['K_trans']  # Total KE includes all motion

    # Temperature
    results['T'] = instantaneous_temperature(velocities, masses)

    # Potential energy (if forces available)
    if forces is not None and reference_positions is not None:
        results['U'] = potential_energy_from_forces(positions, forces, reference_positions)
        results['E_total'] = results['K_total'] + results['U']

        # Virial
        results['virial'] = virial_tensor(positions, forces)
    else:
        results['U'] = None
        results['E_total'] = None
        results['virial'] = None

    return results


def compute_energetics_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    forces: Optional[np.ndarray] = None,
    reference_positions: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute energetic parameters for entire trajectory.

    Args:
        positions: (n_frames, n_atoms, 3) positions
        velocities: (n_frames, n_atoms, 3) velocities
        masses: (n_atoms,) masses
        forces: Optional (n_frames, n_atoms, 3) forces
        reference_positions: Optional reference structure
        verbose: Print progress

    Returns:
        results: Dictionary with arrays:
            - 'K_trans': (n_frames,) translational KE
            - 'K_rot': (n_frames,) rotational KE
            - 'K_total': (n_frames,) total KE
            - 'T': (n_frames,) temperature
            - 'U': (n_frames,) potential energy (if forces/ref provided)
            - 'E_total': (n_frames,) total energy

    Example:
        >>> results = compute_energetics_trajectory(
        ...     traj['positions'],
        ...     traj['velocities'],
        ...     traj['masses'],
        ...     forces=traj['forces'],
        ...     reference_positions=traj['positions'][0]
        ... )
        >>>
        >>> # Check energy conservation
        >>> E = results['E_total']
        >>> drift = (E[-1] - E[0]) / E[0] * 100
        >>> print(f"Energy drift: {drift:.2f}%")
    """
    n_frames = len(positions)

    if verbose:
        print(f"Computing energetics for {n_frames} frames...")

    results = {
        'K_trans': np.zeros(n_frames),
        'K_rot': np.zeros(n_frames),
        'K_total': np.zeros(n_frames),
        'T': np.zeros(n_frames),
    }

    has_forces = forces is not None and reference_positions is not None
    if has_forces:
        results['U'] = np.zeros(n_frames)
        results['E_total'] = np.zeros(n_frames)

    for i in range(n_frames):
        if verbose and i % max(1, n_frames // 10) == 0:
            print(f"  Frame {i}/{n_frames}")

        frame_forces = forces[i] if forces is not None else None

        energetics = compute_energetics(
            positions[i],
            velocities[i],
            frame_forces,
            masses,
            reference_positions
        )

        results['K_trans'][i] = energetics['K_trans']
        results['K_rot'][i] = energetics['K_rot']
        results['K_total'][i] = energetics['K_total']
        results['T'][i] = energetics['T']

        if has_forces:
            results['U'][i] = energetics['U']
            results['E_total'][i] = energetics['E_total']

    if verbose:
        print(f"  ✓ Complete")
        print(f"\nSummary:")
        print(f"  Temperature:   {np.mean(results['T']):.1f} ± {np.std(results['T']):.1f} K")
        print(f"  K_trans:       {np.mean(results['K_trans']):.2f} ± {np.std(results['K_trans']):.2f} kcal/mol")
        print(f"  K_rot:         {np.mean(results['K_rot']):.2f} ± {np.std(results['K_rot']):.2f} kcal/mol")

        if has_forces:
            E_mean = np.mean(results['E_total'])
            E_std = np.std(results['E_total'])
            drift = (results['E_total'][-1] - results['E_total'][0]) / E_mean * 100
            print(f"  E_total:       {E_mean:.2f} ± {E_std:.2f} kcal/mol")
            print(f"  Energy drift:  {drift:.3f}%")

    return results


if __name__ == '__main__':
    # Example usage
    print("Energetics Module")
    print("=================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.observables.energetics import compute_energetics_trajectory")
    print("from protein_orientation.io.gromacs import load_gromacs_trajectory")
    print()
    print("# Load trajectory with velocities and forces")
    print("traj = load_gromacs_trajectory('system.gro', 'traj.trr')")
    print()
    print("if traj['has_velocities']:")
    print("    results = compute_energetics_trajectory(")
    print("        traj['positions'],")
    print("        traj['velocities'],")
    print("        traj['masses'],")
    print("        forces=traj['forces'] if traj['has_forces'] else None")
    print("    )")
    print()
    print("    print(f'Mean temperature: {np.mean(results[\"T\"]):.1f} K')")
