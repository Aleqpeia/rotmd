#!/usr/bin/python
"""
Protein Dynamics Analysis from MD Trajectories with Forces and Velocities

This module analyzes protein dynamics using force and velocity data from
uncompressed MD trajectories. It computes:

1. Angular velocity and acceleration
2. Torques and force decomposition
3. Mechanical work and power
4. Vibrational modes and relaxation timescales
5. Energy dissipation and damping

Theoretical Background
======================

Rigid Body Dynamics
-------------------
For a protein treated as a rigid body:

Angular momentum: L = I·ω
Torque: τ = dL/dt = I·α + ω × (I·ω)

Where:
- I = moment of inertia tensor
- ω = angular velocity vector
- α = angular acceleration vector

Mechanical Work
---------------
Work done by forces and torques:

W_trans = ∫ F · v dt
W_rot = ∫ τ · ω dt

Power:
P = F · v + τ · ω

Energy Dissipation
------------------
Frictional damping from solvent:

F_friction = -γ_trans · v
τ_friction = -γ_rot · ω

Relaxation timescales:
τ_trans = m / γ_trans
τ_rot = I / γ_rot

Requirements
------------
- Uncompressed trajectories with velocities (.trr format in GROMACS)
- Optional: force data (requires rerun with force output or .trr with forces)
- MDAnalysis with velocity/force support

References
----------
- Goldstein, H. (2002). "Classical Mechanics" (3rd ed.)
- Landau & Lifshitz (1976). "Mechanics" (3rd ed.)
- Phillips et al. (2005). "Scalable molecular dynamics with NAMD"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.signal import correlate, find_peaks
from scipy.optimize import curve_fit

# Physical constants (SI units internally, convert for output)
KB = 1.380649e-23  # Boltzmann constant (J/K)
NA = 6.02214076e23  # Avogadro's number
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit to kg
ANGSTROM_TO_M = 1e-10  # Angstrom to meters
PS_TO_S = 1e-12  # Picoseconds to seconds
KCAL_TO_J = 4184.0  # kcal to Joules


class ProteinDynamics:
    """
    Analyze dynamics of a protein from atomic positions, velocities, and forces.

    Computes rigid body and internal dynamics properties from MD trajectory data.

    Attributes
    ----------
    positions : np.ndarray
        Atomic positions (n_atoms, 3) in Angstroms
    velocities : np.ndarray
        Atomic velocities (n_atoms, 3) in Angstrom/ps
    forces : np.ndarray, optional
        Atomic forces (n_atoms, 3) in kcal/(mol·Angstrom)
    masses : np.ndarray
        Atomic masses (n_atoms,) in amu
    temperature : float
        System temperature in Kelvin
    """

    def __init__(self,
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 masses: np.ndarray,
                 forces: Optional[np.ndarray] = None,
                 temperature: float = 310.15):
        """
        Initialize protein dynamics analyzer.

        Parameters
        ----------
        positions : np.ndarray
            Atomic positions (n_atoms, 3) in Angstroms
        velocities : np.ndarray
            Atomic velocities (n_atoms, 3) in Angstrom/ps
        masses : np.ndarray
            Atomic masses (n_atoms,) in amu
        forces : np.ndarray, optional
            Atomic forces (n_atoms, 3) in kcal/(mol·Angstrom)
        temperature : float
            Temperature in Kelvin
        """
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.masses = np.array(masses)
        self.forces = np.array(forces) if forces is not None else None
        self.temperature = temperature
        self.n_atoms = len(positions)

        # Compute center of mass
        self.total_mass = np.sum(self.masses)
        self.com = np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / self.total_mass
        self.com_velocity = np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0) / self.total_mass

        # Compute moment of inertia tensor
        self.I_tensor = self._compute_inertia_tensor()
        self.I_eigenvalues, self.I_eigenvectors = np.linalg.eigh(self.I_tensor)
        self.principal_axis = self.I_eigenvectors[:, np.argmin(self.I_eigenvalues)]  # Long axis

    def _compute_inertia_tensor(self) -> np.ndarray:
        """
        Compute moment of inertia tensor about center of mass.

        Returns
        -------
        I : np.ndarray
            Inertia tensor (3, 3) in amu·Angstrom²
        """
        r = self.positions - self.com
        I = np.zeros((3, 3))

        for i in range(self.n_atoms):
            r_vec = r[i]
            m = self.masses[i]

            # I_ij = m * (r² δ_ij - r_i r_j)
            r_sq = np.dot(r_vec, r_vec)
            I += m * (r_sq * np.eye(3) - np.outer(r_vec, r_vec))

        return I

    def compute_angular_velocity(self) -> np.ndarray:
        """
        Compute angular velocity vector ω from atomic velocities.

        Uses the relation: v_i = v_com + ω × r_i
        Solved via least squares: ω = (∑ m_i [r_i]× [r_i]×)^(-1) (∑ m_i [r_i]× v'_i)

        Where [r]× is the skew-symmetric cross-product matrix.

        Returns
        -------
        omega : np.ndarray
            Angular velocity vector (3,) in rad/ps
        """
        r = self.positions - self.com
        v_rel = self.velocities - self.com_velocity

        # Build system for least squares
        # ω × r = -r × ω, use [r]× matrix
        A = np.zeros((3 * self.n_atoms, 3))
        b = np.zeros(3 * self.n_atoms)

        for i in range(self.n_atoms):
            # Skew-symmetric matrix [r_i]×
            r_cross = np.array([
                [0, -r[i, 2], r[i, 1]],
                [r[i, 2], 0, -r[i, 0]],
                [-r[i, 1], r[i, 0], 0]
            ])

            A[3*i:3*(i+1), :] = r_cross * np.sqrt(self.masses[i])
            b[3*i:3*(i+1)] = v_rel[i] * np.sqrt(self.masses[i])

        # Solve weighted least squares
        omega, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        return omega

    def compute_angular_momentum(self, omega: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute angular momentum L = I · ω

        Parameters
        ----------
        omega : np.ndarray, optional
            Angular velocity. If None, computed from velocities.

        Returns
        -------
        L : np.ndarray
            Angular momentum vector (3,) in amu·Angstrom²/ps
        """
        if omega is None:
            omega = self.compute_angular_velocity()

        return self.I_tensor @ omega

    def compute_torque(self) -> Optional[np.ndarray]:
        """
        Compute total torque τ = ∑ r_i × F_i about center of mass.

        Returns
        -------
        tau : np.ndarray or None
            Torque vector (3,) in kcal·Angstrom/mol, or None if forces unavailable
        """
        if self.forces is None:
            return None

        r = self.positions - self.com
        tau = np.sum(np.cross(r, self.forces), axis=0)

        return tau

    def compute_kinetic_energy(self) -> Dict[str, float]:
        """
        Compute translational and rotational kinetic energy.

        Returns
        -------
        energies : dict
            Dictionary with:
            - 'translational': (1/2) M v_com² (kcal/mol)
            - 'rotational': (1/2) ω · I · ω (kcal/mol)
            - 'total': sum of above
        """
        # Translational kinetic energy
        # KE_trans = (1/2) M v²
        # Units: amu * (Angstrom/ps)² -> kcal/mol
        v_com_magnitude = np.linalg.norm(self.com_velocity)
        KE_trans_au = 0.5 * self.total_mass * v_com_magnitude**2

        # Convert: amu·Å²/ps² to kcal/mol
        # 1 amu·Å²/ps² = 1.03642688e-4 kcal/mol
        conversion = 1.03642688e-4
        KE_trans = KE_trans_au * conversion

        # Rotational kinetic energy
        # KE_rot = (1/2) ω · I · ω
        omega = self.compute_angular_velocity()
        L = self.I_tensor @ omega
        KE_rot_au = 0.5 * np.dot(omega, L)
        KE_rot = KE_rot_au * conversion

        return {
            'translational': float(KE_trans),
            'rotational': float(KE_rot),
            'total': float(KE_trans + KE_rot),
            'translational_expected': 1.5 * 0.001987 * self.temperature,  # (3/2)kT
            'rotational_expected': 1.5 * 0.001987 * self.temperature,  # (3/2)kT for 3D
        }

    def compute_mechanical_power(self) -> Optional[Dict[str, float]]:
        """
        Compute instantaneous mechanical power P = F·v + τ·ω

        Returns
        -------
        power : dict or None
            Dictionary with:
            - 'translational': F_com · v_com (kcal/(mol·ps))
            - 'rotational': τ · ω (kcal/(mol·ps))
            - 'total': sum of above
            Returns None if forces unavailable
        """
        if self.forces is None:
            return None

        # Total force
        F_total = np.sum(self.forces, axis=0)

        # Translational power: P_trans = F · v
        P_trans = np.dot(F_total, self.com_velocity)

        # Rotational power: P_rot = τ · ω
        tau = self.compute_torque()
        omega = self.compute_angular_velocity()
        P_rot = np.dot(tau, omega)

        return {
            'translational': float(P_trans),
            'rotational': float(P_rot),
            'total': float(P_trans + P_rot),
        }

    def decompose_forces(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Decompose forces into components along principal axes.

        Returns
        -------
        components : dict or None
            Dictionary with:
            - 'total_force': Total force vector (3,)
            - 'principal_components': Forces along principal axes (3,)
            - 'principal_axes': The principal axes themselves (3, 3)
            Returns None if forces unavailable
        """
        if self.forces is None:
            return None

        F_total = np.sum(self.forces, axis=0)

        # Project onto principal axes
        principal_components = self.I_eigenvectors.T @ F_total

        return {
            'total_force': F_total,
            'principal_components': principal_components,
            'principal_axes': self.I_eigenvectors,
        }

    def to_dict(self) -> Dict:
        """Export dynamics properties to dictionary."""
        omega = self.compute_angular_velocity()
        L = self.compute_angular_momentum(omega)
        KE = self.compute_kinetic_energy()

        result = {
            'com': self.com.tolist(),
            'com_velocity': self.com_velocity.tolist(),
            'angular_velocity': omega.tolist(),
            'angular_velocity_magnitude': float(np.linalg.norm(omega)),
            'angular_momentum': L.tolist(),
            'principal_axis': self.principal_axis.tolist(),
            'kinetic_energy': KE,
        }

        # Add force-dependent quantities if available
        if self.forces is not None:
            tau = self.compute_torque()
            power = self.compute_mechanical_power()
            force_decomp = self.decompose_forces()

            result.update({
                'torque': tau.tolist(),
                'torque_magnitude': float(np.linalg.norm(tau)),
                'mechanical_power': power,
                'force_decomposition': {
                    'total_force': force_decomp['total_force'].tolist(),
                    'principal_components': force_decomp['principal_components'].tolist(),
                },
            })

        return result


class TrajectoryDynamicsAnalyzer:
    """
    Analyze protein dynamics over entire MD trajectory.

    Computes time-dependent properties, autocorrelation functions,
    and identifies characteristic timescales.
    """

    def __init__(self, temperature: float = 310.15):
        """
        Initialize trajectory dynamics analyzer.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        """
        self.temperature = temperature
        self.frames_data = []
        self.times = []

    def analyze_frame(self,
                     positions: np.ndarray,
                     velocities: np.ndarray,
                     masses: np.ndarray,
                     forces: Optional[np.ndarray] = None,
                     frame: int = 0,
                     time_ps: float = 0.0) -> Dict:
        """
        Analyze single trajectory frame.

        Parameters
        ----------
        positions : np.ndarray
            Atomic positions (n_atoms, 3)
        velocities : np.ndarray
            Atomic velocities (n_atoms, 3)
        masses : np.ndarray
            Atomic masses (n_atoms,)
        forces : np.ndarray, optional
            Atomic forces (n_atoms, 3)
        frame : int
            Frame number
        time_ps : float
            Time in picoseconds

        Returns
        -------
        frame_data : dict
            Dynamics properties for this frame
        """
        dynamics = ProteinDynamics(
            positions, velocities, masses, forces, self.temperature
        )

        frame_data = dynamics.to_dict()
        frame_data['frame'] = frame
        frame_data['time_ps'] = time_ps

        self.frames_data.append(frame_data)
        self.times.append(time_ps)

        return frame_data

    def compute_angular_velocity_autocorrelation(self, max_lag: Optional[int] = None) -> Dict:
        """
        Compute angular velocity autocorrelation function C(t) = ⟨ω(0)·ω(t)⟩

        Used to extract rotational relaxation timescale.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag in frames. If None, uses min(len(data)/4, 1000)

        Returns
        -------
        acf_data : dict
            Dictionary with 'lags', 'acf', and 'relaxation_time'
        """
        if len(self.frames_data) < 10:
            warnings.warn("Too few frames for meaningful autocorrelation")
            return {}

        # Extract angular velocity time series
        omega_series = np.array([f['angular_velocity'] for f in self.frames_data])

        # Compute autocorrelation for each component and average
        if max_lag is None:
            max_lag = min(len(omega_series) // 4, 1000)

        acf = np.zeros(max_lag)
        for i in range(3):
            omega_i = omega_series[:, i]
            omega_i -= np.mean(omega_i)  # Remove mean
            acf_i = correlate(omega_i, omega_i, mode='full')
            acf_i = acf_i[len(acf_i)//2:len(acf_i)//2 + max_lag]
            acf_i /= acf_i[0]  # Normalize
            acf += acf_i / 3.0

        # Time lags
        dt = np.mean(np.diff(self.times)) if len(self.times) > 1 else 1.0
        lags = np.arange(max_lag) * dt

        # Fit exponential decay to get relaxation time
        # C(t) = exp(-t/τ)
        try:
            def exp_decay(t, tau):
                return np.exp(-t / tau)

            # Fit to first half where signal is good
            fit_range = max_lag // 2
            popt, _ = curve_fit(exp_decay, lags[:fit_range], acf[:fit_range],
                              p0=[lags[fit_range//2]], bounds=(0, np.inf))
            relaxation_time = popt[0]
        except Exception:
            # Find 1/e point
            idx = np.where(acf < 1.0/np.e)[0]
            relaxation_time = lags[idx[0]] if len(idx) > 0 else lags[-1]

        return {
            'lags_ps': lags.tolist(),
            'acf': acf.tolist(),
            'relaxation_time_ps': float(relaxation_time),
        }

    def compute_power_spectrum(self, quantity: str = 'angular_velocity') -> Dict:
        """
        Compute power spectrum of a dynamical quantity.

        Identifies characteristic frequencies and vibrational modes.

        Parameters
        ----------
        quantity : str
            Quantity to analyze: 'angular_velocity', 'com_velocity', 'torque'

        Returns
        -------
        spectrum : dict
            Dictionary with 'frequencies' (1/ps) and 'power'
        """
        if len(self.frames_data) < 10:
            return {}

        # Extract time series
        if quantity == 'angular_velocity':
            data = np.array([f['angular_velocity'] for f in self.frames_data])
        elif quantity == 'com_velocity':
            data = np.array([f['com_velocity'] for f in self.frames_data])
        elif quantity == 'torque' and 'torque' in self.frames_data[0]:
            data = np.array([f['torque'] for f in self.frames_data])
        else:
            warnings.warn(f"Quantity {quantity} not available")
            return {}

        # Compute power spectrum via FFT
        # Average over 3 components
        dt = np.mean(np.diff(self.times)) if len(self.times) > 1 else 1.0
        n = len(data)

        power = np.zeros(n // 2)
        for i in range(3):
            series = data[:, i] - np.mean(data[:, i])
            fft = np.fft.rfft(series)
            power += np.abs(fft[:n//2])**2 / 3.0

        freqs = np.fft.rfftfreq(n, dt)[:n//2]

        # Find peaks
        peaks, properties = find_peaks(power, prominence=np.max(power)*0.1)

        return {
            'frequencies_inv_ps': freqs.tolist(),
            'power': power.tolist(),
            'peak_frequencies': freqs[peaks].tolist() if len(peaks) > 0 else [],
            'peak_periods_ps': (1.0 / freqs[peaks]).tolist() if len(peaks) > 0 else [],
        }

    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics over entire trajectory.

        Returns
        -------
        summary : dict
            Mean, std for key dynamical properties
        """
        if not self.frames_data:
            return {}

        # Extract time series
        omega_mag = np.array([f['angular_velocity_magnitude'] for f in self.frames_data])
        KE_rot = np.array([f['kinetic_energy']['rotational'] for f in self.frames_data])
        KE_trans = np.array([f['kinetic_energy']['translational'] for f in self.frames_data])

        summary = {
            'n_frames': len(self.frames_data),
            'angular_velocity_magnitude': {
                'mean': float(np.mean(omega_mag)),
                'std': float(np.std(omega_mag)),
                'min': float(np.min(omega_mag)),
                'max': float(np.max(omega_mag)),
            },
            'rotational_kinetic_energy': {
                'mean': float(np.mean(KE_rot)),
                'std': float(np.std(KE_rot)),
                'expected': 1.5 * 0.001987 * self.temperature,
            },
            'translational_kinetic_energy': {
                'mean': float(np.mean(KE_trans)),
                'std': float(np.std(KE_trans)),
                'expected': 1.5 * 0.001987 * self.temperature,
            },
        }

        # Add force-dependent statistics if available
        if 'torque_magnitude' in self.frames_data[0]:
            tau_mag = np.array([f['torque_magnitude'] for f in self.frames_data])
            power_tot = np.array([f['mechanical_power']['total'] for f in self.frames_data])

            summary.update({
                'torque_magnitude': {
                    'mean': float(np.mean(tau_mag)),
                    'std': float(np.std(tau_mag)),
                    'min': float(np.min(tau_mag)),
                    'max': float(np.max(tau_mag)),
                },
                'mechanical_power': {
                    'mean': float(np.mean(power_tot)),
                    'std': float(np.std(power_tot)),
                },
            })

        return summary
