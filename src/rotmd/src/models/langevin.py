"""
Langevin Dynamics for Protein Orientation

This module provides Langevin integrators for simulating and validating
protein rotational dynamics models.

Key Features:
- Overdamped Langevin dynamics on SO(3)
- Anisotropic friction tensor support
- PMF-driven dynamics
- Validation against MD trajectories
- Generating synthetic trajectories

Theoretical Background:
- Overdamped Langevin: γ dθ/dt = -dV/dθ + √(2γkT) η(t)
- η(t): white noise with <η(t)η(t')> = δ(t-t')
- On SO(3): special care for metric and drift terms

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict


class LangevinIntegrator:
    """
    Overdamped Langevin integrator for protein orientation dynamics.

    Integrates: γ ω = -∇V(θ,ψ,φ) + √(2γkT) η(t)
    where ω is angular velocity, V is potential (PMF)
    """

    def __init__(self,
                 potential: Callable[[np.ndarray], float],
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 friction: float = 1.0,
                 temperature: float = 300.0,
                 timestep: float = 0.001):
        """
        Initialize Langevin integrator.

        Args:
            potential: V(euler) - potential energy function (kcal/mol)
            gradient: ∇V(euler) - gradient function (kcal/mol/rad)
                     If None, uses numerical differentiation
            friction: Friction coefficient γ (amu/ps)
            temperature: Temperature in Kelvin
            timestep: Integration timestep (ps)
        """
        self.potential = potential
        self.gradient = gradient if gradient is not None else self._numerical_gradient
        self.friction = friction
        self.temperature = temperature
        self.timestep = timestep

        # Boltzmann constant
        self.kB = 0.001987204  # kcal/(mol·K)
        self.kT = self.kB * temperature

        # Noise amplitude: σ = √(2γkT·dt)
        self.noise_amplitude = np.sqrt(2 * friction * self.kT * timestep)

    def _numerical_gradient(self, euler: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Compute gradient using finite differences."""
        grad = np.zeros(3)
        V0 = self.potential(euler)

        for i in range(3):
            euler_plus = euler.copy()
            euler_plus[i] += h
            grad[i] = (self.potential(euler_plus) - V0) / h

        return grad

    def step(self, euler: np.ndarray) -> np.ndarray:
        """
        Perform one Langevin integration step.

        Args:
            euler: (3,) current Euler angles [phi, theta, psi]

        Returns:
            euler_new: (3,) updated Euler angles

        Notes:
            - Uses Euler-Maruyama scheme
            - Includes metric tensor corrections for SO(3)
        """
        phi, theta, psi = euler

        # Compute force: F = -∇V
        force = -self.gradient(euler)

        # Deterministic update: dθ = (F/γ)·dt
        deterministic = (force / self.friction) * self.timestep

        # Stochastic update: dθ = √(2kT/γ·dt) · η
        random = self.noise_amplitude / np.sqrt(self.friction) * np.random.randn(3)

        # Metric tensor corrections for SO(3) (Stratonovich interpretation)
        # Drift term: (kT/γ) ∂_i g^{ij}
        # For ZYZ Euler angles: g_ij = diag(sin²θ, 1, sin²θ)
        if np.abs(np.sin(theta)) > 1e-6:
            drift_theta = (self.kT / self.friction) * 2 * np.cos(theta) / np.sin(theta) * self.timestep
        else:
            drift_theta = 0.0

        drift = np.array([0.0, drift_theta, 0.0])

        # Update
        euler_new = euler + deterministic + random + drift

        # Periodic boundary conditions
        euler_new[0] = euler_new[0] % (2 * np.pi)  # phi ∈ [0, 2π]
        euler_new[2] = euler_new[2] % (2 * np.pi)  # psi ∈ [0, 2π]

        # Reflect theta at boundaries
        if euler_new[1] < 0:
            euler_new[1] = -euler_new[1]
            euler_new[0] = (euler_new[0] + np.pi) % (2 * np.pi)
            euler_new[2] = (euler_new[2] + np.pi) % (2 * np.pi)
        elif euler_new[1] > np.pi:
            euler_new[1] = 2 * np.pi - euler_new[1]
            euler_new[0] = (euler_new[0] + np.pi) % (2 * np.pi)
            euler_new[2] = (euler_new[2] + np.pi) % (2 * np.pi)

        return euler_new

    def simulate(self,
                initial_state: np.ndarray,
                n_steps: int,
                save_interval: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Langevin simulation.

        Args:
            initial_state: (3,) initial Euler angles
            n_steps: Number of integration steps
            save_interval: Save every N steps

        Returns:
            times: (n_frames,) time values in ps
            trajectory: (n_frames, 3) Euler angle trajectory

        Example:
            >>> integrator = LangevinIntegrator(potential, friction=100.0)
            >>> times, traj = integrator.simulate([0, np.pi/2, 0], n_steps=10000)
        """
        n_frames = n_steps // save_interval
        trajectory = np.zeros((n_frames, 3))
        times = np.zeros(n_frames)

        euler = initial_state.copy()
        frame_idx = 0

        for step in range(n_steps):
            euler = self.step(euler)

            if step % save_interval == 0:
                trajectory[frame_idx] = euler
                times[frame_idx] = step * self.timestep
                frame_idx += 1

        return times, trajectory


class AnisotropicLangevin(LangevinIntegrator):
    """
    Langevin integrator with anisotropic friction tensor.

    Extends base class to handle different friction coefficients
    for different Euler angles.
    """

    def __init__(self,
                 potential: Callable[[np.ndarray], float],
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 friction_tensor: np.ndarray = None,
                 temperature: float = 300.0,
                 timestep: float = 0.001):
        """
        Initialize anisotropic Langevin integrator.

        Args:
            potential: V(euler) - potential energy function
            gradient: ∇V(euler) - gradient function
            friction_tensor: (3,) diagonal friction coefficients [γ_φ, γ_θ, γ_ψ]
                           If None, uses isotropic γ = 1.0
            temperature: Temperature in Kelvin
            timestep: Integration timestep (ps)
        """
        if friction_tensor is None:
            friction_tensor = np.ones(3)

        super().__init__(potential, gradient, friction=1.0, temperature=temperature, timestep=timestep)

        self.friction_tensor = friction_tensor

        # Noise amplitudes for each angle
        self.noise_amplitudes = np.sqrt(2 * friction_tensor * self.kT * timestep)

    def step(self, euler: np.ndarray) -> np.ndarray:
        """Perform one anisotropic Langevin step."""
        phi, theta, psi = euler

        # Compute force
        force = -self.gradient(euler)

        # Deterministic update with anisotropic friction
        deterministic = (force / self.friction_tensor) * self.timestep

        # Stochastic update with anisotropic noise
        random = self.noise_amplitudes / np.sqrt(self.friction_tensor) * np.random.randn(3)

        # Metric drift (only for theta)
        if np.abs(np.sin(theta)) > 1e-6:
            drift_theta = (self.kT / self.friction_tensor[1]) * 2 * np.cos(theta) / np.sin(theta) * self.timestep
        else:
            drift_theta = 0.0

        drift = np.array([0.0, drift_theta, 0.0])

        # Update
        euler_new = euler + deterministic + random + drift

        # Periodic boundaries (same as base class)
        euler_new[0] = euler_new[0] % (2 * np.pi)
        euler_new[2] = euler_new[2] % (2 * np.pi)

        if euler_new[1] < 0:
            euler_new[1] = -euler_new[1]
            euler_new[0] = (euler_new[0] + np.pi) % (2 * np.pi)
            euler_new[2] = (euler_new[2] + np.pi) % (2 * np.pi)
        elif euler_new[1] > np.pi:
            euler_new[1] = 2 * np.pi - euler_new[1]
            euler_new[0] = (euler_new[0] + np.pi) % (2 * np.pi)
            euler_new[2] = (euler_new[2] + np.pi) % (2 * np.pi)

        return euler_new


def validate_against_trajectory(md_trajectory: np.ndarray,
                               md_times: np.ndarray,
                               potential: Callable,
                               friction: float,
                               temperature: float = 300.0,
                               n_trials: int = 10,
                               verbose: bool = True) -> Dict:
    """
    Validate Langevin model against MD trajectory.

    Compares statistical properties of MD trajectory with
    Langevin simulations using same potential and friction.

    Args:
        md_trajectory: (n_frames, 3) MD Euler angles
        md_times: (n_frames,) MD timestamps
        potential: PMF from MD trajectory
        friction: Friction coefficient to test
        temperature: Temperature in Kelvin
        n_trials: Number of Langevin trajectories to generate
        verbose: Print comparison

    Returns:
        validation: Dictionary with comparison metrics

    Notes:
        - Compares means, variances, autocorrelation functions
        - Good agreement validates Langevin model
    """
    from ..analysis.correlations import autocorrelation_function

    # MD statistics
    md_mean = np.mean(md_trajectory, axis=0)
    md_std = np.std(md_trajectory, axis=0)

    md_acf = np.zeros((3, len(md_times) // 4))
    for i in range(3):
        md_acf[i] = autocorrelation_function(md_trajectory[:, i], max_lag=len(md_times) // 4)

    # Langevin statistics (average over trials)
    integrator = LangevinIntegrator(potential, friction=friction, temperature=temperature)

    langevin_means = []
    langevin_stds = []
    langevin_acfs = []

    for trial in range(n_trials):
        # Random initial state from MD
        initial_idx = np.random.randint(len(md_trajectory))
        initial_state = md_trajectory[initial_idx]

        # Simulate
        dt = md_times[1] - md_times[0]
        n_steps = len(md_times)
        times, traj = integrator.simulate(initial_state, n_steps, save_interval=1)

        langevin_means.append(np.mean(traj, axis=0))
        langevin_stds.append(np.std(traj, axis=0))

        acf_trial = np.zeros((3, len(times) // 4))
        for i in range(3):
            acf_trial[i] = autocorrelation_function(traj[:, i], max_lag=len(times) // 4)
        langevin_acfs.append(acf_trial)

    langevin_mean = np.mean(langevin_means, axis=0)
    langevin_std = np.mean(langevin_stds, axis=0)
    langevin_acf = np.mean(langevin_acfs, axis=0)

    # Compute discrepancies
    mean_error = np.linalg.norm(md_mean - langevin_mean)
    std_error = np.linalg.norm(md_std - langevin_std)
    acf_error = np.mean([np.linalg.norm(md_acf[i] - langevin_acf[i]) for i in range(3)])

    validation = {
        'md_mean': md_mean,
        'langevin_mean': langevin_mean,
        'mean_error': mean_error,
        'md_std': md_std,
        'langevin_std': langevin_std,
        'std_error': std_error,
        'acf_error': acf_error
    }

    if verbose:
        print("Langevin Model Validation")
        print("=" * 50)
        print(f"Mean error: {mean_error:.4f} rad")
        print(f"Std error: {std_error:.4f} rad")
        print(f"ACF error: {acf_error:.4f}")

        if mean_error < 0.1 and std_error < 0.1:
            print("  → Good agreement with MD")
        else:
            print("  → Poor agreement - check PMF and friction")

    return validation


if __name__ == '__main__':
    # Example usage
    print("Langevin Dynamics Module")
    print("========================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.models.langevin import LangevinIntegrator")
    print()
    print("# Define potential (e.g., from PMF)")
    print("def V(euler):")
    print("    phi, theta, psi = euler")
    print("    return 2.0 * (1 - np.cos(2*theta))  # Simple double-well")
    print()
    print("# Create integrator")
    print("integrator = LangevinIntegrator(V, friction=100.0, temperature=300.0)")
    print()
    print("# Run simulation")
    print("times, traj = integrator.simulate([0, np.pi/2, 0], n_steps=100000)")
    print()
    print("# Analyze trajectory")
    print("theta = traj[:, 1]")
    print("print(f'Mean θ: {np.mean(theta):.2f} rad')")
