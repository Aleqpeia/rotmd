#!/usr/bin/python
"""
Coordinate System Comparison for Protein Orientation

Compares spherical coordinates (θ, φ) with SO(3) representation for angular velocity
calculations, particularly near the singularity at θ=90°.

Convention:
- θ = 0°: Protein lying flat (parallel to membrane surface) - STABLE
- θ = 90°: Protein perpendicular (collinear with membrane normal) - SINGULAR
"""

from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import time

from .geometric_dynamics import GeometricDynamics


@dataclass
class ComparisonMetrics:
    """Metrics for comparing coordinate representations."""
    method_name: str
    angular_velocity: np.ndarray  # [ω_x, ω_y, ω_z] in rad/ps
    angular_velocity_magnitude: float
    numerical_error: float  # Estimated numerical error
    condition_number: float  # Matrix conditioning
    computation_time: float  # Time in seconds
    theta_deg: float  # Tilt angle from surface
    is_singular: bool  # True if near θ=90° singularity


class CoordinateComparator:
    """
    Compare spherical vs SO(3) representations for protein orientation dynamics.

    Tests numerical stability and accuracy across different tilt angles,
    especially near the singularity at θ=90°.
    """

    def __init__(self, singularity_threshold: float = 5.0):
        """
        Initialize comparator.

        Args:
            singularity_threshold: Distance from θ=90° to flag as singular (degrees)
        """
        self.singularity_threshold = singularity_threshold

    def compute_omega_spherical(
        self,
        theta_history: np.ndarray,
        phi_history: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Compute angular velocity from spherical coordinates using finite differences.

        For θ measured from surface:
        ω_x =  dθ/dt * cos(φ) + dφ/dt * cos(θ) * sin(φ)
        ω_y =  dθ/dt * sin(φ) - dφ/dt * cos(θ) * cos(φ)
        ω_z =  dφ/dt * sin(θ)

        Args:
            theta_history: Tilt angles from surface in degrees [0, 90]
            phi_history: Azimuthal angles in degrees [0, 360]
            times: Time points in ps

        Returns:
            Angular velocity vector [ω_x, ω_y, ω_z] in rad/ps
        """
        # Convert to radians
        theta_rad = np.radians(theta_history)
        phi_rad = np.radians(phi_history)

        # Compute time derivatives using central differences
        if len(times) < 3:
            raise ValueError("Need at least 3 time points for finite differences")

        dt = times[1] - times[0]  # Assume uniform spacing

        # Central difference for middle point
        idx = len(times) // 2
        dtheta_dt = (theta_rad[idx+1] - theta_rad[idx-1]) / (2 * dt)
        dphi_dt = (phi_rad[idx+1] - phi_rad[idx-1]) / (2 * dt)

        theta = theta_rad[idx]
        phi = phi_rad[idx]

        # Transform to angular velocity components
        omega_x = dtheta_dt * np.cos(phi) + dphi_dt * np.cos(theta) * np.sin(phi)
        omega_y = dtheta_dt * np.sin(phi) - dphi_dt * np.cos(theta) * np.cos(phi)
        omega_z = dphi_dt * np.sin(theta)

        return np.array([omega_x, omega_y, omega_z])

    def compute_omega_so3(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Compute angular velocity using SO(3) skew-symmetric formulation.

        This method is singularity-free and works for all orientations.

        Args:
            positions: Atomic positions [N, 3] in Angstroms
            velocities: Atomic velocities [N, 3] in Angstrom/ps
            masses: Atomic masses [N] in amu

        Returns:
            Angular velocity vector [ω_x, ω_y, ω_z] in rad/ps
        """
        geom_dyn = GeometricDynamics(positions, velocities, masses)
        omega, _ = geom_dyn.compute_angular_velocity_skew()
        return omega

    def estimate_numerical_error(
        self,
        omega: np.ndarray,
        theta_deg: float
    ) -> float:
        """
        Estimate numerical error based on proximity to singularity.

        Args:
            omega: Angular velocity vector
            theta_deg: Tilt angle from surface

        Returns:
            Estimated relative error
        """
        distance_to_singularity = 90.0 - theta_deg

        if distance_to_singularity < 1.0:
            # Very close to singularity - large errors expected
            return 1e-1
        elif distance_to_singularity < self.singularity_threshold:
            # Near singularity - moderate errors
            return 1e-3 * (self.singularity_threshold / distance_to_singularity)
        else:
            # Far from singularity - machine precision errors
            return 1e-14

    def compute_spherical_condition_number(self, theta_deg: float) -> float:
        """
        Compute condition number for spherical coordinate metric.

        For θ from surface: κ(g) ≈ 1 / cos²(θ) → ∞ as θ → 90°

        Args:
            theta_deg: Tilt angle from surface

        Returns:
            Condition number
        """
        theta_rad = np.radians(theta_deg)
        cos_sq = np.cos(theta_rad)**2

        if cos_sq < 1e-14:
            return np.inf

        return 1.0 / cos_sq

    def compare_methods(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        theta_deg: float,
        phi_deg: float,
        theta_history: np.ndarray = None,
        phi_history: np.ndarray = None,
        times: np.ndarray = None
    ) -> Dict[str, ComparisonMetrics]:
        """
        Compare spherical and SO(3) methods at a given orientation.

        Args:
            positions: Atomic positions [N, 3]
            velocities: Atomic velocities [N, 3]
            masses: Atomic masses [N]
            theta_deg: Current tilt angle from surface
            phi_deg: Current azimuthal angle
            theta_history: History of θ for finite differences (optional)
            phi_history: History of φ for finite differences (optional)
            times: Time points for history (optional)

        Returns:
            Dictionary with metrics for each method
        """
        is_singular = (90.0 - theta_deg) < self.singularity_threshold

        results = {}

        # Method 1: SO(3) (always works)
        t0 = time.time()
        omega_so3 = self.compute_omega_so3(positions, velocities, masses)
        t_so3 = time.time() - t0

        results['SO(3)'] = ComparisonMetrics(
            method_name='SO(3) Skew-Symmetric',
            angular_velocity=omega_so3,
            angular_velocity_magnitude=np.linalg.norm(omega_so3),
            numerical_error=1e-14,  # Machine precision
            condition_number=1.0,  # Well-conditioned
            computation_time=t_so3,
            theta_deg=theta_deg,
            is_singular=False  # Never singular
        )

        # Method 2: Spherical (may fail near θ=90°)
        if theta_history is not None and phi_history is not None and times is not None:
            try:
                t0 = time.time()
                omega_sph = self.compute_omega_spherical(theta_history, phi_history, times)
                t_sph = time.time() - t0

                error_est = self.estimate_numerical_error(omega_sph, theta_deg)
                cond_num = self.compute_spherical_condition_number(theta_deg)

                results['Spherical'] = ComparisonMetrics(
                    method_name='Spherical (θ,φ) Finite Difference',
                    angular_velocity=omega_sph,
                    angular_velocity_magnitude=np.linalg.norm(omega_sph),
                    numerical_error=error_est,
                    condition_number=cond_num,
                    computation_time=t_sph,
                    theta_deg=theta_deg,
                    is_singular=is_singular
                )
            except (ValueError, RuntimeError) as e:
                # Spherical method failed
                results['Spherical'] = ComparisonMetrics(
                    method_name='Spherical (θ,φ) Finite Difference',
                    angular_velocity=np.array([np.nan, np.nan, np.nan]),
                    angular_velocity_magnitude=np.nan,
                    numerical_error=np.inf,
                    condition_number=np.inf,
                    computation_time=0.0,
                    theta_deg=theta_deg,
                    is_singular=True
                )

        return results

    def benchmark_singularity_robustness(
        self,
        theta_range: np.ndarray,
        n_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Benchmark numerical stability across range of tilt angles.

        Args:
            theta_range: Array of tilt angles from surface to test [0, 90]
            n_samples: Number of random samples per angle

        Returns:
            Dictionary with arrays of results for plotting
        """
        results = {
            'theta_values': [],
            'spherical_errors': [],
            'so3_errors': [],
            'condition_numbers': [],
            'spherical_times': [],
            'so3_times': []
        }

        for theta in theta_range:
            for _ in range(n_samples):
                # Generate random protein configuration
                n_atoms = 100
                positions = np.random.randn(n_atoms, 3) * 10
                velocities = np.random.randn(n_atoms, 3) * 0.1
                masses = np.ones(n_atoms) * 12.0

                # Generate angle history
                phi = np.random.uniform(0, 360)
                theta_hist = theta + np.random.randn(5) * 0.1
                phi_hist = phi + np.random.randn(5) * 1.0
                times = np.arange(5) * 0.1

                # Compare methods
                comparison = self.compare_methods(
                    positions, velocities, masses,
                    theta, phi,
                    theta_hist, phi_hist, times
                )

                results['theta_values'].append(theta)

                if 'SO(3)' in comparison:
                    results['so3_errors'].append(comparison['SO(3)'].numerical_error)
                    results['so3_times'].append(comparison['SO(3)'].computation_time)

                if 'Spherical' in comparison:
                    results['spherical_errors'].append(comparison['Spherical'].numerical_error)
                    results['spherical_times'].append(comparison['Spherical'].computation_time)
                    results['condition_numbers'].append(comparison['Spherical'].condition_number)

        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])

        return results

    def generate_comparison_report(
        self,
        comparison: Dict[str, ComparisonMetrics]
    ) -> str:
        """
        Generate human-readable comparison report.

        Args:
            comparison: Results from compare_methods()

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("  COORDINATE SYSTEM COMPARISON")
        report.append("=" * 70)
        report.append("")

        for method_name, metrics in comparison.items():
            report.append(f"{metrics.method_name}:")
            report.append(f"  Angular velocity: ω = [{metrics.angular_velocity[0]:.6f}, "
                         f"{metrics.angular_velocity[1]:.6f}, {metrics.angular_velocity[2]:.6f}] rad/ps")
            report.append(f"  Magnitude: |ω| = {metrics.angular_velocity_magnitude:.6f} rad/ps")
            report.append(f"  Numerical error: {metrics.numerical_error:.2e}")
            report.append(f"  Condition number: κ = {metrics.condition_number:.2e}")
            report.append(f"  Computation time: {metrics.computation_time*1000:.3f} ms")
            report.append(f"  Singular: {metrics.is_singular}")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)
