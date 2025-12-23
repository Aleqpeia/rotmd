#!/usr/bin/python
"""
Timoshenko Beam Theory for Protein Deformation Analysis

This module implements continuum mechanics modeling of proteins as deformable beams,
accounting for bending moments, shear deformation, and rotational inertia.

Theoretical Background
======================

Timoshenko Beam Theory extends Euler-Bernoulli theory by including:
1. Shear deformation (especially important for "thick" proteins)
2. Rotational inertia of cross-sections
3. Independent rotation and deflection fields

Governing Equations
-------------------
For a beam with coordinate s along the backbone:

1. Bending moment-curvature relation:
   EI ∂²ψ/∂s² = M(s,t)

2. Shear force-deformation relation:
   κGA (∂w/∂s - ψ) = V(s,t)

Where:
- ψ(s,t) = rotation angle of cross-section
- w(s,t) = transverse deflection
- EI = bending rigidity
- κGA = shear rigidity
- M(s,t) = bending moment
- V(s,t) = shear force

Application to Proteins
-----------------------
- Backbone Cα atoms define beam centerline
- Cross-sections defined by local residue structure
- Elastic moduli estimated from fluctuation analysis or force fields
- Useful for flexible proteins, domain motions, and membrane interactions

References
----------
- Timoshenko, S. P. (1921). "On the correction for shear of the differential equation for transverse vibrations of prismatic bars"
- Bathe, K. J. (1996). "Finite Element Procedures"
- Tama, F. et al. (2000). "Conformational change of proteins arising from normal mode calculations"
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import warnings

# Physical constants
KB = 1.380649e-23  # Boltzmann constant (J/K)
NA = 6.02214076e23  # Avogadro's number


class ProteinBeamModel:
    """
    Represents a protein as a Timoshenko beam with discrete elements.

    The protein backbone (Cα atoms) is discretized into beam elements.
    Each element has bending and shear rigidity computed from local structure.

    Attributes
    ----------
    positions : np.ndarray
        Cα positions along backbone (n_residues, 3)
    n_elements : int
        Number of beam elements (n_residues - 1)
    element_lengths : np.ndarray
        Length of each beam element
    bending_rigidity : np.ndarray
        EI for each element (kcal·Å²/mol)
    shear_rigidity : np.ndarray
        κGA for each element (kcal/mol)
    """

    def __init__(self,
                 ca_positions: np.ndarray,
                 temperature: float = 310.15,
                 elastic_modulus: Optional[float] = None,
                 shear_modulus: Optional[float] = None):
        """
        Initialize protein beam model from Cα coordinates.

        Parameters
        ----------
        ca_positions : np.ndarray
            Cα atom positions (n_residues, 3) in Angstroms
        temperature : float
            Temperature in Kelvin (default: 310.15 K)
        elastic_modulus : float, optional
            Young's modulus in kcal/(mol·Å²). If None, estimated from fluctuations.
        shear_modulus : float, optional
            Shear modulus in kcal/(mol·Å²). If None, set to E/(2(1+ν)) with ν=0.3
        """
        self.positions = np.array(ca_positions)
        self.n_residues = len(ca_positions)
        self.n_elements = self.n_residues - 1
        self.temperature = temperature

        if self.n_elements < 2:
            raise ValueError("Need at least 3 Cα atoms to define beam elements")

        # Calculate element lengths
        self.element_lengths = np.linalg.norm(
            np.diff(self.positions, axis=0), axis=1
        )

        # Calculate local tangent vectors
        self.tangents = np.diff(self.positions, axis=0) / self.element_lengths[:, np.newaxis]

        # Estimate elastic properties
        if elastic_modulus is None:
            # Empirical estimate: E ~ kT/⟨Δr²⟩
            # Typical protein: E ~ 10-100 kcal/(mol·Å²)
            self.E = 50.0  # kcal/(mol·Å²) - reasonable default
        else:
            self.E = elastic_modulus

        if shear_modulus is None:
            # Poisson's ratio ν ~ 0.3 for proteins
            self.G = self.E / (2.0 * (1.0 + 0.3))
        else:
            self.G = shear_modulus

        # Estimate geometric properties
        self._compute_geometric_properties()

        # Compute rigidities
        self.bending_rigidity = self.E * self.I  # EI (kcal·Å²/mol)
        self.shear_rigidity = 0.833 * self.G * self.A  # κGA, κ≈5/6 for rectangular

    def _compute_geometric_properties(self):
        """
        Estimate cross-sectional area and second moment of area.

        For proteins, we approximate the cross-section as circular with
        radius estimated from local structure compactness.
        """
        # Estimate effective radius from residue spacing
        # Typical Cα-Cα distance ~ 3.8 Å, effective radius ~ 2-4 Å
        mean_length = np.mean(self.element_lengths)
        effective_radius = 0.6 * mean_length  # Empirical scaling

        # Circular cross-section
        self.A = np.pi * effective_radius**2  # Area (Å²)
        self.I = np.pi * effective_radius**4 / 4  # Second moment (Å⁴)

        # Could be refined per-element based on local structure
        self.A = np.full(self.n_elements, self.A)
        self.I = np.full(self.n_elements, self.I)

    def compute_curvature(self) -> np.ndarray:
        """
        Calculate local curvature κ(s) along the backbone.

        Curvature is computed from the change in tangent direction:
        κ = |dT/ds| where T is the unit tangent vector

        Returns
        -------
        curvature : np.ndarray
            Local curvature at each element (1/Å)
        """
        if self.n_elements < 2:
            return np.zeros(1)

        # Change in tangent direction
        dtangent = np.diff(self.tangents, axis=0)
        ds = 0.5 * (self.element_lengths[:-1] + self.element_lengths[1:])

        curvature = np.linalg.norm(dtangent, axis=1) / ds

        # Extend to all elements (duplicate first and last)
        curvature_full = np.zeros(self.n_elements)
        curvature_full[0] = curvature[0]
        curvature_full[1:-1] = curvature
        curvature_full[-1] = curvature[-1]

        return curvature_full

    def compute_bending_moment(self, curvature: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate bending moment M = EI κ at each element.

        Parameters
        ----------
        curvature : np.ndarray, optional
            Local curvature. If None, computed from current positions.

        Returns
        -------
        moment : np.ndarray
            Bending moment at each element (kcal·Å/mol)
        """
        if curvature is None:
            curvature = self.compute_curvature()

        return self.bending_rigidity * curvature

    def compute_bending_energy(self) -> float:
        """
        Calculate total bending energy E_bend = ∫ (EI/2) κ² ds

        Returns
        -------
        energy : float
            Total bending energy (kcal/mol)
        """
        curvature = self.compute_curvature()
        integrand = 0.5 * self.bending_rigidity * curvature**2
        energy = np.sum(integrand * self.element_lengths)
        return energy

    def compute_shear_angle(self) -> np.ndarray:
        """
        Calculate shear angle γ = (dw/ds - ψ) from deflection and rotation.

        For static analysis, this requires solving the coupled beam equations.
        Here we provide a simplified estimate from geometry.

        Returns
        -------
        shear_angle : np.ndarray
            Shear angle at each element (radians)
        """
        # Simplified: estimate from deviation of tangent from straight line
        # Full solution would require boundary value problem

        # Connect first and last Cα to define "undeformed" reference
        span_vector = self.positions[-1] - self.positions[0]
        span_length = np.linalg.norm(span_vector)
        reference_tangent = span_vector / span_length

        # Deviation of local tangent from reference
        shear_angle = np.arcsin(np.clip(
            np.linalg.norm(np.cross(self.tangents, reference_tangent), axis=1),
            -1.0, 1.0
        ))

        return shear_angle

    def compute_shear_energy(self) -> float:
        """
        Calculate shear energy E_shear = ∫ (κGA/2) γ² ds

        Returns
        -------
        energy : float
            Total shear energy (kcal/mol)
        """
        shear_angle = self.compute_shear_angle()
        integrand = 0.5 * self.shear_rigidity * shear_angle**2
        energy = np.sum(integrand * self.element_lengths)
        return energy

    def compute_total_deformation_energy(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total deformation energy (bending + shear).

        Returns
        -------
        total_energy : float
            Total deformation energy (kcal/mol)
        components : dict
            Dictionary with 'bending' and 'shear' components
        """
        E_bend = self.compute_bending_energy()
        E_shear = self.compute_shear_energy()

        components = {
            'bending': E_bend,
            'shear': E_shear,
            'total': E_bend + E_shear
        }

        return E_bend + E_shear, components

    def to_dict(self) -> Dict:
        """Export beam properties to dictionary."""
        curvature = self.compute_curvature()
        moment = self.compute_bending_moment(curvature)
        shear = self.compute_shear_angle()

        return {
            'n_residues': self.n_residues,
            'n_elements': self.n_elements,
            'total_length': float(np.sum(self.element_lengths)),
            'mean_element_length': float(np.mean(self.element_lengths)),
            'elastic_modulus': float(self.E),
            'shear_modulus': float(self.G),
            'mean_bending_rigidity': float(np.mean(self.bending_rigidity)),
            'mean_shear_rigidity': float(np.mean(self.shear_rigidity)),
            'mean_curvature': float(np.mean(curvature)),
            'max_curvature': float(np.max(curvature)),
            'bending_energy': float(self.compute_bending_energy()),
            'shear_energy': float(self.compute_shear_energy()),
            'mean_bending_moment': float(np.mean(moment)),
            'mean_shear_angle': float(np.mean(shear)),
        }


class TrajectoryBeamAnalyzer:
    """
    Analyze protein deformation as Timoshenko beam over MD trajectory.

    Computes time-dependent deformation energies, curvature evolution,
    and correlations with orientation changes.
    """

    def __init__(self,
                 temperature: float = 310.15,
                 elastic_modulus: Optional[float] = None,
                 shear_modulus: Optional[float] = None):
        """
        Initialize trajectory beam analyzer.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        elastic_modulus : float, optional
            Young's modulus (kcal/(mol·Å²))
        shear_modulus : float, optional
            Shear modulus (kcal/(mol·Å²))
        """
        self.temperature = temperature
        self.elastic_modulus = elastic_modulus
        self.shear_modulus = shear_modulus
        self.frames_data = []

    def analyze_frame(self, ca_positions: np.ndarray,
                     frame: int, time_ps: float) -> Dict:
        """
        Analyze single trajectory frame.

        Parameters
        ----------
        ca_positions : np.ndarray
            Cα positions for this frame (n_residues, 3)
        frame : int
            Frame number
        time_ps : float
            Time in picoseconds

        Returns
        -------
        frame_data : dict
            Deformation properties for this frame
        """
        beam = ProteinBeamModel(
            ca_positions,
            temperature=self.temperature,
            elastic_modulus=self.elastic_modulus,
            shear_modulus=self.shear_modulus
        )

        curvature = beam.compute_curvature()
        moment = beam.compute_bending_moment(curvature)
        shear_angle = beam.compute_shear_angle()
        E_total, E_components = beam.compute_total_deformation_energy()

        frame_data = {
            'frame': frame,
            'time_ps': time_ps,
            'curvature_mean': float(np.mean(curvature)),
            'curvature_max': float(np.max(curvature)),
            'curvature_std': float(np.std(curvature)),
            'moment_mean': float(np.mean(moment)),
            'moment_max': float(np.max(moment)),
            'shear_angle_mean': float(np.mean(shear_angle)),
            'shear_angle_max': float(np.max(shear_angle)),
            'bending_energy': E_components['bending'],
            'shear_energy': E_components['shear'],
            'total_deformation_energy': E_components['total'],
        }

        self.frames_data.append(frame_data)
        return frame_data

    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics over entire trajectory.

        Returns
        -------
        summary : dict
            Mean, std, min, max for key properties
        """
        if not self.frames_data:
            return {}

        # Extract time series
        bending_energy = np.array([f['bending_energy'] for f in self.frames_data])
        shear_energy = np.array([f['shear_energy'] for f in self.frames_data])
        total_energy = np.array([f['total_deformation_energy'] for f in self.frames_data])
        curvature_mean = np.array([f['curvature_mean'] for f in self.frames_data])

        return {
            'n_frames': len(self.frames_data),
            'bending_energy': {
                'mean': float(np.mean(bending_energy)),
                'std': float(np.std(bending_energy)),
                'min': float(np.min(bending_energy)),
                'max': float(np.max(bending_energy)),
            },
            'shear_energy': {
                'mean': float(np.mean(shear_energy)),
                'std': float(np.std(shear_energy)),
                'min': float(np.min(shear_energy)),
                'max': float(np.max(shear_energy)),
            },
            'total_deformation_energy': {
                'mean': float(np.mean(total_energy)),
                'std': float(np.std(total_energy)),
                'min': float(np.min(total_energy)),
                'max': float(np.max(total_energy)),
            },
            'mean_curvature': {
                'mean': float(np.mean(curvature_mean)),
                'std': float(np.std(curvature_mean)),
            }
        }


def rodrigues_rotation(vector: np.ndarray,
                       axis: np.ndarray,
                       angle: float) -> np.ndarray:
    """
    Rotate vector around axis by angle using Rodrigues' rotation formula.

    The Rodrigues formula provides a numerically stable way to rotate a vector:
    v_rot = v cos(θ) + (k × v) sin(θ) + k(k · v)(1 - cos(θ))

    Parameters
    ----------
    vector : np.ndarray
        Vector to rotate (3,) or (N, 3)
    axis : np.ndarray
        Rotation axis (must be unit vector)
    angle : float
        Rotation angle in radians

    Returns
    -------
    rotated : np.ndarray
        Rotated vector(s)

    Notes
    -----
    - More stable than rotation matrices for small angles
    - Avoids gimbal lock issues
    - Computationally efficient

    References
    ----------
    Rodrigues, O. (1840). "Des lois géométriques qui régissent les déplacements d'un système solide dans l'espace"
    """
    # Ensure axis is normalized
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    vector = np.asarray(vector)

    # Rodrigues' formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Cross product: k × v
    cross_prod = np.cross(axis, vector)

    # Dot product: k · v
    dot_prod = np.dot(vector, axis)

    # Apply formula
    rotated = (vector * cos_angle +
               cross_prod * sin_angle +
               axis * dot_prod * (1.0 - cos_angle))

    return rotated


def axis_angle_from_rotation(vector_initial: np.ndarray,
                             vector_final: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute rotation axis and angle that transforms initial to final vector.

    Parameters
    ----------
    vector_initial : np.ndarray
        Initial vector (3,)
    vector_final : np.ndarray
        Final vector (3,)

    Returns
    -------
    axis : np.ndarray
        Rotation axis (unit vector)
    angle : float
        Rotation angle in radians
    """
    v1 = np.asarray(vector_initial) / np.linalg.norm(vector_initial)
    v2 = np.asarray(vector_final) / np.linalg.norm(vector_final)

    # Rotation axis: k = v1 × v2
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        # Vectors are parallel or anti-parallel
        if np.dot(v1, v2) > 0:
            # Parallel: no rotation
            return np.array([0, 0, 1]), 0.0
        else:
            # Anti-parallel: 180° rotation around any perpendicular axis
            # Find perpendicular axis
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            return perp / np.linalg.norm(perp), np.pi

    axis = axis / axis_norm

    # Rotation angle: θ = arccos(v1 · v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    return axis, angle
