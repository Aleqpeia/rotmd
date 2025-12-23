#!/usr/bin/python
"""
Membrane Interface Utilities

This module provides utilities for defining and working with membrane planes
in MD simulations for protein orientation analysis.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.groups import AtomGroup

from . import leaflet_util


def get_membrane_center_z(
    universe: mda.Universe,
    membrane_sel: str,
    method: str = 'leaflet',
    frame_idx: int = 0
) -> float:
    """
    Get membrane center z-coordinate.

    Args:
        universe: MDAnalysis Universe object
        membrane_sel: Selection string for membrane atoms
        method: Method for determining center ('leaflet', 'density', or 'simple')
        frame_idx: Frame index to use for calculation

    Returns:
        Z-coordinate of membrane center in Angstroms
    """
    universe.trajectory[frame_idx]
    membrane_atoms = universe.select_atoms(membrane_sel)

    if len(membrane_atoms) == 0:
        raise ValueError(f"Membrane selection '{membrane_sel}' matched no atoms")

    if method == 'leaflet':
        # Use leaflet assignment to find center between leaflets
        try:
            leaflets = leaflet_util.assign_leaflet(universe, membrane_atoms)
            if len(leaflets) >= 2 and len(leaflets[0]) > 0 and len(leaflets[1]) > 0:
                upper_z = leaflets[0].center_of_mass()[2]
                lower_z = leaflets[1].center_of_mass()[2]
                center_z = 0.5 * (upper_z + lower_z)
            else:
                # Fallback to simple method
                center_z = membrane_atoms.center_of_mass()[2]
        except (SystemExit, Exception) as e:
            # Leaflet assignment failed, use z-position based method
            print(f"Warning: Leaflet assignment failed ({e}), using z-position based method")
            z_positions = membrane_atoms.positions[:, 2]
            center_z = float(np.median(z_positions))

    elif method == 'density':
        # Use density profile to find center
        z_positions = membrane_atoms.positions[:, 2]
        # Use median to be robust to outliers
        center_z = float(np.median(z_positions))

    elif method == 'simple':
        # Simple center of mass
        center_z = membrane_atoms.center_of_mass()[2]

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'leaflet', 'density', or 'simple'")

    return float(center_z)


def get_membrane_normal(
    universe: mda.Universe,
    membrane_sel: str,
    method: str = 'z_axis'
) -> np.ndarray:
    """
    Get membrane normal vector.

    Args:
        universe: MDAnalysis Universe object
        membrane_sel: Selection string for membrane atoms
        method: Method for determining normal ('z_axis' or 'fit')

    Returns:
        Unit normal vector (3D numpy array)
    """
    if method == 'z_axis':
        # Assume membrane is aligned with z-axis (standard for most MD setups)
        return np.array([0.0, 0.0, 1.0])

    elif method == 'fit':
        # Fit plane to membrane atoms and extract normal
        membrane_atoms = universe.select_atoms(membrane_sel)
        positions = membrane_atoms.positions

        # Fit plane using PCA
        centered = positions - positions.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]  # Third principal component is normal to plane

        # Ensure normal points in positive z direction
        if normal[2] < 0:
            normal = -normal

        return normal / np.linalg.norm(normal)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'z_axis' or 'fit'")


def get_membrane_profile(
    universe: mda.Universe,
    membrane_sel: str,
    bin_size: float = 1.0,
    n_frames: int = 10
) -> Dict[str, np.ndarray]:
    """
    Get membrane density profile along z-axis.

    Args:
        universe: MDAnalysis Universe object
        membrane_sel: Selection string for membrane atoms
        bin_size: Bin size for histogram in Angstroms
        n_frames: Number of frames to sample for profile

    Returns:
        Dictionary containing:
            - 'z_bins': Z-coordinate bin centers
            - 'density': Number density at each bin
            - 'z_min': Minimum z coordinate
            - 'z_max': Maximum z coordinate
    """
    membrane_atoms = universe.select_atoms(membrane_sel)

    # Sample frames uniformly
    n_total_frames = len(universe.trajectory)
    frame_indices = np.linspace(0, n_total_frames - 1, min(n_frames, n_total_frames), dtype=int)

    all_z_positions = []
    for frame_idx in frame_indices:
        universe.trajectory[frame_idx]
        all_z_positions.extend(membrane_atoms.positions[:, 2])

    all_z_positions = np.array(all_z_positions)

    # Create histogram
    z_min = all_z_positions.min()
    z_max = all_z_positions.max()
    n_bins = int((z_max - z_min) / bin_size) + 1

    density, bin_edges = np.histogram(all_z_positions, bins=n_bins, range=(z_min, z_max))
    z_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        'z_bins': z_bins,
        'density': density.astype(float),
        'z_min': float(z_min),
        'z_max': float(z_max),
        'bin_size': bin_size
    }


def get_membrane_thickness(
    universe: mda.Universe,
    membrane_sel: str,
    method: str = 'leaflet',
    frame_idx: int = 0
) -> float:
    """
    Estimate membrane thickness.

    Args:
        universe: MDAnalysis Universe object
        membrane_sel: Selection string for membrane atoms
        method: Method for determining thickness ('leaflet' or 'density')
        frame_idx: Frame index to use for calculation

    Returns:
        Membrane thickness in Angstroms
    """
    universe.trajectory[frame_idx]
    membrane_atoms = universe.select_atoms(membrane_sel)

    if method == 'leaflet':
        # Distance between leaflet centers of mass
        try:
            leaflets = leaflet_util.assign_leaflet(universe, membrane_atoms)
            if len(leaflets) >= 2 and len(leaflets[0]) > 0 and len(leaflets[1]) > 0:
                upper_z = leaflets[0].center_of_mass()[2]
                lower_z = leaflets[1].center_of_mass()[2]
                thickness = abs(upper_z - lower_z)
            else:
                # Fallback to z-range method
                z_positions = membrane_atoms.positions[:, 2]
                thickness = z_positions.max() - z_positions.min()
        except (SystemExit, Exception) as e:
            # Leaflet assignment failed, use z-range
            print(f"Warning: Leaflet assignment failed for thickness ({e}), using z-range method")
            z_positions = membrane_atoms.positions[:, 2]
            thickness = z_positions.max() - z_positions.min()

    elif method == 'density':
        # Use density profile peak-to-peak distance
        profile = get_membrane_profile(universe, membrane_sel, bin_size=0.5, n_frames=1)
        density = profile['density']
        z_bins = profile['z_bins']

        # Find peaks (local maxima in density)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density, height=density.mean())

        if len(peaks) >= 2:
            # Distance between outermost peaks
            thickness = z_bins[peaks[-1]] - z_bins[peaks[0]]
        else:
            # Fallback to full width at half maximum
            half_max = density.max() / 2
            above_half = z_bins[density > half_max]
            if len(above_half) > 0:
                thickness = above_half.max() - above_half.min()
            else:
                thickness = 0.0

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'leaflet' or 'density'")

    return float(thickness)


class MembraneReference:
    """
    Container for membrane reference frame information.

    This class stores all geometric properties of the membrane needed
    for protein orientation analysis.
    """

    def __init__(
        self,
        center_z: float,
        normal: np.ndarray,
        thickness: float,
        box_dimensions: Optional[np.ndarray] = None
    ):
        """
        Initialize membrane reference frame.

        Args:
            center_z: Z-coordinate of membrane center
            normal: Membrane normal unit vector
            thickness: Membrane thickness in Angstroms
            box_dimensions: Simulation box dimensions [x, y, z, alpha, beta, gamma]
        """
        self.center_z = center_z
        self.normal = normal / np.linalg.norm(normal)  # Ensure normalized
        self.thickness = thickness
        self.box_dimensions = box_dimensions

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'center_z': float(self.center_z),
            'normal': self.normal.tolist(),
            'thickness': float(self.thickness)
        }

        if self.box_dimensions is not None:
            result['box_dimensions'] = {
                'x': float(self.box_dimensions[0]),
                'y': float(self.box_dimensions[1]),
                'z': float(self.box_dimensions[2]),
                'alpha': float(self.box_dimensions[3]),
                'beta': float(self.box_dimensions[4]),
                'gamma': float(self.box_dimensions[5])
            }

        return result

    @classmethod
    def from_universe(
        cls,
        universe: mda.Universe,
        membrane_sel: str,
        center_method: str = 'density',
        normal_method: str = 'z_axis',
        thickness_method: str = 'density',
        frame_idx: int = 0
    ) -> 'MembraneReference':
        """
        Create MembraneReference from Universe.

        Args:
            universe: MDAnalysis Universe
            membrane_sel: Selection string for membrane
            center_method: Method for center calculation
            normal_method: Method for normal calculation
            thickness_method: Method for thickness calculation
            frame_idx: Frame to use for calculations

        Returns:
            MembraneReference object
        """
        universe.trajectory[frame_idx]

        center_z = get_membrane_center_z(universe, membrane_sel, center_method, frame_idx)
        normal = get_membrane_normal(universe, membrane_sel, normal_method)
        thickness = get_membrane_thickness(universe, membrane_sel, thickness_method, frame_idx)

        box_dimensions = universe.dimensions if universe.dimensions is not None else None

        return cls(center_z, normal, thickness, box_dimensions)
