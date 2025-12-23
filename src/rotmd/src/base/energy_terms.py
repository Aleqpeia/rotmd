#!/usr/bin/python
"""
Energy Terms for 3D Pendulum Model

This module implements energy calculations for protein-membrane interactions:
- Hydrophobic energy (based on SASA and residue hydrophobicity)
- Electrostatic energy (Coulomb interactions with distance-dependent dielectric)
- Per-residue energy contributions for mutation analysis
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.groups import AtomGroup
import contextlib
import os


# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY_SCALE = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
    # Additional residue names
    'HSD': -3.2, 'HSE': -3.2, 'HSP': -3.2,  # Histidine variants
    'GLYM': -0.4,  # N-myristoylated glycine
}

# Formal charges at physiological pH
RESIDUE_CHARGES = {
    'ARG': +1.0, 'LYS': +1.0, 'HIS': +0.5, 'ASP': -1.0, 'GLU': -1.0,
    'HSD': 0.0, 'HSE': 0.0, 'HSP': +1.0,
    # Neutral residues
    'ALA': 0.0, 'ASN': 0.0, 'CYS': 0.0, 'GLN': 0.0, 'GLY': 0.0,
    'ILE': 0.0, 'LEU': 0.0, 'MET': 0.0, 'PHE': 0.0, 'PRO': 0.0,
    'SER': 0.0, 'THR': 0.0, 'TRP': 0.0, 'TYR': 0.0, 'VAL': 0.0,
    'GLYM': 0.0,
}


class HydrophobicEnergy:
    """
    Calculate hydrophobic energy based on residue burial and hydrophobicity.

    The energy favors burial of hydrophobic residues and exposure of hydrophilic
    residues. This is a simplified model based on residue-level SASA.
    """

    def __init__(
        self,
        reference_sasa: Optional[Dict[str, float]] = None,
        scale_factor: float = 0.01
    ):
        """
        Initialize hydrophobic energy calculator.

        Args:
            reference_sasa: Dictionary of reference SASA values per residue type
                           If None, use approximate values
            scale_factor: Energy scale factor (kcal/mol per Å² per hydrophobicity unit)
        """
        self.scale_factor = scale_factor

        # Approximate reference SASA values for fully exposed residues (Å²)
        self.reference_sasa = reference_sasa or {
            'ALA': 113, 'ARG': 241, 'ASN': 158, 'ASP': 151, 'CYS': 140,
            'GLN': 189, 'GLU': 183, 'GLY': 85, 'HIS': 194, 'ILE': 182,
            'LEU': 180, 'LYS': 211, 'MET': 204, 'PHE': 218, 'PRO': 143,
            'SER': 122, 'THR': 146, 'TRP': 259, 'TYR': 229, 'VAL': 160,
        }

    def calculate_residue_burial(
        self,
        residue: mda.core.groups.Residue,
        current_sasa: float
    ) -> float:
        """
        Calculate burial fraction for a residue.

        Args:
            residue: MDAnalysis Residue object
            current_sasa: Current SASA value in Ų

        Returns:
            Burial fraction (0 = fully exposed, 1 = fully buried)
        """
        ref_sasa = self.reference_sasa.get(residue.resname, 150.0)
        burial = 1.0 - min(current_sasa / ref_sasa, 1.0)
        return max(0.0, burial)

    def calculate_residue_energy(
        self,
        residue: mda.core.groups.Residue,
        burial_fraction: float
    ) -> float:
        """
        Calculate hydrophobic energy for a residue.

        Negative energy = favorable (hydrophobic buried or hydrophilic exposed)
        Positive energy = unfavorable (hydrophobic exposed or hydrophilic buried)

        Args:
            residue: MDAnalysis Residue object
            burial_fraction: Fraction buried (0-1)

        Returns:
            Energy in kcal/mol
        """
        hydrophobicity = HYDROPHOBICITY_SCALE.get(residue.resname, 0.0)

        # Energy = hydrophobicity * burial
        # Positive hydrophobicity (hydrophobic) * high burial = negative energy (favorable)
        # Negative hydrophobicity (hydrophilic) * high burial = positive energy (unfavorable)
        energy = -hydrophobicity * burial_fraction * self.scale_factor

        return energy

    def calculate_per_residue_sasa(
        self,
        protein_atoms: AtomGroup
    ) -> Dict[int, float]:
        """
        Calculate SASA for each residue.

        Args:
            protein_atoms: MDAnalysis AtomGroup

        Returns:
            Dictionary mapping resid to SASA value
        """
        try:
            import freesasa
            import tempfile
            import sys

            # Suppress freesasa warnings using both methods:
            # 1. Set freesasa verbosity to silent
            # 2. Redirect stderr at file descriptor level (works for C libraries)
            # Save original stderr file descriptor by duplicating it
            original_stderr_fd = sys.stderr.fileno()
            saved_stderr_fd = os.dup(original_stderr_fd)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            
            try:
                # Set freesasa to silent mode
                freesasa.setVerbosity(freesasa.silent)
                
                # Redirect stderr at file descriptor level (lower level than contextlib)
                os.dup2(devnull_fd, original_stderr_fd)
                
                # Write atoms to temporary PDB file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                    tmp_pdb = tmp.name
                    protein_atoms.write(tmp_pdb)

                try:
                    # Calculate SASA from PDB
                    structure = freesasa.Structure(tmp_pdb)
                    result = freesasa.calc(structure)

                    # Extract per-residue SASA using residueAreas
                    residue_sasa = {}
                    res_areas = result.residueAreas()

                    for residue in protein_atoms.residues:
                        resid = int(residue.resid)
                        resname = residue.resname

                        try:
                            # Access residue area by residue name and ID
                            if resname in res_areas and resid in res_areas[resname]:
                                area = res_areas[resname][resid]
                                residue_sasa[resid] = area.total
                            else:
                                residue_sasa[resid] = 0.0
                        except:
                            residue_sasa[resid] = 0.0

                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_pdb):
                        os.remove(tmp_pdb)
                    
                    # Restore original stderr from saved copy
                    os.dup2(saved_stderr_fd, original_stderr_fd)
                    os.close(saved_stderr_fd)
                    os.close(devnull_fd)

                return residue_sasa
            except Exception:
                # Restore stderr even on error
                try:
                    os.dup2(saved_stderr_fd, original_stderr_fd)
                    os.close(saved_stderr_fd)
                    os.close(devnull_fd)
                except:
                    pass
                raise

        except ImportError:
            print("Warning: freesasa not available, cannot calculate SASA")
            return {}
        except Exception as e:
            print(f"Warning: SASA calculation failed: {e}")
            return {}

    def calculate_total_energy(
        self,
        protein_atoms: AtomGroup,
        residue_sasa: Optional[Dict[int, float]] = None
    ) -> Tuple[float, Dict[int, float]]:
        """
        Calculate total hydrophobic energy and per-residue contributions.

        Args:
            protein_atoms: MDAnalysis AtomGroup
            residue_sasa: Pre-calculated SASA values (if None, will calculate)

        Returns:
            Tuple of (total_energy, per_residue_energy_dict)
        """
        if residue_sasa is None:
            residue_sasa = self.calculate_per_residue_sasa(protein_atoms)

        if not residue_sasa:
            return 0.0, {}

        total_energy = 0.0
        per_residue_energy = {}

        for residue in protein_atoms.residues:
            resid = int(residue.resid)
            if resid in residue_sasa:
                burial = self.calculate_residue_burial(residue, residue_sasa[resid])
                energy = self.calculate_residue_energy(residue, burial)
                per_residue_energy[resid] = energy
                total_energy += energy

        return total_energy, per_residue_energy


class ElectrostaticEnergy:
    """
    Calculate electrostatic energy using Coulomb's law with distance-dependent dielectric.

    This simplified model calculates protein-membrane electrostatic interactions.
    """

    def __init__(
        self,
        membrane_surface_potential: float = -10.0,  # mV
        dielectric_constant: float = 80.0,
        distance_cutoff: float = 15.0,  # Angstroms
        temperature: float = 310.15  # Kelvin
    ):
        """
        Initialize electrostatic energy calculator.

        Args:
            membrane_surface_potential: Membrane surface potential in mV
            dielectric_constant: Relative dielectric constant
            distance_cutoff: Maximum distance for interactions in Angstroms
            temperature: Temperature in Kelvin
        """
        self.membrane_potential = membrane_surface_potential / 1000.0  # Convert to V
        self.epsilon = dielectric_constant
        self.cutoff = distance_cutoff
        self.temperature = temperature

        # Coulomb constant in kcal·Å/(mol·e²)
        self.k_coulomb = 332.0636

        # Debye length for ionic screening (approximate for physiological conditions)
        # λ_D ≈ 10 Å for 0.1 M salt
        self.debye_length = 10.0

    def calculate_residue_charge(self, residue: mda.core.groups.Residue) -> float:
        """
        Get formal charge of a residue.

        Args:
            residue: MDAnalysis Residue object

        Returns:
            Formal charge (in elementary charges)
        """
        return RESIDUE_CHARGES.get(residue.resname, 0.0)

    def calculate_membrane_interaction_energy(
        self,
        protein_atoms: AtomGroup,
        membrane_center_z: float,
        protein_com_z: float
    ) -> Tuple[float, Dict[int, float]]:
        """
        Calculate electrostatic interaction with membrane surface.

        Uses a simplified model where charged residues interact with the
        membrane surface potential based on their distance from the membrane.

        Args:
            protein_atoms: MDAnalysis AtomGroup
            membrane_center_z: Z-coordinate of membrane center
            protein_com_z: Z-coordinate of protein center of mass

        Returns:
            Tuple of (total_energy, per_residue_energy_dict)
        """
        total_energy = 0.0
        per_residue_energy = {}

        # Membrane surface positions (approximate as ±20 Å from center)
        upper_surface_z = membrane_center_z + 20.0
        lower_surface_z = membrane_center_z - 20.0

        # Determine which surface is closer to protein
        if protein_com_z > membrane_center_z:
            surface_z = upper_surface_z
        else:
            surface_z = lower_surface_z

        for residue in protein_atoms.residues:
            charge = self.calculate_residue_charge(residue)

            if abs(charge) < 0.01:  # Skip neutral residues
                continue

            # Distance from residue COM to membrane surface
            res_com = residue.atoms.center_of_mass()
            distance = abs(res_com[2] - surface_z)

            if distance > self.cutoff:
                continue

            # Screened Coulomb potential with distance-dependent dielectric
            # U = k * q * V * exp(-r/λ_D) / (ε * r)
            screening = np.exp(-distance / self.debye_length)
            epsilon_r = self.epsilon * (distance / 10.0)  # Distance-dependent
            epsilon_r = max(epsilon_r, 4.0)  # Minimum dielectric constant

            # Energy from interaction with surface potential
            # Simplified: E = q * ΔV * screening_factor
            energy = charge * self.membrane_potential * screening * 23.06  # Convert to kcal/mol
            energy /= (epsilon_r / 80.0)  # Scale by dielectric

            per_residue_energy[int(residue.resid)] = energy
            total_energy += energy

        return total_energy, per_residue_energy

    def calculate_protein_self_energy(
        self,
        protein_atoms: AtomGroup
    ) -> Tuple[float, Dict[int, float]]:
        """
        Calculate electrostatic self-energy of protein (simplified).

        This is a rough estimate of charge-charge interactions within the protein.

        Args:
            protein_atoms: MDAnalysis AtomGroup

        Returns:
            Tuple of (total_energy, per_residue_contribution_dict)
        """
        charged_residues = []

        for residue in protein_atoms.residues:
            charge = self.calculate_residue_charge(residue)
            if abs(charge) > 0.01:
                com = residue.atoms.center_of_mass()
                charged_residues.append((residue, charge, com))

        total_energy = 0.0
        per_residue_energy = {int(r.resid): 0.0 for r, _, _ in charged_residues}

        # Pairwise interactions
        for i, (res_i, q_i, pos_i) in enumerate(charged_residues):
            for j, (res_j, q_j, pos_j) in enumerate(charged_residues[i+1:], start=i+1):
                distance = np.linalg.norm(pos_i - pos_j)

                if distance < 3.0:  # Avoid singularity
                    distance = 3.0

                # Coulomb energy with distance-dependent dielectric
                epsilon_r = self.epsilon * (distance / 10.0)
                epsilon_r = max(epsilon_r, 4.0)

                energy = self.k_coulomb * q_i * q_j / (epsilon_r * distance)

                # Distribute energy equally
                resid_i = int(res_i.resid)
                resid_j = int(res_j.resid)
                per_residue_energy[resid_i] += energy / 2.0
                per_residue_energy[resid_j] += energy / 2.0
                total_energy += energy

        return total_energy, per_residue_energy


class TotalEnergy:
    """
    Calculate total energy combining all terms.
    """

    def __init__(
        self,
        hydrophobic_weight: float = 1.0,
        electrostatic_weight: float = 1.0,
        temperature: float = 310.15
    ):
        """
        Initialize total energy calculator.

        Args:
            hydrophobic_weight: Weight for hydrophobic term
            electrostatic_weight: Weight for electrostatic term
            temperature: Temperature in Kelvin
        """
        self.hydrophobic = HydrophobicEnergy()
        self.electrostatic = ElectrostaticEnergy(temperature=temperature)
        self.hydrophobic_weight = hydrophobic_weight
        self.electrostatic_weight = electrostatic_weight

    def calculate(
        self,
        protein_atoms: AtomGroup,
        membrane_center_z: float,
        residue_sasa: Optional[Dict[int, float]] = None
    ) -> Dict:
        """
        Calculate all energy terms.

        Args:
            protein_atoms: MDAnalysis AtomGroup
            membrane_center_z: Z-coordinate of membrane center
            residue_sasa: Pre-calculated SASA values

        Returns:
            Dictionary with energy components and per-residue contributions
        """
        protein_com_z = protein_atoms.center_of_mass()[2]

        # Hydrophobic energy
        e_hydrophobic, hydro_per_res = self.hydrophobic.calculate_total_energy(
            protein_atoms, residue_sasa
        )

        # Electrostatic energy (membrane interaction)
        e_electrostatic, elec_per_res = self.electrostatic.calculate_membrane_interaction_energy(
            protein_atoms, membrane_center_z, protein_com_z
        )

        # Total weighted energy
        total = (self.hydrophobic_weight * e_hydrophobic +
                self.electrostatic_weight * e_electrostatic)

        # Combine per-residue energies
        all_resids = set(list(hydro_per_res.keys()) + list(elec_per_res.keys()))
        per_residue = {}
        for resid in all_resids:
            hydro = hydro_per_res.get(resid, 0.0)
            elec = elec_per_res.get(resid, 0.0)
            per_residue[resid] = {
                'hydrophobic': hydro,
                'electrostatic': elec,
                'total': self.hydrophobic_weight * hydro + self.electrostatic_weight * elec
            }

        return {
            'total': total,
            'hydrophobic': e_hydrophobic,
            'electrostatic': e_electrostatic,
            'per_residue': per_residue
        }
