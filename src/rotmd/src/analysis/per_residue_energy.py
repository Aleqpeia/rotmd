"""
Per-Residue Energy Analysis

Identifies residues with highest energetic contribution to protein dynamics.
Useful for finding mutation hotspots and key structural elements.

Key Features:
- Per-residue energy decomposition (PMF contributions)
- Statistical ranking of highest-energy residues
- Comparison between WT and mutant
- Visualization of energy profiles

Author: Mykyta Bobylyow
Date: 2025
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
from numpy.typing import ArrayLike, NDArray
from dataclasses import dataclass
import warnings


@dataclass
class ResidueEnergyProfile:
    """
    Energy profile for a single residue.

    Attributes
    ----------
    residue_id : int
        Residue number
    residue_name : str
        Residue 3-letter code (e.g., 'ALA', 'LYS')
    mean_energy : float
        Time-averaged energy (kcal/mol)
    std_energy : float
        Standard deviation of energy
    max_energy : float
        Maximum energy observed
    min_energy : float
        Minimum energy observed
    """
    residue_id: int
    residue_name: str
    energy_series: NDArray
    mean_energy: float
    std_energy: float
    max_energy: float
    min_energy: float


def compute_residue_energies(
    positions: NDArray,
    residue_ids: NDArray,
    residue_names: List[str],
    energy_function: Optional[callable] = None
) -> List[ResidueEnergyProfile]:
    """
    Compute per-residue energy profiles from trajectory.

    Parameters
    ----------
    positions : ndarray, shape (n_frames, n_atoms, 3)
        Atomic positions over time (Angstroms)
    residue_ids : ndarray, shape (n_atoms,)
        Residue ID for each atom
    residue_names : list of str
        Residue names corresponding to unique residue_ids
    energy_function : callable, optional
        Function to compute residue energy from positions
        If None, uses simple distance-based estimate

    Returns
    -------
    profiles : list of ResidueEnergyProfile
        Energy profile for each residue, sorted by mean_energy

    Notes
    -----
    This is a simplified analysis. For production use, integrate with:
    - GROMACS energy decomposition (gmx energy -odh)
    - APBS electrostatic calculations
    - MM-PBSA per-residue contributions

    Example
    -------
    >>> profiles = compute_residue_energies(
    ...     positions, residue_ids, residue_names
    ... )
    >>> top5 = profiles[:5]
    >>> for p in top5:
    ...     print(f"Residue {p.residue_id} ({p.residue_name}): {p.mean_energy:.2f} kcal/mol")
    """
    n_frames = len(positions)
    unique_residues = np.unique(residue_ids)
    n_residues = len(unique_residues)

    profiles = []

    for res_id in unique_residues:
        # Get atoms belonging to this residue
        mask = residue_ids == res_id
        res_positions = positions[:, mask, :]  # (n_frames, n_atoms_in_res, 3)

        # Compute energy time series for this residue
        if energy_function is not None:
            energies = np.array([energy_function(res_positions[i]) for i in range(n_frames)])
        else:
            # Default: use RMS displacement as proxy for energy
            # (larger fluctuations ~ higher energy)
            com = np.mean(res_positions, axis=1)  # Center of mass per frame
            rms_displacement = np.std(np.linalg.norm(com - np.mean(com, axis=0), axis=1))
            # Scale to energy-like units (this is just a heuristic!)
            energies = rms_displacement * np.ones(n_frames)

        # Get residue name
        res_idx = np.where(unique_residues == res_id)[0][0]
        res_name = residue_names[res_idx] if res_idx < len(residue_names) else 'UNK'

        # Create profile
        profile = ResidueEnergyProfile(
            residue_id=int(res_id),
            residue_name=res_name,
            energy_series=energies,
            mean_energy=float(np.mean(energies)),
            std_energy=float(np.std(energies)),
            max_energy=float(np.max(energies)),
            min_energy=float(np.min(energies))
        )

        profiles.append(profile)

    # Sort by mean energy (descending)
    profiles.sort(key=lambda p: p.mean_energy, reverse=True)

    return profiles


def identify_hotspot_residues(
    profiles: List[ResidueEnergyProfile],
    threshold_percentile: float = 75.0,
    min_count: int = 5
) -> List[ResidueEnergyProfile]:
    """
    Identify hotspot residues with highest energetic contribution.

    Parameters
    ----------
    profiles : list of ResidueEnergyProfile
        Residue energy profiles
    threshold_percentile : float
        Energy percentile cutoff (default: 90th percentile)
    min_count : int
        Minimum number of residues to return

    Returns
    -------
    hotspots : list of ResidueEnergyProfile
        Highest-energy residues

    Example
    -------
    >>> hotspots = identify_hotspot_residues(profiles, threshold_percentile=95)
    >>> print(f"Found {len(hotspots)} hotspot residues")
    """
    energies = np.array([p.mean_energy for p in profiles])
    threshold = np.percentile(energies, threshold_percentile)

    hotspots = [p for p in profiles if p.mean_energy >= threshold]

    # Ensure minimum count
    if len(hotspots) < min_count:
        hotspots = profiles[:min_count]

    return hotspots


def compare_residue_energies(
    profiles_wt: List[ResidueEnergyProfile],
    profiles_mut: List[ResidueEnergyProfile]
) -> Dict[int, Tuple[float, float, float]]:
    """
    Compare per-residue energies between WT and mutant.

    Parameters
    ----------
    profiles_wt : list
        Residue profiles for wild-type
    profiles_mut : list
        Residue profiles for mutant

    Returns
    -------
    comparison : dict
        Mapping residue_id → (energy_wt, energy_mut, difference)

    Example
    -------
    >>> comparison = compare_residue_energies(profiles_wt, profiles_mut)
    >>> for res_id, (e_wt, e_mut, diff) in comparison.items():
    ...     if abs(diff) > 2.0:  # Large change
    ...         print(f"Residue {res_id}: ΔE = {diff:.2f} kcal/mol")
    """
    # Create dictionaries for lookup
    wt_dict = {p.residue_id: p.mean_energy for p in profiles_wt}
    mut_dict = {p.residue_id: p.mean_energy for p in profiles_mut}

    # Find common residues
    common_ids = set(wt_dict.keys()) & set(mut_dict.keys())

    comparison = {}
    for res_id in common_ids:
        e_wt = wt_dict[res_id]
        e_mut = mut_dict[res_id]
        diff = e_mut - e_wt

        comparison[res_id] = (e_wt, e_mut, diff)

    return comparison


def plot_residue_energy_profile(
    profiles: List[ResidueEnergyProfile],
    title: str = 'Per-Residue Energy Profile',
    top_n: int = 20,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> 'Figure':
    """
    Plot per-residue energy as bar chart.

    Parameters
    ----------
    profiles : list
        Residue energy profiles (sorted by energy)
    title : str
        Plot title
    top_n : int
        Number of top residues to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        File formats

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Example
    -------
    >>> fig = plot_residue_energy_profile(
    ...     profiles, top_n=15,
    ...     save_path='figures/residue_energy'
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("Matplotlib required")

    from ..visualization._plot_utils import setup_publication_style, save_publication_figure

    setup_publication_style()

    # Select top N residues
    top_profiles = profiles[:top_n]

    residue_labels = [f"{p.residue_id} ({p.residue_name})" for p in top_profiles]
    mean_energies = [p.mean_energy for p in top_profiles]
    std_energies = [p.std_energy for p in top_profiles]

    fig, ax = plt.subplots(figsize=figsize)

    # Multi-plot: bar chart (left), timeseries (right) arranged side by side
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.80)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[1, 0])

    # Bar chart
    x = np.arange(len(residue_labels))
    bars = ax_bar.bar(x, mean_energies, yerr=std_energies,
                      capsize=3, color='steelblue', alpha=0.7,
                      edgecolor='black', linewidth=0.5)

    # Highlight top 3
    for i in range(min(3, len(bars))):
        bars[i].set_color('coral')

    # Styling for bar chart
    ax_bar.set_xlabel('Residue', fontsize=12)
    ax_bar.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax_bar.set_title(title, fontsize=14)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(residue_labels, rotation=45, ha='right', fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis='y')

    # Timeseries for top 3 residues (right plot)
    colors = ['coral', 'darkorange', 'gold']
    plotted_any = False
    for idx, (profile, color) in enumerate(zip(top_profiles[:3], colors)):
        if hasattr(profile, "energy_series"):
            ts = np.asarray(profile.energy_series)
            ax_ts.plot(ts, label=f"{profile.residue_id} ({profile.residue_name})", color=color, lw=1.8)
            plotted_any = True

    if plotted_any:
        ax_ts.set_xlabel("Frame", fontsize=12)
        ax_ts.set_ylabel("Energy (kcal/mol)", fontsize=12)
        ax_ts.set_title("Timeseries  High-Std Residues", fontsize=13)
        ax_ts.tick_params(axis='both', labelsize=9)
        ax_ts.legend(fontsize=9, loc='upper left', frameon=True)
        ax_ts.grid(True, alpha=0.3)
    else:
        # Hide axis if no data
        ax_ts.set_visible(False)

    fig.tight_layout()


    # Styling
    ax.set_xlabel('Residue', fontsize=12)
    ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(residue_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


def plot_residue_energy_comparison(
    profiles_wt: List[ResidueEnergyProfile],
    profiles_mut: List[ResidueEnergyProfile],
    wt_label: str = 'Wild-Type',
    mut_label: str = 'Mutant',
    top_n: int = 20,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> 'Figure':
    """
    Plot side-by-side comparison of residue energies.

    Parameters
    ----------
    profiles_wt : list
        WT residue profiles
    profiles_mut : list
        Mutant residue profiles
    wt_label, mut_label : str
        Labels for systems
    top_n : int
        Number of residues to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path
    save_formats : tuple
        Save formats

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Example
    -------
    >>> fig = plot_residue_energy_comparison(
    ...     profiles_wt, profiles_mut,
    ...     wt_label='WT', mut_label='N75K',
    ...     save_path='figures/residue_comparison'
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required")

    from ..visualization._plot_utils import setup_publication_style, save_publication_figure

    setup_publication_style()

    # Get top residues from combined list
    all_profiles = profiles_wt + profiles_mut
    all_profiles.sort(key=lambda p: p.mean_energy, reverse=True)
    top_residue_ids = [p.residue_id for p in all_profiles[:top_n]]
    top_residue_ids = sorted(set(top_residue_ids))  # Unique, sorted

    # Extract energies for these residues
    wt_dict = {p.residue_id: (p.mean_energy, p.std_energy, p.residue_name)
               for p in profiles_wt}
    mut_dict = {p.residue_id: (p.mean_energy, p.std_energy, p.residue_name)
                for p in profiles_mut}

    residue_labels = []
    wt_energies = []
    wt_stds = []
    mut_energies = []
    mut_stds = []

    for res_id in top_residue_ids:
        if res_id in wt_dict:
            res_name = wt_dict[res_id][2]
        elif res_id in mut_dict:
            res_name = mut_dict[res_id][2]
        else:
            res_name = 'UNK'

        residue_labels.append(f"{res_id} ({res_name})")

        if res_id in wt_dict:
            wt_energies.append(wt_dict[res_id][0])
            wt_stds.append(wt_dict[res_id][1])
        else:
            wt_energies.append(0)
            wt_stds.append(0)

        if res_id in mut_dict:
            mut_energies.append(mut_dict[res_id][0])
            mut_stds.append(mut_dict[res_id][1])
        else:
            mut_energies.append(0)
            mut_stds.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(residue_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, wt_energies, width,
                   yerr=wt_stds, capsize=2,
                   label=wt_label, color='steelblue', alpha=0.7,
                   edgecolor='black', linewidth=0.5)

    bars2 = ax.bar(x + width/2, mut_energies, width,
                   yerr=mut_stds, capsize=2,
                   label=mut_label, color='coral', alpha=0.7,
                   edgecolor='black', linewidth=0.5)

    # Styling
    ax.set_xlabel('Residue', fontsize=12)
    ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax.set_title('Per-Residue Energy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(residue_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


if __name__ == '__main__':
    print("Per-Residue Energy Analysis Module")
    print("=" * 60)
    print("\nFunctions:")
    print("  - compute_residue_energies()")
    print("  - identify_hotspot_residues()")
    print("  - compare_residue_energies()")
    print("  - plot_residue_energy_profile()")
    print("  - plot_residue_energy_comparison()")
    print("\nUse cases:")
    print("  - Identify mutation hotspots")
    print("  - Compare energetic changes WT vs mutant")
    print("  - Find key structural residues")
    print("=" * 60)
