"""
Simple Metrics Visualization

Basic time series plots for RMSD, radius of gyration, and other structural metrics.
Designed for quick visualization without complex phase space analysis.

Author: Mykyta Bobylyow
Date: 2025
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
from numpy.typing import NDArray
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

from ._plot_utils import setup_publication_style, save_publication_figure


def plot_rmsd_timeseries(
    times: NDArray,
    rmsd: NDArray,
    title: str = 'RMSD vs Time',
    xlabel: str = 'Time (ps)',
    ylabel: str = 'RMSD (Å)',
    color: str = 'steelblue',
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot RMSD time series.

    Parameters
    ----------
    times : ndarray, shape (n,)
        Time values in picoseconds
    rmsd : ndarray, shape (n,)
        RMSD values in Angstroms
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color : str
        Line color
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save figure (without extension)
    save_formats : tuple
        File formats to save ('png', 'pdf', 'svg')

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Example
    -------
    >>> from protein_orientation.observables.structural import compute_rmsd_trajectory
    >>> rmsd = compute_rmsd_trajectory(positions, reference=positions[0])
    >>> fig = plot_rmsd_timeseries(times, rmsd, save_path='figures/rmsd')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot time series
    ax.plot(times, rmsd, color=color, linewidth=1.5, alpha=0.8)

    # Add mean line
    mean_rmsd = np.mean(rmsd)
    ax.axhline(mean_rmsd, color='red', linestyle='--', linewidth=1.0,
               label=f'Mean: {mean_rmsd:.2f} Å', alpha=0.7)

    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


def plot_rg_timeseries(
    times: NDArray,
    rg: NDArray,
    title: str = 'Radius of Gyration vs Time',
    xlabel: str = 'Time (ps)',
    ylabel: str = r'$R_g$ (Å)',
    color: str = 'forestgreen',
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot radius of gyration time series.

    Parameters
    ----------
    times : ndarray, shape (n,)
        Time values in picoseconds
    rg : ndarray, shape (n,)
        Radius of gyration values in Angstroms
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color : str
        Line color
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save figure (without extension)
    save_formats : tuple
        File formats to save ('png', 'pdf', 'svg')

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Example
    -------
    >>> from protein_orientation.observables.structural import radius_of_gyration
    >>> rg = np.array([radius_of_gyration(pos, masses) for pos in positions])
    >>> fig = plot_rg_timeseries(times, rg, save_path='figures/rg')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot time series
    ax.plot(times, rg, color=color, linewidth=1.5, alpha=0.8)

    # Add mean line
    mean_rg = np.mean(rg)
    ax.axhline(mean_rg, color='red', linestyle='--', linewidth=1.0,
               label=f'Mean: {mean_rg:.2f} Å', alpha=0.7)

    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


def plot_rmsd_rg_comparison(
    times: NDArray,
    rmsd_wt: NDArray,
    rmsd_mut: NDArray,
    rg_wt: NDArray,
    rg_mut: NDArray,
    wt_label: str = 'Wild-Type',
    mut_label: str = 'Mutant',
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot RMSD and Rg comparison between WT and mutant in side-by-side panels.

    Parameters
    ----------
    times : ndarray, shape (n,)
        Time values (same for both systems)
    rmsd_wt : ndarray, shape (n,)
        RMSD for wild-type
    rmsd_mut : ndarray, shape (n,)
        RMSD for mutant
    rg_wt : ndarray, shape (n,)
        Rg for wild-type
    rg_mut : ndarray, shape (n,)
        Rg for mutant
    wt_label : str
        Label for wild-type
    mut_label : str
        Label for mutant
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        File formats to save

    Returns
    -------
    fig : Figure
        Matplotlib figure with two subplots

    Example
    -------
    >>> fig = plot_rmsd_rg_comparison(
    ...     times, rmsd_wt, rmsd_n75k, rg_wt, rg_n75k,
    ...     wt_label='Wild-Type', mut_label='N75K',
    ...     save_path='figures/metrics_comparison'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: RMSD
    ax1.plot(times, rmsd_wt, color='steelblue', linewidth=1.5,
             label=wt_label, alpha=0.8)
    ax1.plot(times, rmsd_mut, color='coral', linewidth=1.5,
             label=mut_label, alpha=0.8)

    ax1.axhline(np.mean(rmsd_wt), color='steelblue', linestyle='--',
                linewidth=1.0, alpha=0.5)
    ax1.axhline(np.mean(rmsd_mut), color='coral', linestyle='--',
                linewidth=1.0, alpha=0.5)

    ax1.set_xlabel('Time (ps)', fontsize=12)
    ax1.set_ylabel('RMSD (Å)', fontsize=12)
    ax1.set_title('RMSD Comparison', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rg
    ax2.plot(times, rg_wt, color='forestgreen', linewidth=1.5,
             label=wt_label, alpha=0.8)
    ax2.plot(times, rg_mut, color='darkorange', linewidth=1.5,
             label=mut_label, alpha=0.8)

    ax2.axhline(np.mean(rg_wt), color='forestgreen', linestyle='--',
                linewidth=1.0, alpha=0.5)
    ax2.axhline(np.mean(rg_mut), color='darkorange', linestyle='--',
                linewidth=1.0, alpha=0.5)

    ax2.set_xlabel('Time (ps)', fontsize=12)
    ax2.set_ylabel(r'$R_g$ (Å)', fontsize=12)
    ax2.set_title('Radius of Gyration Comparison', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


def plot_metric_distribution(
    values_wt: NDArray,
    values_mut: NDArray,
    metric_name: str = 'Metric',
    metric_units: str = 'Å',
    wt_label: str = 'Wild-Type',
    mut_label: str = 'Mutant',
    bins: int = 50,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot histogram distribution comparison between WT and mutant.

    Parameters
    ----------
    values_wt : ndarray
        Metric values for wild-type
    values_mut : ndarray
        Metric values for mutant
    metric_name : str
        Name of metric (e.g., 'RMSD', 'Rg')
    metric_units : str
        Units for x-axis
    wt_label : str
        Label for wild-type
    mut_label : str
        Label for mutant
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        File formats to save

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Example
    -------
    >>> fig = plot_metric_distribution(
    ...     rmsd_wt, rmsd_n75k,
    ...     metric_name='RMSD', metric_units='Å',
    ...     save_path='figures/rmsd_distribution'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    ax.hist(values_wt, bins=bins, alpha=0.6, color='steelblue',
            label=f'{wt_label} (μ={np.mean(values_wt):.2f} {metric_units})',
            density=True, edgecolor='black', linewidth=0.5)
    ax.hist(values_mut, bins=bins, alpha=0.6, color='coral',
            label=f'{mut_label} (μ={np.mean(values_mut):.2f} {metric_units})',
            density=True, edgecolor='black', linewidth=0.5)

    # Add mean lines
    ax.axvline(np.mean(values_wt), color='steelblue', linestyle='--',
               linewidth=2, alpha=0.8)
    ax.axvline(np.mean(values_mut), color='coral', linestyle='--',
               linewidth=2, alpha=0.8)

    # Styling
    ax.set_xlabel(f'{metric_name} ({metric_units})', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'{metric_name} Distribution', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


def plot_multi_metric_panel(
    times: NDArray,
    metrics_wt: Dict[str, NDArray],
    metrics_mut: Dict[str, NDArray],
    metric_names: List[str] = ['rmsd', 'rg'],
    metric_labels: Optional[Dict[str, str]] = None,
    wt_label: str = 'Wild-Type',
    mut_label: str = 'Mutant',
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('png',)
) -> Figure:
    """
    Plot multiple metrics in a grid layout.

    Parameters
    ----------
    times : ndarray
        Time values
    metrics_wt : dict
        Dictionary of metric arrays for wild-type
        Keys should match metric_names
    metrics_mut : dict
        Dictionary of metric arrays for mutant
    metric_names : list
        List of metric keys to plot
    metric_labels : dict, optional
        Custom y-axis labels for each metric
        Example: {'rmsd': 'RMSD (Å)', 'rg': r'$R_g$ (Å)'}
    wt_label : str
        Label for wild-type
    mut_label : str
        Label for mutant
    figsize : tuple, optional
        Figure size (defaults to auto based on n_metrics)
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        File formats to save

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Example
    -------
    >>> metrics_wt = {'rmsd': rmsd_wt, 'rg': rg_wt, 'asphericity': asph_wt}
    >>> metrics_mut = {'rmsd': rmsd_mut, 'rg': rg_mut, 'asphericity': asph_mut}
    >>> fig = plot_multi_metric_panel(
    ...     times, metrics_wt, metrics_mut,
    ...     metric_names=['rmsd', 'rg', 'asphericity'],
    ...     save_path='figures/metrics_panel'
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    setup_publication_style()

    n_metrics = len(metric_names)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    if figsize is None:
        figsize = (12, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Default labels
    default_labels = {
        'rmsd': 'RMSD (Å)',
        'rg': r'$R_g$ (Å)',
        'asphericity': 'Asphericity',
        'acylindricity': 'Acylindricity',
        'end_to_end': 'End-to-End (Å)'
    }

    if metric_labels is None:
        metric_labels = default_labels

    colors_wt = ['steelblue', 'forestgreen', 'mediumpurple', 'darkgoldenrod']
    colors_mut = ['coral', 'darkorange', 'orchid', 'gold']

    for i, metric in enumerate(metric_names):
        ax = axes[i]

        if metric not in metrics_wt or metric not in metrics_mut:
            ax.text(0.5, 0.5, f'Metric "{metric}" not found',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        color_wt = colors_wt[i % len(colors_wt)]
        color_mut = colors_mut[i % len(colors_mut)]

        # Plot time series
        ax.plot(times, metrics_wt[metric], color=color_wt, linewidth=1.5,
                label=wt_label, alpha=0.8)
        ax.plot(times, metrics_mut[metric], color=color_mut, linewidth=1.5,
                label=mut_label, alpha=0.8)

        # Mean lines
        ax.axhline(np.mean(metrics_wt[metric]), color=color_wt,
                  linestyle='--', linewidth=1.0, alpha=0.5)
        ax.axhline(np.mean(metrics_mut[metric]), color=color_mut,
                  linestyle='--', linewidth=1.0, alpha=0.5)

        # Styling
        ax.set_xlabel('Time (ps)', fontsize=11)
        ylabel = metric_labels.get(metric, metric.upper())
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{metric.upper()}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


if __name__ == '__main__':
    print("Metrics Visualization Module")
    print("=" * 60)
    print("\nSimple plotting functions for structural metrics:")
    print("  - plot_rmsd_timeseries()")
    print("  - plot_rg_timeseries()")
    print("  - plot_rmsd_rg_comparison()")
    print("  - plot_metric_distribution()")
    print("  - plot_multi_metric_panel()")
    print("\nDesigned for quick visualization without complex phase space analysis.")
    print("=" * 60)
