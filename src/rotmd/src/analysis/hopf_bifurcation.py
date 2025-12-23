"""
Hopf Bifurcation Analysis for Protein Orientation Dynamics

Detects Hopf bifurcations where stable fixed points lose stability and
periodic orbits (limit cycles) emerge.

Key Features:
- Stability analysis of fixed points
- Hopf bifurcation detection
- Limit cycle identification from Poincaré sections
- Integration with phase space and transition analysis

Physics:
- Supercritical Hopf: Stable limit cycle emerges (soft transition)
- Subcritical Hopf: Unstable limit cycle collapses (hard transition)
- Critical for understanding perpendicular → parallel transitions

Author: Mykyta Bobylyow
Date: 2025-12-21
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


@dataclass
class LimitCycleClassification:
    """Classification of limit cycle properties."""

    # Existence
    has_cycle: bool

    # Stability (from Floquet analysis or Poincaré return map)
    stability: str  # 'stable', 'unstable', 'semi-stable', 'unknown'

    # Origin
    origin: str  # 'hopf_supercritical', 'hopf_subcritical', 'homoclinic', 'saddle_node', 'unknown'

    # Dynamical type
    dynamics_type: str  # 'simple', 'relaxation', 'nearly_harmonic', 'chaotic'

    # Quantitative properties
    amplitude: Optional[float]  # rad
    frequency: Optional[float]  # rad/ps
    period: Optional[float]  # ps
    harmonicity: Optional[float]  # Deviation from pure sine (0=harmonic, 1=relaxation)


@dataclass
class HopfBifurcationResult:
    """Results from Hopf bifurcation analysis."""

    # Bifurcation detection
    has_hopf: bool
    bifurcation_energy: Optional[float]

    # Fixed point stability
    fixed_point_stable: bool
    eigenvalue_real: float
    eigenvalue_imag: float

    # Limit cycle properties
    has_limit_cycle: bool
    cycle_amplitude: Optional[float]
    cycle_frequency: Optional[float]

    # Classification
    bifurcation_type: str  # 'supercritical', 'subcritical', or 'none'
    cycle_classification: Optional[LimitCycleClassification] = None


def analyze_fixed_point_stability(
    theta: np.ndarray,
    omega: np.ndarray,
    energy: np.ndarray,
    energy_window: Tuple[float, float],
    verbose: bool = True
) -> Dict:
    """
    Analyze stability of fixed point in energy window.

    Parameters
    ----------
    theta : ndarray
        Tilt angle (radians)
    omega : ndarray
        Angular velocity (rad/ps)
    energy : ndarray
        Total energy (kcal/mol)
    energy_window : tuple
        (E_min, E_max) energy range to analyze
    verbose : bool
        Print analysis

    Returns
    -------
    result : dict
        Stability metrics including:
        - fixed_point: (theta_fp, omega_fp)
        - eigenvalues: Complex eigenvalues
        - stable: Boolean stability

    Notes
    -----
    Fixed point at perpendicular (θ≈90°) loses stability via Hopf bifurcation.
    Leads to oscillatory motion (limit cycle).
    """
    # Extract data in energy window
    mask = (energy >= energy_window[0]) & (energy <= energy_window[1])
    theta_window = theta[mask]
    omega_window = omega[mask]

    if len(theta_window) < 10:
        return {'stable': None, 'eigenvalues': None}

    # Estimate fixed point (mean position)
    theta_fp = np.mean(theta_window)
    omega_fp = np.mean(omega_window)

    # Linearize dynamics around fixed point
    # dθ/dt = ω
    # dω/dt = -γω - (dV/dθ)

    # Estimate damping from velocity autocorrelation
    omega_centered = omega_window - omega_fp
    if len(omega_centered) > 1:
        acf = np.correlate(omega_centered, omega_centered, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]

        # Exponential decay: exp(-γt)
        t = np.arange(len(acf))
        try:
            popt, _ = curve_fit(lambda t, gamma: np.exp(-gamma * t),
                              t[:min(20, len(t))],
                              acf[:min(20, len(acf))],
                              p0=[0.1], bounds=(0, 10))
            gamma = popt[0]
        except:
            gamma = 0.1
    else:
        gamma = 0.1

    # Estimate restoring force gradient (dV/dθ)
    theta_var = np.var(theta_window)
    if theta_var > 0:
        # Spring constant from equipartition: k = kT/⟨θ²⟩
        k = 1.0 / theta_var if theta_var < 1.0 else 0.1
    else:
        k = 0.1

    # Jacobian matrix at fixed point:
    # J = [[0, 1],
    #      [-k, -γ]]

    # Eigenvalues: λ = -γ/2 ± √(γ²/4 - k)
    discriminant = (gamma**2 / 4) - k

    if discriminant >= 0:
        # Real eigenvalues (overdamped)
        lambda1 = -gamma/2 + np.sqrt(discriminant)
        lambda2 = -gamma/2 - np.sqrt(discriminant)
        eigenvalues = [lambda1, lambda2]
        stable = lambda1 < 0  # Both must be negative for stability
    else:
        # Complex eigenvalues (underdamped) - Hopf candidate
        real_part = -gamma / 2
        imag_part = np.sqrt(-discriminant)
        eigenvalues = [complex(real_part, imag_part),
                      complex(real_part, -imag_part)]
        stable = real_part < 0  # Stable if real part negative

    result = {
        'fixed_point': (theta_fp, omega_fp),
        'eigenvalues': eigenvalues,
        'stable': stable,
        'damping': gamma,
        'stiffness': k
    }

    if verbose:
        print(f"  Fixed point: θ={np.degrees(theta_fp):.1f}°, ω={omega_fp:.2f} rad/ps")
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Stable: {stable}")

    return result


def detect_limit_cycle(
    theta: np.ndarray,
    omega: np.ndarray,
    energy: np.ndarray,
    energy_window: Tuple[float, float]
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Detect limit cycle (periodic orbit) in phase space.

    Parameters
    ----------
    theta : ndarray
        Tilt angle
    omega : ndarray
        Angular velocity
    energy : ndarray
        Total energy
    energy_window : tuple
        Energy range to analyze

    Returns
    -------
    has_cycle : bool
        True if limit cycle detected
    amplitude : float or None
        Cycle amplitude in θ
    frequency : float or None
        Oscillation frequency (rad/ps)

    Notes
    -----
    Limit cycles appear as closed loops in (θ, ω) phase space.
    Signature of supercritical Hopf bifurcation.
    """
    mask = (energy >= energy_window[0]) & (energy <= energy_window[1])
    theta_window = theta[mask]
    omega_window = omega[mask]

    if len(theta_window) < 50:
        return False, None, None

    # Detect periodicity in θ time series
    theta_mean = np.mean(theta_window)
    theta_centered = theta_window - theta_mean

    # Find peaks (oscillations)
    peaks, properties = find_peaks(theta_centered,
                                   height=0.1,  # At least 0.1 rad amplitude
                                   distance=5)  # At least 5 frames apart

    if len(peaks) < 3:
        return False, None, None

    # Estimate amplitude
    amplitude = np.mean(np.abs(theta_centered[peaks]))

    # Estimate frequency from peak spacing
    if len(peaks) > 1:
        peak_spacing = np.mean(np.diff(peaks))
        frequency = 2 * np.pi / peak_spacing  # rad/ps (assuming 1 ps per frame)
    else:
        frequency = None

    # Criterion: at least 3 peaks with consistent amplitude
    has_cycle = len(peaks) >= 3 and amplitude > 0.1

    return has_cycle, amplitude, frequency


def classify_limit_cycle(
    theta: np.ndarray,
    omega: np.ndarray,
    times: Optional[np.ndarray] = None,
    amplitude: Optional[float] = None,
    frequency: Optional[float] = None,
    has_hopf: bool = False,
    hopf_type: str = 'unknown'
) -> LimitCycleClassification:
    """
    Classify limit cycle by stability, origin, and dynamics type.

    Parameters
    ----------
    theta : ndarray
        Tilt angle trajectory
    omega : ndarray
        Angular velocity trajectory
    times : ndarray, optional
        Time points (ps). If None, assumes unit spacing.
    amplitude : float, optional
        Pre-computed cycle amplitude (rad)
    frequency : float, optional
        Pre-computed cycle frequency (rad/ps)
    has_hopf : bool
        Whether Hopf bifurcation was detected
    hopf_type : str
        Type of Hopf bifurcation if detected

    Returns
    -------
    classification : LimitCycleClassification
        Complete classification of the limit cycle

    Notes
    -----
    Classification schemes:

    1. Stability (via Poincaré return map):
       - stable: Points converge to cycle
       - unstable: Points diverge from cycle
       - semi-stable: Mixed behavior

    2. Origin:
       - hopf_supercritical: Smooth emergence from fixed point
       - hopf_subcritical: Jump to distant attractor
       - homoclinic: Formed from saddle separatrix
       - saddle_node: Cycle collision

    3. Dynamics type:
       - simple: Single isolated orbit
       - relaxation: Fast/slow (stiff) oscillation
       - nearly_harmonic: Close to sinusoidal
       - chaotic: Irregular/aperiodic
    """
    if times is None:
        times = np.arange(len(theta))

    dt = np.mean(np.diff(times))

    # Check if cycle exists
    if amplitude is None or frequency is None:
        # Recompute from data
        theta_centered = theta - np.mean(theta)
        peaks, _ = find_peaks(np.abs(theta_centered), height=0.1, distance=5)

        if len(peaks) < 3:
            return LimitCycleClassification(
                has_cycle=False,
                stability='unknown',
                origin='unknown',
                dynamics_type='none',
                amplitude=None,
                frequency=None,
                period=None,
                harmonicity=None
            )

        amplitude = np.mean(np.abs(theta_centered[peaks]))
        peak_spacing = np.mean(np.diff(peaks)) * dt
        frequency = 2 * np.pi / peak_spacing if peak_spacing > 0 else None

    period = 2 * np.pi / frequency if frequency is not None and frequency > 0 else None

    # 1. Stability Classification (from Poincaré return map)
    stability = _classify_stability_poincare(theta, omega)

    # 2. Origin Classification
    if has_hopf:
        origin = f'hopf_{hopf_type}'
    else:
        # Heuristics for other origins
        if amplitude > 1.0:  # Large amplitude suggests homoclinic
            origin = 'homoclinic'
        else:
            origin = 'unknown'

    # 3. Dynamics Type Classification
    dynamics_type = _classify_dynamics_type(theta, omega, amplitude)

    # 4. Harmonicity (0 = pure sine, 1 = relaxation oscillation)
    harmonicity = _compute_harmonicity(theta, frequency, dt)

    return LimitCycleClassification(
        has_cycle=True,
        stability=stability,
        origin=origin,
        dynamics_type=dynamics_type,
        amplitude=amplitude,
        frequency=frequency,
        period=period,
        harmonicity=harmonicity
    )


def _classify_stability_poincare(theta: np.ndarray, omega: np.ndarray) -> str:
    """
    Classify stability using Poincaré return map.

    Stable cycle: consecutive crossings converge
    Unstable cycle: consecutive crossings diverge
    """
    # Find zero crossings of theta (Poincaré section)
    theta_centered = theta - np.mean(theta)
    sign_changes = np.where(np.diff(np.sign(theta_centered)))[0]

    if len(sign_changes) < 4:
        return 'unknown'

    # Get omega values at crossings
    omega_crossings = omega[sign_changes]

    # Check convergence: do consecutive crossings approach each other?
    differences = np.abs(np.diff(omega_crossings))

    if len(differences) < 2:
        return 'unknown'

    # Trend: decreasing differences → stable, increasing → unstable
    trend = np.polyfit(np.arange(len(differences)), differences, deg=1)[0]

    if trend < -0.01:  # Converging
        return 'stable'
    elif trend > 0.01:  # Diverging
        return 'unstable'
    else:
        return 'semi-stable'


def _classify_dynamics_type(
    theta: np.ndarray,
    omega: np.ndarray,
    amplitude: float
) -> str:
    """
    Classify dynamics type from phase space trajectory.

    - simple: smooth closed orbit
    - relaxation: sharp transitions (high omega variance)
    - nearly_harmonic: circular in phase space
    - chaotic: irregular trajectory
    """
    # Compute phase space characteristics
    omega_std = np.std(omega)
    omega_mean = np.abs(np.mean(omega))

    # Check for relaxation oscillation (fast/slow dynamics)
    if omega_std / (omega_mean + 1e-10) > 5.0:
        return 'relaxation'

    # Check for nearly harmonic (circular phase portrait)
    theta_centered = theta - np.mean(theta)
    omega_centered = omega - np.mean(omega)

    # Circularity: correlation should be ~0 for circle
    correlation = np.abs(np.corrcoef(theta_centered, omega_centered)[0, 1])

    if correlation < 0.1 and amplitude < 0.5:
        return 'nearly_harmonic'

    # Check for chaos (irregular peaks)
    peaks, _ = find_peaks(np.abs(theta_centered), height=amplitude * 0.5)

    if len(peaks) > 3:
        peak_spacings = np.diff(peaks)
        spacing_cv = np.std(peak_spacings) / (np.mean(peak_spacings) + 1e-10)

        if spacing_cv > 0.3:  # High variability → chaotic
            return 'chaotic'

    return 'simple'


def _compute_harmonicity(
    theta: np.ndarray,
    frequency: Optional[float],
    dt: float
) -> Optional[float]:
    """
    Compute harmonicity: 0 = pure sine, 1 = relaxation oscillation.

    Method: Fit sine wave and compute residual energy.
    """
    if frequency is None or frequency <= 0:
        return None

    times = np.arange(len(theta)) * dt
    theta_mean = np.mean(theta)
    theta_centered = theta - theta_mean

    # Fit sine wave: θ(t) = A·sin(2πft + φ)
    def sine_model(t, A, phi):
        return A * np.sin(2 * np.pi * frequency * t + phi)

    try:
        popt, _ = curve_fit(
            sine_model, times, theta_centered,
            p0=[np.std(theta_centered), 0],
            maxfev=1000
        )

        # Compute fit residuals
        theta_fit = sine_model(times, *popt)
        residual_energy = np.sum((theta_centered - theta_fit)**2)
        total_energy = np.sum(theta_centered**2)

        # Harmonicity: 0 = perfect fit, 1 = poor fit (relaxation)
        harmonicity = residual_energy / (total_energy + 1e-10)

        return float(np.clip(harmonicity, 0, 1))

    except:
        return None


def detect_hopf_bifurcation(
    theta: np.ndarray,
    omega: np.ndarray,
    energy: np.ndarray,
    n_energy_bins: int = 10,
    verbose: bool = True
) -> HopfBifurcationResult:
    """
    Detect Hopf bifurcation by analyzing stability vs energy.

    Parameters
    ----------
    theta : ndarray
        Tilt angle trajectory
    omega : ndarray
        Angular velocity trajectory
    energy : ndarray
        Total energy trajectory
    n_energy_bins : int
        Number of energy windows to analyze
    verbose : bool
        Print detailed analysis

    Returns
    -------
    result : HopfBifurcationResult
        Complete bifurcation analysis

    Notes
    -----
    Hopf bifurcation occurs when:
    1. Fixed point changes from stable to unstable
    2. Limit cycle emerges at critical energy
    3. Complex eigenvalues cross imaginary axis

    For N75K perpendicular state: expect Hopf bifurcation
    leading to unstable oscillations.
    """
    E_min, E_max = np.percentile(energy, [10, 90])
    energy_bins = np.linspace(E_min, E_max, n_energy_bins + 1)

    stability_vs_energy = []
    limit_cycle_vs_energy = []

    if verbose:
        print("\nHopf Bifurcation Analysis")
        print("=" * 60)
        print(f"Energy range: {E_min:.2f} to {E_max:.2f} kcal/mol")
        print(f"Analyzing {n_energy_bins} energy windows...\n")

    for i in range(n_energy_bins):
        E_window = (energy_bins[i], energy_bins[i+1])
        E_center = (E_window[0] + E_window[1]) / 2

        if verbose:
            print(f"Energy window {i+1}: {E_window[0]:.2f} - {E_window[1]:.2f} kcal/mol")

        # Analyze fixed point stability
        stability = analyze_fixed_point_stability(
            theta, omega, energy, E_window, verbose=verbose
        )

        # Detect limit cycle
        has_cycle, amplitude, frequency = detect_limit_cycle(
            theta, omega, energy, E_window
        )

        if verbose and has_cycle:
            print(f"  → Limit cycle detected! Amplitude={amplitude:.3f} rad, f={frequency:.3f} rad/ps")

        stability_vs_energy.append((E_center, stability['stable'], stability['eigenvalues']))
        limit_cycle_vs_energy.append((E_center, has_cycle, amplitude, frequency))

        if verbose:
            print()

    # Detect bifurcation: stability change + limit cycle emergence
    has_hopf = False
    bifurcation_energy = None
    bifurcation_type = 'none'

    for i in range(len(stability_vs_energy) - 1):
        E_curr, stable_curr, eigs_curr = stability_vs_energy[i]
        E_next, stable_next, eigs_next = stability_vs_energy[i + 1]

        _, has_cycle_curr, amp_curr, _ = limit_cycle_vs_energy[i]
        _, has_cycle_next, amp_next, _ = limit_cycle_vs_energy[i + 1]

        # Check for stability change
        if stable_curr and not stable_next:
            # Fixed point loses stability
            if has_cycle_next:
                # Limit cycle emerges → supercritical Hopf
                has_hopf = True
                bifurcation_energy = (E_curr + E_next) / 2
                bifurcation_type = 'supercritical'
                break

    # Use last window for final state
    E_final, stable_final, eigs_final = stability_vs_energy[-1]
    _, has_cycle_final, amp_final, freq_final = limit_cycle_vs_energy[-1]

    # Extract eigenvalue parts
    if eigs_final and isinstance(eigs_final[0], complex):
        eigenvalue_real = eigs_final[0].real
        eigenvalue_imag = eigs_final[0].imag
    else:
        eigenvalue_real = eigs_final[0] if eigs_final else 0.0
        eigenvalue_imag = 0.0

    # Classify limit cycle if present
    cycle_classification = None
    if has_cycle_final:
        cycle_classification = classify_limit_cycle(
            theta, omega,
            times=None,  # Will use frame indices
            amplitude=amp_final,
            frequency=freq_final,
            has_hopf=has_hopf,
            hopf_type=bifurcation_type
        )

    result = HopfBifurcationResult(
        has_hopf=has_hopf,
        bifurcation_energy=bifurcation_energy,
        fixed_point_stable=stable_final,
        eigenvalue_real=eigenvalue_real,
        eigenvalue_imag=eigenvalue_imag,
        has_limit_cycle=has_cycle_final,
        cycle_amplitude=amp_final,
        cycle_frequency=freq_final,
        bifurcation_type=bifurcation_type,
        cycle_classification=cycle_classification
    )

    if verbose:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if has_hopf:
            print(f"✓ Hopf bifurcation detected at E ≈ {bifurcation_energy:.2f} kcal/mol")
            print(f"  Type: {bifurcation_type}")
        else:
            print("✗ No Hopf bifurcation detected")

        print(f"\nFinal state (high energy):")
        print(f"  Fixed point stable: {stable_final}")
        print(f"  Limit cycle present: {has_cycle_final}")
        if has_cycle_final and cycle_classification:
            print(f"  Cycle amplitude: {amp_final:.3f} rad")
            print(f"  Cycle frequency: {freq_final:.3f} rad/ps")
            print(f"\nLimit Cycle Classification:")
            print(f"  Stability: {cycle_classification.stability}")
            print(f"  Origin: {cycle_classification.origin}")
            print(f"  Dynamics type: {cycle_classification.dynamics_type}")
            if cycle_classification.harmonicity is not None:
                harm_pct = cycle_classification.harmonicity * 100
                print(f"  Harmonicity: {harm_pct:.1f}% deviation from sine")
            if cycle_classification.period is not None:
                print(f"  Period: {cycle_classification.period:.2f} ps")

    return result


def plot_hopf_bifurcation_diagram(
    theta: np.ndarray,
    omega: np.ndarray,
    energy: np.ndarray,
    result: HopfBifurcationResult,
    save_path: Optional[str] = None,
    save_formats: Tuple[str, ...] = ('pdf', 'png')
) -> 'Figure':
    """
    Plot bifurcation diagram showing Hopf transition.

    Parameters
    ----------
    theta, omega, energy : ndarray
        Phase space trajectory
    result : HopfBifurcationResult
        Bifurcation analysis results
    save_path : str, optional
        Path to save figure
    save_formats : tuple
        Output formats

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required")

    from ..visualization._plot_utils import setup_publication_style, save_publication_figure

    setup_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Phase portrait with limit cycle highlighted
    ax = axes[0, 0]
    ax.scatter(np.degrees(theta), omega, s=1, alpha=0.3, c=energy,
              cmap='viridis', rasterized=True)
    ax.set_xlabel('Tilt Angle θ (°)', fontsize=11)
    ax.set_ylabel('Angular Velocity ω (rad/ps)', fontsize=11)
    ax.set_title('Phase Portrait', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 2. Energy vs amplitude (bifurcation diagram)
    ax = axes[0, 1]
    n_bins = 20
    E_bins = np.linspace(energy.min(), energy.max(), n_bins)
    amplitudes = []
    E_centers = []

    for i in range(len(E_bins) - 1):
        mask = (energy >= E_bins[i]) & (energy < E_bins[i+1])
        if np.sum(mask) > 10:
            theta_bin = theta[mask]
            amp = np.std(theta_bin)
            amplitudes.append(amp)
            E_centers.append((E_bins[i] + E_bins[i+1]) / 2)

    ax.plot(E_centers, amplitudes, 'o-', markersize=4, linewidth=2, color='steelblue')
    if result.has_hopf and result.bifurcation_energy:
        ax.axvline(result.bifurcation_energy, color='red', linestyle='--',
                  linewidth=2, label=f'Hopf at E={result.bifurcation_energy:.2f}')
        ax.legend()
    ax.set_xlabel('Total Energy (kcal/mol)', fontsize=11)
    ax.set_ylabel('Amplitude (rad)', fontsize=11)
    ax.set_title('Bifurcation Diagram', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. Time series of theta
    ax = axes[1, 0]
    t = np.arange(len(theta))
    ax.plot(t, np.degrees(theta), linewidth=0.5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('θ (°)', fontsize=11)
    ax.set_title('Tilt Angle Time Series', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Hopf Bifurcation Analysis\n"
    summary_text += "=" * 40 + "\n\n"

    if result.has_hopf:
        summary_text += f"✓ Hopf bifurcation detected\n"
        summary_text += f"  Energy: {result.bifurcation_energy:.2f} kcal/mol\n"
        summary_text += f"  Type: {result.bifurcation_type}\n\n"
    else:
        summary_text += "✗ No Hopf bifurcation\n\n"

    summary_text += f"Fixed point stability: {result.fixed_point_stable}\n"
    summary_text += f"Eigenvalues: {result.eigenvalue_real:.3f} ± {result.eigenvalue_imag:.3f}i\n\n"

    if result.has_limit_cycle:
        summary_text += f"✓ Limit cycle present\n"
        summary_text += f"  Amplitude: {result.cycle_amplitude:.3f} rad\n"
        summary_text += f"  Frequency: {result.cycle_frequency:.3f} rad/ps\n"
    else:
        summary_text += "✗ No limit cycle\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.tight_layout()

    if save_path:
        save_publication_figure(fig, save_path, formats=save_formats, close=False)

    return fig


if __name__ == '__main__':
    print("Hopf Bifurcation Analysis Module")
    print("=" * 60)
    print("\nDetects Hopf bifurcations in protein orientation dynamics")
    print("\nFunctions:")
    print("  - detect_hopf_bifurcation()")
    print("  - analyze_fixed_point_stability()")
    print("  - detect_limit_cycle()")
    print("  - plot_hopf_bifurcation_diagram()")
    print("\nFor perpendicular (θ≈90°) N75K state:")
    print("  Expect supercritical Hopf → unstable oscillations")
