# Protein Orientation Analysis Toolkit

A comprehensive Python toolkit for analyzing protein rotational dynamics from molecular dynamics (MD) trajectories.

## Features

### Core Capabilities
- **Euler Angle Extraction**: Extract ZYZ Euler angles (φ, θ, ψ) from MD trajectories
- **Angular Momentum Analysis**: Compute and decompose angular momentum into spin and nutation components
- **Torque Validation**: Validate Euler's equation dL/dt = τ
- **Rotational Diffusion**: Analyze anisotropic rotational diffusion tensors
- **PMF Computation**: Calculate potential of mean force F(θ,ψ) with proper Jacobian corrections

### Advanced Analysis
- **Transition State Theory**: Transmission coefficients, reactive flux, committor probabilities
- **Non-equilibrium Thermodynamics**: Detailed balance tests, entropy production
- **Langevin Dynamics**: Validate MD against overdamped Langevin models
- **Correlation Functions**: Autocorrelation and cross-correlation analysis
- **Friction Coefficients**: Extract rotational friction from velocity autocorrelation

### Visualization
- Phase space plots (E vs L, 3D phase space)
- PMF heatmaps and 3D surfaces with minima identification
- Torque vector fields
- Power spectra and autocorrelation functions
- Trajectory animations with state coloring

## Installation

### From source
```bash
cd /path/to/sta
pip install -e .
```

### With optional dependencies
```bash
# Visualization
pip install -e ".[viz]"

# Configuration files (YAML)
pip install -e ".[config]"

# HDF5 support
pip install -e ".[hdf5]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from protein_orientation import analyze_trajectory

# Run complete analysis pipeline
results = analyze_trajectory(
    topology='system.gro',
    trajectory='traj.trr',
    output_dir='analysis_results',
    temperature=300.0,
    save_plots=True
)

print(f"Diffusion coefficient: {results['D_msad']:.3f} rad²/ps")
print(f"Mean nutation angle: {np.degrees(results['theta_mean']):.1f}°")
```

### Command Line Interface

```bash
# Full analysis pipeline
protein-orientation analyze system.gro traj.trr -o results/ --plots

# Extract Euler angles only
protein-orientation extract system.gro traj.trr -o euler.npz

# Compute PMF from Euler angles
protein-orientation pmf euler.npz -o pmf.npz --plot

# Analyze diffusion
protein-orientation diffusion euler.npz -o diffusion.json

# Compare two systems
protein-orientation compare wt_results.json n75k_results.json \
    --labels "Wild Type" "N75K Mutant" -o comparison/
```

## Detailed Usage

### 1. Loading Trajectories

```python
from protein_orientation.io.gromacs import load_gromacs_trajectory

# Load with MDAnalysis
traj_data = load_gromacs_trajectory(
    'system.gro',
    'traj.trr',
    selection='protein',
    start=0,
    stop=10000,
    step=10,
    center=True  # Center at origin
)

print(f"Loaded {traj_data['n_frames']} frames")
print(f"Has velocities: {traj_data['has_velocities']}")
print(f"Has forces: {traj_data['has_forces']}")
```

### 2. Extracting Euler Angles

```python
from protein_orientation.core.orientation import extract_orientation_trajectory

euler_angles = extract_orientation_trajectory(
    traj_data['positions'],
    traj_data['masses']
)

phi, theta, psi = euler_angles.T
print(f"Mean θ: {np.degrees(np.mean(theta)):.1f}°")
```

### 3. Computing PMF

```python
from protein_orientation.analysis.pmf import compute_pmf_2d

pmf, theta_bins, psi_bins = compute_pmf_2d(
    euler_angles,
    temperature=300.0,
    bins=(36, 18),  # theta, psi bins
    verbose=True
)

# Find minima
from protein_orientation.analysis.pmf import find_pmf_minima
minima = find_pmf_minima(pmf, theta_bins, psi_bins)

for i, (theta, psi, energy) in enumerate(minima):
    print(f"Minimum {i+1}: θ={np.degrees(theta):.1f}°, "
          f"ψ={np.degrees(psi):.1f}°, F={energy:.2f} kcal/mol")
```

### 4. Analyzing Diffusion

```python
from protein_orientation.observables.diffusion import analyze_diffusion

results = analyze_diffusion(
    euler_angles,
    traj_data['times'],
    verbose=True
)

print(f"Isotropic D: {results['D_msad']:.4f} ± {results['D_msad_err']:.4f} rad²/ps")
print(f"D_φ: {results['D_aniso'][0]:.4f} rad²/ps")
print(f"D_θ: {results['D_aniso'][1]:.4f} rad²/ps")
print(f"D_ψ: {results['D_aniso'][2]:.4f} rad²/ps")
```

### 5. Angular Momentum Analysis

```python
from protein_orientation.observables.angular_momentum import angular_momentum_analysis

L_results = angular_momentum_analysis(
    traj_data['positions'],
    traj_data['velocities'],
    traj_data['masses'],
    traj_data['times'],
    verbose=True
)

print(f"Mean |L|: {L_results['L_mean']:.1f} amu·Ų/ps")
print(f"Mean |L∥|: {L_results['L_parallel_mean']:.1f} amu·Ų/ps")
print(f"Mean |L⊥|: {L_results['L_perp_mean']:.1f} amu·Ų/ps")
```

### 6. Langevin Validation

```python
from protein_orientation.models.langevin import LangevinIntegrator, validate_against_trajectory
from protein_orientation.models.energy import PMFPotential

# Create PMF potential from MD
pmf_potential = PMFPotential.from_histogram(
    euler_angles,
    temperature=300.0
)

# Validate Langevin model
validation = validate_against_trajectory(
    euler_angles,
    traj_data['times'],
    pmf_potential,
    friction=100.0,  # amu/ps
    temperature=300.0,
    n_trials=10
)

print(f"Mean error: {validation['mean_error']:.4f} rad")
print(f"Std error: {validation['std_error']:.4f} rad")
```

### 7. Visualization

```python
from protein_orientation.visualization.surfaces import plot_pmf_heatmap
from protein_orientation.visualization.phase_space import plot_energy_vs_angular_momentum

# PMF heatmap
plot_pmf_heatmap(
    pmf, theta_bins, psi_bins,
    vmax=10.0,
    mark_minima=True,
    save_path='pmf_heatmap.png'
)

# Phase space
L_mag = np.linalg.norm(L_results['L'], axis=1)
plot_energy_vs_angular_momentum(
    energies, L_mag,
    times=traj_data['times'],
    save_path='phase_space.png'
)
```

## Package Structure

```
protein_orientation/
├── core/
│   ├── trajectory.py       # Trajectory I/O
│   ├── inertia.py          # Inertia tensor
│   └── orientation.py      # Euler angles, quaternions
├── observables/
│   ├── angular_momentum.py # L and dL/dt
│   ├── angular_velocity.py # ω from rotation matrices
│   ├── torque.py           # τ = r × F
│   └── diffusion.py        # Rotational diffusion
├── analysis/
│   ├── correlations.py     # ACF, cross-correlation
│   ├── friction.py         # Friction from ACF
│   ├── pmf.py              # Free energy landscapes
│   ├── transitions.py      # TST, reactive flux
│   └── nonequilibrium.py   # Detailed balance, entropy
├── models/
│   ├── langevin.py         # Langevin integrator
│   └── energy.py           # PMF, harmonic, SASA
├── io/
│   ├── gromacs.py          # XTC/TRR readers
│   └── output.py           # JSON, NPZ, HDF5
├── visualization/
│   ├── phase_space.py      # E-L plots, Poincaré sections
│   ├── surfaces.py         # PMF heatmaps, 3D surfaces
│   └── spectra.py          # ACF, power spectra
├── cli.py                  # Command-line interface
├── config.py               # Configuration management
└── utils.py                # Numerical utilities
```

## Configuration Files

Create a YAML configuration file for reproducible analysis:

```yaml
# analysis_config.yaml
trajectory:
  start: 0
  stop: null  # All frames
  step: 10
  selection: "protein"
  center: true

analysis:
  temperature: 300.0
  pmf_bins: [36, 18, 36]  # phi, theta, psi

output:
  directory: "results"
  save_euler: true
  save_pmf: true
  save_plots: true
  format: "npz"  # or "hdf5"

visualization:
  dpi: 300
  pmf_vmax: 10.0
  colormap: "viridis"
```

Load and use:

```python
from protein_orientation.config import AnalysisConfig

config = AnalysisConfig.load('analysis_config.yaml')
temp = config.get('analysis.temperature')
```

## Theory Background

### Euler Angles (ZYZ Convention)
- **φ**: First rotation about lab Z-axis (0 to 2π)
- **θ**: Nutation angle about new Y-axis (0 to π)
- **ψ**: Spin angle about final Z-axis (0 to 2π)

### Angular Momentum Decomposition
```
L = L∥ + L⊥
```
- **L∥**: Parallel to principal axis (spin)
- **L⊥**: Perpendicular to principal axis (nutation)

### Potential of Mean Force
```
F(θ,ψ) = -kT ln[P(θ,ψ) sin(θ)]
```
The sin(θ) factor is the Jacobian for spherical coordinates.

### Rotational Diffusion
From mean squared angular displacement:
```
<Δθ²(t)> = 6D·t  (isotropic)
```

From velocity autocorrelation (Green-Kubo):
```
D = (1/3) ∫₀^∞ <ω(0)·ω(t)> dt
```

### Euler's Equation
```
dL/dt = τ
```
Where τ = Σᵢ rᵢ × Fᵢ is the total torque.

## Examples

See `examples/` directory for complete workflows:
- `torque_validation_example.py`: Angular momentum and torque analysis
- `pmf_analysis_example.py`: Free energy landscape computation
- `diffusion_analysis_example.py`: Rotational diffusion analysis
- `langevin_validation_example.py`: Model validation

## Citation

If you use this toolkit in your research, please cite:

```
@software{protein_orientation_toolkit,
  author = {Bobylyow, Mykyta},
  title = {Protein Orientation Analysis Toolkit},
  year = {2025},
  version = {0.1.0}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions and bug reports, please open an issue on GitHub.

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7
- MDAnalysis >= 2.0
- Matplotlib >= 3.5 (optional, for visualization)
- h5py >= 3.0 (optional, for HDF5 support)
- PyYAML >= 5.0 (optional, for YAML configs)

## Acknowledgments

Based on the theoretical framework from:
- Ivanov, I. et al. (2013) "Ionizing radiation..." J. Biol. Chem.
- Your research group's publications

Developed as part of the ST-Analyzer project for protein orientation analysis in molecular dynamics simulations.
