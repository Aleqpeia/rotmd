# Protein Orientation Analysis - 3D Pendulum Model

This package implements a 3D pendulum model for analyzing peripheral membrane protein orientation from MD trajectories.

## Overview

The implementation models a peripheral membrane protein as a rigid body (3D pendulum) with restricted motion relative to the membrane plane. It extracts orientation angles, builds free energy landscapes, and identifies preferred tilt angles corresponding to energy minima.

## Features

- **Trajectory-based analysis**: Extract protein orientations from existing MD simulations
- **Free energy landscapes**: Calculate F(θ, φ) using Boltzmann weighting
- **Energy minima identification**: Automatically find and characterize preferred orientations
- **Comprehensive visualization**: Generate publication-quality plots
- **Membrane interface utilities**: Robust membrane plane detection
- **SASA integration**: Optional solvent accessible surface area calculation

## Core Components

### 1. `membrane_interface.py`
Utilities for defining membrane reference frames:
- Membrane center detection (density-based or leaflet-based)
- Membrane normal calculation
- Thickness estimation
- Density profiles

### 2. `orientation_analyzer.py`
Main analysis engine:
- `TrajectoryOrientationAnalyzer`: Extract orientations frame-by-frame
- `FreeEnergyLandscape`: Build F(θ, φ) and find minima
- `OrientationData`: Container for single-frame orientation data

### 3. `visualization.py`
Plotting functions:
- 2D free energy landscape F(θ, φ)
- Marginal distributions F(θ) and F(φ)
- Time series of orientation angles
- Tilt angle histograms
- SASA vs tilt angle correlations

## Usage

### Command Line Interface

```bash
stanalyzer protein_pendulum \
  --tpr system.tpr \
  --traj trajectory.xtc \
  --protein-sel "protein" \
  --membrane-sel "resname POPC POPE" \
  --start-frame 1000 \
  --interval 10 \
  --temperature 310.15 \
  --theta-bins 36 \
  --phi-bins 36 \
  --out hippocalcin_orientation.json
```

**Example: Skipping Equilibration**
```bash
# Skip first 1000 frames (e.g., 5 ns of equilibration if dt=5ps)
stanalyzer protein_pendulum \
  --tpr system.tpr \
  --traj trajectory.xtc \
  --protein-sel "protein" \
  --membrane-sel "resname POPC POPE" \
  --start-frame 1000 \
  --interval 50 \
  --out production_only.json
```

### Parameters

**Required:**
- `--protein-sel`: Selection string for protein (MDAnalysis syntax)
- `--membrane-sel`: Selection string for membrane lipids

**Optional:**
- `--start-frame`: Starting frame for analysis (default: 0, useful for skipping equilibration)
- `--interval`: Frame sampling interval (default: 1)
- `--temperature`: Temperature in Kelvin (default: 310.15)
- `--theta-bins`: Number of bins for tilt angle (default: 36 = 5° bins)
- `--phi-bins`: Number of bins for rotation angle (default: 36 = 10° bins)
- `--calculate-sasa`: Enable SASA calculation (slow)
- `--no-plots`: Disable automatic plot generation
- `--plot-dir`: Directory for output plots

### Python API

```python
from stanalyzer.analysis.protein_pendulum import run_protein_pendulum

run_protein_pendulum(
    protein_sel="protein",
    membrane_sel="resname POPC POPE",
    psf="system.tpr",
    traj="trajectory.xtc",
    interval=10,
    temperature=310.15,
    out="result.json"
)
```

## Output Format

### JSON Structure

```json
{
  "analysis_type": "protein_pendulum",
  "metadata": {
    "protein_selection": "protein",
    "membrane_selection": "resname POPC POPE",
    "frames_analyzed": 1000,
    "interval": 10,
    "temperature_K": 310.15
  },
  "membrane_properties": {
    "center_z": 0.0,
    "normal": [0, 0, 1],
    "thickness": 40.5,
    "box_dimensions": {...}
  },
  "data": [
    {
      "frame": 0,
      "time_ps": 0.0,
      "theta": 25.3,
      "phi": 145.2,
      "z": -22.1,
      "com_x": 50.1,
      "com_y": 48.3,
      "com_z": -22.1,
      "principal_axis": [0.2, 0.1, 0.97],
      "sasa": 12500.0
    }
  ],
  "free_energy_surface": {
    "theta_bins": [...],
    "phi_bins": [...],
    "free_energy_matrix": [[...]],
    "histogram": [[...]],
    "temperature_K": 310.15,
    "n_samples": 1000
  },
  "energy_minima": [
    {
      "theta": 25.0,
      "phi": 145.0,
      "free_energy": 0.0,
      "population": 0.35,
      "prominence": 2.5,
      "representative_frames": [10, 25, 89]
    }
  ],
  "summary": {
    "dominant_tilt_angle": 25.0,
    "mean_tilt_angle": 26.3,
    "std_tilt_angle": 5.2,
    "median_tilt_angle": 25.8,
    "mean_z_distance": -22.1,
    "num_minima": 3
  }
}
```

### Generated Plots

1. **`protein_orientation_landscape.png`**: 2D free energy contour plot F(θ, φ) with marked minima
2. **`protein_orientation_marginals.png`**: 1D profiles F(θ) and F(φ)
3. **`protein_orientation_timeseries.png`**: Time evolution of θ(t), φ(t), and z(t)
4. **`protein_orientation_distribution.png`**: Histogram of tilt angles
5. **`protein_orientation_sasa_vs_tilt.png`**: SASA correlation (if calculated)

## Theory

### 3D Pendulum Model

The protein is modeled as a rigid body with orientation defined by:

- **θ (theta)**: Tilt angle from membrane normal (0-180°)
  - Calculated as angle between protein principal axis and membrane normal
  - 0° = aligned with membrane normal, 90° = parallel to membrane

- **φ (phi)**: Rotation angle around membrane normal (0-360°)
  - Azimuthal angle of principal axis projection onto membrane plane

- **z**: Distance from membrane center (signed)
  - Positive = above membrane center
  - Negative = below membrane center

### Principal Axis Calculation

The protein orientation is determined by its principal axis, calculated from the moment of inertia tensor:

```
I = Σ (r_i ⊗ r_i)
```

The principal axis is the eigenvector corresponding to the smallest eigenvalue (long axis).

### Free Energy Calculation

Free energy is calculated using Boltzmann weighting:

```
P(θ, φ) = N(θ, φ) / Σ N(θ, φ)
F(θ, φ) = -kT ln(P(θ, φ))
```

Where:
- `N(θ, φ)` = histogram count at (θ, φ)
- `k` = Boltzmann constant (0.001987 kcal/(mol·K))
- `T` = temperature (K)

## EF-Hand Domain Analysis

For proteins with multiple domains (like hippocalcin with 4 EF-hands), analyze each separately:

```bash
# EF-hand 1 (example residue range)
stanalyzer protein_pendulum \
  --protein-sel "protein and resid 20-50" \
  --membrane-sel "resname POPC POPE" \
  --out efhand1_orientation.json

# EF-hand 2
stanalyzer protein_pendulum \
  --protein-sel "protein and resid 60-90" \
  --membrane-sel "resname POPC POPE" \
  --out efhand2_orientation.json
```

## Future Extensions

### Phase 2: Energy Terms (Planned)
- Hydrophobic energy (SASA-based)
- Electrostatic interactions
- Membrane deformation penalty

### Phase 3: Monte Carlo Sampling (Planned)
- Grid-based exploration of orientation space
- Umbrella sampling for high-resolution PMF
- Independent validation of trajectory-based results

### Phase 4: Deformable Body Model (Planned)
- Configuration-dependent SASA
- Side-chain conformational effects
- Ensemble-averaged landscapes

### Phase 5: Mutation Analysis (Planned)
- Single amino acid mutation significance
- ΔΔG estimation
- Predictive modeling for mutant orientations

## Performance

**Typical performance on workstation:**
- Trajectory loading: ~1-5 seconds
- Orientation extraction: ~10-50 ms/frame (without SASA)
- Orientation extraction: ~1-2 s/frame (with SASA)
- Free energy calculation: ~0.1-1 seconds
- Plot generation: ~2-5 seconds

**Recommendations:**
- Use `--interval` to sample large trajectories (e.g., every 10th frame)
- Skip SASA for initial exploratory analysis
- For 10,000 frame trajectory with interval=10: ~1-3 hours total

## Dependencies

- **MDAnalysis**: Trajectory handling
- **NumPy**: Numerical operations
- **SciPy**: Signal processing (minima detection, smoothing)
- **Matplotlib**: Plotting
- **freesasa** (optional): SASA calculation

## References

### Theoretical Background
- Pendulum model for membrane proteins
- Principal component analysis for molecular orientation
- Free energy from probability distributions

### Related Work
- Helix tilt analysis in `helix_tilt_rotation_angle.py`
- Cholesterol tilt in `chol_tilt.py`
- Membrane thickness in `thickness.py`

## Authors

Implemented as part of st-analyzer fork for peripheral membrane protein orientation analysis, specifically targeting hippocalcin and other neuronal calcium sensors.

## License

Same as parent st-analyzer project.
