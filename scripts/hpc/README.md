# HPC Workflow: PLUMED Metadynamics for Protein Orientation

Enhanced sampling of peripheral protein orientation dynamics using GROMACS + PLUMED.

## Quick Start

```bash
# From your simulation directory (with md.tpr, md.cpt):
cd /path/to/your/simulation

# Copy workflow scripts
cp -r /home/efyis/projects/rotmd/scripts/hpc ./

# Run full workflow
./hpc/workflow.sh all md.tpr md.cpt
```

## Collective Variables

| CV | Description | Biased | Range |
|----|-------------|--------|-------|
| `com_z` | COM along membrane normal | Yes | -3 to 3 nm |
| `theta` | Tilt angle (principal axis vs z) | Yes | 20° to 90° |
| `phi` | Spin angle (precession) | **No** (tracked only) | -180° to 180° |

## Workflow Steps

### 1. Prepare
```bash
./hpc/workflow.sh prepare md.tpr md.cpt plumed_run
```
Creates `plumed_run/` with:
- `plumed.dat` - PLUMED input
- `reference.pdb` - Structure for MOLINFO
- `topol.tpr` - Extended TPR

### 2. Submit
```bash
cd plumed_run
./hpc/workflow.sh submit
```
Submits GPU-accelerated metadynamics job.

### 3. Monitor
```bash
./hpc/workflow.sh status

# Real-time output
tail -f logs/metad_*.out

# Quick COLVAR check
tail COLVAR
wc -l HILLS
```

### 4. Continue (if needed)
```bash
./hpc/workflow.sh continue
```
Automatically finds latest checkpoint.

### 5. Analyze
```bash
./hpc/workflow.sh analyze
```
Reconstructs FES from HILLS, computes:
- 2D FES(z, θ)
- 1D projections F(z), F(θ)
- Convergence analysis
- Minima identification

### 6. Plot
```bash
./hpc/workflow.sh plot
```
Generates publication-ready figures:
- `fes_2d.png` - 2D free energy surface
- `fes_1d.png` - 1D projections
- `colvar_timeseries.png` - CV time series
- `phi_dynamics.png` - Spin analysis (for torque-momentum model)

## Files Structure

```
plumed_run/
├── plumed.dat          # PLUMED input
├── reference.pdb       # Structure
├── topol.tpr           # GROMACS run input
├── metad.xtc           # Trajectory
├── metad.edr           # Energies
├── metad.cpt           # Checkpoint
├── COLVAR              # CV time series
├── HILLS               # Deposited hills
├── WALLS               # Wall restraint log
├── logs/               # SLURM logs
└── analysis/
    ├── fes.dat         # 2D FES
    ├── fes_z.dat       # F(z)
    ├── fes_theta.dat   # F(θ)
    ├── fes_minima.dat  # Minima coordinates
    └── *.png           # Plots
```

## Customization

### Adjust PLUMED parameters

Edit `hpc/plumed_orientation.dat`:

```plumed
# Change hill height/pace for faster/slower exploration
METAD ... HEIGHT=1.2 PACE=500 ...

# Adjust theta walls
LOWER_WALLS ARG=theta AT=20 ...  # Change to 10 for wider range
UPPER_WALLS ARG=theta AT=90 ...  # Change to 80 for narrower

# Change bias factor for well-tempered metadynamics
BIASFACTOR=15  # Higher = slower exploration, better convergence
```

### Adjust SLURM resources

Edit `hpc/02_run_metad.slurm`:

```bash
#SBATCH --time=72:00:00      # Wall time
#SBATCH --gres=gpu:1         # GPU count
#SBATCH --cpus-per-task=4    # Threads per MPI rank
#SBATCH --ntasks-per-node=4  # MPI ranks
```

## Expected Results

For your WT vs N75K comparison:

| System | Expected θ minimum | F(θ) barrier |
|--------|-------------------|--------------|
| WT | ~25° | Low (self-stabilizing) |
| N75K | ~56° | Higher (dissipative) |

The **φ dynamics** analysis will show:
- WT: damped oscillations (feedback loop active)
- N75K: higher spin frequency (less damping)

## Troubleshooting

### Job fails immediately
```bash
# Check PLUMED input
plumed driver --plumed plumed.dat --mf_xtc test.xtc --mc mc.dat
```

### FES doesn't converge
- Increase simulation time
- Reduce `HEIGHT` or increase `PACE`
- Check if system escapes CV range

### GPU errors
```bash
# Try CPU-only
#SBATCH --gres=gpu:0
mpirun gmx_mpi mdrun ... -nb cpu -pme cpu
```

## References

- PLUMED manual: https://www.plumed.org/doc
- Well-tempered metadynamics: Barducci et al., PRL 2008
- GROMACS-PLUMED: https://www.plumed.org/doc-v2.9/user-doc/html/gromacs.html
