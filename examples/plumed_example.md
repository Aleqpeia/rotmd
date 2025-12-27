# PLUMED Integration Examples

This guide shows how to use rotmd to generate PLUMED input files for enhanced sampling of protein orientation in GROMACS simulations.

## Quick Start

### Generate Basic PLUMED Input

```bash
# Generate plumed.dat for theta and psi CVs
rotmd plumed -o plumed.dat -a "1-500" --cvs theta psi

# Use in GROMACS
gmx mdrun -s topol.tpr -plumed plumed.dat
```

### With Metadynamics

```bash
# Enable metadynamics biasing
rotmd plumed -o plumed_metad.dat -a "1-500" \
    --cvs theta psi \
    --metad \
    --hills-height 1.5 \
    --hills-stride 1000
```

### With Wall Potentials

```bash
# Restrict theta to 20-160 degrees
rotmd plumed -o plumed_walls.dat -a "1-500" \
    --cvs theta \
    --wall-min 20 \
    --wall-max 160
```

## Complete Workflow

### 1. Run MD Simulation to Get Atom Indices

First, identify your protein atom indices:

```bash
# View atom indices in GRO file
grep -n "^ATOM" system.gro | head -20

# Or use VMD/PyMOL to get selection
```

### 2. Generate PLUMED Input

```python
from rotmd.io.plumed import generate_plumed_input

# Basic orientation tracking
generate_plumed_input(
    output_path='plumed.dat',
    protein_atoms='1-500',  # Adjust to your system
    cv_names=['theta', 'psi'],
    output_stride=100
)
```

### 3. Run GROMACS with PLUMED

```bash
# Energy minimization (optional)
gmx mdrun -v -deffnm em

# Equilibration with PLUMED (NVT)
gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt -plumed plumed.dat

# Production MD with metadynamics
gmx grompp -f md.mdp -c nvt.gro -p topol.top -o md.tpr
gmx mdrun -v -deffnm md -plumed plumed.dat
```

### 4. Analyze Results

```python
from rotmd.io.plumed import read_colvar_file
import numpy as np
import matplotlib.pyplot as plt

# Read COLVAR file
data = read_colvar_file('COLVAR')

# Plot time series
plt.figure(figsize=(10, 4))
plt.plot(data['time'], data['theta'])
plt.xlabel('Time (ps)')
plt.ylabel('θ (degrees)')
plt.title('Protein Tilt Angle')
plt.savefig('theta_timeseries.png')

# Statistics
print(f"Mean theta: {data['theta'].mean():.1f}°")
print(f"Std theta: {data['theta'].std():.1f}°")
```

## Advanced Examples

### Example 1: Membrane Protein Insertion

Track protein orientation during membrane insertion:

```python
from rotmd.io.plumed import generate_plumed_input

# Monitor tilt angle with walls to prevent complete insertion
generate_plumed_input(
    output_path='insertion_plumed.dat',
    protein_atoms='1-500',
    cv_names=['theta'],
    wall_theta_min=30.0,  # Prevent lying flat
    wall_theta_max=150.0,  # Prevent flip
    wall_kappa=200.0,  # Strong walls (kJ/mol/rad²)
    output_stride=50
)
```

**MDP file additions:**
```
; In your .mdp file
pull = yes
pull-ngroups = 1
pull-ncoords = 1
pull-group1-name = Protein
pull-coord1-type = umbrella
pull-coord1-geometry = direction
pull-coord1-dim = N N Y
pull-coord1-vec = 0 0 1
```

### Example 2: Metadynamics to Sample Orientation Space

Explore free energy landscape F(θ, ψ):

```python
from rotmd.io.plumed import generate_plumed_input

generate_plumed_input(
    output_path='metad_2d.dat',
    protein_atoms='1-500',
    cv_names=['theta', 'psi'],
    metadynamics=True,
    hills_height=1.2,  # kJ/mol
    hills_width={'theta': 5.0, 'psi': 10.0},  # degrees
    hills_stride=500,  # Every 500 steps = 1 ps
    output_stride=100
)
```

**Analysis:**
```bash
# After simulation, compute FES from HILLS file
plumed sum_hills --hills HILLS --mintozero --stride 500
```

```python
# Plot free energy surface
import numpy as np
import matplotlib.pyplot as plt

# Load FES
fes = np.loadtxt('fes.dat')
theta = fes[:, 0]
psi = fes[:, 1]
F = fes[:, 2]

# Reshape to 2D grid
n_theta = len(np.unique(theta))
n_psi = len(np.unique(psi))
theta_grid = theta.reshape(n_theta, n_psi)
psi_grid = psi.reshape(n_theta, n_psi)
F_grid = F.reshape(n_theta, n_psi)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(theta_grid, psi_grid, F_grid, levels=20, cmap='viridis')
plt.colorbar(label='F(θ,ψ) [kJ/mol]')
plt.xlabel('θ (degrees)')
plt.ylabel('ψ (degrees)')
plt.title('Free Energy Landscape')
plt.savefig('fes_2d.png', dpi=300)
```

### Example 3: Umbrella Sampling Along Theta

Sample specific tilt angles:

```python
# Generate series of PLUMED files for umbrella windows
import numpy as np
from rotmd.io.plumed import PlumedConfig, PlumedWriter

theta_windows = np.arange(20, 160, 10)  # 20, 30, ..., 150 degrees

for theta_0 in theta_windows:
    config = PlumedConfig(
        protein_selection='1-500',
        cv_names=['theta'],
        output_stride=100
    )

    writer = PlumedWriter(config)

    # Add custom restraint
    lines = writer._generate_header()
    lines.extend(writer._generate_group_definition())
    lines.extend(writer._generate_com_definition())
    lines.extend(writer._generate_gyration_definition())
    lines.extend(writer._generate_theta_cv())

    # Add restraint
    lines.extend([
        f"# Umbrella restraint at theta = {theta_0}°",
        f"RESTRAINT ...",
        f"    ARG=theta",
        f"    AT={theta_0}",
        f"    KAPPA=500.0",  # kJ/mol/rad²
        f"... RESTRAINT",
        ""
    ])

    lines.extend(writer._generate_print_statement())

    # Write file
    with open(f'plumed_theta_{theta_0:.0f}.dat', 'w') as f:
        f.write('\n'.join(lines))

print(f"Generated {len(theta_windows)} umbrella window PLUMED files")
```

**GROMACS workflow:**
```bash
# Run each window
for theta in 20 30 40 50 60 70 80 90 100 110 120 130 140 150; do
    gmx grompp -f md.mdp -c nvt.gro -p topol.top -o md_${theta}.tpr
    gmx mdrun -v -deffnm md_${theta} -plumed plumed_theta_${theta}.dat
done

# Analyze with WHAM
# Create wham_metadata.txt with paths to COLVAR files
# Run WHAM analysis
```

### Example 4: Custom Membrane Normal

For tilted membranes:

```python
from rotmd.io.plumed import PlumedConfig, PlumedWriter
import numpy as np

# Define tilted membrane normal (e.g., 30° from z-axis)
tilt = np.radians(30)
membrane_normal = np.array([np.sin(tilt), 0, np.cos(tilt)])

config = PlumedConfig(
    protein_selection='1-500',
    membrane_normal=membrane_normal,
    cv_names=['theta'],
    output_stride=100
)

writer = PlumedWriter(config)
writer.generate_plumed_input('plumed_tilted.dat')
```

## PLUMED File Structure

A typical generated `plumed.dat` looks like:

```plumed
# PLUMED input file for protein orientation analysis
# Generated by rotmd

RESTART

# Define protein atom group
protein: GROUP ATOMS=1-500

# Compute center of mass
com: COM ATOMS=protein

# Compute gyration tensor for principal axes
gyration: GYRATION ATOMS=protein TYPE=GTPC_VECTOR
gyr_eig: GYRATION ATOMS=protein TYPE=GTPC

# Define orientation collective variables
# Theta: Tilt angle relative to membrane normal
nterm: COM ATOMS=1-50
cterm: COM ATOMS=450-500
paxis: DISTANCE ATOMS=nterm,cterm COMPONENTS
paxis_norm: MATHEVAL ARG=paxis.x,paxis.y,paxis.z FUNC=sqrt(x*x+y*y+z*z) PERIODIC=NO
dot_product: MATHEVAL ARG=paxis.x,paxis.y,paxis.z,paxis_norm FUNC=(0.000*x+0.000*y+1.000*z)/w PERIODIC=NO
theta: MATHEVAL ARG=dot_product FUNC=acos(x)*57.2958 PERIODIC=NO

# Metadynamics bias
METAD ...
    ARG=theta
    SIGMA=5.0
    HEIGHT=1.2
    PACE=500
    TEMP=300.0
    BIASFACTOR=10
    FILE=HILLS
... METAD

# Output collective variables
PRINT ARG=theta STRIDE=100 FILE=COLVAR FMT=%12.6f
```

## Tips and Best Practices

### 1. Atom Selection
- Include **all protein atoms** for accurate inertia tensor
- Use N-terminal and C-terminal groups that span the protein
- Avoid including membrane or solvent atoms

### 2. CV Selection
- **Theta alone**: For simple tilt angle tracking
- **Theta + Psi**: For full orientation (recommended)
- **All three (φ, θ, ψ)**: For complete 3D orientation

### 3. Metadynamics Parameters
- **hills_height**: 1-2 kJ/mol for proteins (adjust based on barrier height)
- **hills_width**: ~5° for theta, ~10° for phi/psi
- **hills_stride**: Deposit every 1-2 ps (500-1000 MD steps)
- **biasfactor**: 10-20 for well-tempered metadynamics

### 4. Wall Potentials
- Use walls to prevent unphysical orientations
- Soft walls: kappa = 50-100 kJ/mol/rad²
- Hard walls: kappa = 200-500 kJ/mol/rad²

### 5. Simulation Length
- **Equilibrium sampling**: 100-500 ns
- **Metadynamics**: 50-200 ns (until FES converges)
- **Umbrella sampling**: 20-50 ns per window

## Troubleshooting

### PLUMED errors

**"Action GYRATION: keyword TYPE has value GTPC_VECTOR which is not allowed"**
- Older PLUMED version, remove `TYPE=GTPC_VECTOR`

**"Cannot find atoms 1-500"**
- Check your protein has correct atom indices
- Use `gmx editconf -f system.gro -n` to create index file

### Slow convergence

- Increase `hills_height` for faster exploration
- Decrease `hills_width` for finer resolution
- Use well-tempered metadynamics with `BIASFACTOR`

### Unphysical orientations

- Add wall potentials at reasonable limits
- Check if protein is breaking apart (increase constraints)

## References

- PLUMED documentation: https://www.plumed.org
- PLUMED tutorials: https://www.plumed.org/doc-v2.8/user-doc/html/tutorials.html
- Metadynamics review: Barducci et al., WIREs Comput Mol Sci (2011)
- rotmd GitHub: https://github.com/yourusername/rotmd
