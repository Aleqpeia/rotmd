#!/bin/bash
#SBATCH --job-name=fes_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/fes_%j.out
#SBATCH --error=logs/fes_%j.err

# =============================================================================
# FES Reconstruction and Analysis
# =============================================================================

set -e

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

module purge
module load gromacs/mpi-plumed_2024.3_gcc

mkdir -p analysis

echo "============================================="
echo "FES Analysis"
echo "============================================="
echo "Start time: $(date)"

# -----------------------------------------------------------------------------
# 1. Reconstruct FES from HILLS
# -----------------------------------------------------------------------------
echo ""
echo "1. Reconstructing FES(z, theta)..."

plumed sum_hills \
    --hills HILLS \
    --outfile analysis/fes.dat \
    --mintozero \
    --bin 100,90 \
    --min -3.0,0 \
    --max 3.0,90

# FES at different times (convergence check)
echo "   Computing FES at different times..."
for stride in 100000 500000 1000000; do
    plumed sum_hills \
        --hills HILLS \
        --outfile "analysis/fes_${stride}.dat" \
        --mintozero \
        --stride $stride \
        --bin 100,90 \
        --min -3.0,0 \
        --max 3.0,90 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# 2. 1D projections
# -----------------------------------------------------------------------------
echo ""
echo "2. Computing 1D FES projections..."

# FES(z) - marginalize over theta
plumed sum_hills \
    --hills HILLS \
    --outfile analysis/fes_z.dat \
    --mintozero \
    --idw com_z \
    --bin 100 \
    --min -3.0 \
    --max 3.0

# FES(theta) - marginalize over z
plumed sum_hills \
    --hills HILLS \
    --outfile analysis/fes_theta.dat \
    --mintozero \
    --idw theta \
    --bin 90 \
    --min 0 \
    --max 90

# -----------------------------------------------------------------------------
# 3. Block analysis for error estimation
# -----------------------------------------------------------------------------
echo ""
echo "3. Block analysis..."

# Split HILLS into blocks
TOTAL_HILLS=$(wc -l < HILLS)
BLOCK_SIZE=$((TOTAL_HILLS / 5))

for i in 1 2 3 4 5; do
    START=$(( (i-1) * BLOCK_SIZE + 1 ))
    END=$(( i * BLOCK_SIZE ))
    sed -n "${START},${END}p" HILLS > "analysis/HILLS_block_${i}"

    plumed sum_hills \
        --hills "analysis/HILLS_block_${i}" \
        --outfile "analysis/fes_block_${i}.dat" \
        --mintozero \
        --bin 100,90 \
        --min -3.0,0 \
        --max 3.0,90 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# 4. Extract minimum energy structures
# -----------------------------------------------------------------------------
echo ""
echo "4. Finding FES minima..."

python3 << 'EOF'
import numpy as np

# Load FES
data = np.loadtxt('analysis/fes.dat')
z = data[:, 0]
theta = data[:, 1]
fes = data[:, 2]

# Reshape to 2D grid
n_z = len(np.unique(z))
n_theta = len(np.unique(theta))
Z = z.reshape(n_theta, n_z)
T = theta.reshape(n_theta, n_z)
F = fes.reshape(n_theta, n_z)

# Find global minimum
min_idx = np.argmin(fes)
z_min, theta_min, fes_min = z[min_idx], theta[min_idx], fes[min_idx]

print(f"Global minimum:")
print(f"  z = {z_min:.2f} nm")
print(f"  theta = {theta_min:.1f} deg")
print(f"  FES = {fes_min:.2f} kJ/mol")

# Find local minima (simple approach)
from scipy.ndimage import minimum_filter
local_min = (F == minimum_filter(F, size=5))
min_coords = np.where(local_min)

print(f"\nLocal minima found: {len(min_coords[0])}")
for i in range(min(5, len(min_coords[0]))):
    ti, zi = min_coords[0][i], min_coords[1][i]
    print(f"  z={Z[ti,zi]:.2f} nm, theta={T[ti,zi]:.1f} deg, FES={F[ti,zi]:.2f} kJ/mol")

# Save minima
np.savetxt('analysis/fes_minima.dat',
           np.column_stack([Z[local_min], T[local_min], F[local_min]]),
           header='z(nm) theta(deg) FES(kJ/mol)')
EOF

# -----------------------------------------------------------------------------
# 5. COLVAR statistics
# -----------------------------------------------------------------------------
echo ""
echo "5. COLVAR statistics..."

python3 << 'EOF'
import numpy as np

# Load COLVAR (skip header)
data = np.loadtxt('COLVAR', comments='#')
time = data[:, 0]
com_z = data[:, 1]
theta = data[:, 2]
phi = data[:, 3]

print(f"Trajectory length: {time[-1]/1000:.1f} ns")
print(f"Frames: {len(time)}")
print(f"\nCV statistics:")
print(f"  com_z: {com_z.mean():.3f} +/- {com_z.std():.3f} nm")
print(f"  theta: {theta.mean():.1f} +/- {theta.std():.1f} deg")
print(f"  phi:   {phi.mean():.1f} +/- {phi.std():.1f} deg")

print(f"\nCV ranges:")
print(f"  com_z: [{com_z.min():.3f}, {com_z.max():.3f}] nm")
print(f"  theta: [{theta.min():.1f}, {theta.max():.1f}] deg")
print(f"  phi:   [{phi.min():.1f}, {phi.max():.1f}] deg")

# Save for plotting
np.savez('analysis/colvar_stats.npz',
         time=time, com_z=com_z, theta=theta, phi=phi)
EOF

echo ""
echo "============================================="
echo "Analysis complete: $(date)"
echo "Results in: analysis/"
echo "============================================="
ls -lh analysis/
