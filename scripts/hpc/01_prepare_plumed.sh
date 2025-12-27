#!/bin/bash
# =============================================================================
# Step 1: Prepare PLUMED run from existing simulation
# =============================================================================
set -e

# Configuration
TPR_FILE="${1:-md.tpr}"
CPT_FILE="${2:-md.cpt}"
OUTPUT_DIR="${3:-plumed_run}"
NSTEPS="${4:-50000000}"  # 100 ns at 2fs timestep

echo "============================================="
echo "Preparing PLUMED enhanced sampling run"
echo "============================================="
echo "Input TPR: $TPR_FILE"
echo "Checkpoint: $CPT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Steps: $NSTEPS"
echo ""

# Load modules
module purge
module load gromacs/mpi-plumed_2024.3_gcc

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Copy PLUMED input
cp ../plumed_orientation.dat plumed.dat

# Extract reference structure for MOLINFO
echo "Extracting reference structure..."
gmx_mpi editconf -f "../${TPR_FILE}" -o reference.pdb 2>/dev/null || \
gmx editconf -f "../${TPR_FILE}" -o reference.pdb

# Extend/modify TPR for new run with more steps
echo "Preparing TPR for PLUMED run..."
gmx_mpi convert-tpr -s "../${TPR_FILE}" -extend 100000 -o topol.tpr 2>/dev/null || \
gmx_mpi grompp -f ../mdp/md_plumed.mdp -c "../${CPT_FILE%.cpt}.gro" -p ../topol.top -o topol.tpr -maxwarn 2

# Verify PLUMED input
echo "Validating PLUMED input..."
plumed driver --plumed plumed.dat --mf_xtc ../md.xtc --timestep 0.002 --trajectory-stride 1000 --mc mc.dat 2>&1 | head -20 || true

echo ""
echo "Preparation complete. Files in: $OUTPUT_DIR/"
echo "Next: sbatch 02_run_metad.slurm"
