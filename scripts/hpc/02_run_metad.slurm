#!/bin/bash
#SBATCH --job-name=metad_orient
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/metad_%j.out
#SBATCH --error=logs/metad_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com

# =============================================================================
# GROMACS + PLUMED Metadynamics for Protein Orientation
# =============================================================================

set -e

# Configuration
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TPR="topol.tpr"
CPT="${1:-}"  # Optional checkpoint for restart

echo "============================================="
echo "GROMACS + PLUMED Metadynamics"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working dir: $WORKDIR"
echo "Start time: $(date)"
echo ""

cd "$WORKDIR"
mkdir -p logs

# Load modules
module purge
module load gromacs/mpi-plumed_2024.3_gcc

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_MAXBACKUP=-1  # Disable backups

# Print environment
echo "Modules loaded:"
module list 2>&1
echo ""
echo "PLUMED version:"
plumed --version
echo ""

# Check required files
for f in "$TPR" plumed.dat; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# Run command
if [[ -n "$CPT" && -f "$CPT" ]]; then
    echo "Restarting from checkpoint: $CPT"
    RESTART_OPTS="-cpi $CPT -append"
else
    echo "Starting new simulation"
    RESTART_OPTS=""
fi

echo "Starting GROMACS+PLUMED..."
echo ""

mpirun -np $SLURM_NTASKS gmx_mpi mdrun \
    -s "$TPR" \
    -deffnm metad \
    -plumed plumed.dat \
    -ntomp $OMP_NUM_THREADS \
    -nb gpu \
    -pme gpu \
    -bonded gpu \
    -update gpu \
    -pin on \
    -v \
    $RESTART_OPTS

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "============================================="
    echo "Simulation completed successfully"
    echo "End time: $(date)"
    echo "============================================="

    # Quick analysis
    echo ""
    echo "COLVAR summary:"
    tail -5 COLVAR

    echo ""
    echo "HILLS count:"
    wc -l HILLS
else
    echo "ERROR: Simulation failed!"
    exit 1
fi
