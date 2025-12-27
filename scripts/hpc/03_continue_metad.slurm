#!/bin/bash
#SBATCH --job-name=metad_continue
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/metad_cont_%j.out
#SBATCH --error=logs/metad_cont_%j.err

# =============================================================================
# Continue PLUMED Metadynamics from checkpoint
# =============================================================================

set -e

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

module purge
module load gromacs/mpi-plumed_2024.3_gcc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_MAXBACKUP=-1

# Find latest checkpoint
CPT=$(ls -t metad*.cpt 2>/dev/null | head -1)

if [[ -z "$CPT" ]]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi

echo "Continuing from: $CPT"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

mpirun -np $SLURM_NTASKS gmx_mpi mdrun \
    -s topol.tpr \
    -deffnm metad \
    -plumed plumed.dat \
    -cpi "$CPT" \
    -append \
    -ntomp $OMP_NUM_THREADS \
    -nb gpu \
    -pme gpu \
    -bonded gpu \
    -update gpu \
    -pin on \
    -v

echo "Continuation complete: $(date)"
echo "HILLS count: $(wc -l < HILLS)"
