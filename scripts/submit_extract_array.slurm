#!/bin/bash
#SBATCH --job-name=rotmd_extract_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/rotmd_extract_array_%j_%a.out
#SBATCH --error=logs/rotmd_extract_array_%j_%a.err
#SBATCH --array=0-3

# Set up environment
module load gcc/11.3.0
module load python/3.12
source /path/to/venv/bin/activate

# Define trajectory list
declare -a TRAJECTORIES=(
    "~/hotarchive/archive/wtZYZorient/wtcmplx_dt100"
    "~/hotarchive/archive/wtZYZorient/wtcmplx_dt200"
    "~/hotarchive/archive/wtZYZorient/wtcmplx_dt300"
    "~/hotarchive/archive/wtZYZorient/wtcmplx_dt400"
)

# Get current trajectory from array index
TRAJ_BASE="${TRAJECTORIES[$SLURM_ARRAY_TASK_ID]}"
TRAJ_FILE="${TRAJ_BASE}.trr"
OUTPUT_DIR="wtextract_${SLURM_ARRAY_TASK_ID}"

echo "Processing trajectory: $TRAJ_FILE"
echo "Array task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR" logs

# Run extraction
rotmd ~/hotarchive/archive/wtZYZorient/wtcomplex.tpr \
      "$TRAJ_FILE" \
      -d "$OUTPUT_DIR" \
      -f "data_${SLURM_ARRAY_TASK_ID}.npz" \
      --chunked \
      --chunk-size 2000 \
      --step 10

# Monitor exit status
if [ $? -eq 0 ]; then
    echo "✓ Task $SLURM_ARRAY_TASK_ID completed"
    ls -lh "$OUTPUT_DIR/"
else
    echo "✗ Task $SLURM_ARRAY_TASK_ID failed"
    exit 1
fi
