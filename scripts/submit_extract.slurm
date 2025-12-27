#!/bin/bash
#SBATCH --job-name=rotmd_extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/rotmd_extract_%j.out
#SBATCH --error=logs/rotmd_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

# Set up environment
module load gcc/11.3.0
module load python/3.12
source /path/to/venv/bin/activate  # Adjust to your virtualenv

# Create output directory
mkdir -p wtextract logs

# Run extraction with chunked processing for large files
echo "Starting rotmd extraction job..."
echo "Topology: ~/hotarchive/archive/wtZYZorient/wtcomplex.tpr"
echo "Trajectory: ~/hotarchive/archive/wtZYZorient/wtcmplx_dt100.trr"
echo "Output: wtextract/data.npz"
echo ""

rotmd ~/hotarchive/archive/wtZYZorient/wtcomplex.tpr \
      ~/hotarchive/archive/wtZYZorient/wtcmplx_dt100.trr \
      -d wtextract \
      -f data.npz \
      --chunked \
      --chunk-size 1000 \
      --step 10

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ Job completed successfully"
    ls -lh wtextract/
else
    echo "✗ Job failed with status $?"
    exit 1
fi
