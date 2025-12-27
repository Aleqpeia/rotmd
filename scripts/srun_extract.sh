#!/bin/bash
# Quick SLURM execution templates for rotmd extract

# ============================================================================
# OPTION 1: Direct srun command (fastest for interactive use)
# ============================================================================
# Run this for small files (<10k frames):
srun --job-name=rotmd_extract \
     --partition=gpu \
     --cpus-per-task=16 \
     --time=01:00:00 \
     rotmd ~/hotarchive/archive/wtZYZorient/wtcomplex.tpr \
           ~/hotarchive/archive/wtZYZorient/wtcmplx_dt100.trr \
           -d wtextract \
           -f data.npz \
           --step 10

# ============================================================================
# OPTION 2: srun with chunked processing (for large files, 1TB+)
# ============================================================================
# Run this for huge trajectories:
srun --job-name=rotmd_extract_chunked \
     --partition=gpu \
     --cpus-per-task=16 \
     --time=12:00:00 \
     rotmd ~/hotarchive/archive/wtZYZorient/wtcomplex.tpr \
           ~/hotarchive/archive/wtZYZorient/wtcmplx_dt100.trr \
           -d wtextract \
           -f data.npz \
           --chunked \
           --chunk-size 2000 \
           --step 10

# ============================================================================
# OPTION 3: Low-priority background job (sbatch)
# ============================================================================
# Use this for long runs that don't need immediate results:
sbatch --job-name=rotmd_extract \
       --partition=gpu \
       --cpus-per-task=16 \
       --time=24:00:00 \
       --output=logs/extract_%j.log \
       scripts/submit_extract.slurm

# ============================================================================
# OPTION 4: Array job (process multiple trajectories in parallel)
# ============================================================================
# For processing multiple files at once:
sbatch --job-name=rotmd_array \
       --partition=gpu \
       --cpus-per-task=16 \
       --time=24:00:00 \
       --array=0-3 \
       scripts/submit_extract_array.slurm

# ============================================================================
# OPTION 5: Dependency chains (run post-processing after extraction)
# ============================================================================
# Submit extraction, then analysis when it's done:
JOB_ID=$(sbatch --parsable scripts/submit_extract.slurm)
echo "Submitted extraction job: $JOB_ID"
sbatch --dependency=afterok:$JOB_ID scripts/analyze_results.slurm
