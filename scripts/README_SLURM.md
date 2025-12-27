# SLURM Execution Guide for rotmd

Quick reference for running rotmd extract on HPC clusters with SLURM.

## Quick Start

### Option 1: srun (Interactive, Best for Testing)
```bash
srun --job-name=rotmd_extract \
     --partition=gpu \
     --cpus-per-task=16 \
     --time=02:00:00 \
     rotmd ~/hotarchive/archive/wtZYZorient/wtcomplex.tpr \
           ~/hotarchive/archive/wtZYZorient/wtcmplx_dt100.trr \
           -d wtextract \
           -f data.npz
```

### Option 2: sbatch (Batch Job, Best for Large Runs)
```bash
sbatch scripts/submit_extract.slurm
```

Check status:
```bash
squeue -j <job_id>
tail -f logs/rotmd_extract_*.out
```

### Option 3: Array Job (Multiple Files)
```bash
sbatch scripts/submit_extract_array.slurm
```

Monitor all tasks:
```bash
squeue -j <job_id>
ls -lh wtextract_*/
```

---

## Common SLURM Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--job-name` | `rotmd_extract` | Job identifier in queue |
| `--partition` | `gpu` | Change to `cpu`, `mem`, etc. based on cluster |
| `--cpus-per-task` | `16` | Adjust to available cores (numba uses all) |
| `--time` | `02:00:00` | Wall-clock limit (HH:MM:SS) |
| `--nodes` | `1` | Use 1 node (rotmd doesn't parallelize across nodes) |
| `--ntasks` | `1` | Single task (rotmd is single-process) |
| `--output` | `logs/...out` | Standard output file |
| `--error` | `logs/...err` | Standard error file |

---

## For Your 1TB Trajectory

### Recommended sbatch Settings
```bash
sbatch --job-name=rotmd_huge \
       --partition=gpu \
       --cpus-per-task=16 \
       --time=24:00:00 \
       --mem=64G \
       scripts/submit_extract.slurm
```

**With chunked processing:**
```bash
rotmd topology.tpr trajectory.trr \
    -d results/ \
    -f huge.npz \
    --chunked \
    --chunk-size 2000 \
    --step 10
```

**Time estimate:**
- ~100M frames × 10-field = 1 billion frame operations
- Numba: ~500-1000 frames/sec on 16 cores
- Total: ~1-2 million seconds = 11-23 days
- **Recommendation:** Submit with `--time=30:00:00` and patience 🚀

---

## Workflow Examples

### Single File, Full Processing
```bash
sbatch --job-name=rotmd \
       --time=24:00:00 \
       --mem=64G \
       scripts/submit_extract.slurm
```

### Multiple Files in Parallel (Array Job)
```bash
# Edit scripts/submit_extract_array.slurm to add your files
# Then:
sbatch scripts/submit_extract_array.slurm
```

### Extract → Analyze Chain (Dependency)
```bash
# First submit extraction
JOB1=$(sbatch --parsable scripts/submit_extract.slurm)

# Then submit analysis (runs after extraction completes)
sbatch --dependency=afterok:$JOB1 scripts/analyze_results.slurm

echo "Extraction: $JOB1"
echo "Analysis will run after extraction"
```

---

## Debugging

### Check job status
```bash
squeue -u $USER
squeue -j <job_id>
```

### View output in real-time
```bash
tail -f logs/rotmd_extract_*.out
```

### Cancel job if needed
```bash
scancel <job_id>
scancel -u $USER  # Cancel all your jobs
```

### Check estimated time for job
```bash
sstat -j <job_id> --format=JobID,MaxVMSize,AveVMSize,AveCPU,TotalCPU
```

---

## Performance Tips

1. **Use chunked processing** for trajectories >100k frames
   ```bash
   --chunked --chunk-size 2000
   ```

2. **Don't oversample** - use `--step 10` or higher if possible
   ```bash
   --step 10  # Keep every 10th frame instead of every frame
   ```

3. **Allocate enough memory** - numba needs RAM for parallel processing
   ```bash
   --mem=64G  # For 1TB trajectory processing
   ```

4. **Use GPU partition if available** - faster for some operations
   ```bash
   --partition=gpu
   ```

5. **Request enough cores** - numba parallelizes across all available
   ```bash
   --cpus-per-task=16  # Or more if available
   ```

---

## Advanced: Custom Script

Create `run_extraction.sh`:
```bash
#!/bin/bash
set -e  # Exit on error

TOPOLOGY="$1"
TRAJECTORY="$2"
OUTPUT_DIR="${3:-results}"
CHUNK_SIZE="${4:-2000}"

echo "Extracting: $TRAJECTORY"
echo "Output: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE frames"

mkdir -p "$OUTPUT_DIR" logs

rotmd "$TOPOLOGY" "$TRAJECTORY" \
    -d "$OUTPUT_DIR" \
    -f data.npz \
    --chunked \
    --chunk-size "$CHUNK_SIZE" \
    --step 10

echo "✓ Completed"
```

Run with:
```bash
srun --cpus-per-task=16 --time=24:00:00 \
     scripts/run_extraction.sh \
     wtcomplex.tpr \
     wtcmplx_dt100.trr \
     wtextract \
     2000
```

---

## References

- SLURM Documentation: https://slurm.schedmd.com/
- srun: https://slurm.schedmd.com/srun.html
- sbatch: https://slurm.schedmd.com/sbatch.html
- squeue: https://slurm.schedmd.com/squeue.html
