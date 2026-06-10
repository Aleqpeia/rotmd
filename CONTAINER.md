# Container guide — rotmd

This document covers:

1. Building the Docker image
2. Running the CLI with Docker (local development/testing)
3. Converting the image for Apptainer/Singularity (HPC)
4. Thread-count binding in SLURM jobs
5. Future JupyterLab-as-library path

---

## 1. Build the Docker image

```bash
# From the repo root
docker build \
  --target runtime \
  -t rotmd:latest \
  .

# Pin a specific tag for reproducible HPC use
docker build --target runtime -t rotmd:0.1.0 .
```

The multi-stage build has three targets:

| Target    | Purpose                                    | Ships to HPC? |
|-----------|--------------------------------------------|---------------|
| `builder` | Intermediate; has compiler/headers         | No            |
| `runtime` | Production CLI image                       | Yes           |
| `dev`     | Extends runtime with pytest/ruff/mypy etc. | No            |

The default `docker build` without `--target` builds up to the last stage
defined in the file (`dev`), which is heavier. Always specify
`--target runtime` for the image you push or convert for Apptainer.

---

## 2. Run locally with Docker

Trajectories and topology files must be bind-mounted in. Never copy data
into the image — MD trajectories can be hundreds of gigabytes.

```bash
# Extract a chunk
docker run --rm \
  -v /path/to/data:/data:ro \
  -v /path/to/output:/work \
  rotmd:latest extract \
    /data/system.gro \
    /data/traj_chunk.xtc \
    --reference /data/reference.gro \
    -o /work/chunk_001.npz

# Merge chunks
docker run --rm \
  -v /path/to/output:/work \
  rotmd:latest merge \
    /work/chunk_*.npz \
    -o /work/merged.npz
```

`/work` is the default `WORKDIR` inside the container, making it a natural
mount point for output files.

---

## 3. HPC: Apptainer/Singularity

HPC clusters typically run Apptainer (formerly Singularity) and do not have
a Docker daemon. Convert the image before transfer or pull it directly from
a registry.

### 3a. Build .sif from a local Docker image

On a machine that has Docker and Apptainer installed:

```bash
# Build the Docker image first
docker build --target runtime -t rotmd:latest .

# Convert to a Singularity Image Format file
apptainer build rotmd.sif docker-daemon://rotmd:latest
```

Transfer `rotmd.sif` to the cluster (scp/rsync).

### 3b. Build .sif directly from a registry

If you push the image to Docker Hub or GHCR:

```bash
# Push (from a machine with Docker)
docker tag rotmd:latest ghcr.io/<org>/rotmd:0.1.0
docker push ghcr.io/<org>/rotmd:0.1.0

# On the HPC login node (no Docker needed)
apptainer pull rotmd.sif docker://ghcr.io/<org>/rotmd:0.1.0
# or
apptainer build rotmd.sif docker://ghcr.io/<org>/rotmd:0.1.0
```

### 3c. Running under Apptainer

```bash
# Basic: bind data directory read-only, output directory read-write
apptainer exec \
  --bind /scratch/user/data:/data:ro \
  --bind /scratch/user/results:/work \
  rotmd.sif \
  rotmd extract \
    /data/system.gro \
    /data/traj_chunk.xtc \
    --reference /data/reference.gro \
    -o /work/chunk_001.npz
```

`apptainer run rotmd.sif` invokes the container's default CMD (`rotmd --help`).
`apptainer exec rotmd.sif <cmd>` bypasses the ENTRYPOINT and runs `<cmd>` directly.
For the CLI use `exec`; the `PATH` inside the image is set so `rotmd` is found
without any activation step.

---

## 4. Thread-count binding in SLURM

numba's parallel kernels (and the OpenBLAS-backed numpy/scipy) will try to
use all cores by default. On shared HPC nodes this causes oversubscription
and degrades performance for yourself and other users.

**Always set these three variables to match `$SLURM_CPUS_PER_TASK`:**

```bash
# In your SLURM job script (after #SBATCH --cpus-per-task=N):
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

With Apptainer, pass them as `--env` flags or export before calling:

```bash
apptainer exec \
  --env NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  --env OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  --env OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  --bind /scratch/user/data:/data:ro \
  --bind /scratch/user/results:/work \
  rotmd.sif \
  rotmd extract ...
```

### Example SLURM array job

```bash
#!/bin/bash
#SBATCH --job-name=rotmd_extract
#SBATCH --array=0-23          # 24 trajectory chunks
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

CHUNK=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
DATA=/scratch/$USER/md_data
RESULTS=/scratch/$USER/results

apptainer exec \
  --env NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  --env OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  --env OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  --bind ${DATA}:/data:ro \
  --bind ${RESULTS}:/work \
  rotmd.sif \
  rotmd extract \
    /data/system.gro \
    /data/traj_${CHUNK}.xtc \
    --reference /data/reference.gro \
    -o /work/chunk_${CHUNK}.npz
```

After all array tasks complete:

```bash
apptainer exec \
  --bind ${RESULTS}:/work \
  rotmd.sif \
  rotmd merge /work/chunk_*.npz -o /work/merged.npz
```

---

## 5. Future: JupyterLab as library

The `rotmd` package is installed into the system Python with no virtual-env
wrapping, so it is importable anywhere `python3` reaches:

```python
# Inside any Python process that has the same interpreter
import rotmd
from rotmd.core.kernels import angular_momentum_kernel
```

To add a JupyterLab layer:

1. Install the `jupyter` poetry group on top of the runtime image:
   ```dockerfile
   FROM rotmd:latest AS jupyter
   RUN pip install --no-cache-dir \
       "jupyterlab>=4.0" "ipywidgets>=8.1" "nglview>=3.1"
   EXPOSE 8888
   ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
   ```
   Or use `poetry install --with jupyter` if you add it as a second stage.

2. On HPC with Apptainer, bind the data and start the server:
   ```bash
   apptainer exec \
     --bind /scratch/$USER:/work \
     rotmd-jupyter.sif \
     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```
   Then forward port 8888 from the compute node to your laptop via SSH tunnel.

The `jupyter` optional group is already declared in `pyproject.toml`
(`jupyterlab ^4.0`, `ipywidgets ^8.1`, `nglview ^3.1`).

---

## Design notes

**Why `python:3.12-slim` + poetry (not micromamba/conda-forge)?**

The committed `poetry.lock` (Poetry 2.3.4, Python `>=3.10,<3.14`) is the
reproducibility anchor for this project. Using it directly via
`poetry install --only main` guarantees exactly the versions that have been
tested. A conda-forge approach would require manually mirroring pins and
losing this guarantee.

The two packages that are sdist-only on Linux (`pytim 1.0.6` and
`freesasa 2.2.1`) are compiled in the builder stage with `build-essential`.
All other packages (`numba 0.62.1`, `MDAnalysis 2.10.0`, `scipy`, etc.) have
`manylinux` wheels in the lock, so they install quickly. The multi-stage
build strips the compiler from the runtime image.

`pytim` is **not** on conda-forge under its own name (it ships only a
`.tar.gz` sdist on PyPI), which makes a pure conda-forge approach impossible
without a separate pip install step — messier than `slim + poetry`.
