# syntax=docker/dockerfile:1.9
# =============================================================================
# rotmd — multi-stage Dockerfile
#
# Stage 1 (builder): installs build toolchain + all Python dependencies via
#   poetry install --only main  (honours the committed poetry.lock exactly).
#   pytim and freesasa are sdist-only on Linux, so build-essential + Cython
#   are required here but stripped from the runtime image.
#
# Stage 2 (runtime): copies only the populated venv and source install; no
#   compiler, no headers, ~300 MB lighter than the builder.
#
# Stage 3 (dev): extends runtime with the dev dependency group and dev tools.
#   Used by .devcontainer/ — not deployed to HPC.
#
# HPC / Apptainer notes:
#   - No data is baked in. Bind-mount trajectories at runtime.
#   - Thread counts are NOT hardcoded. Set on the SLURM command line:
#       --env NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
#       --env OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#       --env OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
#   - The PATH line below makes `rotmd` findable when Apptainer exec bypasses
#     the Docker ENTRYPOINT (apptainer never runs ENTRYPOINT/CMD unless
#     `apptainer run` is used and the runscript is explicit).
# =============================================================================

# ── Pinned base ──────────────────────────────────────────────────────────────
# python:3.12-slim is a Debian bookworm-slim image.  We target 3.12 because:
#   - numba 0.62.1 ships manylinux wheels for cp312
#   - MDAnalysis 2.10.0 ships manylinux wheels for cp312
#   - pytim/freesasa are sdist-only; 3.12 has the best ABI stability window
#   - 3.13 is the riskier edge with numba + Cython extensions (not tested)
# Pin to 3.12 — do not parametrize further. If you change this, also update
# the COPY --from=builder paths below (they reference the version literally).
FROM python:3.12-slim AS builder

# ── Build toolchain (compile pytim + freesasa from sdist) ───────────────────
# Needed at build time only; stripped from the runtime stage.
#   - build-essential: gcc, g++, make — required for pytim (Cython/C) and
#     freesasa (C) which are sdist-only on Linux in the committed lock.
#   - git: defensive; some pip extras may need it for VCS deps.
# h5py ships manylinux wheels (bundled libhdf5) so no libhdf5-dev needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Poetry (build-time only, not shipped in runtime) ─────────────────────────
# Pinned to the same major version that generated the committed lock.
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "poetry==2.3.4"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# ── Install into the system Python (no venv — cleaner for copy-forward) ──────
# We copy the package source so poetry can build the editable install, then
# install only the main (non-dev) group from the committed lock.
WORKDIR /build
COPY pyproject.toml poetry.lock README.md ./
COPY src/ ./src/

# IMPORTANT: `poetry install --only main` would install the root package
# (rotmd itself) in editable mode — a `.pth` pointer to /build/src that
# becomes a dangling reference once we copy to the runtime stage.
#
# Fix: install all *dependencies* from the lock without the root package,
# then install rotmd itself as a real wheel (pip build via poetry-core).
# This puts actual .py files into site-packages, safe to COPY forward.
RUN poetry install --only main --no-root --no-interaction --no-ansi \
 && pip install --no-deps --no-cache-dir .

# =============================================================================
# Runtime stage — no compiler, no headers.
#
# NOTE: Poetry and its dependency tree (~40-80 MB: cleo, dulwich, keyring,
# tomlkit…) are present in the runtime image because Poetry was installed
# via pip into the system site-packages, which we COPY forward.  Poetry is
# not needed at runtime and its console_scripts are not on PATH, but the libs
# are present in site-packages.  This is acceptable given the total stack
# size (~1 GB scientific libraries); removing it would require a separate
# venv approach that is harder to verify without a build environment.
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="rotmd" \
      org.opencontainers.image.description="Orientational/angular-momentum analysis for membrane-protein MD" \
      org.opencontainers.image.source="https://github.com/efyis/rotmd"

# ── Minimal runtime system libs ──────────────────────────────────────────────
# libgomp1: OpenMP runtime required by numba parallel kernels and scipy/numpy
#           (libgomp is a shared library not bundled in any wheel)
# h5py 3.x wheels bundle their own libhdf5, so no system hdf5 is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy the full Python installation from builder ───────────────────────────
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Make the installed CLI discoverable regardless of activation state.
# This is critical for Apptainer: `apptainer exec rotmd.sif rotmd ...`
# bypasses the Docker ENTRYPOINT and never activates any environment, so
# the PATH must point directly to the installed scripts.
ENV PATH="/usr/local/bin:${PATH}"

# ── Thread-count defaults (can be overridden at runtime) ─────────────────────
# Do NOT hardcode to a number here — SLURM jobs must pass their allocation:
#   singularity exec --env NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK ...
# Setting 0 is "use all available" (numba default).  We leave these unset so
# the user's environment takes precedence.  Document override in CONTAINER.md.

# ── Working directory for I/O (bind-mount trajectories here) ─────────────────
WORKDIR /work

# ── Entrypoint ────────────────────────────────────────────────────────────────
# `rotmd` is the console_scripts entry point installed by poetry.
# Usage: docker run rotmd extract <topo> <traj> -o out.npz --reference <ref>
#        docker run rotmd merge chunk*.npz -o merged.npz
ENTRYPOINT ["rotmd"]
CMD ["--help"]

# =============================================================================
# Dev stage — extends runtime with the dev dependency group + dev tools.
# Used by .devcontainer; not pushed as the production image.
# The workspace is bind-mounted at /workspace at container start, so live
# source edits on the host are immediately visible inside the container.
# =============================================================================
FROM runtime AS dev

# Re-install build tools needed for dev installs (compiler for pytim/freesasa)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        vim \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "poetry==2.3.4"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Copy sources so poetry can install the dev group at image-build time.
# (The devcontainer bind-mount overlays /workspace at run time, but we bake
# the dev group in so the container is useful immediately on first open.)
WORKDIR /workspace
COPY pyproject.toml poetry.lock README.md ./
COPY src/ ./src/

# Install dev dependencies (not root — same --no-root pattern as builder),
# then install rotmd itself as an EDITABLE install.  Editable matters in the
# dev image: the workspace is bind-mounted over /workspace at run time, so the
# .pth pointer resolves to the live host src.  Any process — including the bare
# `python -c ...` subprocess that tests/test_kernels.py spawns, which does NOT
# inherit pytest's `pythonpath=src` — then imports the live source, so host
# edits are visible without a rebuild.  (Builder/runtime stages stay a real
# wheel; production has no src mount.)
RUN poetry install --with dev --no-root --no-interaction --no-ansi \
 && pip install --no-deps --no-cache-dir -e .

# Clear the inherited `rotmd` ENTRYPOINT (and its `--help` CMD) so the dev
# container runs commands directly instead of treating them as arguments.
#
# ENTRYPOINT [] resets the inherited entrypoint to nothing; CMD ["/bin/bash"]
# makes an interactive shell the *default* command. Crucially this keeps the
# command overridable:
#   podman run rotmd:dev               -> /bin/bash      (interactive shell)
#   podman run rotmd:dev pytest -q     -> pytest -q      (runs directly)
#   podman run rotmd:dev ruff check .  -> ruff check .   (runs directly)
# With ENTRYPOINT ["/bin/bash"] the args become a *script* for bash, so
# `pytest` is parsed as shell ("import: command not found") and `/bin/bash`
# is exec'd as a script ("cannot execute binary file").
ENTRYPOINT []
CMD ["/bin/bash"]