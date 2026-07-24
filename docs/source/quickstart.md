# Quickstart

## Installation

rotmd is managed with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/Aleqpeia/rotmd.git
cd rotmd
poetry install
```

This installs the core dependencies (NumPy, SciPy, MDAnalysis, numba, …). A
few CLI subcommands shell out to external binaries that Poetry can't install:
`apbs` needs `pdb2pqr30` and `apbs` on `PATH`, and `coulomb` needs a working
GROMACS (`gmx`) install.

To build this documentation site locally:

```bash
poetry install --with docs
poetry run sphinx-build -b html docs/source docs/build/html
```

then open `docs/build/html/index.html`.

## The CLI

Every `rotmd` subcommand operates on one artifact and writes one artifact —
see {doc}`pipeline` for how they chain together. A minimal run looks like:

```bash
# 1. Extract one trajectory chunk to .npz (one call per chunk/SLURM array task)
rotmd extract system.gro traj.xtc \
    --reference system.gro \
    -o chunks/rep0_000.npz

# 2. Concatenate chunks along time (skip if there's only one)
rotmd merge chunks/rep0_*.npz -o work/rep0.npz

# 3. Decide where equilibration ends
rotmd equilibrate work/rep0.npz -o work/rep0.window.json

# 4. Run an analysis over the post-equilibration window
rotmd dccm work/rep0.npz --window work/rep0.window.json -o analysis/rep0.dccm.npz
```

Other analysis subcommands follow the same `--window` convention:
`dssp` (secondary-structure occupancy), `local` (salt bridges / H-bonds / RMSF
around a residue of interest), `apbs` (Poisson-Boltzmann electrostatics),
`coulomb` (per-residue Coulomb/LJ decomposition), `compare` (ΔDCCM between two
systems with bootstrap significance), `methods` (auto-generated methods
section from `.mdp`/topology files), and `plot-equil` (equilibration
diagnostic figures). Run `rotmd <subcommand> --help` for the full flag list —
the {doc}`API reference <api/index>` documents the Python functions each one
calls.

## As a library

Everything the CLI does is also a plain function, importable directly — e.g.
the DCCM comparison the `compare` subcommand runs:

```python
from rotmd.analysis.compare import compare_dccm

result = compare_dccm(dcc_a, dcc_b, resids, site=75, exclude_within=5)
print(result["max_distal_change"])
```

`rotmd.analysis` also carries a standalone correlation/friction/PMF/transition
toolkit (`correlations.py`, `friction.py`, `pmf.py`, `transitions.py`,
`nonequilibrium.py`) that isn't wired into a CLI subcommand yet — see their
API reference pages for the angular-velocity ACF → friction → PMF workflow
they support.
