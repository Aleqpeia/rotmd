# The pipeline

rotmd's pipeline is a DAG of small, single-purpose stages, each reading the
artifact the previous one wrote rather than re-deriving it. That's a
deliberate constraint (see `rotmd_analysis_plan.md` in the repo root for the
full design rationale): re-reading a raw trajectory is the expensive
operation, so every stage after `extract` works from a compact `.npz` +
JSON sidecar instead.

```{code-block} text
extract (per chunk) ──> merge ──> equilibrate ──> window.json
                                                      │
                        ┌─────────────────┬───────────┼──────────────┬──────────────┐
                        v                 v            v              v              v
                      dccm              dssp         local           apbs        coulomb
                        │                 │            │              │              │
                        └─────────────────┴─────dcc/dssp of two systems───────────────┘
                                                      │
                                                      v
                                                  compare (ΔDCCM, effect sizes)

methods  (independent — reads .mdp/topology directly, no window needed)
```

## extract — per-frame observables from a trajectory chunk

One call processes one trajectory chunk (the natural unit of a SLURM array
job) and writes one `.npz`. Per frame it computes, via {mod}`rotmd.core` and
{mod}`rotmd.observables`: the inertia tensor, principal axes/moments, ZYZ
Euler angles and membrane tilt, angular momentum/torque/angular velocity
(parallel/perpendicular/z decomposed), RMSD (whole-protein and per-domain —
see {mod}`rotmd.analysis.domains`), radius of gyration and shape parameters,
and — when the trajectory carries velocities and forces — kinetic energy and
(if `freesasa` is available) polar/nonpolar energetics. Optional stages
degrade rather than fail: a positions-only `.xtc` still gives orientation,
structural, and CA output.

`ca_coords` is written raw (COM-centered but **not** rotationally aligned) —
alignment is a downstream choice, because different analyses want different
references (e.g. DCCM aligns on a stable core to avoid leaking rigid-body
rotation into the correlation map; per-domain RMSD self-superposes each
domain independently).

## merge — concatenate chunks

Chunks concatenate along time into one replica-level `.npz`. Purely
mechanical, no physics — kept as a separate command so the SLURM-array unit
of work (one chunk) doesn't have to match the analysis unit (one replica).

## equilibrate — decide the production window

Every later analysis needs to know which frames are actual equilibrium
sampling and which are still relaxing from the starting structure.
`equilibrate` detects the production start `t0` on a scalar order parameter
(default the backbone RMSD) via `pymbar.timeseries.detect_equilibration`,
which also returns `g` (the statistical inefficiency — how many frames make
up one effectively-independent sample, which drives block sizes for every
bootstrap error bar computed downstream) and `n_eff`. The result is written
to `window.json`, the single source of truth: **no analysis stage reads a
different window than the one `equilibrate` decided**, and none of them
re-derive it independently. `--pool` combines several replicas' windows into
one conservative common window when systems need to be length-matched before
comparison.

## The analysis stages

All of these slice `time_ps >= t0_ps` from `window.json` and, except for
`dssp` and `local`'s heavy-atom neighbor search (which need atoms beyond CA),
never re-read the trajectory — see {mod}`rotmd.analysis` for why that
boundary is drawn where it is.

- **`dccm`** ({mod}`rotmd.analysis.dccm`) — CA-CA displacement correlations
  after RMS-fitting every frame to the average structure over the stable
  alignment core. Misalignment here leaks rigid-body rotation straight into
  the correlation map, which matters more for rotmd than most MD analysis
  tools since rotational dynamics is the whole point.
- **`dssp`** ({mod}`rotmd.analysis.dssp`) — per-frame secondary structure via
  MDAnalysis's pure-Python DSSP, reduced to per-residue occupancy. Occupancy
  rows sum to exactly 1 by construction. One of only two stages that must
  re-read the trajectory (backbone N/C/O atoms aren't in the CA-only extract
  output).
- **`local`** — salt-bridge occupancy, local hydrogen-bond network, and
  windowed RMSF around a residue of interest (e.g. a mutation site) — the
  direct mechanistic readout of a local sequence change.
- **`apbs`** — ensemble Poisson-Boltzmann electrostatics (PDB2PQR + APBS)
  averaged over sampled frames from the window; needs external `pdb2pqr30`
  and `apbs` binaries.
- **`coulomb`** — per-residue Coulomb/LJ decomposition via
  `gmx mdrun -rerun` with `energygrps` set, giving site-shell interaction
  energies with block-averaged errors (block size from the window's `g`).

## compare — is the difference real?

`compare` ({mod}`rotmd.analysis.compare`) answers "what did the mutation (or
condition change) actually change?", with uncertainty attached rather than as
a bare point estimate. Per-system replicas are pooled in Fisher-z space (a
plain average of correlations is biased toward zero, worse for the strong
correlations the map is about — {func}`rotmd.analysis.compare.pool_matrices`
is why), the two pooled maps are differenced into ΔDCCM, and a block
bootstrap over `g`-sized blocks attaches a confidence interval to every cell.
A "distal" view masks residues near the site of interest so long-range,
allosteric rewiring isn't drowned out by the (expected, uninformative) local
signal. Scalar per-replica metrics get the same treatment via
{func}`rotmd.analysis.compare.effect_size_table` (Cohen's d with a bootstrap
CI).

## methods — provenance, not by hand

`methods` ({mod}`rotmd.analysis.methods`) is the one stage independent of the
window: it parses the `.mdp` files and topology that actually produced the
trajectories and renders both a machine-readable `methods.json` and a prose
`methods.md`. Generating the methods section from the inputs — rather than
transcribing it — removes the class of error where the write-up says one
timestep and the `.mdp` used another; a value absent from the `.mdp` is
reported as absent, never silently defaulted to what GROMACS would have used.

## The standalone dynamics toolkit

{mod}`rotmd.analysis.correlations`, {mod}`rotmd.analysis.friction`,
{mod}`rotmd.analysis.pmf`, {mod}`rotmd.analysis.transitions`, and
{mod}`rotmd.analysis.nonequilibrium` implement a second, related workflow —
autocorrelation functions and correlation times, orientation-dependent
friction γ(θ, ψ), potential-of-mean-force landscapes over Euler angles (with
the SO(3) Jacobian correction applied), transition-state kinetics, and
non-equilibrium diagnostics. These aren't wired into a `rotmd` subcommand yet;
they're used directly as a Python API against angular velocity / momentum
trajectories, most naturally the ones {mod}`rotmd.core` extracts per frame.
