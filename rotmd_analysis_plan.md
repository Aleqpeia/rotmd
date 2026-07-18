# rotmd — Analysis Extension Plan (reviewer response, HPCA WT vs N75K)

**Purpose.** Address five reviewer comments by extending `rotmd` on the `slim-extract`
branch. This document is written for coding agents (Claude Code / equivalent). Treat each
**Task card** as an independent unit of work with explicit inputs, outputs, acceptance
criteria, and execution site (login vs compute node).

**Non-negotiable invariants (do not violate `slim-extract` philosophy):**
- Output is `.npz` only. No NetCDF, no torch. JSON sidecars for metadata only.
- RMSD and Rg stay always-on in `extract`.
- New analyses are separate subcommands that *consume* extracted `.npz` + `window.json`;
  they never re-read the trajectory unless the card explicitly says so (DSSP/electrostatics
  are the two exceptions — they need atoms beyond CA).
- Keep LOC lean. Prefer MDAnalysis + numpy over new heavy deps. Any new dependency must be
  justified in the PR description.

---

## 0. Traceability matrix (reviewer point → task)

| # | Reviewer comment (RU) | Deliverable | Task |
|---|------------------------|-------------|------|
| 1 | нет протоколов и описания методов | auto-generated `methods.{json,md}` from `.mdp`+topology | **T7** |
| 2 | нет графиков РМСД → эквилибрация | RMSD/Rg/per-domain plots + auto `t0` detection | **T2, T3** |
| 3 | не указан участок траектории для анализа | `window.json` per system, consumed by all analyses | **T3** |
| 4 | кросс-корреляционные карты, дистальные движения | DCCM (+LMI), ΔDCCM WT−mut with bootstrap significance | **T4, T6** |
| 5 | электростатика/локальные изменения + DSSP | salt-bridge/H-bond/RMSF, per-residue Coulomb, APBS PB; DSSP occupancy | **T5a–c, T8** |

---

## 1. Systems & data conventions

Two systems, each with `R` replicas:

- `wt`   — hippocalcin, wild type
- `n75k` — N75K mutant (Asn→Lys, neutral→+1; expect electrostatic + local restructuring)

Site of interest: `resid 75` and its EF-hand / myristoyl-pocket neighborhood.

**Path layout (agents must follow):**
```
runs/{system}/rep{r}/            # md.tpr, md.xtc (or chunked), *.mdp, topol.top
work/{system}/rep{r}/            # extracted chunks + merged npz
work/{system}/rep{r}/window.json # equilibration window (T3), SINGLE SOURCE OF TRUTH
analysis/{system}/               # dccm.npz, dssp.npz, elec.*  (per-replica then pooled)
compare/                         # ΔDCCM, effect sizes, figures
methods/                         # methods.json, methods.md  (T7)
```

**Metadata sidecar** (`work/.../meta.json`, one per replica) — written by `extract`,
extended by later tasks:
```json
{
  "system": "n75k", "replica": 2,
  "n_frames": 100000, "dt_ps": 10.0,
  "sel_ca": "protein and name CA", "n_ca": 191,
  "sel_align": "backbone and resid 20-90",     // stable core, NOT flexible termini
  "forcefield": "CHARMM36m", "water": "TIP3P",
  "window": {"t0_ps": 210000.0, "g": 3.4, "n_eff": 26411}  // filled by T3
}
```

---

## 2. Subcommand surface (target CLI)

```
rotmd extract   ...   # existing; extend arrays (T1)
rotmd merge     ...   # existing; chunk concat (unchanged, verify T1 arrays flow through)
rotmd equilibrate ... # NEW (T3) -> window.json
rotmd dccm      ...   # NEW (T4)
rotmd dssp      ...   # NEW (T5c/T8)
rotmd elec      ...   # NEW (T5a/T5b)
rotmd compare   ...   # NEW (T6)
rotmd methods   ...   # NEW (T7)
```

---

## 3. Task DAG (execution order & dependencies)

```
T1 extract-arrays ─┬─> T2 rmsd/rg-plots ──> T3 equilibrate ──> window.json
                   │                                         ├─> T4 dccm ──┐
                   │                                         ├─> T5a saltbridge/hbond/rmsf
                   │                                         ├─> T5b per-res Coulomb (rerun)
                   │                                         ├─> T8 dssp
                   │                                         └─> T5c APBS PB (optional/heavy)
T7 methods (independent, reads .mdp/top)                                    │
                                                     T4×(wt,n75k) ──> T6 compare (ΔDCCM, Cohen's d)
```
`window.json` is the join point: **no analysis (T4/T5/T8) may start before T3 for that system.**

---

## 4. Task cards

### T1 — Extend `extract` arrays
**Site:** compute (SLURM array over chunks). **Reads:** trajectory. **Writes:** chunk `.npz`.
- Keep always-on `rmsd_bb (F,)`, `rg (F,)`.
- Add `time_ps (F,)`, `ca_coords (F, n_ca, 3) float32` (raw, unaligned — alignment happens
  downstream so T4 and T2 can choose their own reference/selection).
- Add per-domain RMSD: `rmsd_dom (F, n_dom)` with domain masks in `meta.json`
  (EF1..EF4 / N-lobe / C-lobe). Domains addressing reviewer's "different segments →
  different results" spatially, complementing T3's temporal window.
- **Accept:** `merge` concatenates the new arrays without shape errors; RMSD/Rg identical
  to pre-change values on a regression trajectory (bit-for-bit on `rmsd_bb`).

### T2 — RMSD / Rg / per-domain plots  *(reviewer #2)*
**Site:** login. **Reads:** merged `.npz`. **Writes:** `analysis/{sys}/rep{r}/equil_*.png`.
- Plot `rmsd_bb`, `rg`, each `rmsd_dom` column vs `time_ps`. Overlay running mean + the
  detected `t0` (from T3) as a vertical line once available.
- Multi-replica overlay per system on one axis.
- **Accept:** one figure per replica + one pooled overlay per system; `t0` line rendered.

### T3 — `equilibrate` (window detection)  *(reviewer #2, #3)*
**Site:** login (cheap). **Reads:** merged `.npz`. **Writes:** `window.json`, updates `meta.json`.
- Detect production start `t0` on a scalar order parameter (default `rmsd_bb`; allow `rg`
  or a user column). Use `pymbar.timeseries.detect_equilibration(A_t)` →
  `(t0_idx, g, n_eff)`. `g` = statistical inefficiency (drives block size for all error bars).
- Cross-check with reverse-cumulative-average / block-average plateau; log both, prefer
  pymbar, warn on >20% disagreement.
- Report per replica; also emit a pooled decision (max `t0` across replicas for a
  conservative common window when systems must be length-matched).
- **Accept:** `window.json` has `t0_ps, g, n_eff`; downstream tasks read only from it;
  re-running is deterministic.

> Design note: T4/T5/T8 must slice frames as `time_ps >= t0_ps` from `window.json`. This is
> the mechanism that makes the analysis-segment choice explicit and reproducible (#3), and
> forces WT and mutant onto comparable, equilibrated windows.

### T4 — `dccm` (dynamic cross-correlation)  *(reviewer #4)*
**Site:** compute (short). **Reads:** merged `.npz` + `window.json`. **Writes:** `analysis/{sys}/rep{r}/dccm.npz`.
- CA-only. Slice window. RMS-fit each frame to the **average structure over `sel_align`**
  (stable core — this is essential; misalignment leaks rigid-body rotation into DCCM, and
  rotmd deals with rotational dynamics, so be explicit that alignment removes exactly that).
- Displacements `Δr_i(t) = r_i(t) − <r_i>`. Compute
  `C_ij = <Δr_i·Δr_j> / sqrt(<|Δr_i|²><|Δr_j|²>)`, shape `(n_ca, n_ca)`.
- Store `dcc (N,N)`, `resids (N,)`, and `frames_used`, `t0_ps` for provenance.
- **Optional LMI** (captures orthogonal correlated motion DCCM misses): compute via
  `correlationplus` if the dep is approved, else defer. Store as `lmi (N,N)`.
- **Accept:** `dcc` symmetric, diagonal ≈ 1, uses only post-`t0` frames.

### T5a — salt bridges / H-bonds / RMSF  *(reviewer #5, cheap, high-value for N→K)*
**Site:** login/compute (light). **Reads:** trajectory + `window.json`. **Writes:** `analysis/{sys}/rep{r}/local.npz` + JSON.
- RMSF per residue on the window (MDAnalysis `RMSF` after alignment).
- Salt bridges: K75 side-chain N (`resid 75 and name NZ`) within 4.0 Å of any
  Asp/Glu carboxylate O; report occupancy (%) and partner residues. For WT (Asn75) track
  its H-bond partners instead. This is the direct mechanistic readout of the charge change.
- H-bond network within 10 Å of `resid 75` (MDAnalysis `HydrogenBondAnalysis`); report
  gained/lost bonds WT vs mutant.
- **Accept:** occupancy tables per replica; residue indexing consistent with T4 `resids`.

### T5b — per-residue Coulomb decomposition  *(reviewer #5, medium)*
**Site:** compute. **Reads:** trajectory + tpr. **Writes:** `analysis/{sys}/rep{r}/coulomb.npz`.
- Use `gmx mdrun -rerun` on windowed frames with `energygrps` = {res75, shell residues,
  rest} to get short-range Coulomb/LJ contributions of residue 75 to its environment;
  parse with `gmx energy` / `panedr`. Report mean ± block error (block size from `g`).
- **Accept:** ΔE (mut−wt) reported with uncertainty; note this needs an energygrps rerun,
  so budget wall-time and disk.

### T5c — APBS Poisson–Boltzmann surface  *(reviewer #5, optional/heavy)*
**Site:** compute. **Reads:** representative/averaged frames. **Writes:** `analysis/{sys}/elec_pb.*` + figure.
- Sample ~50–100 windowed frames (or cluster and take medoids). PDB2PQR (CHARMM ff) → APBS
  linearized PB → per-residue electrostatic solvation/interaction energy + potential surface.
- Produce the ΔG_elec(mut−wt) and a potential-surface figure for the thesis.
- **Accept:** ensemble-averaged, not single-frame; report N frames used.

### T6 — `compare` (ΔDCCM + effect sizes)  *(reviewer #4, #5)*
**Site:** login. **Reads:** T4/T5 outputs for both systems. **Writes:** `compare/*.npz`, figures.
- Pool replicas per system (concatenate `dcc` via Fisher-z average, or average matrices;
  document choice). Length-match windows across systems before pooling.
- `ddccm = dcc[n75k] − dcc[wt]`; render heatmap + a "distal changes" view masking the
  `|resid_i − 75| < k` band to surface long-range (allosteric) rewiring the reviewer asked about.
- **Significance:** block-bootstrap over `g`-sized blocks per system; flag `ddccm` entries
  outside the bootstrap CI. Do not interpret ΔDCCM cells without CIs.
- Scalar per-residue metrics (RMSF, SASA, salt-bridge occupancy): report **Cohen's d**
  (consistent with existing pipeline) with pooled-SD and bootstrap CI.
- **Accept:** ΔDCCM heatmap + significance mask; effect-size table; all keyed to `resids`.

### T7 — `methods` (auto-provenance)  *(reviewer #1)*
**Site:** login. **Reads:** `*.mdp`, `topol.top`, tpr header. **Writes:** `methods/methods.{json,md}`.
- Parse EM/NVT/NPT/MD `.mdp`: integrator, `dt`, `nsteps` (→ ns), thermostat + `tau_t` + `ref_t`,
  barostat + `tau_p` + `ref_p`, `coulombtype` (PME) + `rcoulomb`/`rvdw`, `constraints`/LINCS,
  `nstlist`. From topology/tpr: force field (CHARMM36m), water model, ion counts, N atoms, box.
- Render a Methods paragraph (`methods.md`) + machine-readable `methods.json`. Auto-generation
  eliminates transcription error and *is* the methods section the reviewer wants.
- **Accept:** paragraph names ff/water/thermostat/barostat/PME/cutoffs/constraints and the
  equilibration protocol; values cross-checked against `.mdp` by a unit test on a fixture.

### T8 — `dssp` (secondary structure)  *(reviewer #5)*
**Site:** compute (light). **Reads:** trajectory + `window.json`. **Writes:** `analysis/{sys}/rep{r}/dssp.npz`.
- Per-frame DSSP over the window via `MDAnalysis.analysis.dssp.DSSP` (pure-python, no extra
  dep) — fallback `gmx dssp` if MDA version < 2.8. Store `codes (F, n_res)` (int-encoded) and
  `occupancy (n_res, n_classes)`.
- In `compare`: per-residue Δoccupancy WT vs mutant; highlight residues near site 75 and the
  EF-hand loops (helix↔coil transitions are the expected N75K signature).
- **Accept:** occupancy sums to 1 per residue; Δoccupancy figure produced.

---

## 5. SLURM sketch (dependency chain)

```bash
# Per system/replica. Chunked extract as an array, then a linear analysis chain.
EX=$(sbatch --parsed --array=0-$((NCHUNK-1)) extract.slurm)          # T1
MG=$(sbatch --parsed --dependency=afterok:$EX     merge.slurm)       # merge
EQ=$(sbatch --parsed --dependency=afterok:$MG     equilibrate.slurm) # T3 -> window.json
sbatch --dependency=afterok:$EQ dccm.slurm      # T4
sbatch --dependency=afterok:$EQ dssp.slurm      # T8
sbatch --dependency=afterok:$EQ local.slurm     # T5a
sbatch --dependency=afterok:$EQ coulomb.slurm   # T5b (energygrps rerun; larger wall-time)
# T5c APBS: separate, after frame sampling. T7 methods: independent, run anytime.
# T6 compare: after T4/T5/T8 for BOTH systems (afterok on both dccm jobs).
```
Resource notes: T1 array = trajectory-bound (I/O + mem for `ca_coords`); DCCM is
`O(F·N²)` but N≈191 so trivial; T5b/T5c are the wall-time sinks — size partitions accordingly.

---

## 6. Definition of done (whole plan)

- Every analysis reads its window from `window.json`; grepping the codebase shows no
  hard-coded frame ranges.
- WT and N75K figures share identical selections, alignment core, and window length.
- ΔDCCM and all scalar comparisons carry uncertainty (bootstrap CI / Cohen's d), never bare
  point differences.
- `methods.md` regenerates from `.mdp` and passes its fixture test.
- Regression test confirms `rmsd_bb`/`rg` unchanged vs pre-T1 baseline.
```
