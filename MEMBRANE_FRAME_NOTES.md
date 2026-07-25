# Membrane frame, protein–lipid coupling, and biasable orientation CVs

Design notes for extending rotmd from a protein-only orientation pipeline to one
that measures orientation **against the membrane it is actually bound to**, and
that produces collective variables metadynamics can bias.

Companion to `MEMBRANE_NOTES.md` (curvature; on `master`/`release`). Where the
two touch — the surface, and what may be differentiated on it — §7 reconciles
them explicitly.

Nothing here is implemented yet. Every empirical number is marked **[measure]**
and produced by `scripts/verify_membrane_frame.py`.

Maths is written for MathJax (`dollarmath` + `amsmath` are enabled in
`docs/source/conf.py`); code blocks are literal PLUMED or Python input.

**Notation.** Vectors are bold ($\mathbf{r}$), unit vectors carry a hat
($\hat{\mathbf{n}}$), matrices are upright sans ($\mathsf{R}$, $\mathsf{S}$).
Frame index $t$, atom index $i$. $\mathbf{r}_i^{(0)}$ is a reference-structure
position, always COM-centred. Angles are radians throughout; degrees appear only
in printed output and on plot axes.

---

## 0. The system, as it actually is

Measured from `~/Data/n75kcomplex.gro` (247 689 atoms, box
$12.81 \times 12.81 \times 14.43$ nm):

| Component | Count |
|---|---|
| Protein | hippocalcin, 192 residues, `GLYM` = N-myristoylated Gly1 |
| DPPC | 380 |
| CHL1 (cholesterol) | 250 |
| POPC | 95 |
| **DMPI2 / POPI2 (PIP₂)** | **20 / 5 = 25** |
| K⁺ / Cl⁻ / Na⁺ / **Ca²⁺** | 248 / 188 / 38 / **5** |
| TIP3 | 53 387 |

Geometry: bilayer P–P thickness $\approx 4.5$ nm, centre $z \approx 6.4$ nm;
protein COM at $z \approx 3.0$ nm, i.e. **peripherally bound beneath the lower
leaflet**.

The `.gro` resname field is 5 characters, so `DMPI2`/`POPI2` are truncations of
`DMPI25`/`POPI25`. The atom names settle the chemistry: `P4` with `OP42/OP43/OP44`,
`P5` with `OP52/OP53/OP54` **and `HP52`**, plus the backbone `P` with `O11–O14`.
That is **PI(4,5)P₂ with one protonated 5-phosphate oxygen**, net $-4e$. Tails run
`C22–C214` / `C32–C314` — dimyristoyl (14:0/14:0), confirming `DM`. There is no
`SAPI25` in this system.

Two facts drive everything below.

**All 25 PIP₂ are in the protein-facing (lower) leaflet.** Backbone P at
$z \approx 4.20$ nm (POPI2) / $4.34$ nm (DMPI2); inositol P4/P5 at
$z \approx 3.70\text{–}3.85$ nm. The bisphosphate headgroups therefore project
**$\approx 0.5$ nm off-plane, downward, into the slab the protein occupies** —
carrying $\approx -100e$ of charge out of the headgroup plane and toward the
protein.

That is not an analogy to Li & Buck's "off-plane negative potential" — it is the
same object, present in this box and measurable. Their argument is that
off-plane charge reorients a peripheral protein by pulling or pushing its charged
face. N75K (Asn→Lys, neutral $\to +1$) perturbs exactly that complementarity,
which is a candidate mechanism for WT $\approx 70°$ vs N75K $\approx 90°$ tilt.
§5 turns the argument into a number.

**The protein is anchored by a myristoyl chain** (`GLYM`, Gly1), with 5 Ca²⁺
present — hippocalcin in the Ca²⁺-bound, extruded-myristoyl state. The anchor is
a physically privileged, *directed* reference vector, which §3 would like to
exploit. It was not extracted. See §0.1.

### 0.1 What extraction captured, and the choice hidden inside it

From `wt/merged.npz` directly:

```
positions   (15763, 3082, 3)      masses  (3082,)
ca_coords   (15763,  191, 3)      ca_resids  191 values, running 2 ... 192
```

**`ca_resids` starts at 2. `GLYM` (resid 1) is not in the selection.**
MDAnalysis' `protein` keyword matches a fixed list of standard residue names, and
the non-standard `GLYM` is not on it — so `--selection protein` dropped it
silently. (3 082 also matches `GROUP ATOMS=1-3082` in
`scripts/hpc/plumed_orientation.dat`, so the PLUMED setup inherited the same
selection.)

This is really **two** consequences, and they deserve different verdicts.

**(a) The anchor is unavailable as a reference point — an unambiguous gap.**
§3 needs a directed reference to sign-fix the principal axis, and the anchor is
the natural one. §4 wants an anchor-relative depth. Neither is computable from
the current `.npz`, and no re-analysis recovers it — the atoms are simply not
there. Use resid 2 (the closest anchor-end CA present) as a stand-in meanwhile.

**(b) The anchor is excluded from the inertia tensor — defensible, possibly
correct, but undocumented.** It is tempting to call this a bug; on reflection it
may be the better modelling choice. The myristoyl is a flexible 14-carbon chain
inserted into the bilayer. Including it makes $\theta$ depend on *chain
conformation* — a fast, local degree of freedom — rather than on the orientation
of the folded core, which is what "protein orientation" is meant to mean. Its
lever arm makes this non-negligible: the contribution to the inertia tensor goes
as $m_i r_i^2$, so $\approx 1.5\,\%$ of the mass sitting at the extreme end
contributes disproportionately, and it *moves*.

So the recommendation is not simply "put it back":

- **Extract `GLYM`** so the anchor is available as a reference point and as a
  group PLUMED can address — this is required for (a).
- **Keep the inertia tensor on the folded core by default**, and make the choice
  an explicit, recorded flag rather than an accident of the selection string.
- **[measure]** $\theta$ computed with and without the anchor, on the same
  frames. If the two differ by more than a degree or two, the choice is a
  reportable methodological decision, not a detail.

Note the silver lining: all-atom `positions` for the 3 082 selected atoms **are**
stored, not just CA. So any protein-side geometry within residues 2–192 —
contacts, per-residue distances, charged-face orientation — needs no
re-extraction. Only the lipids and `GLYM` do.

### 0.2 The N75K trajectory is split across a periodic image

**Measured, and it invalidates the entire N75K dataset.** Consecutive Cα atoms
sit $\approx 3.8$ Å apart in *any* conformation — folded, unfolded, molten. Only
a periodic-image split can stretch that bond. Over 300 sampled frames per system:

| | WT | N75K |
|---|---|---|
| max consecutive Cα–Cα | **4.11 Å** | **191.45 Å** |
| frames with any Cα–Cα $> 10$ Å | 0 / 304 | **247 / 301 (82 %)** |
| $R_g$ (min / median / max) | 18.0 / 18.8 / 20.4 Å | 18.0 / **43.0** / **60.9** Å |
| RMSD (median) | 3.8 Å | **41.1 Å** |
| verdict | chain intact | **split** |

$191.45$ Å is not a protein dimension. It matches
$\sqrt{128.1^2 + 144.3^2} = 193.0$ Å, the box diagonal in $x$–$z$ — the molecule
is wrapped in two dimensions. **Every frame in the N75K analysis window (9 155 of
9 155) is affected.**

The corruption is *gradual*, which is why it never announced itself: the largest
single-frame $\Delta R_g$ is only 5.7 Å. As the protein diffuses across the
boundary its residues wrap a few at a time, so $R_g$ creeps from 18 Å at
$t = 0$ to 40–60 Å, wandering up and down as the protein drifts back and forth.
Nothing looks like a step; it looks like a very mobile protein.

**Why it produced physics-shaped output.** The inertia tensor of two blobs a box
apart is a near-perfect symmetric top:

| | $I_1$ | $I_2$ | $I_3$ | $(I_3-I_2)/I_3$ |
|---|---|---|---|---|
| WT | $2.89\times10^6$ | $5.85\times10^6$ | $6.56\times10^6$ | 10.9 % |
| N75K | $5.53\times10^6$ | $5.73\times10^7$ | $6.03\times10^7$ | **2.2 %** (44.6 % of frames below 2 %) |

For N75K $I_2 \approx I_3$, so $\mathbf{v}_c$ is **degenerate**: the two largest
eigenvectors span a plane and neither is individually defined. `eigh` returns
*some* vector in that plane, which then diffuses slowly around it — smooth
frame to frame (median $|\Delta\mathbf{v}_c| = 1.5°$), but with no fixed
body-frame direction at all. That single fact explains every N75K anomaly:

- **"N75K sits at $\approx 90°$ tilt"** — an artifact. $\mathbf{v}_c$ lies in the
  degenerate plane perpendicular to the spurious long axis created by the split.
- **The $\psi$ scatter across the whole circle** — the degenerate eigenvector's
  azimuth, not an orientational preference.
- **$g = 596$–$2253$ for the orientation observables** — the slow random walk of
  a degenerate eigenvector, not slow physics.
- **The proxy fit failing at $37.6°$ median** (§3.3) — no fixed body-frame axis
  exists to fit.

> **Correction.** An earlier conclusion in this project held that N75K's PMF seam
> was genuine, unavoidable physics — a protein truly sitting at the $90°$ pivot
> and crossing it thermally, immune to any post-hoc fold. That was wrong, and the
> smooth frame-to-frame $\theta$ used to support it is exactly what a slowly
> diffusing degenerate eigenvector produces. The seam was an artifact of broken
> coordinates.

**What has to happen.** Reassemble the molecules and re-extract, from the raw
`.trr` and **back to `.trr`**:

```
gmx trjconv -s step7_chunkN.tpr -f step7_chunkN.trr \
            -o whole/step7_chunkN.trr -pbc whole
```

`scripts/hpc/make_whole_array.slurm` runs this as a SLURM array over the 15
chunks. Three constraints make the exact form matter:

- **`.trr` in, `.trr` out.** TRR stores $x$, $v$ and $f$ at full precision;
  `.xtc` stores positions only and would silently discard the velocities and
  forces that $L$, $\tau$, $\omega$, $dL/dt$ and $K_{\text{trans}}$ are built
  from. `-pbc whole` itself only translates atoms by whole box vectors, so $v$
  and $f$ are unaffected by the operation.
- **Never `-fit`.** Fitting rotates coordinates but does *not* rotate velocities
  or forces, decoupling them from the positions. For a package whose entire
  purpose is rotational dynamics that would be worse than the bug it fixes.
- **`-pbc nojump` is a separate, later pass.** It cannot reassemble an
  already-broken molecule, so it must follow `-pbc whole`; and because it
  compares each frame to the previous one, applying it per chunk resets
  continuity at every chunk boundary (concatenate with `gmx trjcat` first).
  rotmd re-centres each frame on the protein COM, so orientation analysis does
  not need it at all — it matters for COM diffusion and for keeping protein and
  membrane in one image, i.e. for §8's membrane stage.

Fix the selection in the same pass (§0.1). Note that GROMACS will drop `GLYM`
from its `Protein` group for exactly the reason MDAnalysis did — it builds that
group from `residuetypes.dat`, which does not list the non-standard residue — so
write the `System` group and let `--selection "protein or resname GLYM"` do the
picking at the rotmd end.

Until then, every N75K number in the project — orientation, tilt, $\psi$, PMF,
friction, transitions, correlation times, **and the equilibration window itself**
($t_0 = 237\,300$ ps and $g = 363$ were fitted to the artifact) — should be
treated as void.

WT is unaffected and its results stand, but note *why*: its protein simply never
crossed a boundary. That is luck, not process, so run the repair on both systems.
Reprocessing WT is also the cleanest validation available — it should reproduce
the existing WT numbers exactly, and if it does not, the repair itself is wrong.

**Guard added.** `cli._check_chain_connectivity` now runs in extract phase 2 and
raises with the `trjconv` command rather than writing a chunk. The check is one
line of arithmetic on data already in hand, and it would have caught this before
any of the analysis was built on it.

---

## 1. Why the lab $z$-axis is not the membrane normal

rotmd currently hardcodes the normal to $\hat{\mathbf{z}} = (0,0,1)$:
`cli.py:329`, `core/vector_observables.py:220-221`, and the `'z_axis'` default of
`base/membrane_interface.get_membrane_normal` (`membrane_interface.py:92`). A
working SVD plane-fit sits directly beneath it at lines 94–108 and is never
called.

Three ways that bites:

1. **Undulations.** A 12.8 nm patch supports bending modes whose local normal
   wanders from the box axis. The excursion grows with patch area.
2. **Local deformation.** A peripherally bound protein presses on the leaflet;
   the normal directly beneath it is the most perturbed one in the box, and it is
   precisely the one tilt is measured against.
3. **PIP₂ clustering.** 25 tetra-anions in one leaflet, attracted to a charged
   protein face, are a local composition and area-per-lipid anomaly — a
   deformation correlated *with the very coordinate being measured*. Error that
   correlates with the observable does not average out.

This is falsifiable, so treat it as a hypothesis, not a premise:

> **[measure]** the distribution of
> $\angle\bigl(\hat{\mathbf{n}}_{\text{fp}},\, \hat{\mathbf{z}}\bigr)$ over the
> production window, both systems.

Decision rule: median $\lesssim 2°$ → lab $z$ remains adequate as a *biasing*
coordinate (PLUMED cannot compute $\hat{\mathbf{n}}_{\text{fp}}$ anyway, §3);
median $\gtrsim 5\text{–}8°$ → the tilt PMFs computed so far carry a systematic,
orientation-correlated error and must be recomputed against the local normal.

Note the asymmetry that makes this tractable: **the biasing coordinate and the
analysis coordinate need not be the same.** §4 uses that deliberately.

---

## 2. The Willard–Chandler interface

### 2.1 Definition

Given atoms at $\{\mathbf{r}_i\}$, the coarse-grained density field is

$$
\bar\rho(\mathbf{r}) \;=\; \sum_i \phi_\xi\bigl(|\mathbf{r} - \mathbf{r}_i|\bigr),
\qquad
\phi_\xi(s) \;=\; \bigl(2\pi\xi^2\bigr)^{-3/2}\exp\!\left(-\frac{s^2}{2\xi^2}\right)
$$

and the instantaneous interface is the level set

$$
\mathcal{I} \;=\; \bigl\{\, \mathbf{r} \;:\; \bar\rho(\mathbf{r}) = c \,\bigr\},
\qquad c = \tfrac{1}{2}\rho_{\text{bulk}} .
$$

The point of the construction is that it defines a surface **per frame**, with no
time averaging and no assumption that the surface is a graph $z(x,y)$. Everything
useful follows from the field rather than from the mesh:

$$
\hat{\mathbf{n}}(\mathbf{r}) \;=\; -\,\frac{\nabla\bar\rho(\mathbf{r})}{\bigl|\nabla\bar\rho(\mathbf{r})\bigr|},
\qquad
\nabla\phi_\xi(s) \;=\; -\,\frac{\mathbf{r}}{\xi^{2}}\,\phi_\xi(s) .
$$

The gradient is **analytic** — normals never require finite differences of a
fitted surface. That distinction carries the whole argument of §7.

### 2.2 The one real parameter: which group, and what $\xi$

$\xi$ is `alpha` in pytim. Willard & Chandler's $\xi = 2.4$ Å is calibrated for
**water**, where it approximates the bulk correlation length.

Applied to *lipid headgroups* that choice is wrong, and wrong in a way this repo
has already ruled on. Lipids sit $\approx 8$ Å apart. A 2.4 Å kernel resolves
individual headgroups and produces a surface that dips between them — inventing
structure at a scale the sampling does not support. That is the same objection
`MEMBRANE_NOTES.md` raises against grid-fit curvature, and it applies here too.

Two defensible constructions:

| Field built from | $\xi$ | Pro | Con |
|---|---|---|---|
| **Lipid heavy atoms** | $\approx 3$ Å | Directly the membrane surface; unaffected by the protein | Sparse sampler; $\xi$ is a compromise, not derived |
| **Water oxygens** | 2.4 Å (as published) | Dense, well-conditioned, $\xi$ physically justified; captures the water-accessible surface a peripheral protein actually "sees", off-plane PIP₂ roughness included | **The protein excludes water too**, carving a cavity into exactly the region of interest |

**Recommendation: build from lipid heavy atoms at $\xi \approx 3$ Å**, and
validate against a water-built surface *in the protein-free region of the box*,
where the water construction is uncompromised. If the two agree there to within
the mesh spacing, the lipid-built surface is trustworthy under the protein too.

Do not average this choice away: report which group and which $\xi$ produced any
number derived from the surface.

### 2.3 What we take from it

**Footprint-averaged normal.** Not the per-vertex normal. With
$\mathbf{x}^{\perp}$ denoting lateral $(x,y)$ coordinates and $\mathbf{x}_p$ the
protein's projected centre, let

$$
F \;=\; \bigl\{\, v \;:\; \|\mathbf{x}^{\perp}_v - \mathbf{x}^{\perp}_p\| < R_{\text{fp}} \,\bigr\},
\qquad
\hat{\mathbf{n}}_{\text{fp}} \;=\; \frac{\sum_{v \in F} \hat{\mathbf{n}}_v}{\bigl\|\sum_{v \in F} \hat{\mathbf{n}}_v\bigr\|}
$$

with $R_{\text{fp}}$ the protein's lateral radius of gyration. Averaging over the
$\sim 10^2$ vertices in a footprint is what makes this robust. It is a **first**
derivative of a smoothed field, area-averaged — a very different statistical
object from the per-lipid second derivative §7 refuses to take from a grid.

**Signed proximal distance.** For an atom at $\mathbf{r}$ with nearest surface
point $\mathbf{p}$,

$$
\zeta(\mathbf{r}) \;=\; \hat{\mathbf{n}}_{\mathbf{p}} \cdot (\mathbf{r} - \mathbf{p}) .
$$

Nearest-*vertex* lookup (KD-tree) is accurate only to $O(\text{mesh spacing})$;
refine by projecting onto the adjacent triangle when the number matters. This is
the capillary-wave-corrected replacement for a lab-frame $z$ histogram: it removes
the $\pm 3\text{–}5$ Å undulation smearing that broadens every naive density
profile.

**Leaflet assignment** from $\operatorname{sgn}(\hat{\mathbf{n}}_z)$ on each
connected component of the isosurface — more robust than the $z$-bisection in
`base/leaflet_util.py:74`, which assumes the normal is $\hat{\mathbf{z}}$ (and
whose failure path is `sys.exit(1)`, lines 190/201/215/335/347/366 — unusable
inside a library call).

### 2.4 Using pytim

pytim is already declared in `pyproject.toml` and imported nowhere in the repo.
Illustrative shape only — **verify attribute names against the installed version
before relying on them**:

```python
import pytim
inter = pytim.WillardChandler(universe, group=lipids, alpha=3.0, mesh=1.5)
verts, faces, normals = ...   # from inter.triangulated_surface — CHECK the actual return
```

Three honest caveats:

1. **Normals and signed distances are ours to compute regardless.** pytim hands
   back a mesh; footprint averaging, the sign convention and the KD-tree lookup
   are not in the box.
2. **`scikit-image` is absent from `wheels/`.** pytim's marching cubes needs it.
   The air-gapped HPC install therefore needs a new wheel — a deployment task, not
   an afterthought.
3. **Per-frame cost is the binding constraint.** Budget it on a 100-frame slice
   before committing to a full pass over 15 763 frames. If it does not fit, the
   fallback is a per-leaflet height field $h(x,y)$ from an FFT-smoothed 2-D
   density, which yields the normal and thickness at a small fraction of the cost
   and loses only the non-graph generality and per-atom $\zeta$.

---

## 3. A directed orientation axis that PLUMED can bias

The centrepiece. One construction resolves the N75K seam *and* unblocks
metadynamics.

### 3.1 Two problems with the same root

rotmd's $\theta$ is the ZYZ nutation angle of $\mathbf{v}_c$, the largest-moment
eigenvector of the inertia tensor. `np.linalg.eigh` fixes no sign, so
$\mathbf{v}_c$ is a **headless line**: $\mathbf{v}_c$ and $-\mathbf{v}_c$ are the
same physical object. Hence `membrane_tilt_angle` folding to $[0,\pi/2]$, hence
`fold_tilt_and_psi`, hence the seam at $90°$, hence N75K — which sits at
$\approx 90°$ — being pathological rather than merely interesting.

**Problem A (analysis).** Folding a continuous domain onto a restricted range
creates a seam wherever the trajectory crosses the fold, and no post-hoc
algorithm removes it — folding is not invertible. This is a real limitation of
the folded coordinate and is reason enough to prefer a directed axis.

It is *not*, however, the explanation for what N75K's PMF looked like. That was
broken coordinates (§0.2). The smooth frame-to-frame $\theta$ once taken as
evidence of "genuine thermal motion through the pivot" is the signature of a
degenerate eigenvector diffusing in its plane. Both facts point the same way —
use a directed axis — but only one of them was ever about physics.

**Problem B (sampling).** PLUMED cannot compute inertia eigenvectors at all.
`GYRATION` emits **scalars** — `GTPC_1` is a number, not a vector — so
`gyr.eigvec1z` in `scripts/hpc/plumed_orientation.dat` does not exist and that
file cannot run (§9). And where a fold *is* written into a CV, as in
`theta: MATHEVAL ... FUNC=acos(abs(x))`, it is far worse than in analysis. Writing
$x = \cos\theta$,

$$
\frac{d}{dx}\arccos|x| \;=\; -\,\frac{\operatorname{sgn}(x)}{\sqrt{1-x^{2}}}
$$

which flips sign discontinuously at $x = 0$. The metadynamics force is
$-\,\partial B/\partial\theta \cdot \partial\theta/\partial x$, so:

> Biasing this CV applies a **discontinuous force exactly at N75K's equilibrium
> orientation.**

Both problems are the same missing ingredient: **a direction**.

### 3.2 The construction

Sign-fix $\mathbf{v}_c$ once against an unambiguous reference
$\mathbf{r}_{\text{ref}}$. The myristoyl anchor is the natural choice,
$\mathbf{r}_{\text{ref}} = \mathbf{R}_{\text{core}} - \mathbf{R}_{\text{GLYM}}$ —
but it is not in the current `.npz` (§0.1), so until re-extraction the substitute
is $\mathbf{r}_{\text{ref}} = \mathbf{R}_{\text{centroid}} - \mathbf{r}_{\mathrm{CA}(2)}$:

$$
\mathbf{v}_c \;\longleftarrow\; \mathbf{v}_c \, \operatorname{sgn}\bigl(\mathbf{v}_c \cdot \mathbf{r}_{\text{ref}}\bigr),
\qquad
\theta \;=\; \arccos\bigl(\hat{\mathbf{v}}_c \cdot \hat{\mathbf{n}}\bigr) \;\in\; [0,\pi].
$$

No fold, no seam, and the SO(3) measure reverts to plain $\sin\theta$ on
$[0,\pi]$ — the existing `angle_kind="theta"` path in `analysis/pmf.py`:

$$
F(\theta,\psi) \;=\; -k_BT \ln\!\left[\frac{P(\theta,\psi)}{\sin\theta}\right].
$$

N75K at $90°$ becomes an ordinary interior point of the domain.

This is the right *analysis* coordinate and it is what you asked for. It is still
not biasable. The rest of this section shows it has an excellent biasable proxy —
in fact an exactly equivalent one for a rigid body.

### 3.3 The rigid-body linear proxy

For a quasi-rigid body the principal axis is a **fixed body-frame vector**:
$\mathbf{v}_c(t) = \mathsf{R}(t)\,\mathbf{u}$ with $\mathbf{u}$ constant. Any
weighted sum of centred atomic positions transforms identically:

$$
\mathbf{m}(t) \;=\; \sum_i w_i\bigl(\mathbf{r}_i(t) - \mathbf{R}_{\text{COM}}(t)\bigr)
\;=\; \mathsf{R}(t)\sum_i w_i\bigl(\mathbf{r}^{(0)}_i - \mathbf{R}^{(0)}_{\text{COM}}\bigr).
$$

So if fixed weights $\mathbf{w}$ satisfy
$\mathsf{X}_0^{\mathsf{T}}\mathbf{w} = \mathbf{u}$ on the reference structure —
where $\mathsf{X}_0 \in \mathbb{R}^{n\times 3}$ stacks the centred reference
coordinates — then $\mathbf{m}(t) \parallel \mathbf{v}_c(t)$ at **every** frame,
exactly, for as long as the body is rigid.

That system is 3 equations in $n$ unknowns. Its minimum-norm solution is

$$
\mathbf{w} \;=\; \mathsf{X}_0\bigl(\mathsf{X}_0^{\mathsf{T}}\mathsf{X}_0\bigr)^{-1}\mathbf{u}
\;=\; \mathsf{X}_0\,\mathsf{S}^{-1}\mathbf{u},
\qquad
\mathsf{S} \;=\; \mathsf{X}_0^{\mathsf{T}}\mathsf{X}_0 \;=\; \sum_i \mathbf{r}_i\mathbf{r}_i^{\mathsf{T}} ,
$$

and it simplifies, because the scatter tensor $\mathsf{S}$ and the inertia tensor
$\mathsf{I}$ share eigenvectors. For unit masses

$$
\mathsf{I} \;=\; \sum_i\bigl(|\mathbf{r}_i|^2\mathbb{1} - \mathbf{r}_i\mathbf{r}_i^{\mathsf{T}}\bigr)
\;=\; \operatorname{tr}(\mathsf{S})\,\mathbb{1} - \mathsf{S},
$$

so $\mathsf{S}\mathbf{u} = s\,\mathbf{u}$ exactly when
$\mathsf{I}\mathbf{u} = \bigl(\operatorname{tr}\mathsf{S} - s\bigr)\mathbf{u}$.
Therefore $\mathsf{S}^{-1}\mathbf{u} = \mathbf{u}/s$, and

$$
\boxed{\;w_i \;\propto\; \mathbf{r}^{(0)}_i \cdot \hat{\mathbf{u}}\;}
$$

**The weight of an atom is simply its coordinate along the principal axis in the
reference structure**, and $\mathbf{m}$ is the first moment of the structure along
that axis. This is about as interpretable as a fitted CV gets.

(Note the eigenvalue ordering inverts between the two tensors: the *largest*-moment
axis of $\mathsf{I}$ is the *smallest*-eigenvalue axis of $\mathsf{S}$. Harmless
here — it is the same eigenvector — but worth knowing when reading
`core/inertia.py`, whose docstring claims $v_c$ is "typically the longest axis",
which is backwards for a prolate body.)

Two consequences fall out for free:

- $\sum_i w_i = 0$ **automatically**, because the reference coordinates are
  COM-centred:
  $\sum_i \mathbf{r}^{(0)}_i \cdot \hat{\mathbf{u}} = \bigl(\sum_i \mathbf{r}^{(0)}_i\bigr)\cdot\hat{\mathbf{u}} = 0$.
  So the COM term drops and $\mathbf{m} = \sum_i w_i \mathbf{r}_i$ needs no
  recentring — it is origin-independent.
- Splitting $\mathbf{w}$ into positive and negative parts, the two partial weight
  sums are **equal** (they must cancel), so

  $$
  \mathbf{m} \;\propto\; \mathbf{C}^{+} - \mathbf{C}^{-},
  \qquad
  \mathbf{C}^{\pm} = \frac{\sum_{i \in \pm} |w_i|\,\mathbf{r}_i}{\sum_{i \in \pm} |w_i|}
  $$

  with **no** scale factor. The proxy is a plain vector between two weighted
  centres — exactly what PLUMED's `CENTER` computes.

For a real, flexible protein, refine $\mathbf{w}$ by ridge least squares over the
production window rather than trusting one reference frame:

$$
\mathbf{w}^{\star} \;=\; \arg\min_{\mathbf{w}}
\;\sum_t \bigl\| \mathsf{X}(t)^{\mathsf{T}}\mathbf{w} - \hat{\mathbf{v}}_c(t) \bigr\|^{2}
\;+\; \lambda\|\mathbf{w}\|^{2} .
$$

With $\sim 7000$ frames $\times\, 3$ components against 191 CA weights this is
heavily overdetermined and well posed. Fit on one half of the window, evaluate on
the other — an in-sample residual proves nothing.

Crucially, **the residual is a measurement, not a nuisance**: for a perfectly
rigid body it is zero, so whatever survives is the protein's internal flexibility
projected onto the orientation coordinate.

**Measured** — $\angle\bigl(\mathbf{m}(t), \mathbf{v}_c(t)\bigr)$ over the
production window. N75K is reported only to show what a broken system looks like
(§0.2); its numbers carry no information about the method.

| Proxy | WT median / p95 | N75K median / p95 |
|---|---|---|
| **Fitted weights, held out** | **3.27° / 8.94°** | 37.59° / 109.09° |
| Fitted weights, in-sample | 1.82° / 4.61° | 23.47° / 121.03° |
| Closed form $w_i \propto \mathbf{r}_i\cdot\hat{\mathbf{u}}$ | 5.84° / 12.31° | 137.91° / 169.70° |
| N-lobe $\to$ C-lobe COM | 91.94° / 93.00° | 84.42° / 97.03° |
| $\mathrm{CA}(2) \to$ centroid | 75.71° / 81.43° | 75.58° / 88.60° |

**The fitted proxy passes on WT at 3.27° median**, comfortably inside the $5°$
criterion, and the held-out figure is only $1.5°$ worse than in-sample — so it is
fitting the body frame, not the noise. The construction works.

Two things worth reading off this table.

**The closed form is a good sanity check but not the production answer.** Its
$5.84°$ against the fitted $3.27°$ is the price of trusting a single reference
frame; the trajectory fit absorbs flexibility the reference cannot know about.

**The two-group proxies fail at $\approx 90°$ — and that is a fact about
$\mathbf{v}_c$, not about them.** They are orthogonal to it, and necessarily so.
The largest-moment eigenvector of $\mathsf{I}$ is the *smallest*-eigenvalue
eigenvector of $\mathsf{S}$, i.e. the protein's **shortest** axis, while any
lobe-to-lobe vector runs along its longest. So:

> rotmd's $\theta$ has always been the angle between the membrane normal and the
> protein's **short** axis. That is a defensible coordinate — but it is not what
> "the long axis of the protein tilts by $\theta$" would suggest, and it is worth
> stating explicitly in any figure caption. `core/inertia.py`'s docstring claim
> that $v_c$ is "typically the longest axis" is backwards, and this table is the
> measurement that shows it.

| Proxy | PLUMED input | Evaluable now? |
|---|---|---|
| Fitted weights $\mathbf{w}^{\star}$ | two `CENTER`, $\sim 191$ weights | **yes** — and it passes |
| N-lobe $\to$ C-lobe COM | two `COM`, via `analysis/domains.py` | yes — but targets the long axis |
| $\mathrm{CA}(2) \to$ centroid | two `COM` | yes — same objection |
| Myristoyl $\to$ core COM | two `COM` | **no** — `GLYM` not extracted (§0.1) |

### 3.4 PLUMED encoding

```
WHOLEMOLECULES ENTITY0=1-3082          # REQUIRED: CENTER/COM are meaningless
                                       # on a molecule broken across PBC

gp: CENTER ATOMS=<atoms with w_i > 0> WEIGHTS=<w_i>
gm: CENTER ATOMS=<atoms with w_i < 0> WEIGHTS=<|w_i|>
ax: DISTANCE ATOMS=gm,gp COMPONENTS

th: CUSTOM ARG=ax.x,ax.y,ax.z FUNC=acos(z/sqrt(x*x+y*y+z*z)) PERIODIC=NO
```

No `abs()`. `PERIODIC=NO` is correct here and is *not* the §9 bug: $\theta$
genuinely lives on $[0,\pi]$ with distinct endpoints. It is $\psi$ that is
periodic.

For $\psi$, project a second body vector $\mathbf{b}(t)$ (an independently fitted
proxy) into the tangent plane and take its angle against a fixed in-plane
reference $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2)$:

$$
\mathbf{b}_{\perp} \;=\; \mathbf{b} - (\mathbf{b}\cdot\hat{\mathbf{n}})\,\hat{\mathbf{n}},
\qquad
\psi \;=\; \operatorname{atan2}\bigl(\mathbf{b}_{\perp}\cdot\hat{\mathbf{e}}_2,\;\; \mathbf{b}_{\perp}\cdot\hat{\mathbf{e}}_1\bigr)
$$

declared `PERIODIC=-pi,pi`. That declaration is mandatory — it is what makes
`METAD` deposit Gaussians that wrap correctly instead of tearing the landscape at
$\pm\pi$.

---

## 4. Depth: $D_z$ to bias, $\zeta$ to analyse

Li & Buck's second orientation parameter is $D_z$, the $z$-separation between a
protein subdomain COM and the membrane COM. It is PLUMED-native — two groups and
one `DISTANCE ... COMPONENTS`, taking `.z` — and PBC-safe when the groups are made
whole.

Two cautions:

- **Never use an absolute box coordinate.** `com.z` is meaningless under PBC and
  drifts with the whole system. It must be a *difference*. (Live bug, §9.2.)
- **Reference the bound leaflet, not the whole bilayer.** A whole-membrane COM is
  ill-defined if the bilayer wanders across the periodic $z$ boundary. Use the
  lower-leaflet phosphates — the leaflet the protein is actually on.

The analysis-side upgrade is $\zeta$ (§2.3), which removes capillary-wave smearing
that $D_z$ cannot.

**Recommended split: bias on $(D_z, \theta)$; reweight onto
$(\zeta, \theta_{\text{local}})$.** Bussi & Laio endorse this explicitly —
metadynamics "can also be used to reconstruct the free energy along non-biased
variables" by reweighting — with the one caveat that the analysed variables must
be sufficiently sampled. That caveat is exactly §6.3's problem, so check it rather
than assume it.

This resolves the tension in §1 cleanly: PLUMED cannot compute a WC normal, and it
does not have to. Bias with the cheap approximate coordinate, analyse in the good
one.

---

## 5. Making the off-plane potential quantitative

Li & Buck *assert* that off-plane charge sets orientation; their Figs 5b/7b are
lab-frame charge profiles $\rho_q(z)$. With $\zeta$ from §2.3 and one integration,
the assertion becomes a number.

Bin charge against the intrinsic coordinate,

$$
\rho_q(\zeta) \;=\; \sum_i q_i\, \delta(\zeta - \zeta_i),
$$

then integrate the 1-D Poisson equation
$\dfrac{d^2\varphi}{d\zeta^2} = -\dfrac{\rho_q(\zeta)}{\varepsilon_0}$ twice, with
$\varphi(\zeta_0) = \varphi'(\zeta_0) = 0$ deep in the bulk:

$$
\varphi(\zeta) \;=\; -\frac{1}{\varepsilon_0}\int_{\zeta_0}^{\zeta} (\zeta - \zeta')\,\rho_q(\zeta')\,d\zeta' .
$$

Differentiating once gives
$\varphi'(\zeta) = -\varepsilon_0^{-1}\int_{\zeta_0}^{\zeta}\rho_q(\zeta')\,d\zeta'$ —
the boundary term vanishes because the integrand carries the factor
$(\zeta - \zeta')$ — and once more returns $-\rho_q/\varepsilon_0$. Numerically it
is a double cumulative sum, the same scheme as `gmx potential`, so it is
externally cross-checkable.

Report in $k_BT/e$ at 310 K, three ways:

1. whole protein-facing leaflet;
2. restricted to the PIP₂ solvation shell — the "off-plane" part proper, the
   $\zeta > 0$ tail where §0 already located the P4/P5 phosphates;
3. **WT vs N75K**, the comparison that carries the biology.

State the limitation plainly: this is a **mean-field electrostatic potential, not
a PMF**. It excludes the entropic cost of lipid reorganisation and ion
redistribution, and must not be quoted as a binding free energy. Its value is that
it makes a qualitative published argument testable on this system.

---

## 6. Protein–lipid coupling

### 6.1 Observables

Lipid side, per frame: $n_{\text{PIP2}}$ in the contact shell; PIP₂ radial
distribution about the footprint; footprint charge density; local thickness; local
area per lipid. Protein side: $\theta$, $\psi$, $\zeta$.

`analysis/local.py:30 contact_occupancy` already does PBC-aware contact counting
via `distance_array(..., box=ts.dimensions)` and reports per-partner occupancy —
reuse it rather than writing a second contact routine.

### 6.2 The estimators

**(a) Lagged cross-correlation.** `analysis/correlations.py` already has
cross-correlation. For mean-free series $A$, $B$,

$$
C_{AB}(\tau) \;=\; \frac{\langle\, \delta A(t)\,\delta B(t+\tau) \,\rangle_t}{\sigma_A \sigma_B}
$$

between $n_{\text{PIP2}}(t)$ and $\theta(t)$ answers the actual question — *does
PIP₂ recruitment precede reorientation, or follow it?* — from the sign of
$\arg\max_\tau |C_{AB}(\tau)|$. Cheapest, most interpretable, most robust to short
samples. Attach block-bootstrap errors using $g$ from `window.json`.

**(b) Joint PMF $F(\theta, n_{\text{PIP2}})$.** Thermodynamic coupling rather than
kinetic. `analysis/pmf.py:compute_pmf_2d` almost does this, but note the API
constraint: it applies the Jacobian to **axis 0 only** and hardwires
`psi_edges = linspace(0, 2*pi, psi_bins+1)`. A non-angular second axis therefore
needs an explicit edges/Jacobian parameter — a small, well-contained change. The
$\sin\theta$ factor must apply to $\theta$ and **not** to a particle count.

**(c) TICA — conditional.** The slow collective mode, and Bussi & Laio's named
route to a learned CV (their refs 50–51). With mean-free features
$\mathbf{y}(t)$, symmetrise the lagged covariance and solve the generalised
eigenproblem

$$
\mathsf{C}(\tau) \;=\; \tfrac{1}{2}\bigl[\tilde{\mathsf{C}}(\tau) + \tilde{\mathsf{C}}(\tau)^{\mathsf{T}}\bigr],
\qquad
\mathsf{C}(\tau)\,\mathbf{v}_k \;=\; \lambda_k\, \mathsf{C}(0)\,\mathbf{v}_k ,
$$

with implied timescales $t_k = -\tau/\ln\lambda_k$; scan $\tau$ for a plateau. The
leading eigenvector's **loadings** are the deliverable — "the slow mode is 60 %
tilt, 30 % PIP₂ count" is a readable result in a way a neural CV is not.

Two hard requirements: angles enter as **unit-vector components
$(\cos\psi, \sin\psi)$, never as raw $\psi$** — the $0/2\pi$ wrap would otherwise
manufacture a spurious slow mode, the same trap as the fold in §3; and the input
must be the equilibrated window, since TICA assumes reversible stationary
dynamics.

### 6.3 The sampling problem, stated before any of this is run

**Measured**, $n_{\text{eff}} = N/g$, on the equilibrated window. The orientation
observables were taken directly rather than inherited from RMSD, and $\psi$'s
inefficiency is taken on $\cos\psi$ / $\sin\psi$ because the wrapped series would
read its own $0/2\pi$ jump as decorrelation:

| Observable | WT $g$ | WT $n_{\text{eff}}$ | N75K $g$ | N75K $n_{\text{eff}}$ |
|---|---|---|---|---|
| $\cos\theta$ | 1089.1 | **6.6** | 596.1 | 15.4 |
| tilt (folded) | 121.9 | 58.9 | 322.7 | 28.4 |
| $\cos\psi$ | 1118.9 | **6.4** | 697.6 | 13.1 |
| $\sin\psi$ | 278.4 | 25.8 | 2252.7 | 4.1 |
| RMSD (reference) | 105.75 | 67.9 | 363.4 | 25.2 |

Read WT only; N75K is broken (§0.2). Three conclusions, and the first is bad news.

**Orientation decorrelates an order of magnitude more slowly than RMSD.**
$g = 1089$ against RMSD's $106$. So the guess that the orientational numbers
"might be substantially better" than the RMSD proxy was wrong in the unhelpful
direction: WT has $n_{\text{eff}} \approx 6.6$ **independent orientational
samples** across a $30 \times 36$ PMF's 1 080 bins. That is not an
under-converged free-energy surface; it is roughly one configuration.

**The folded coordinate flatters itself.** tilt reports $n_{\text{eff}} = 58.9$
against $\cos\theta$'s $6.6$ — a factor of nine "better" from the same
trajectory. Folding does not improve sampling; it *discards the branch
information*, which is precisely the slowly decorrelating part. A folded
coordinate will always look better sampled than the quantity it came from, and
that apparent gain is an artifact. One more reason to prefer the directed axis of
§3, where the number is at least honest.

**This reprices the whole estimator ladder.** (a) lagged cross-correlation
remains defensible with block errors; (b) a 2-D joint PMF needs $n_{\text{eff}}$
in the hundreds and is out of reach; (c) TICA on ~7 independent samples would be
pure overfitting and must not be attempted on this data.

So **every unbiased free-energy surface in this project is a snapshot, not a
converged PMF** — and that is now the strongest argument in the file for doing
the metadynamics. It is also a reminder that WT and N75K failed for entirely
different reasons: N75K's coordinates were wrong, WT's are right and merely
sampled ~7 times.

---

## 7. Curvature stays discrete

`MEMBRANE_NOTES.md` rejects curvature from a fitted $z(x,y)$ grid, because second
derivatives of a fit through headgroups $0.65$ nm apart are dominated by grid
spacing and smoothing rather than by the membrane. That ruling stands, and §2's
smoothed density field is exactly the kind of object it warns about.

The division of labour is by **derivative order**:

| Quantity | Order | Source | Why it is safe there |
|---|---|---|---|
| Normal $\hat{\mathbf{n}}$ | 1st | WC field (§2.3) | Analytic gradient, then averaged over $\sim 10^2$ vertices in the footprint |
| Curvature $K$, $H$ | 2nd | Delaunay/Voronoi (`MEMBRANE_NOTES.md`) | Angle defect is *exact* for a polyhedral surface; resolution set by the lipids, not by a grid |

First derivatives of a smoothed field, area-averaged over many lipids, are a
different statistical proposition from per-lipid second derivatives. Both notes
hold at once.

Keep the discrete machinery as specified there — Delaunay for the angles, Voronoi
(Meyer mixed area) for the areas, the angle defect

$$
K_i \;=\; \frac{2\pi - \sum_j \theta_j}{A_i}
$$

for Gaussian curvature, the cotangent Laplacian for $H$, Gauss–Bonnet pooling of
numerators and areas separately, and Banchoff's discrete Morse index for saddles.
Keep especially its free correctness test: over a periodic patch (a torus,
$\chi = 0$),

$$
\sum_i K_i A_i \;=\; 2\pi\chi \;=\; 0 \qquad \text{exactly.}
$$

A pipeline whose total angle defect is not $\approx 0$ has a bug — a missing
periodic image, a double-counted boundary lipid, a leaflet mixed in. The WC route
has no comparable self-check, which is a further reason not to source curvature
from it.

---

## 8. What a `rotmd membrane` stage would store

The blocker: **no lipid coordinate survives extraction.** `extract` reads only
`--selection` (`io/gromacs.py:95,154`); the bilayer is collapsed to one frame-0
scalar `membrane_center_z` (`io/gromacs.py:321`) used solely by the optional
freesasa stage, and the normal it computes is discarded (`cli.py:395-399`). Any
lipid observable requires a new pass over the trajectory — a compute-node job, not
a login-node one.

Storing lipid coordinates is not an option: 750 lipids at full atomic detail is
trajectory-sized, and the project's stated purpose is compressed observables.
Store derived quantities instead:

| Field | Shape | Bytes/frame |
|---|---|---|
| `membrane_normal` | (3,) | 24 |
| `zeta_com`, `Dz`, `thickness_local` | (1,) each | 24 |
| `n_pip2_contact` | (1,) | 8 |
| `pip2_min_distance` | (25,) | 200 |
| `charge_profile_footprint` | (n_bins,) | ~800 |
| `leaflet_height_field` *(optional)* | (2, 32, 32) | 8 192 |

$\approx 1$ kB/frame without the height field, $\approx 9$ kB with — **16 MB to
145 MB** for 15 763 frames. Comparable to existing artifacts and consistent with
the pipeline's design.

Follow the established conventions: `.npz` plus a `meta.json` sidecar recording
which group and which $\xi$ built the surface; one chunk per invocation with
`merge` handling concatenation; radians internally and degrees only at plot axes
(`viz/core.py:285 angle_grid_degrees`).

**Settle the protein selection in the same pass.** `--selection protein` drops
`GLYM` (§0.1). Re-extract with `--selection "protein or resname GLYM"` so the
anchor is *available*, but record explicitly whether it enters the inertia tensor
— §0.1 argues the folded core is the better default, and that decision should be
in `meta.json`, not implicit in a selection string.

Worth adding a guard to `extract` so this class of ambiguity is loud rather than
silent: warn when `--selection` matches fewer residues than the topology contains,
in the same spirit as the existing `_check_membrane_sel` pre-flight
(`cli.py:207-224`).

---

## 9. PLUMED inputs: the blocking bug list

Both existing inputs would produce wrong science. Each item is a fix, not a style
note.

**`scripts/hpc/plumed_orientation.dat`**

1. **`gyr.eigvec1z` / `.eigvec1x` / `.eigvec1y` do not exist.** `GYRATION` returns
   scalars; `GTPC_1` is a number, not a vector. $\theta$ and $\varphi$ in this file
   are fictional and it will not run. *Verify with*
   `plumed driver --plumed ... --mf_xtc ...` *before anything else.*
2. **`com_z: COMBINE ARG=com.z COEFFICIENTS=0.1`** — the comment says "convert to
   nm", but PLUMED works in nm natively, so this is a **10× scale error**. It also
   uses an **absolute box coordinate** as an insertion depth, meaningless under
   PBC. Both fixed by $D_z$ (§4). Note the downstream damage:
   `GRID_MIN/MAX=-3.0,3.0` with `SIGMA=0.05` are sized for the wrong quantity.
3. **`theta: MATHEVAL ... FUNC=acos(abs(x))`** — folds to $[0°, 90°]$, giving a
   **discontinuous biasing force at exactly N75K's equilibrium tilt** (§3.1).
   Fixed by §3.
4. **`phi: ... atan2(y,x) ... PERIODIC=NO`** — $\varphi$ is periodic. Currently
   only printed, so merely wrong in analysis; fatal the moment it is biased, and
   $\psi$ is a CV you want to bias.

**`plumed.dat`**

5. **`TYPE=GTPC_VECTOR`** is not a valid `GYRATION` type; `gyration` and `gyr_eig`
   are computed and then never used.
6. $\theta$ is actually built from the **N$\to$C terminus vector**
   (`nterm`/`cterm`), which is not rotmd's inertia $\theta$ and not a good proxy
   for it for an EF-hand protein. §3.3 gives the principled replacement — and,
   usefully, quantifies how good any such two-group proxy is.

**Both files**

7. **Walls `AT=20` / `AT=90` encode the *old* tilt convention.** Under the current
   convention ($0°$ = axis $\parallel$ normal, $90°$ = axis in plane) with
   WT $\approx 70°$ and N75K $\approx 90°$, an upper wall at $90°$ sits **on
   N75K's population maximum**. It would distort the FES precisely where the
   WT/mutant difference lives. Re-derive both walls from the measured
   distributions and place them where the density is negligible.
8. `scripts/hpc/README.md:135-144` ("WT ~25°, N75K ~56°") is in the old convention
   and contradicts current results. Restate or delete — a stale expectation table
   is worse than none.

**Missing throughout:** `WHOLEMOLECULES` before any `COM`/`CENTER` (§3.4).

### Which variant, per Bussi & Laio

Their decision tree (Fig. 5c) applied here: the relevant variables are known and
few, so **well-tempered metadynamics on 2 CVs $(D_z, \theta)$** is the right
default. The bias grows as

$$
B_{t+1}(s) \;=\; B_t(s) + w\,\exp\!\left(-\frac{B_t(s_t)}{\Delta T}\right)
\exp\!\left(-\frac{(s - s_t)^2}{2\sigma^2}\right),
$$

converging to $-\dfrac{\Delta T}{T + \Delta T}\,F(s)$ with bias factor
$\gamma = (T+\Delta T)/T$ — provably, and without having to declare in advance the
region where the free energy should be estimated.

Keep `CALC_RCT` (already present) for reweighting onto
$\zeta$ / $\theta_{\text{local}}$ (§4).

For convergence, follow their explicit warning: in well-tempered metadynamics the
bias necessarily flattens, so **a smooth bias is not evidence of convergence**.
Check instead that transitions between the relevant minima continue, via block
analysis of the biased-CV histogram. If the CV distinguishes the metastable states
but not the transition state between them, the bias will not accelerate anything
and convergence will be no better than plain MD — the diagnostic is a trajectory
that diffuses freely in the CV yet only ever explores one basin.

---

## References

- Willard & Chandler (2010), *Instantaneous liquid interfaces*, J. Phys. Chem. B
  **114**, 1954 — the $\xi = 2.4$ Å field and the $\rho_{\text{bulk}}/2$ level set.
- Sega, Hantal, Fábián & Jedlovszky, *pytim* — the implementation used here.
- Bussi & Laio (2020), *Using metadynamics to explore complex free-energy
  landscapes*, Nat. Rev. Phys. **2**, 200 — CV selection, well-tempered
  convergence diagnostics, reweighting to unbiased variables, TICA-based CVs.
- Li & Buck (2019), bioRxiv 565945 — myristoylated cell-penetrating peptides, the
  $(D_z, \theta)$ orientation maps and the off-plane-potential argument of §5.
- Meyer, Desbrun, Schröder & Barr (2003) — mixed area, cotangent Laplacian (§7).
- Banchoff (1970) — discrete Morse index for saddles (§7).
