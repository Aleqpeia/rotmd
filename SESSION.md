# Session notes

Deferred / incidental issues noticed while working. These are not
necessarily blocking; recorded here so they can be addressed later.

> **Next planned work:** see `PLAN.md` for the post-merge plan to make
> `rotmd.core` the authoritative import surface (namespace migration +
> `VectorObservable` consolidation).

## Statistical / scientific caveats for the crossings regression

- **PIP2 / cholesterol are confounded** in the current two-composition
  design (`simple` = DPPC/DOPC, `complex` = DPPC/DOPC/CHOL/PIP2). Any
  "complex vs simple" effect is the *combined* CHOL+PIP2 effect; the
  `has_PIP2` and `has_CHOL` covariates cannot be separated until a third
  composition (e.g. CHOL without PIP2) is added. `analyze_counts` drops
  constant predictors and warns on perfect collinearity, but it cannot
  manufacture the missing contrast.
- **Statistical power is set by the number of independent replicas**, not by
  the number of frames. Aggregating to one count per replica (current
  default) avoids within-trajectory autocorrelation but means a few replicas
  per composition give limited power. Consider more replicas before trusting
  marginal p-values.
- Overdispersion is expected (crossings cluster in time). The regression uses
  scikit-learn's `PoissonRegressor` for point estimates and a **cluster
  bootstrap over replicas** for CIs / p-values, which is robust to
  overdispersion without needing a Negative-Binomial model. The Pearson
  dispersion is still reported as a diagnostic. The composition effect is
  assessed with a label **permutation test** rather than a likelihood-ratio
  test (scikit-learn exposes no likelihood). Trade-off: no closed-form NB or
  exact LR test, but no distributional assumption beyond the Poisson mean
  model either.

## Core module cleanup (done)

- `core/__init__.py` now declares an explicit public API (geometry,
  orientation, trajectory IO, the immutable observable system, vector-math
  primitives, backend dispatch, functional utilities) with a matching
  `__all__`, instead of only re-exporting three inertia helpers.
- `core/kernels.py`: dropped the PyTorch backend entirely (GPU path will be
  JAX). The dispatcher now supports `numba` (default) and `jax` (planned;
  `jax_kernels` not yet written, so selecting it raises a clear error).
  `torch_kernels.py` was already deleted from the working tree.

## Bugs fixed during core cleanup

- `core/orientation.py::extract_orientation_trajectory` returned
  `(rotation_matrix, euler_angles)` while its docstring, examples, and the
  only caller (`cli.py`) expected `(euler_angles, rotation_matrix)`. Fixed
  the return order to match the documented contract. **Check
  `src/rotmd/__init__.py::analyze_trajectory`**, which unpacks this function
  as a single array — it was relying on the (now corrected) behaviour and
  may need its own follow-up fix.
- `core/trajectory.py::load_trajectory` reset `velocities_list` /
  `forces_list` to `[]` *inside* the per-frame loop, discarding every frame
  but the last. Removed the in-loop reset so all frames accumulate.

## Pre-existing issues still outstanding

- `src/rotmd/__init__.py::analyze_trajectory` unpacks
  `extract_orientation_trajectory` (a `(euler, R)` tuple) as a single array,
  and passes mismatched arguments to `analyze_diffusion`. Not fixed here to
  avoid widening scope, but now more visible after the return-order fix.
- `src/rotmd/observables/diffusion.py::analyze_diffusion` is annotated to
  return a `Dict` but the implementation does not match the docstring's
  promised keys.
- Two `VectorObservable` implementations coexist: the immutable one in
  `core/observables_classes.py` (canonical, re-exported from `rotmd.core`,
  used by the CLI) and the older procedural one in
  `core/vector_observables.py` (still used by `observables/unified.py`).
  They should be unified to avoid drift; only the immutable one is part of
  the `rotmd.core` public API for now. **Decision locked (see `PLAN.md`
  Phase 0):** the immutable one is canonical; the procedural class is slated
  for removal in Phase 2.
- **`observables/unified.py` is broken at runtime** (found while locking
  PLAN.md Phase 0). `compute_angular_momentum`, `compute_torque`, and
  `compute_angular_velocity_from_inertia` call
  `decompose_vector_parallel(vec, axis, normal, times, name=..., verbose=...)`
  with 6 args, expecting a `VectorObservable` back, but
  `core/vector_observables.py::decompose_vector_parallel` takes
  `(vectors, reference_axis)` and returns a tuple. The intended function is
  `create_vector_observable(...)`, which also rejects `verbose=`. As a
  result the re-exported public `compute_all_observables`
  (`rotmd/__init__.py`, `observables/__init__.py`) raises `TypeError`. Fix
  is folded into PLAN.md Phase 2.
- `src/rotmd/io/output.py::save_results_npz` writes one file per key with
  inconsistent suffix handling, and computes `file_size` from `filename`
  rather than the path it actually wrote, so the reported size is wrong.
- `cli.py::main` routes any unrecognised subcommand to the legacy `extract`
  path, so a mistyped subcommand silently runs extraction instead of erroring.

## `analysis/nonequilibrium.py` correctness issues (found while scoping a
## possible lipid-bilayer-collective-behaviour use)

The math here is framed for protein rotation but is mostly generic. Several
functions, however, do **not** compute the quantity they advertise and should
be fixed (or removed) before being relied on for bilayer analysis:

- **`time_reversal_asymmetry` is a no-op disguise.** `backward_diffs` is
  defined as `observable[:-lag] - observable[lag:]`, i.e. exactly
  `-forward_diffs`. So `(forward - backward)**2 == 4 * forward_diffs**2` and
  the returned value is just `4 * <(O(t+lag) - O(t))**2>` — the mean-squared
  displacement of the observable, not any time-reversal asymmetry. A single
  scalar series cannot exhibit time-reversal-symmetry breaking through this
  construction at all; a real measure needs the cross-correlation of two
  observables with *different* time-reversal parity (or the area-enclosing
  current in a 2D collective-variable plane).
- **`irreversibility_index` measures ~nothing for the documented self-use.**
  The value histogram of a stationary series is invariant under time reversal,
  so if `trajectory_backward` is the time-reversed `trajectory_forward`
  (as the docstring suggests), the two flattened histograms are identical and
  `KL == 0`, `KS == 0` by construction. It only does something when the two
  inputs come from genuinely independent forward/reverse-protocol runs, and
  even then `.flatten()` discards the temporal/joint structure that carries
  the irreversibility signal.
- **`phase_space_compressibility` differentiates along atom index.**
  `np.gradient(velocities[:, dim])` and `np.gradient(positions[:, dim])` take
  finite differences over the *atom ordering* axis, which is arbitrary, then
  divides them. This is not a divergence of the velocity field and is not a
  meaningful phase-space compression rate.

Conceptual caveat (not a bug): a passive bilayer in equilibrium MD obeys
detailed balance, has zero entropy production, and satisfies FDT *by
construction*. The genuinely informative non-equilibrium signal for
"collective behaviour" is **broken detailed balance in projected
collective-variable space** (probability currents / curl in a 2D CV plane,
à la Battle et al. 2016), which `test_detailed_balance` only weakly
approximates for a 1D discrete-state sequence. A 2D-current primitive is the
missing piece if we pursue this direction.

## PSA (path similarity) — if needed, use the upstream package

If path-similarity analysis (Hausdorff / Fréchet path metrics, Seyler 2015) is
ever required, add `pathsimanalysis` as an **optional** dependency and wrap it,
mirroring the `twodanalysis`-backed `membrane` submodule pattern in
`analysis/__init__.py`. Do **not** re-vendor the source: it is GPLv2+, which is
incompatible with this project's MIT license, and the maintained package
already carries fixes (e.g. the recursive `discrete_frechet` recursion-limit
issue and matplotlib `tick_params` breakage in older snapshots).

## `analysis/transitions.py` caveats

- `transmission_coefficient` now returns an honest, bounded **recrossing
  estimate** (reactive fraction of barrier excursions), not the
  Bennett--Chandler transmission coefficient (the reactive-flux correlation
  plateau). If a true Bennett--Chandler kappa is needed, it must be computed
  from a dividing surface with velocities, which this discrete-state path does
  not provide.
- The module is standalone (not exported from `analysis/__init__.py`, no
  callers). Decide whether to expose it in the package API.

## `visualization/` review (concept good, execution outdated)

Project-specific plot helpers (phase portraits, L∥/L⊥ phase space, PMF
surfaces, Poincaré sections, ACF/spectra). Worth keeping, but needs a refactor.

Design:
- **No function returns `(fig, ax)`**; each ends in a terminal side effect, and
  inconsistently: `spectra.py`/`surfaces.py` call `plt.show()`, `phase_space.py`
  calls `plt.close()` (so those plots are invisible interactively unless
  `save_path` is given). Kills composability/testing. Refactor to take
  `ax=None` and return `(fig, ax)`, leaving show/save to the caller.
- `HAS_MATPLOTLIB` try/except + import-time `warnings.warn` is dead code:
  `matplotlib ^3.8` is a hard dependency. Remove the guards.
- `print("✓ Saved to ...")` from library code; use logging or nothing.

Broken integration:
- `rotmd/__init__.py::analyze_trajectory(save_plots=True)` imports
  `plot_trajectory_with_states` from `visualization.phase_space`, which **does
  not exist**. The `ImportError` is swallowed by the surrounding `except`, so
  `save_plots=True` silently produces no plots.
- All three `__main__` examples use the stale `protein_orientation.*` package
  name (should be `rotmd`).

Correctness bugs:
- `plot_energy_phase_space`: `pmf`/`*_bins`/`times` are `Optional=None` but used
  unconditionally (crashes on the documented call without bins); `c=energy[1::]`
  is length `n-1` vs `psi`/`theta` length `n` → scatter shape mismatch.
- `plot_poincare_section_improved`: `phi`/`theta = euler[1:-1]` (len n-2) but
  `omega_theta = angular_velocities[:,1]` (len n); boolean-indexing the latter
  with an (n-2) mask errors. Detects sections by proximity `|φ-φ₀|<tol` instead
  of sign-change crossings. Mutable default arg `section_angles=[...]`.
- `plot_phase_portrait_2d` / `..._with_vector_field`: per-frame `ax.plot` loop
  for time coloring (use `LineCollection`); hardcoded 0–2π xticks.
- `plot_phase_portrait_with_vector_field`: "vector field" uses a single
  `mean_torque` constant (comment admits unit conversion not done) → trivial
  constant field, not the flow the docstring claims.
- `plot_pmf_heatmap`: plots bins in radians but labels axes "degrees" and places
  minima markers via `np.degrees(...)` → mislabeled axes, misplaced markers;
  diverging `seismic` cmap for sequential PMF. Inconsistent with
  `plot_pmf_contour` (which uses degrees).
- `plot_autocorrelation`: bare `except:`.

## Dependency / env notes

- `pyproject.toml` declares `click` and `pydantic` but the source uses
  `argparse` and plain dataclasses; they appear unused.
- Added `pandas` and `scikit-learn` to `[tool.poetry.dependencies]` for the
  crossings regression (statsmodels was considered but dropped to avoid the
  extra dependency; scikit-learn's `PoissonRegressor` plus resampling covers
  the inference). Run `poetry lock && poetry install` inside the `devenv
  shell` to materialise them (the sandbox used for development has no Python
  scientific stack, so the new tests could not be executed here).
