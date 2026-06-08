# PLAN: Make `rotmd.core` the authoritative integration surface

> Pick this up in a fresh session **after the current branch is merged**.
> It is self-contained: every target is referenced by file (and line at time
> of writing). Re-grep before editing in case line numbers have shifted.

## Goal & rationale

`src/rotmd/core/__init__.py` already declares a clean, curated public API
with `__all__` (geometry, orientation, trajectory IO, the immutable
observable system, vector-math primitives, backend dispatch, functional
utilities). However, almost every internal module bypasses it and imports
from submodules directly (e.g. `from rotmd.core.inertia import ...`). The
only consumer that goes through the package namespace today is `kernels`.

This plan makes the curated namespace **the real consumption path**, so:

- `__all__` becomes the single enforced contract.
- API and actual usage stop drifting apart.
- The duplicate `VectorObservable` is consolidated.

## Current state (migration targets)

External (non-`core`) consumers importing `core` submodules directly:

- `src/rotmd/__init__.py` — `.core.trajectory`, `.core.inertia`, `.core.orientation`
- `src/rotmd/observables/unified.py` — `core.vector_observables`, `core.inertia`
- `src/rotmd/observables/diffusion.py` — `..core.orientation` (3 lazy imports)
- `src/rotmd/observables/energetics.py` — `..core.inertia`
- `src/rotmd/observables/structural.py` — `..core.inertia` (2 sites)
- `src/rotmd/cli.py` — `core.functional`, `core.observables_classes`,
  `core import kernels`, `core.orientation`, `core.inertia`
- `src/rotmd/io/gromacs.py` — `core import kernels` (already namespace-correct)

## Critical constraint: no intra-`core` cycles

Migrate **only external consumers**. Modules *inside* `core` must keep
submodule-relative imports (e.g. `vector_observables.py` -> `from . import
kernels`; `observables_classes.py` -> `from rotmd.core.vector_observables
import ...`). If a core submodule imported `from rotmd.core import X`, it
would re-enter `core/__init__.py` mid-initialization and risk a circular
import. This is the most important rule of the plan.

## Phase 0 — Decisions (LOCKED 2026-06-08, pre-merge)

1. **`VectorObservable` duplication -> immutable wins.** The canonical type
   is the immutable `observables_classes.VectorObservable` (re-exported from
   `rotmd.core`, used by the CLI and the one that actually works). The
   procedural `vector_observables.VectorObservable` + `create_vector_observable`
   + `compute_spin_nutation_ratio` will be removed in Phase 2; only the free
   math functions (`decompose_vector_parallel`, `compute_magnitudes`,
   `compute_cross_product_trajectory`) stay in `vector_observables`.

   > **Discovered while locking this:** `observables/unified.py` is already
   > broken at runtime. `compute_angular_momentum` / `compute_torque` /
   > `compute_angular_velocity_from_inertia` call
   > `decompose_vector_parallel(vec, axis, normal, times, name=..., verbose=...)`
   > — 6 args expecting a `VectorObservable` back — but that function takes
   > `(vectors, reference_axis)` and returns a tuple. The intended call is
   > `create_vector_observable(...)`, which also does not accept `verbose=`.
   > So the re-exported public `compute_all_observables` raises `TypeError`.
   > This makes the procedural path effectively dead, reinforcing decision 1.

2. **Imports stay eager (no PEP 562 laziness now).** `import rotmd.core`
   continues to pull `numba` + `xarray` via `observables_classes` /
   `vector_observables`. Top-level `import rotmd` already triggers these, so
   there is no new cost. Revisit only if cold-import latency becomes a
   problem.

3. **`__all__` is complete for all consumers — verified exhaustively.**
   Enumerated every `rotmd.core` import across the whole repo (src, tests,
   examples) and cross-referenced each against `core.__all__`. Result: every
   imported name resolves to an exported name. Consumer-by-consumer:

   | Consumer | Symbols imported from `core` | In `__all__`? |
   | --- | --- | --- |
   | `rotmd/__init__.py` | `TrajectoryData`, `load_trajectory`, `validate_trajectory`, `inertia_tensor`, `principal_axes`, `extract_orientation_trajectory`, `rotation_matrix_to_euler_zyz`, `euler_zyz_to_rotation_matrix`, `rotation_matrix_to_quaternion`, `quaternion_to_rotation_matrix`, `compute_angular_displacement`, `unwrap_euler_angles` | yes |
   | `observables/structural.py` | `inertia_tensor`, `principal_axes` | yes |
   | `observables/energetics.py` | `inertia_tensor`, `principal_axes` | yes |
   | `observables/diffusion.py` | `compute_angular_displacement`, `unwrap_euler_angles` | yes |
   | `observables/unified.py` | `compute_cross_product_trajectory`, `decompose_vector_parallel`, `VectorObservable`, `compute_center_of_mass` | yes* |
   | `cli.py` | `Pipeline`, `Maybe`, `compute_all_observables_functional`, `kernels`, `extract_orientation_trajectory`, `principal_axes` | yes |
   | `io/gromacs.py` | `kernels` | yes |
   | `tests/`, `examples/` | (none import `rotmd.core`) | n/a |

   **\*Critical caveat for Phase 1/2:** `unified.py`'s `VectorObservable`
   binds to the *procedural* `vector_observables.VectorObservable`, whereas
   `core.VectorObservable` (in `__all__`) is the *immutable*
   `observables_classes.VectorObservable` — a different class with a
   different attribute API (`.vector`/`.magnitude` arrays vs `.data` /
   `.magnitude.data`). Therefore `unified.py` must **not** be migrated with a
   blind `from ..core import VectorObservable`; it is handled by the Phase 2
   rewrite instead. Every other consumer can be safely switched to
   `from ..core import ...` in Phase 1.

## Phase 1 — Migrate external consumers to the namespace

Mechanical, low-risk rewrites; verify each by import + tests:

- `rotmd/__init__.py`: collapse the three `from .core.* import ...` blocks
  into a single `from .core import (...)`.
- `observables/diffusion.py`, `energetics.py`, `structural.py`: change
  `from ..core.orientation import ...` / `from ..core.inertia import ...`
  to `from ..core import ...`.
- `observables/unified.py`: `from ..core import compute_center_of_mass,
  decompose_vector_parallel, compute_magnitudes,
  compute_cross_product_trajectory` (after Phase 2 removes the class dep).
- `cli.py`: route `Pipeline`/`Maybe`,
  `compute_all_observables_functional`, and the lazy
  `orientation`/`inertia` imports through `from rotmd.core import ...`.
  Keep `from rotmd.core import kernels as K` as-is.
- `io/gromacs.py`: no change.

## Phase 2 — Consolidate the duplicate observable (and fix the broken path)

Per Phase 0, `observables/unified.py` is currently broken, so this is a fix
plus a consolidation:

1. Reimplement `compute_all_observables` (and the per-observable helpers, if
   kept) as a thin wrapper over
   `core.compute_all_observables_functional`, returning the immutable
   `VectorObservable`s. This both fixes the `TypeError` and removes the
   dependency on the procedural class. Watch for attribute-name changes:
   callers using `.vector` / `.magnitude` (arrays) must move to the
   immutable API (`.data` / `.magnitude.data`).
2. Remove `VectorObservable`, `create_vector_observable`,
   `compute_spin_nutation_ratio` from `vector_observables.py` (keep the math
   functions), updating that module's `__all__`.
3. Update the top-level / `observables` re-exports and mark the redundancy
   resolved in `SESSION.md`.

## Phase 3 — Tighten the contract

- Add a test asserting every name in `rotmd.core.__all__` is importable.
- Add a guard test (grep/AST) that non-`core` modules do not import
  `rotmd.core.<submodule>` directly, preventing regression.
- Add a `from rotmd import core` smoke import to the suite.

## Phase 4 — Verification

- In the toolbx venv (sandbox lacks numba/xarray):
  `python -c "import rotmd, rotmd.core, rotmd.cli"`.
- Run `pytest` (esp. observables + the CLI extract path).
- `git diff --stat` review; commit per repo conventions (one commit per
  phase so Phase 2 can be reverted independently).

## Risks & mitigations

- **Circular imports** -> Phase 1 touches only non-core modules.
- **Hidden reliance on a non-exported symbol** -> caught in Phase 0.3 and
  the Phase 3 import smoke test.
- **Behavior change from the `VectorObservable` swap** -> covered by
  observables tests; keep Phase 2 as its own commit.

## Out of scope (tracked in `SESSION.md`)

- `analyze_trajectory` unpacking bug in `rotmd/__init__.py`.
- `diffusion.analyze_diffusion` return-type mismatch.
- Implementing the `jax` kernel backend (`rotmd.core.jax_kernels`).
