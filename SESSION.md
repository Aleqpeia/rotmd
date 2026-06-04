# Session notes

Incidental bugs / oddities found while reviewing and adding tests. Fix
opportunistically; not all are blocking.

## Bugs

- **`observables/unified.py` — extract is broken (FIXED).** `compute_angular_momentum`,
  `compute_torque`, and `compute_angular_velocity_from_inertia` called
  `decompose_vector_parallel(vec, axis, membrane_normal, times, name=..., verbose=...)`,
  but `core/vector_observables.decompose_vector_parallel` only accepts
  `(vectors, reference_axis)` and returns a tuple. The factory that builds a
  `VectorObservable` is `create_vector_observable(...)`. Result: `rotmd extract`
  died at step [5/6] with `TypeError: ... unexpected keyword argument 'name'`.

- **`observables/unified.py::compute_time_derivative` — dLdt was not decomposed
  (FIXED).** It used to return a `VectorObservable` whose `parallel`/`perp`/
  `z_component` (and magnitudes) were all zeros, so every `dLdt_parallel*`/
  `dLdt_perp*`/`dLdt_z*` column in the chunk schema was identically zero. Now it
  accepts an optional `reference_axis`/`membrane_normal` and `compute_all_observables`
  decomposes dL/dt against L's long axis.

- **`observables/unified.py::compute_time_derivative` — `forward`/`backward` were
  broken (FIXED).** `np.diff(vector, prepend=vector[0:1])` (shape `(n,3)`) was
  divided by `np.diff(times, prepend=times[0])` (shape `(n,)`), which both fails to
  broadcast *and* puts a zero (`times[0] - times[0]`) in the denominator at the
  duplicated edge → division by zero / NaN. These methods were untested. Rewrote
  them to use consecutive one-sided slopes
  `np.diff(v)/np.diff(t)[:,None]` with one edge slope duplicated to keep length `n`,
  and added regression tests covering uniform + non-uniform spacing.

- **`core/orientation.py::rotation_matrix_to_euler_zyz` — wrong φ/ψ (FIXED).**
  The off-diagonal atan2 formulas were swapped/mangled, so recovered φ and ψ were
  incorrect (θ was fine). Cross-checked against `scipy ... as_euler('ZYZ')`:
  forward `euler_zyz_to_rotation_matrix` matched scipy, but the inverse did not
  round-trip. Since `extract` stores `phi/psi` from this function, the stored Euler
  angles were wrong. Fixed to `phi=atan2(R[1,2],R[0,2])`, `psi=atan2(R[2,1],-R[2,0])`.

## Stale / inconsistent

- **`RUNTIME.md`** documents a torch/jax backend selector (`set_backend`,
  `get_backend`, `torch_kernels.py`, `numba_kernels.py`) that no longer exists.
  The real runtime is `core/kernels.py` (numba-optional, `ROTMD_NUMBA` env). Docs
  should be rewritten to match.

- **`numpy_kernels.py`** is staged in git (`A`) but absent on disk; `kernels.py`
  is the actual implementation.

- **`observables/potential.py`** requires `freesasa` (raised in
  `HydrophobicEnergy.__init__`, used by `TotalEnergy`), but `freesasa` is not a
  declared dependency/extra in `pyproject.toml`. Without it,
  `compute_trajectory_energies` cannot run. Consider adding a `sasa` extra and/or
  a graceful no-SASA fallback.

## Minor

- `core/orientation.py` and several observable modules still carry
  `if __name__ == "__main__"` demo blocks referencing the old
  `protein_orientation.*` package name.
- Docstring in `unified.py` calls principal-axis index 0 the "longest axis";
  `principal_axes` returns moments ascending, so index 0 is the *smallest*
  moment (which is the long axis for a rod). Wording is confusing.
