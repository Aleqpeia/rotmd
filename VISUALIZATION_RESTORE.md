# Visualization restore — defects found

The legacy `src/rotmd/visualization/` (`origin/altextract`, 1 493 lines across
`phase_space.py`, `spectra.py`, `surfaces.py`) is restored as `src/rotmd/viz/`,
rebuilt on a shared plotting API. This file records what was wrong with it,
because the whole point of the exercise is quality control of routines that get
trusted by eye.

Each item below was reproduced against the legacy code before being fixed, not
inferred from reading it. Every fix has a regression test in `tests/test_viz.py`.

## Defects

| # | Where | Defect | Evidence |
|---|---|---|---|
| 1 | `phase_space.plot_poincare_section_improved` | Sliced `euler_angles[1:-1]` but `angular_velocities[:]`, then indexed the length-`n` array with a mask built from the length-`n-2` one. | `IndexError: boolean index did not match indexed array ... size of axis is 2000` on well-formed input. The function could never run. |
| 2 | `surfaces.plot_pmf_heatmap` | Drew the mesh in **radians** and the minima markers in **degrees** — a factor of 57.3. | Mesh spans 0–6.28; markers land at x=184.6, y=86.9. Autoscale stretches the axis to 193, compressing the PMF into 3 % of the panel with the stars outside it. |
| 3 | `phase_space.plot_energy_phase_space` | Same radian/degree mix: radian-valued PMF contours behind a degree-valued scatter, on axes pinned to 0–360 / 0–180. | The landscape collapses into an invisible sliver in the corner of every figure the function produced. |
| 4 | `spectra.plot_spectral_density` | Took `real(fft(acf)) * dt` of a one-sided ACF — no factor of 2 for folding the negative lags. | For `C(t)=exp(-t/τ)`, τ=5: reported `S(0)=5.025`, truth `2τ=10`. Ratio **0.503**. Every value was half. |
| 5 | `spectra.plot_spectral_density` | Rectangle rule, so a constant `+C(0)·dt` bias on top of #4. | Residual of exactly `0.0500` at `dt=0.05`, `C(0)=1`, after the factor of 2 was corrected. |
| 6 | `surfaces._find_local_minima` | `values == minimum_filter(values, size=3)` is true for every cell of a flat region. | 25 markers on a single 5×5 flat basin. Broad, undersampled basins — the common case — came out sprayed. |
| 7 | `phase_space.plot_phase_portrait_with_vector_field` | `dω/dt` set to one global **mean** torque, so the "vector field" was constant everywhere. | The field cannot show the structure the function exists to show. The unit conversion its own comment describes was also never applied. |
| 8 | `phase_space.plot_energy_phase_space` | Coloured by `energy[1:]` against full-length angle arrays. | Every point's colour shifted by one frame, or a hard length error. |
| 9 | `phase_space.plot_energy_phase_space` | `pmf`, `theta_bins`, `psi_bins` documented as optional, dereferenced unconditionally. | `TypeError` on the documented call. |
| 10 | `spectra`, `surfaces` (all functions) | `plt.show()` then no close. | No-op on a headless node, and leaks every figure. |
| 11 | `phase_space` (all functions) | Bare `plt.close()` — closes whatever figure is *current*, not the one built. | Wrong figure closed whenever anything else is open. |
| 12 | all three modules | Trajectory colouring as `for i in range(n): ax.plot(...)`. | One `Line2D` per frame: 25 000 artists at this package's trajectory lengths. Minutes to render, unopenable vector output. |
| 13 | `spectra.plot_autocorrelation` | Bare `except:` around the curve fit. | Swallows `KeyboardInterrupt` and `SystemExit`; a failed fit is reported only as a warning. |
| 14 | `surfaces.plot_torque_vector_field` | `dpi=900` where every sibling used 300. | ~9× the file size, silently. |

Two further points are behavioural rather than defects, and are noted in the
module docstrings: the Poincaré "crossing" test was a proximity threshold
(`|φ − φ₀| < tol`), which counts a slow approach many times and misses a fast
transit entirely; and phase portraits hard-coded x ticks from 0 to 2π even for
θ, a polar angle confined to [0, π], stretching the axis to twice the occupied
range.

## What replaced it

`rotmd/viz/core.py` holds the API. A plot is a function that draws on axes it
is handed, and the `@figure` decorator gives it three calling modes:

```python
plot_pmf_heatmap(pmf, tb, pb, output="pmf.png")   # -> Path, figure closed
plot_pmf_heatmap(pmf, tb, pb)                     # -> Figure, for a notebook
plot_pmf_heatmap(pmf, tb, pb, ax=ax)              # -> draws into your axes
```

The third mode is what makes plots composable — `plot_multi_panel_summary`
now assembles the single-panel plots instead of reimplementing them. It was
impossible before, because every legacy function created and closed its own
figure.

The decorator also owns the things that were copied and drifted: the Agg
backend, the house style (`RC`), output directory creation, dpi, and closing
the figure even when the drawing code raises. Plots register by name, so a CLI
flag or config string reaches one without importing its module:

```python
from rotmd.viz import render, available
render("pmf-heatmap", pmf, theta_bins, psi_bins, output="pmf.png")
```

Shared helpers exist specifically to make defects #2, #3, #8 and #12
unrepresentable: `angle_grid_degrees` is the only way to build an angle mesh
and always returns degrees; `label_orientation_axes` sets matching labels and
limits; `time_coloured_path` draws a coloured path as a single `LineCollection`.

`analysis/plots.py` (the T2/T3 reviewer figures) takes the backend and save
helpers from `viz.core` but keeps its own signatures. It passes `output`
positionally, which the decorator makes keyword-only; migrating it would churn
every call site in `cli.py` and `tests/test_plots.py` for a cosmetic gain, on
figures that are currently in front of a reviewer.
