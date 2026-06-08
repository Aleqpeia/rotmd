# Session notes

Deferred / incidental issues noticed while working. These are not
necessarily blocking; recorded here so they can be addressed later.

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

## Pre-existing bugs spotted (not fixed)

- `src/rotmd/__init__.py::analyze_trajectory` appears to unpack
  `extract_orientation_trajectory` (which returns a `(euler, R)` tuple) as a
  single array, and passes mismatched arguments to `analyze_diffusion`.
- `src/rotmd/observables/diffusion.py::analyze_diffusion` is annotated to
  return a `Dict` but the implementation does not match the docstring's
  promised keys.
- Two `VectorObservable` implementations coexist
  (`core/vector_observables.py` vs `core/observables_classes.py`); they
  should probably be unified to avoid drift.
- `src/rotmd/io/output.py::save_results_npz` writes one file per key with
  inconsistent suffix handling, and computes `file_size` from `filename`
  rather than the path it actually wrote, so the reported size is wrong.
- `cli.py::main` routes any unrecognised subcommand to the legacy `extract`
  path, so a mistyped subcommand silently runs extraction instead of erroring.

## Dependency / env notes

- `pyproject.toml` declares `click` and `pydantic` but the source uses
  `argparse` and plain dataclasses; they appear unused.
- Added `pandas` and `scikit-learn` to `[tool.poetry.dependencies]` for the
  crossings regression (statsmodels was considered but dropped to avoid the
  extra dependency; scikit-learn's `PoissonRegressor` plus resampling covers
  the inference). Run `poetry lock && poetry install` inside the `devenv
  shell` to materialise them (the sandbox used for development has no Python
  scientific stack, so the new tests could not be executed here).
