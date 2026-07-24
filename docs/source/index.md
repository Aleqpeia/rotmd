# rotmd

**rotmd** analyzes how a protein bound to a membrane surface reorients over
the course of a molecular dynamics trajectory — its tilt and spin relative to
the membrane normal, the angular momentum and friction that drive those
motions, and how all of that changes between conditions (e.g. wild type vs. a
point mutant).

The toolkit is organized around one pipeline:

```{code-block} text
extract  ->  merge  ->  equilibrate  ->  analyze (dccm / dssp / local / apbs / coulomb / …)  ->  compare / methods
```

`extract` turns a raw GROMACS trajectory chunk into per-frame observables
(inertia tensor, Euler angles, angular momentum, RMSD, energetics, …) written
to a single `.npz` + `meta.json` sidecar; everything downstream consumes that
`.npz` and the equilibration window it settles on, rather than re-reading the
trajectory. See {doc}`pipeline` for the full walkthrough and {doc}`quickstart`
to run it.

```{toctree}
:maxdepth: 2
:caption: Guide

quickstart
pipeline
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
