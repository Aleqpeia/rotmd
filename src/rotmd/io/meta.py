"""JSON sidecars for the extract pipeline: ``meta.json`` and ``window.json``.

The ``.npz`` stays the only numeric artifact (a slim-extract invariant); these
sidecars carry the *metadata* needed to interpret it — selections, domain
definitions, provenance, and the equilibration window that every post-extract
analysis slices on.

Two files, two lifetimes:

``<chunk>.meta.json``
    Written by ``rotmd extract`` next to each chunk, merged by ``rotmd merge``
    into one sidecar per replica. Extended in place by later stages.

``window.json``
    Written by ``rotmd equilibrate``. The **single source of truth** for the
    production window: no analysis may hard-code a frame range, it reads
    ``t0_ps`` from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class _NpEncoder(json.JSONEncoder):
    """Make numpy scalars/arrays JSON-serializable.

    ``np.int64`` and friends are not instances of ``int``/``float``, so a plain
    ``json.dump`` of anything derived from an array raises TypeError. Every
    value in these sidecars comes from numpy somewhere, so this is not optional.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, cls=_NpEncoder)
        fh.write("\n")
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return json.load(fh)


def meta_path(npz_path: str | Path) -> Path:
    """``work/rep1/chunk3.npz`` -> ``work/rep1/chunk3.meta.json``."""
    return Path(npz_path).with_suffix(".meta.json")


def update_meta(path: str | Path, **updates: Any) -> dict:
    """Shallow-merge ``updates`` into an existing sidecar and rewrite it.

    Later stages (T3 writes ``window``) extend the sidecar rather than
    replacing it, so a missing file is an error worth surfacing: it means
    ``extract`` never ran for this replica.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no metadata sidecar at {path} — run `rotmd extract` first")
    meta = read_json(path)
    meta.update(updates)
    write_json(path, meta)
    return meta


def merge_meta(metas: list[dict], time_ps: np.ndarray) -> dict:
    """Collapse per-chunk sidecars into one per-replica sidecar.

    Chunks of one replica share every selection and domain definition, so the
    merged sidecar is chunk 0's with the frame-count and time-span fields
    recomputed over the concatenation. Disagreement on a selection means the
    chunks are not from the same run, which would silently corrupt every
    downstream comparison — so it raises.
    """
    if not metas:
        raise ValueError("at least one metadata sidecar required")

    merged = dict(metas[0])
    keys = ("selection", "sel_ca", "sel_align", "domains", "system", "replica")
    for other in metas[1:]:
        for k in keys:
            if other.get(k) != merged.get(k):
                raise ValueError(
                    f"chunk metadata disagrees on {k!r}: "
                    f"{merged.get(k)!r} vs {other.get(k)!r} — chunks are not one replica"
                )

    merged["n_frames"] = len(time_ps)
    merged["t_start_ps"] = float(time_ps[0])
    merged["t_end_ps"] = float(time_ps[-1])
    merged["dt_ps"] = float(np.median(np.diff(time_ps))) if len(time_ps) > 1 else None
    merged["n_chunks"] = len(metas)
    return merged
