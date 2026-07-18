"""Residue-domain definitions for per-domain RMSD (T1) and figure annotation.

A whole-protein RMSD averages a rigid core together with flexible termini and
so plateaus at a value that belongs to neither. Splitting it per domain is what
makes "different segments give different answers" visible instead of hidden --
the spatial complement to the temporal window that ``rotmd equilibrate`` picks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# A domain maps to one or more inclusive [first_resid, last_resid] spans.
DomainSpec = dict[str, list[tuple[int, int]]]


def _parse_range(text: str, name: str) -> tuple[int, int]:
    if "-" not in text:
        raise ValueError(f"domain {name!r}: expected 'start-end', got {text!r}")
    lo_s, _, hi_s = text.partition("-")
    try:
        lo, hi = int(lo_s), int(hi_s)
    except ValueError as exc:
        raise ValueError(f"domain {name!r}: non-integer resid in {text!r}") from exc
    if lo > hi:
        raise ValueError(f"domain {name!r}: start {lo} > end {hi}")
    return lo, hi


def parse_domains(spec: str | None) -> DomainSpec:
    """Parse a domain specification into ``{name: [(first, last), ...]}``.

    Accepts either an inline spec or a path to a JSON file, so a quick
    exploratory run stays a one-liner while a production run keeps its domain
    definitions in a reviewable, version-controlled file::

        "EF1:20-35,EF2:56-70,N-lobe:1-90+100-110"   # '+' joins discontiguous spans
        "domains.json"  ->  {"EF1": [20, 35], "N-lobe": [[1, 90], [100, 110]]}

    Resids are **inclusive on both ends**, matching how residues are numbered in
    a PDB and how MDAnalysis' ``resid A:B`` selection behaves.
    """
    if not spec:
        return {}

    candidate = Path(spec)
    if candidate.suffix == ".json" or candidate.exists():
        raw = json.loads(candidate.read_text())
        out: DomainSpec = {}
        for name, value in raw.items():
            # Accept both [start, end] and [[s, e], [s, e]].
            pairs = [value] if value and np.ndim(value[0]) == 0 else value
            out[name] = [(int(a), int(b)) for a, b in pairs]
        return out

    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"domain spec {item!r}: expected 'name:start-end'")
        name, _, ranges = item.partition(":")
        name = name.strip()
        out[name] = [_parse_range(r.strip(), name) for r in ranges.split("+")]
    return out


def domain_masks(resids: np.ndarray, domains: DomainSpec) -> dict[str, np.ndarray]:
    """Boolean mask over ``resids`` for each domain.

    An empty domain is an error, not a warning: it would otherwise produce a
    silent all-NaN RMSD column that only surfaces as a blank panel in a figure
    much later.
    """
    resids = np.asarray(resids)
    masks = {}
    for name, spans in domains.items():
        mask = np.zeros(len(resids), dtype=bool)
        for lo, hi in spans:
            mask |= (resids >= lo) & (resids <= hi)
        if not mask.any():
            raise ValueError(
                f"domain {name!r} ({spans}) selects no residues; "
                f"available resid range is {resids.min()}-{resids.max()}"
            )
        masks[name] = mask
    return masks
