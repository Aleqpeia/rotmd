"""
NPZ chunk I/O for the ``rotmd extract`` / ``rotmd merge`` CLI.

One ``rotmd extract`` invocation writes one compressed ``.npz`` per trajectory
chunk (the canonical schema documented in :mod:`rotmd.cli`); ``rotmd merge``
concatenates those chunks along the time axis. This module is the whole on-disk
contract for the extract worktree — nothing else reads or writes results.

Author: Mykyta Bobylyow
Date: 2025
"""

from pathlib import Path

import numpy as np


def save_npz(path: str | Path, data: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def merge_npz(inputs: list[Path], output: Path) -> Path:
    if not inputs:
        raise ValueError("at least one input chunk required")

    # Open chunks lazily: np.load returns an NpzFile that only decompresses a
    # key on access, so we never hold every array of every chunk in memory at
    # once. Peak memory is then dominated by the merged result itself (which we
    # must materialize for np.savez_compressed) plus one key's worth of sources.
    handles = [np.load(p) for p in inputs]
    try:
        # Every chunk must have time_ps.
        for f, p in zip(handles, inputs):
            if "time_ps" not in f.files:
                raise ValueError(f"{p} has no time_ps")

        # All chunks must share the same set of keys.
        ref_keys = set(handles[0].files)
        for f, p in zip(handles, inputs):
            if set(f.files) != ref_keys:
                missing = ref_keys - set(f.files)
                extra = set(f.files) - ref_keys
                parts = []
                if missing:
                    parts.append(f"missing: {sorted(missing)}")
                if extra:
                    parts.append(f"extra: {sorted(extra)}")
                raise ValueError(f"key mismatch with {p}: {'; '.join(parts)}")

        # Determine which keys are time-varying (first dim matches time_ps length).
        def _is_time_varying(arr: np.ndarray, n: int) -> bool:
            return arr.ndim >= 1 and arr.shape[0] == n

        # Detect time-invariant keys from the first chunk.
        n0 = len(handles[0]["time_ps"])
        invariant_keys = {
            k for k in ref_keys if k != "time_ps" and not _is_time_varying(handles[0][k], n0)
        }

        # Check time monotonicity across chunks.
        all_t = np.concatenate([f["time_ps"] for f in handles])
        if np.any(np.diff(all_t) <= 0):
            raise ValueError("time_ps is not strictly monotonic across chunks")

        # Check invariant arrays are consistent across chunks.
        for k in invariant_keys:
            ref = handles[0][k]
            for f in handles[1:]:
                if f[k].shape != ref.shape or not np.allclose(f[k], ref):
                    raise ValueError(f"time-invariant key {k} differs between chunks")

        # Concatenate time-varying arrays one key at a time so only a single
        # key's sources are decompressed simultaneously.
        merged: dict[str, np.ndarray] = {"time_ps": all_t}
        for k in (ref_keys - {"time_ps"} - invariant_keys):
            merged[k] = np.concatenate([f[k] for f in handles], axis=0)
        for k in invariant_keys:
            merged[k] = np.array(handles[0][k])

        return save_npz(output, merged)
    finally:
        for f in handles:
            f.close()
