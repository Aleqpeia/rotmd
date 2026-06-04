"""Tests for io.output: npz save/load round-trip and chunk merging."""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.io.output import load_npz, merge_npz, save_npz


def _chunk(t0, n_frames=3, n_atoms=4, masses=None):
    """A minimal chunk dict with a per-frame and a time-invariant array."""
    t = np.arange(n_frames, dtype=float) + t0
    if masses is None:
        masses = np.linspace(1.0, 2.0, n_atoms)
    return {
        "time_ps": t,
        "positions": np.ones((n_frames, n_atoms, 3)) * t0,
        "rmsd": t * 0.1,
        "masses": masses,
    }


def test_save_load_roundtrip(tmp_path):
    data = _chunk(0.0)
    p = save_npz(tmp_path / "out.npz", data)
    assert p.exists()
    loaded = load_npz(p)
    assert set(loaded) == set(data)
    for k in data:
        assert loaded[k] == pytest.approx(data[k])


def test_save_npz_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "chunk.npz"
    save_npz(target, _chunk(0.0))
    assert target.exists()


def test_merge_concatenates_time_and_keeps_invariants(tmp_path):
    a = save_npz(tmp_path / "a.npz", _chunk(0.0))
    b = save_npz(tmp_path / "b.npz", _chunk(3.0))
    out = merge_npz([a, b], tmp_path / "merged.npz")
    merged = load_npz(out)
    assert merged["time_ps"].shape == (6,)
    assert merged["positions"].shape == (6, 4, 3)
    # Time-invariant array is taken once, not concatenated.
    assert merged["masses"].shape == (4,)
    assert merged["time_ps"] == pytest.approx(np.arange(6.0))


def test_merge_requires_at_least_one_input(tmp_path):
    with pytest.raises(ValueError):
        merge_npz([], tmp_path / "x.npz")


def test_merge_requires_time_ps(tmp_path):
    p = save_npz(tmp_path / "no_time.npz", {"foo": np.arange(3.0)})
    with pytest.raises(ValueError, match="time_ps"):
        merge_npz([p], tmp_path / "out.npz")


def test_merge_rejects_key_mismatch(tmp_path):
    a = save_npz(tmp_path / "a.npz", _chunk(0.0))
    bad = _chunk(3.0)
    del bad["rmsd"]
    b = save_npz(tmp_path / "b.npz", bad)
    with pytest.raises(ValueError, match="key mismatch"):
        merge_npz([a, b], tmp_path / "out.npz")


def test_merge_rejects_nonmonotonic_time(tmp_path):
    # Second chunk starts before the first ends -> time not strictly increasing.
    a = save_npz(tmp_path / "a.npz", _chunk(0.0))
    b = save_npz(tmp_path / "b.npz", _chunk(1.0))
    with pytest.raises(ValueError, match="monotonic"):
        merge_npz([a, b], tmp_path / "out.npz")


def test_merge_rejects_inconsistent_invariant(tmp_path):
    a = save_npz(tmp_path / "a.npz", _chunk(0.0))
    b = save_npz(tmp_path / "b.npz", _chunk(3.0, masses=np.array([9.0, 9, 9, 9])))
    with pytest.raises(ValueError, match="time-invariant"):
        merge_npz([a, b], tmp_path / "out.npz")


def test_merge_single_chunk(tmp_path):
    a = save_npz(tmp_path / "a.npz", _chunk(0.0))
    out = merge_npz([a], tmp_path / "merged.npz")
    merged = load_npz(out)
    assert merged["time_ps"].shape == (3,)
