"""Tests for manifest parsing and tidy count-table assembly."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("yaml")

from rotmd.analysis.composition import (  # noqa: E402
    Manifest,
    SystemRecord,
    build_count_table,
    load_manifest,
)


def _write_angle_npz(path, theta_deg, n=200):
    """Write a minimal extract-style NPZ with a theta (radians) series."""
    times = np.arange(n, dtype=float)
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    np.savez(path, theta=theta, phi=np.zeros(n), psi=np.zeros(n), times=times)


def test_system_record_from_dict_parses_paths_and_covariates():
    record = SystemRecord.from_dict(
        {
            "label": "complex",
            "replica": 2,
            "npz": "a.npz",
            "covariates": {"has_PIP2": 1, "has_CHOL": 1},
        }
    )
    assert record.label == "complex"
    assert record.npz.name == "a.npz"
    assert record.covariates == {"has_PIP2": 1.0, "has_CHOL": 1.0}


def test_load_manifest_roundtrip(tmp_path):
    manifest_yaml = tmp_path / "m.yaml"
    manifest_yaml.write_text(
        "spacing_deg: 20.0\n"
        "coordinates: [theta]\n"
        "systems:\n"
        "  - label: simple\n"
        "    replica: 1\n"
        f"    npz: {tmp_path / 'simple.npz'}\n"
    )
    manifest = load_manifest(manifest_yaml)
    assert manifest.spacing_deg == 20.0
    assert manifest.coordinates == ["theta"]
    assert len(manifest.systems) == 1


def test_build_count_table_from_theta_npz(tmp_path):
    # Wobbly 'simple' (big sweeps) vs steady 'complex' (small wiggle).
    wobbly = 90.0 + 60.0 * np.sin(np.linspace(0, 20 * np.pi, 200))
    steady = 90.0 + 2.0 * np.sin(np.linspace(0, 20 * np.pi, 200))
    _write_angle_npz(tmp_path / "simple.npz", wobbly)
    _write_angle_npz(tmp_path / "complex.npz", steady)

    manifest = Manifest(
        systems=[
            SystemRecord(
                label="simple", replica=1, npz=tmp_path / "simple.npz",
                covariates={"has_complex": 0},
            ),
            SystemRecord(
                label="complex", replica=1, npz=tmp_path / "complex.npz",
                covariates={"has_complex": 1},
            ),
        ],
        spacing_deg=15.0,
        coordinates=["theta"],
    )
    table = build_count_table(manifest, verbose=False)

    # Two systems x (crossings + direction_changes) = 4 rows.
    assert len(table) == 4
    assert set(table["response"]) == {"crossings", "direction_changes"}

    crossings = table[table["response"] == "crossings"].set_index("label")["count"]
    # The wobbly simple bilayer must show many more crossings than the steady one.
    assert crossings["simple"] > crossings["complex"]
    assert (table["exposure"] == 199.0).all()


def test_build_count_table_rejects_missing_coordinate(tmp_path):
    _write_angle_npz(tmp_path / "s.npz", np.full(50, 90.0))
    manifest = Manifest(
        systems=[SystemRecord(label="x", replica=1, npz=tmp_path / "s.npz")],
        coordinates=["tilt"],  # needs rotation_matrices, which the NPZ lacks
    )
    with pytest.raises(ValueError, match="rotation_matrices"):
        build_count_table(manifest, verbose=False)
