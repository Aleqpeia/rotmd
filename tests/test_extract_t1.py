"""T1: the arrays and sidecar `rotmd extract` gained for the analysis plan.

Covers the CA subsystem (``ca_coords``/``ca_resids``/``ca_align_mask``),
per-domain RMSD, the ``meta.json`` sidecar, and the graceful-degradation paths
that let a positions-only trajectory or a freesasa-less environment still
produce a usable chunk.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

from _helpers import write_synthetic_gromacs
from rotmd import cli
from rotmd.io.meta import meta_path, read_json
from rotmd.io.output import load_npz


def _fake_energies(n_frames, n_residues):
    def _stub(*_args, **_kwargs):
        return {
            "Etot": np.linspace(-10.0, -5.0, n_frames),
            "Epol": np.linspace(-3.0, -1.0, n_frames),
            "Enonpol": np.linspace(-7.0, -4.0, n_frames),
            "per_residue": np.zeros((n_frames, n_residues)),
            "normal": np.array([0.0, 0.0, 1.0]),
        }

    return _stub


def _config(tmp_path, n_frames=5, n_residues=4, traj_kwargs=None, **overrides):
    top, trj, info = write_synthetic_gromacs(
        tmp_path, n_frames=n_frames, n_residues=n_residues, **(traj_kwargs or {})
    )
    defaults = dict(
        topology=Path(top),
        trajectory=Path(trj),
        reference=Path(top),
        output=tmp_path / "chunk.npz",
        selection="all",
        membrane_sel="resname ARG",
        start=0,
        stop=None,
        step=1,
        n_workers=1,
        force=False,
    )
    defaults.update(overrides)
    return cli.ExtractConfig(**defaults), info


def _run(cfg, n_frames, n_residues):
    with mock.patch(
        "rotmd.io.gromacs.compute_trajectory_energies", _fake_energies(n_frames, n_residues)
    ):
        return cli.extract(cfg)


# ---------------------------------------------------------------------------
# CA subsystem
# ---------------------------------------------------------------------------

def test_ca_arrays_written_with_expected_shapes(tmp_path):
    cfg, info = _config(tmp_path, n_frames=5, n_residues=4)
    data = load_npz(_run(cfg, 5, info["n_residues"]))

    n_ca = info["n_atoms"]  # every synthetic atom is named CA
    assert data["ca_coords"].shape == (5, n_ca, 3)
    assert data["ca_coords"].dtype == np.float32
    assert data["ca_resids"].shape == (n_ca,)
    assert data["ca_align_mask"].shape == (n_ca,)
    assert data["ca_align_mask"].dtype == bool
    # No --sel-align given => the whole CA set is the alignment core.
    assert data["ca_align_mask"].all()


def test_ca_coords_match_the_extracted_positions(tmp_path):
    # ca_coords must be a plain slice of `positions`, not a re-read or a
    # re-alignment: downstream stages assume the two share a frame of reference.
    cfg, info = _config(tmp_path, n_frames=4, n_residues=4)
    data = load_npz(_run(cfg, 4, info["n_residues"]))
    np.testing.assert_allclose(
        data["ca_coords"], data["positions"].astype(np.float32), rtol=0, atol=0
    )


def test_sel_align_restricts_the_alignment_core(tmp_path):
    cfg, info = _config(tmp_path, n_frames=4, n_residues=4, sel_align="resid 1-2")
    data = load_npz(_run(cfg, 4, info["n_residues"]))
    mask = data["ca_align_mask"]
    assert mask.sum() == 6  # residues 1-2, three atoms each
    assert set(data["ca_resids"][mask]) == {1, 2}


def test_sel_ca_matching_nothing_raises(tmp_path):
    cfg, _info = _config(tmp_path, sel_ca="name ZZZ")
    with pytest.raises(ValueError, match="selects no atoms"):
        cli.extract(cfg)


def test_sel_align_matching_nothing_raises(tmp_path):
    cfg, _info = _config(tmp_path, sel_align="resid 900-999")
    with pytest.raises(ValueError, match="--sel-align .* selects no atoms"):
        cli.extract(cfg)


def test_membrane_sel_matching_nothing_raises_before_the_expensive_work(tmp_path, capsys):
    """A typo'd --membrane-sel must not surface only at the energy stage.

    It is resolved inside compute_trajectory_energies, i.e. phase 7/7, so
    without an upfront check a long extract does all its loading, structural
    and orientation work and *then* dies having written nothing.
    """
    cfg, _info = _config(tmp_path, membrane_sel="resname ZZZ")
    with pytest.raises(ValueError, match="--membrane-sel .* selects no atoms"):
        cli.extract(cfg)
    assert "[1/7] Loading" not in capsys.readouterr().out


def test_membrane_sel_is_not_checked_when_energy_is_off(tmp_path):
    """--no-energy never resolves the membrane, so it must not demand one."""
    cfg, _info = _config(tmp_path, membrane_sel="resname ZZZ", no_energy=True)
    data = load_npz(cli.extract(cfg))
    assert "E_total" not in data


# ---------------------------------------------------------------------------
# Per-domain RMSD
# ---------------------------------------------------------------------------

def test_rmsd_dom_has_one_column_per_domain_in_spec_order(tmp_path):
    cfg, info = _config(tmp_path, n_frames=5, n_residues=4, domains="B:3-4,A:1-2")
    out = _run(cfg, 5, info["n_residues"])
    data = load_npz(out)
    assert data["rmsd_dom"].shape == (5, 2)
    assert np.all(data["rmsd_dom"] >= 0)
    # Column order follows the spec, and meta records the mapping.
    assert read_json(meta_path(out))["domain_names"] == ["B", "A"]


def test_rmsd_dom_is_empty_when_no_domains_requested(tmp_path):
    cfg, info = _config(tmp_path, n_frames=5, n_residues=4)
    data = load_npz(_run(cfg, 5, info["n_residues"]))
    assert data["rmsd_dom"].shape == (5, 0)


def test_domain_rmsd_is_self_superposed_not_global(tmp_path):
    # A domain aligned on itself can only deviate less than the whole
    # selection aligned on itself, since it optimises over fewer atoms.
    cfg, info = _config(tmp_path, n_frames=5, n_residues=4, domains="A:1-2")
    data = load_npz(_run(cfg, 5, info["n_residues"]))
    assert np.all(data["rmsd_dom"][:, 0] <= data["rmsd"] + 1e-9)


# ---------------------------------------------------------------------------
# Regression: RMSD/Rg must be untouched by T1
# ---------------------------------------------------------------------------

def test_rmsd_and_rg_are_bit_for_bit_unchanged(tmp_path):
    """Acceptance criterion for T1: adding arrays must not perturb rmsd/rg."""
    from rotmd.io.gromacs import load_gromacs_trajectory
    from rotmd.observables.structural import compute_structural_trajectory

    cfg, info = _config(tmp_path, n_frames=6, n_residues=4, domains="A:1-2")
    data = load_npz(_run(cfg, 6, info["n_residues"]))

    traj = load_gromacs_trajectory(
        str(cfg.topology), str(cfg.trajectory), selection=cfg.selection, verbose=False
    )
    reference = cli._load_reference(cfg, traj["masses"])
    expected = compute_structural_trajectory(
        traj["positions"], traj["masses"], reference=reference, verbose=False
    )
    np.testing.assert_array_equal(data["rmsd"], expected["rmsd"])
    np.testing.assert_array_equal(data["rg"], expected["rg"])


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_positions_only_trajectory_still_extracts(tmp_path):
    """A .xtc-style trajectory yields the DCCM path without L/tau/omega."""
    cfg, info = _config(
        tmp_path,
        n_frames=5,
        n_residues=4,
        traj_kwargs={"with_velocities": False, "with_forces": False},
    )
    data = load_npz(_run(cfg, 5, info["n_residues"]))

    # The CA/structural path survives...
    assert data["ca_coords"].shape == (5, info["n_atoms"], 3)
    assert data["rmsd"].shape == (5,)
    # ...while the dynamics-dependent fields are absent rather than zero-filled.
    assert "L_vector" not in data
    assert "K_trans" not in data
    assert "velocities" not in data
    meta = read_json(meta_path(cfg.output))
    assert meta["has_velocities"] is False
    assert meta["has_forces"] is False


def test_no_energy_flag_skips_energy_fields(tmp_path):
    cfg, _info = _config(tmp_path, n_frames=5, n_residues=4, no_energy=True)
    out = cli.extract(cfg)  # no energy mock needed — the stage must not run
    data = load_npz(out)
    assert "E_total" not in data
    assert "E_per_residue" not in data
    assert data["rmsd"].shape == (5,)
    assert read_json(meta_path(out))["has_energy"] is False


def test_missing_freesasa_skips_energy_without_failing(tmp_path):
    cfg, _info = _config(tmp_path, n_frames=4, n_residues=4)
    with mock.patch.object(cli, "_freesasa_available", return_value=False):
        out = cli.extract(cfg)
    assert "E_total" not in load_npz(out)
    assert read_json(meta_path(out))["has_energy"] is False


# ---------------------------------------------------------------------------
# meta.json sidecar
# ---------------------------------------------------------------------------

def test_meta_sidecar_records_provenance(tmp_path):
    cfg, info = _config(
        tmp_path,
        n_frames=5,
        n_residues=4,
        domains="A:1-2",
        sel_align="resid 1-2",
        system="wt",
        replica=3,
    )
    out = _run(cfg, 5, info["n_residues"])
    meta = read_json(meta_path(out))

    assert meta["system"] == "wt"
    assert meta["replica"] == 3
    assert meta["n_frames"] == 5
    assert meta["dt_ps"] == pytest.approx(2.0)
    assert meta["n_ca"] == info["n_atoms"]
    assert meta["n_align"] == 6
    assert meta["domains"] == {"A": [[1, 2]]}
    assert meta["sel_align"] == "resid 1-2"
    assert meta["window"] is None  # filled by T3
    assert meta["source"]["trajectory"] == str(cfg.trajectory)


def test_meta_sel_align_defaults_to_sel_ca(tmp_path):
    cfg, info = _config(tmp_path, n_frames=4, n_residues=4)
    out = _run(cfg, 4, info["n_residues"])
    meta = read_json(meta_path(out))
    assert meta["sel_align"] == meta["sel_ca"] == "name CA"


# ---------------------------------------------------------------------------
# merge carries the new arrays and collapses the sidecars
# ---------------------------------------------------------------------------

def test_merge_dedups_invariant_ca_arrays_and_merges_meta(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=6, n_residues=4)
    common = dict(
        topology=Path(top), trajectory=Path(trj), reference=Path(top),
        selection="all", membrane_sel="resname ARG", step=1,
        n_workers=1, force=False, domains="A:1-2", system="wt", replica=1,
    )
    outs = []
    for i, (start, stop) in enumerate([(0, 3), (3, 6)]):
        cfg = cli.ExtractConfig(
            output=tmp_path / f"c{i}.npz", start=start, stop=stop, **common
        )
        outs.append(_run(cfg, 3, info["n_residues"]))

    merged = tmp_path / "merged.npz"
    cli.merge(outs, merged, force=True)
    data = load_npz(merged)

    # Time-varying arrays concatenate...
    assert data["time_ps"].shape == (6,)
    assert data["ca_coords"].shape == (6, info["n_atoms"], 3)
    assert data["rmsd_dom"].shape == (6, 1)
    # ...while time-invariant CA metadata is stored once.
    assert data["ca_resids"].shape == (info["n_atoms"],)
    assert data["ca_align_mask"].shape == (info["n_atoms"],)

    meta = read_json(meta_path(merged))
    assert meta["n_frames"] == 6
    assert meta["n_chunks"] == 2
    assert meta["system"] == "wt"
    assert meta["domains"] == {"A": [[1, 2]]}


def test_merge_rejects_chunks_from_different_replicas(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=6, n_residues=4)
    common = dict(
        topology=Path(top), trajectory=Path(trj), reference=Path(top),
        selection="all", membrane_sel="resname ARG", step=1,
        n_workers=1, force=False,
    )
    outs = []
    for i, (start, stop, system) in enumerate([(0, 3, "wt"), (3, 6, "n75k")]):
        cfg = cli.ExtractConfig(
            output=tmp_path / f"c{i}.npz", start=start, stop=stop, system=system, **common
        )
        outs.append(_run(cfg, 3, info["n_residues"]))

    with pytest.raises(ValueError, match="not one replica"):
        cli.merge(outs, tmp_path / "merged.npz", force=True)
