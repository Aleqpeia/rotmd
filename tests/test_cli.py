"""End-to-end tests for the `rotmd extract` / `rotmd merge` CLI.

The energy stage (`compute_trajectory_energies`) depends on the optional
`freesasa` package and on process-pool execution, so it is replaced with a
lightweight stub here. Everything else — trajectory loading, orientation,
principal axes, L/tau/omega/dLdt, structural parameters, kinetic energy, and
the npz writer — runs for real, giving high coverage of the extract path.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")

from rotmd import cli  # noqa: E402
from rotmd.io.output import load_npz, save_npz  # noqa: E402
from _helpers import write_synthetic_gromacs  # noqa: E402


def _fake_energies(n_frames, n_residues):
    """Stand-in for compute_trajectory_energies with the right shapes."""

    def _stub(*_args, **_kwargs):
        return {
            "Etot": np.linspace(-10.0, -5.0, n_frames),
            "Epol": np.linspace(-3.0, -1.0, n_frames),
            "Enonpol": np.linspace(-7.0, -4.0, n_frames),
            "per_residue": np.zeros((n_frames, n_residues)),
            "normal": np.array([0.0, 0.0, 1.0]),
        }

    return _stub


def _config(tmp_path, n_frames=5, n_residues=4, **overrides):
    top, trj, info = write_synthetic_gromacs(
        tmp_path, n_frames=n_frames, n_residues=n_residues
    )
    defaults = dict(
        topology=top,
        trajectory=trj,
        reference=top,
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
    # ExtractConfig wants Paths for the path-typed fields.
    from pathlib import Path

    for k in ("topology", "trajectory", "reference", "output"):
        defaults[k] = Path(defaults[k])
    return cli.ExtractConfig(**defaults), info


def test_extract_writes_full_schema(tmp_path):
    cfg, info = _config(tmp_path, n_frames=5, n_residues=4)
    with mock.patch(
        "rotmd.io.gromacs.compute_trajectory_energies",
        _fake_energies(5, info["n_residues"]),
    ):
        out = cli.extract(cfg)
    assert out.exists()
    data = load_npz(out)

    expected_keys = {
        "time_ps", "masses", "positions", "velocities", "forces", "com",
        "inertia_tensor", "axes", "moments", "phi", "theta", "psi",
        "rotation_matrices", "rmsd", "rg", "rg_components", "asphericity",
        "acylindricity", "end_to_end", "E_total", "E_pol", "E_nonpol",
        "E_per_residue", "K_trans",
        "L_vector", "L_magnitude", "tau_vector", "omega_vector", "dLdt_vector",
    }
    assert expected_keys <= set(data)

    n = 5
    assert data["time_ps"].shape == (n,)
    assert data["positions"].shape == (n, info["n_atoms"], 3)
    assert data["inertia_tensor"].shape == (n, 3, 3)
    assert data["axes"].shape == (n, 3, 3)
    assert data["L_vector"].shape == (n, 3)
    assert data["L_magnitude"].shape == (n,)
    assert data["E_total"].shape == (n,)
    # theta lives in [0, pi]; magnitudes are nonnegative.
    assert np.all(data["theta"] >= -1e-9) and np.all(data["theta"] <= np.pi + 1e-9)
    assert np.all(data["L_magnitude"] >= 0)


def test_extract_skips_when_output_exists(tmp_path):
    cfg, info = _config(tmp_path)
    cfg.output.write_bytes(b"sentinel")  # pre-existing file
    with mock.patch("rotmd.io.gromacs.compute_trajectory_energies") as energies:
        energies.side_effect = _fake_energies(5, info["n_residues"])
        out = cli.extract(cfg)
    # Skipped: the stub energy function is never called and the file is untouched.
    energies.assert_not_called()
    assert out.read_bytes() == b"sentinel"


def test_extract_force_overwrites(tmp_path):
    cfg, info = _config(tmp_path, force=True)
    cfg.output.write_bytes(b"sentinel")
    with mock.patch(
        "rotmd.io.gromacs.compute_trajectory_energies",
        _fake_energies(5, info["n_residues"]),
    ):
        out = cli.extract(cfg)
    # Overwritten with a real npz, so it is loadable.
    assert "time_ps" in load_npz(out)


def test_extract_reference_selection_mismatch_raises(tmp_path):
    # _load_reference must reject a reference whose selection atom count differs
    # from the trajectory selection (here: "all" selects more than 3 atoms).
    cfg, _info = _config(tmp_path, selection="all")
    with pytest.raises(ValueError, match="must match"):
        cli._load_reference(cfg, masses=np.ones(3))


def test_main_extract_dispatch(tmp_path):
    top, trj, info = write_synthetic_gromacs(tmp_path, n_frames=4, n_residues=4)
    out = tmp_path / "viamain.npz"
    argv = [
        "extract", top, trj, "--reference", top, "-o", str(out),
        "--selection", "all", "--membrane-sel", "resname ARG", "--n-workers", "1",
    ]
    with mock.patch(
        "rotmd.io.gromacs.compute_trajectory_energies",
        _fake_energies(4, info["n_residues"]),
    ):
        rc = cli.main(argv)
    assert rc == 0
    assert out.exists()


def test_main_merge_dispatch(tmp_path):
    # Build two schema-compatible chunks and merge them through the CLI.
    def chunk(t0):
        t = np.arange(3.0) + t0
        return {"time_ps": t, "positions": np.ones((3, 2, 3)) * t0, "masses": np.ones(2)}

    a = save_npz(tmp_path / "a.npz", chunk(0.0))
    b = save_npz(tmp_path / "b.npz", chunk(3.0))
    out = tmp_path / "merged.npz"
    rc = cli.main(["merge", str(a), str(b), "-o", str(out)])
    assert rc == 0
    assert load_npz(out)["time_ps"].shape == (6,)


def test_main_no_command_returns_error():
    with pytest.raises(SystemExit):
        cli.main([])  # argparse requires a subcommand
