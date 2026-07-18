"""T4: dynamic cross-correlation maps.

Built on trajectories whose correlation structure is imposed by construction —
a pair forced to move together must come out at +1, a pair forced to move in
opposition at -1 — plus the rigid-body test that proves the superposition is
doing its job.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotmd.analysis.dccm import (
    _superpose,
    average_structure,
    compute_dccm,
    cross_correlation,
)


def _random_rotations(n, seed=0):
    """n uniformly random rotation matrices via QR of gaussian matrices."""
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        mats.append(q)
    return np.array(mats)


# ---------------------------------------------------------------------------
# cross_correlation: the normalisation itself
# ---------------------------------------------------------------------------

def test_perfectly_correlated_and_anticorrelated_pairs():
    rng = np.random.default_rng(0)
    n_frames = 500
    d = np.zeros((n_frames, 4, 3))
    signal = rng.normal(size=(n_frames, 3))
    d[:, 0] = signal
    d[:, 1] = 2.0 * signal          # same direction, different amplitude
    d[:, 2] = -signal               # exact opposition
    d[:, 3] = rng.normal(size=(n_frames, 3))  # independent

    dcc = cross_correlation(d)
    assert dcc[0, 1] == pytest.approx(1.0, abs=1e-9)   # amplitude is normalised out
    assert dcc[0, 2] == pytest.approx(-1.0, abs=1e-9)
    assert abs(dcc[0, 3]) < 0.15


def test_map_is_symmetric_with_unit_diagonal_and_bounded():
    rng = np.random.default_rng(1)
    dcc = cross_correlation(rng.normal(size=(300, 12, 3)))
    np.testing.assert_allclose(dcc, dcc.T, atol=0)      # exactly symmetric
    np.testing.assert_allclose(np.diag(dcc), 1.0, atol=1e-12)
    assert dcc.min() >= -1.0 - 1e-12
    assert dcc.max() <= 1.0 + 1e-12


def test_immobile_atom_yields_zero_not_nan():
    rng = np.random.default_rng(2)
    d = rng.normal(size=(100, 3, 3))
    d[:, 1] = 0.0                                       # atom 1 never moves
    dcc = cross_correlation(d)
    assert np.isfinite(dcc).all()
    assert dcc[1, 0] == 0.0
    assert dcc[1, 1] == 0.0


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

def test_superpose_removes_rigid_rotation_and_translation():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(10, 3)) * 5.0
    rots = _random_rotations(40, seed=4)
    shifts = rng.normal(size=(40, 1, 3)) * 20.0
    coords = np.einsum("fij,nj->fni", rots, ref) + shifts

    mask = np.ones(10, dtype=bool)
    aligned = _superpose(coords, ref, mask)
    # Every frame is the reference again, up to floating point.
    for frame in aligned:
        np.testing.assert_allclose(frame, ref, atol=1e-9)


def test_rigid_body_motion_produces_no_apparent_fluctuation():
    """The headline correctness property: tumbling must not look like dynamics."""
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(12, 3)) * 5.0
    rots = _random_rotations(60, seed=6)
    coords = np.einsum("fij,nj->fni", rots, ref) + rng.normal(size=(60, 1, 3)) * 10.0

    result = compute_dccm(coords, np.ones(12, dtype=bool), np.arange(1, 13))
    assert result["rmsf"].max() < 1e-6      # nothing actually moved internally


def test_average_structure_converges_and_is_frame_order_independent():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(8, 3)) * 4.0
    coords = ref + rng.normal(size=(50, 8, 3)) * 0.2
    mask = np.ones(8, dtype=bool)

    _, mean_a, n_iter = average_structure(coords, mask)
    _, mean_b, _ = average_structure(coords[::-1], mask)

    assert n_iter >= 1
    # Shuffling frames must not move the average structure (up to a global
    # rigid transform, which we remove by superposing one onto the other).
    aligned_b = _superpose(mean_b[None], mean_a, mask)[0]
    np.testing.assert_allclose(aligned_b, mean_a, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_dccm end to end
# ---------------------------------------------------------------------------

def _correlated_trajectory(n_frames=400, n_ca=10, seed=8):
    """Atoms 0-1 move together, 2-3 in opposition, the rest independently."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(n_ca, 3)) * 6.0
    disp = rng.normal(size=(n_frames, n_ca, 3)) * 0.3
    shared = rng.normal(size=(n_frames, 3)) * 0.5
    disp[:, 0] += shared
    disp[:, 1] += shared
    disp[:, 2] += shared
    disp[:, 3] -= shared
    return ref + disp, ref


def test_raw_correlation_matches_the_analytic_value():
    """Without superposition the estimator must hit the closed-form answer.

    Atoms 0 and 1 each carry independent noise (variance 0.3**2) plus a shared
    term (variance 0.5**2), so their correlation is exactly
    0.25 / (0.09 + 0.25) = 0.735.
    """
    coords, _ = _correlated_trajectory(n_frames=20000, seed=11)
    raw = coords - coords.mean(axis=0)
    assert cross_correlation(raw)[0, 1] == pytest.approx(0.25 / 0.34, rel=0.05)


def test_compute_dccm_recovers_imposed_correlations():
    coords, _ = _correlated_trajectory()
    dcc = compute_dccm(coords, np.ones(10, dtype=bool), np.arange(1, 11))["dcc"]

    # Atoms 0, 1, 2 share a displacement; atom 3 carries its negative.
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert dcc[i, j] > 0.35, f"co-moving pair {i},{j} came out at {dcc[i, j]:.3f}"
    for i in (0, 1, 2):
        assert dcc[i, 3] < -0.5, f"opposed pair {i},3 came out at {dcc[i, 3]:.3f}"
    assert abs(dcc[5, 6]) < 0.25

    # Superposition attenuates the co-moving signal (0.74 raw -> ~0.46 here)
    # because part of a shared displacement *is* rigid-body motion and is
    # removed by the fit. The ordering must survive regardless.
    assert dcc[0, 1] > abs(dcc[5, 6])


def test_compute_dccm_output_contract():
    coords, _ = _correlated_trajectory()
    resids = np.arange(1, 11)
    result = compute_dccm(coords, np.ones(10, dtype=bool), resids)

    assert result["dcc"].shape == (10, 10)
    np.testing.assert_allclose(result["dcc"], result["dcc"].T, atol=0)
    np.testing.assert_allclose(np.diag(result["dcc"]), 1.0, atol=1e-12)
    np.testing.assert_array_equal(result["resids"], resids)
    assert result["rmsf"].shape == (10,)
    assert int(result["frames_used"]) == 400


def test_frame_mask_restricts_to_the_window():
    coords, _ = _correlated_trajectory(n_frames=400)
    mask = np.zeros(400, dtype=bool)
    mask[300:] = True
    result = compute_dccm(coords, np.ones(10, dtype=bool), np.arange(1, 11), frame_mask=mask)
    assert int(result["frames_used"]) == 100


def test_window_changes_the_map():
    """Guards against the frame mask being silently ignored."""
    coords, _ = _correlated_trajectory(n_frames=400)
    full = compute_dccm(coords, np.ones(10, dtype=bool), np.arange(1, 11))["dcc"]
    mask = np.zeros(400, dtype=bool)
    mask[200:] = True
    part = compute_dccm(coords, np.ones(10, dtype=bool), np.arange(1, 11), frame_mask=mask)["dcc"]
    assert not np.allclose(full, part)


def test_alignment_core_can_be_a_subset():
    coords, _ = _correlated_trajectory(n_ca=10)
    core = np.zeros(10, dtype=bool)
    core[:6] = True
    result = compute_dccm(coords, core, np.arange(1, 11))
    assert result["dcc"].shape == (10, 10)      # fit on 6, measured on all 10
    assert int(result["align_mask"].sum()) == 6


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(ca_coords=np.zeros((5, 4))), "must be"),
        (dict(align_mask=np.ones(3, dtype=bool)), "does not match"),
        (dict(align_mask=np.zeros(10, dtype=bool)), "no atoms"),
    ],
)
def test_input_validation(kwargs, match):
    base = dict(
        ca_coords=np.zeros((5, 10, 3)),
        align_mask=np.ones(10, dtype=bool),
        resids=np.arange(10),
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        compute_dccm(**base)


def test_too_few_frames_in_window_raises():
    coords, _ = _correlated_trajectory(n_frames=50)
    mask = np.zeros(50, dtype=bool)
    mask[-1] = True
    with pytest.raises(ValueError, match="at least 2 frames"):
        compute_dccm(coords, np.ones(10, dtype=bool), np.arange(10), frame_mask=mask)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_dccm_end_to_end(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import write_json
    from rotmd.io.output import load_npz, save_npz

    coords, _ = _correlated_trajectory(n_frames=300, n_ca=10)
    npz = save_npz(tmp_path / "merged.npz", {
        "time_ps": np.arange(300) * 10.0,
        "ca_coords": coords.astype(np.float32),
        "ca_resids": np.arange(1, 11, dtype=np.int32),
        "ca_align_mask": np.ones(10, dtype=bool),
    })
    window = write_json(tmp_path / "window.json", {"t0_ps": 1000.0})
    out = tmp_path / "dccm.npz"

    assert main(["dccm", str(npz), "--window", str(window), "-o", str(out)]) == 0

    result = load_npz(out)
    assert result["dcc"].shape == (10, 10)
    assert int(result["frames_used"]) == 200      # frames at t >= 1000 ps
    assert float(result["t0_ps"]) == 1000.0
    np.testing.assert_array_equal(result["resids"], np.arange(1, 11))


def test_cli_dccm_rejects_npz_without_ca_arrays(tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import write_json
    from rotmd.io.output import save_npz

    npz = save_npz(tmp_path / "old.npz", {"time_ps": np.arange(10.0)})
    window = write_json(tmp_path / "window.json", {"t0_ps": 0.0})
    with pytest.raises(SystemExit, match="predates the CA arrays"):
        main(["dccm", str(npz), "--window", str(window), "-o", str(tmp_path / "d.npz")])
