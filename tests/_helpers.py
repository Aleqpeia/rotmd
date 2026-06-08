"""Shared builders for the rotmd test suite.

These are plain functions (not pytest fixtures) so they can be reused freely
and called with explicit arguments. They construct small, deterministic
synthetic systems — both raw NumPy arrays for the pure-physics tests and, when
MDAnalysis is available, on-disk GROMACS topology+TRR pairs for the I/O and CLI
integration tests.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Raw array systems (no MDAnalysis needed)
# ---------------------------------------------------------------------------

def random_trajectory(
    n_frames: int = 8,
    n_atoms: int = 24,
    seed: int = 0,
    scale: float = 5.0,
) -> dict:
    """A deterministic random trajectory with positions, velocities, forces.

    Masses are strictly positive. Positions are spread out (``scale``) so the
    inertia tensor is well-conditioned and eigen-decomposition is stable.
    """
    rng = np.random.default_rng(seed)
    masses = np.abs(rng.normal(size=n_atoms)) + 1.0
    positions = rng.normal(size=(n_frames, n_atoms, 3)) * scale
    velocities = rng.normal(size=(n_frames, n_atoms, 3))
    forces = rng.normal(size=(n_frames, n_atoms, 3))
    times = np.arange(n_frames, dtype=np.float64) * 2.0
    return {
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "masses": masses,
        "times": times,
        "n_frames": n_frames,
        "n_atoms": n_atoms,
    }


def ellipsoid_positions(
    n_atoms: int = 60, axes: tuple = (1.0, 2.0, 4.0), seed: int = 3
) -> np.ndarray:
    """Points scattered inside an axis-aligned ellipsoid.

    The intended principal-moment ordering is determined by ``axes``: a larger
    extent along an axis means a *smaller* moment of inertia about it. Used to
    check that ``principal_axes`` recovers a known geometry.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n_atoms, 3))
    return pts * np.asarray(axes)


def reference_inertia_tensor(
    positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Independent, vectorized reference implementation of the inertia tensor.

    Deliberately written differently from the production code (einsum identity
    minus outer product) so the tests cross-check the math rather than the
    implementation.
    """
    com = np.average(positions, weights=masses, axis=0)
    r = positions - com
    r2 = np.einsum("ij,ij->i", r, r)
    eye = np.eye(3)
    # I = Σ m (r²·δ - r⊗r)
    return np.einsum("a,ij->ij", masses * r2, eye) - np.einsum(
        "a,ai,aj->ij", masses, r, r
    )


# ---------------------------------------------------------------------------
# On-disk synthetic GROMACS systems (require MDAnalysis)
# ---------------------------------------------------------------------------

def write_synthetic_gromacs(
    directory,
    n_frames: int = 5,
    n_residues: int = 4,
    atoms_per_residue: int = 3,
    seed: int = 7,
    with_velocities: bool = True,
    with_forces: bool = True,
):
    """Write a small topology (.pdb) + trajectory (.trr) under ``directory``.

    Returns ``(topology_path, trajectory_path, info)`` where ``info`` carries
    the ground-truth arrays so tests can assert against them. The TRR carries
    velocities/forces only when requested, which lets us exercise the loader's
    capability-detection branches.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    from pathlib import Path

    directory = Path(directory)
    n_atoms = n_residues * atoms_per_residue
    rng = np.random.default_rng(seed)

    atom_resindex = np.repeat(np.arange(n_residues), atoms_per_residue)
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=atom_resindex,
        trajectory=True,
    )
    u.add_TopologyAttr("names", ["CA"] * n_atoms)
    u.add_TopologyAttr("types", ["C"] * n_atoms)
    u.add_TopologyAttr("masses", np.linspace(2.0, 16.0, n_atoms))
    # Use real residue names so the energy model's lookup tables hit.
    resnames = (["ALA", "ARG", "GLY", "ASP"] * n_residues)[:n_residues]
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("resids", list(range(1, n_residues + 1)))
    u.add_TopologyAttr("segids", ["A"])

    positions = rng.normal(size=(n_frames, n_atoms, 3)) * 6.0
    velocities = rng.normal(size=(n_frames, n_atoms, 3)) if with_velocities else None
    forces = rng.normal(size=(n_frames, n_atoms, 3)) if with_forces else None

    u.load_new(
        positions,
        format=MemoryReader,
        velocities=velocities,
        forces=forces,
        dt=2.0,
    )

    topology = directory / "top.pdb"
    trajectory = directory / "traj.trr"
    u.atoms.write(str(topology))
    with mda.Writer(str(trajectory), n_atoms=n_atoms) as w:
        for _ts in u.trajectory:
            w.write(u.atoms)

    info = {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "n_residues": n_residues,
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "resnames": resnames,
    }
    return str(topology), str(trajectory), info
