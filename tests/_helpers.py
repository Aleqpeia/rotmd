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


def _place_atom(a, b, c, bond, angle_deg, torsion_deg):
    """NeRF: position D from A, B, C plus |CD|, angle(BCD) and torsion(ABCD)."""
    angle, torsion = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d = np.array([
        -bond * np.cos(angle),
        bond * np.sin(angle) * np.cos(torsion),
        bond * np.sin(angle) * np.sin(torsion),
    ])
    return c + d[0] * bc + d[1] * m + d[2] * n


def build_backbone(n_res: int, phi: float = -57.0, psi: float = -47.0, omega: float = 180.0):
    """Ideal protein backbone (N, CA, C, O) at the given torsions.

    Defaults are the canonical alpha-helix, which reproduces a 3.8 Å CA-CA
    spacing and the i -> i+4 O...N hydrogen bond at ~3.0 Å, so DSSP genuinely
    assigns 'H'. Pass ``phi=-139, psi=135`` for an extended/beta strand.
    """
    b_nca, b_cac, b_cn, b_co = 1.458, 1.525, 1.329, 1.231
    a_ncac, a_cacn, a_cnca, a_caco = 111.2, 116.2, 121.7, 120.8

    n_atoms = [np.array([0.0, 0.0, 0.0])]
    ca_atoms = [np.array([b_nca, 0.0, 0.0])]
    theta = np.radians(a_ncac)
    c_atoms = [ca_atoms[0] + b_cac * np.array([-np.cos(theta), np.sin(theta), 0.0])]
    o_atoms = []

    for i in range(n_res):
        o_atoms.append(_place_atom(n_atoms[i], ca_atoms[i], c_atoms[i], b_co, a_caco, psi + 180.0))
        if i == n_res - 1:
            break
        n_atoms.append(_place_atom(n_atoms[i], ca_atoms[i], c_atoms[i], b_cn, a_cacn, psi))
        ca_atoms.append(_place_atom(ca_atoms[i], c_atoms[i], n_atoms[i + 1], b_nca, a_cnca, omega))
        c_atoms.append(_place_atom(c_atoms[i], n_atoms[i + 1], ca_atoms[i + 1], b_cac, a_ncac, phi))

    return np.array(n_atoms), np.array(ca_atoms), np.array(c_atoms), np.array(o_atoms)


def write_protein_trajectory(
    directory,
    n_res: int = 16,
    n_frames: int = 4,
    phi: float = -57.0,
    psi: float = -47.0,
    jitter: float = 0.02,
    seed: int = 5,
):
    """Write a real backbone protein (.pdb + .xtc) that DSSP can analyse.

    The all-CA system from :func:`write_synthetic_gromacs` has no N/C/O, so
    DSSP cannot run on it at all. This builds a genuine helix (or strand) with
    a little per-frame jitter so the trajectory is not perfectly static.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    from pathlib import Path

    directory = Path(directory)
    rng = np.random.default_rng(seed)
    n_atoms = n_res * 4

    n_a, ca, c_a, o_a = build_backbone(n_res, phi=phi, psi=psi)
    base = np.empty((n_atoms, 3))
    base[0::4], base[1::4], base[2::4], base[3::4] = n_a, ca, c_a, o_a

    frames = base[None] + rng.normal(size=(n_frames, n_atoms, 3)) * jitter

    u = mda.Universe.empty(
        n_atoms, n_residues=n_res,
        atom_resindex=np.repeat(np.arange(n_res), 4), trajectory=True,
    )
    u.add_TopologyAttr("names", ["N", "CA", "C", "O"] * n_res)
    u.add_TopologyAttr("elements", ["N", "C", "C", "O"] * n_res)
    u.add_TopologyAttr("types", ["N", "C", "C", "O"] * n_res)
    u.add_TopologyAttr("masses", [14.0, 12.0, 12.0, 16.0] * n_res)
    u.add_TopologyAttr("resnames", ["ALA"] * n_res)
    u.add_TopologyAttr("resids", list(range(1, n_res + 1)))
    u.add_TopologyAttr("segids", ["A"])
    u.load_new(frames, format=MemoryReader, dt=10.0)

    topology = directory / "protein.pdb"
    trajectory = directory / "protein.xtc"
    u.atoms.write(str(topology))
    with mda.Writer(str(trajectory), n_atoms=n_atoms) as w:
        for _ts in u.trajectory:
            w.write(u.atoms)

    return str(topology), str(trajectory), {"n_res": n_res, "n_frames": n_frames}


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
