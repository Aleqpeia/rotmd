"""Poisson-Boltzmann electrostatics via PDB2PQR + APBS (T5c).

Computes the polar solvation free energy ``ΔG_elec`` as an *ensemble* average
over frames sampled from the production window, which is the whole point: a
single-frame PB number on a flexible protein is dominated by whichever
side-chain rotamer that frame happened to catch. Comparing WT and a charge
mutant demands the average and its spread, not one snapshot.

``ΔG_elec`` is the difference between two identical calculations that differ
only in the solvent dielectric — solvated (``sdie = 78.54``) minus a reference
at the solute dielectric (``sdie = pdie``). The grid, centring and charge
assignment cancel between them, so the difference is a real solvation free
energy rather than a grid artefact.

External tools (not Python dependencies):
    pdb2pqr30   assigns CHARMM charges/radii and adds missing hydrogens
    apbs        solves the linearised PB equation
Both are located at run time; their absence is reported, never silently
skipped.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# APBS multigrid requires dime = c*2^(l+1) + 1; these are the usual choices.
_VALID_DIME = (33, 49, 65, 81, 97, 113, 129, 161, 193, 225, 257)

KJ_PER_KCAL = 4.184


def find_tool(name: str, extra: list[str] | None = None) -> str:
    """Locate an external binary needed by the PB pipeline.

    Searches PATH, then the directory holding the running interpreter. That
    second location matters: ``pdb2pqr30`` is a pip console script installed
    beside ``python`` in the virtualenv, so it is *not* on PATH whenever rotmd
    is invoked by absolute interpreter path (which is normal in SLURM scripts).
    """
    import sys

    found = shutil.which(name)
    if found:
        return found

    candidates = [Path(sys.executable).parent / name, *(Path(c) for c in extra or [])]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"{name!r} not found on PATH nor beside the interpreter "
        f"({Path(sys.executable).parent}). Install it — `pip install pdb2pqr` for "
        "pdb2pqr30, `micromamba install -c conda-forge apbs` for apbs — or put it "
        "on PATH."
    )


def read_pqr(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(coordinates, radii)`` from a PQR file.

    PQR is whitespace-delimited with charge and radius as the last two fields,
    so the coordinates are fields -5..-3 counting from the end. Parsing from the
    right survives the wide atom/residue-number columns that break fixed-width
    PDB parsing on large systems.
    """
    coords, radii = [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        coords.append([float(v) for v in fields[-5:-2]])
        radii.append(float(fields[-1]))
    if not coords:
        raise ValueError(f"no ATOM/HETATM records in {path}")
    return np.asarray(coords, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def grid_dimensions(
    coords: np.ndarray,
    radii: np.ndarray,
    fine_spacing: float = 0.5,
    max_dime: int = 161,
) -> dict[str, Any]:
    """APBS ``mg-auto`` grid parameters for a molecule (the psize heuristic).

    ``cglen`` is the coarse box at 1.7x the molecule; ``fglen`` is the focused
    box, the molecule plus 20 Å of solvent. ``dime`` is the smallest legal grid
    reaching ``fine_spacing``, capped at ``max_dime`` so memory stays bounded
    (a 161³ grid is ~4M points).
    """
    lower = (coords - radii[:, None]).min(axis=0)
    upper = (coords + radii[:, None]).max(axis=0)
    extent = upper - lower

    cglen = 1.7 * extent
    fglen = np.minimum(20.0 + extent, cglen)

    dime = []
    for length in fglen:
        needed = length / fine_spacing + 1
        choice = next((v for v in _VALID_DIME if v >= needed), _VALID_DIME[-1])
        dime.append(min(choice, max_dime))

    return {
        "dime": dime,
        "cglen": cglen.tolist(),
        "fglen": fglen.tolist(),
        "centre": ((upper + lower) / 2).tolist(),
        "extent": extent.tolist(),
        "actual_spacing": (fglen / (np.asarray(dime) - 1)).tolist(),
    }


_ELEC_BLOCK = """\
elec name {name}
    mg-auto
    dime {dime[0]} {dime[1]} {dime[2]}
    cglen {cglen[0]:.3f} {cglen[1]:.3f} {cglen[2]:.3f}
    fglen {fglen[0]:.3f} {fglen[1]:.3f} {fglen[2]:.3f}
    cgcent mol 1
    fgcent mol 1
    mol 1
    lpbe
    bcfl sdh
    pdie {pdie}
    sdie {sdie}
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.4
    swin 0.3
    temp {temp}
    ion charge 1 conc {ionic} radius 2.0
    ion charge -1 conc {ionic} radius 2.0
    calcenergy total
    calcforce no
{extra}\
end
"""


def write_apbs_input(
    path: str | Path,
    pqr_name: str,
    grid: dict[str, Any],
    pdie: float = 2.0,
    sdie: float = 78.54,
    temperature: float = 298.15,
    ionic_strength: float = 0.15,
    potential_dx: str | None = None,
) -> Path:
    """Write a two-state APBS input whose ``print`` yields ΔG_elec directly."""
    common = dict(
        dime=grid["dime"], cglen=grid["cglen"], fglen=grid["fglen"],
        pdie=pdie, temp=temperature, ionic=ionic_strength,
    )
    solvated = _ELEC_BLOCK.format(
        name="solv", sdie=sdie,
        extra=(f"    write pot dx {potential_dx}\n" if potential_dx else ""),
        **common,
    )
    # Reference state: identical grid, solvent dielectric collapsed onto the
    # solute's, so everything except the solvation response cancels.
    reference = _ELEC_BLOCK.format(name="ref", sdie=pdie, extra="", **common)

    text = (
        f"read\n    mol pqr {pqr_name}\nend\n"
        f"{solvated}{reference}"
        "print elecEnergy solv - ref end\nquit\n"
    )
    path = Path(path)
    path.write_text(text)
    return path


_ENERGY_RE = re.compile(
    r"Global net (?:ELEC )?energy\s*=\s*([-+0-9.Ee]+)\s*kJ/mol", re.IGNORECASE
)


def parse_apbs_energy(output: str) -> float:
    """Extract ΔG_elec (kJ/mol) from APBS stdout.

    The final ``Global net ELEC energy`` line is the value produced by the
    ``print`` statement, i.e. the solvated-minus-reference difference. Earlier
    ``Total electrostatic energy`` lines are the individual grid energies and
    are deliberately not used — they are not physically meaningful alone.
    """
    matches = _ENERGY_RE.findall(output)
    if not matches:
        raise ValueError(
            "no 'Global net ELEC energy' line in APBS output — the calculation "
            f"likely failed. Tail:\n{output[-800:]}"
        )
    return float(matches[-1])


def run_single_frame(
    pdb_path: str | Path,
    workdir: str | Path,
    forcefield: str = "CHARMM",
    pdb2pqr: str | None = None,
    apbs: str | None = None,
    potential_dx: str | None = None,
    **pb_kwargs: Any,
) -> dict[str, Any]:
    """PDB2PQR + APBS for one frame; returns ΔG_elec in kJ/mol and kcal/mol."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pqr = workdir / "frame.pqr"

    pdb2pqr = pdb2pqr or find_tool("pdb2pqr30")
    apbs = apbs or find_tool("apbs")

    result = subprocess.run(
        [pdb2pqr, f"--ff={forcefield}", "--whitespace", str(pdb_path), str(pqr)],
        capture_output=True, text=True, cwd=workdir,
    )
    if result.returncode != 0 or not pqr.exists():
        raise RuntimeError(f"pdb2pqr failed:\n{result.stderr[-2000:]}")

    coords, radii = read_pqr(pqr)
    grid = grid_dimensions(coords, radii)
    write_apbs_input(
        workdir / "apbs.in", pqr.name, grid, potential_dx=potential_dx, **pb_kwargs
    )

    result = subprocess.run(
        [apbs, "apbs.in"], capture_output=True, text=True, cwd=workdir,
    )
    energy = parse_apbs_energy(result.stdout + result.stderr)
    return {
        "dG_elec_kJ": energy,
        "dG_elec_kcal": energy / KJ_PER_KCAL,
        "grid": grid,
        "n_atoms": len(coords),
    }


def sample_frame_indices(
    n_frames: int, start: int, n_samples: int
) -> np.ndarray:
    """Evenly spaced frame indices across ``[start, n_frames)``.

    Even spacing rather than random sampling: it covers the window uniformly,
    is reproducible without a seed, and cannot cluster by chance in a way that
    biases an ensemble average over a correlated trajectory.
    """
    available = n_frames - start
    if available <= 0:
        raise ValueError(f"window start {start} is beyond the {n_frames} available frames")
    n_samples = int(min(n_samples, available))
    return np.unique(np.linspace(start, n_frames - 1, n_samples).astype(int))


def compute_pb_ensemble(
    topology: str,
    trajectory: str,
    t0_ps: float | None = None,
    n_samples: int = 50,
    selection: str = "protein",
    forcefield: str = "CHARMM",
    workdir: str | Path | None = None,
    potential_frame: bool = True,
    verbose: bool = True,
    **pb_kwargs: Any,
) -> dict[str, Any]:
    """Ensemble-averaged ΔG_elec over frames sampled from the window."""
    import MDAnalysis as mda

    from rotmd.analysis.dssp import first_frame_at_or_after

    universe = mda.Universe(topology, trajectory)
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"selection {selection!r} matched no atoms")

    start = first_frame_at_or_after(universe, t0_ps) if t0_ps is not None else 0
    indices = sample_frame_indices(len(universe.trajectory), start, n_samples)

    pdb2pqr = find_tool("pdb2pqr30")
    apbs = find_tool("apbs")

    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rotmd-apbs-"))
    root.mkdir(parents=True, exist_ok=True)

    energies, times, used = [], [], []
    for count, index in enumerate(indices, start=1):
        ts = universe.trajectory[int(index)]
        frame_dir = root / f"frame{int(index):06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        pdb = frame_dir / "frame.pdb"
        atoms.write(str(pdb))

        # Only the last frame writes a potential map; one .dx is enough for a
        # figure and each is tens of MB.
        dx = "pot" if (potential_frame and count == len(indices)) else None
        try:
            result = run_single_frame(
                pdb, frame_dir, forcefield=forcefield,
                pdb2pqr=pdb2pqr, apbs=apbs, potential_dx=dx, **pb_kwargs
            )
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"  ! frame {int(index)} failed: {str(exc)[:200]}")
            continue

        energies.append(result["dG_elec_kJ"])
        times.append(float(ts.time))
        used.append(int(index))
        if verbose:
            print(
                f"  [{count}/{len(indices)}] frame {int(index)} t={ts.time:.0f} ps  "
                f"ΔG_elec = {result['dG_elec_kcal']:.1f} kcal/mol",
                flush=True,
            )

    if not energies:
        raise RuntimeError("every APBS frame failed — see the messages above")

    values = np.asarray(energies, dtype=np.float64)
    return {
        "dG_elec_kJ": values,
        "dG_elec_kcal": values / KJ_PER_KCAL,
        "frames": np.asarray(used, dtype=np.int64),
        "time_ps": np.asarray(times, dtype=np.float64),
        "mean_kcal": np.float64(values.mean() / KJ_PER_KCAL),
        # Standard error of the mean over sampled frames. Frames are spread
        # across the window and so are far less correlated than adjacent ones,
        # but this is still an optimistic error bar if the window is short.
        "sem_kcal": np.float64(
            values.std(ddof=1) / np.sqrt(len(values)) / KJ_PER_KCAL
            if len(values) > 1 else 0.0
        ),
        "n_frames_used": np.int64(len(values)),
        "n_frames_requested": np.int64(len(indices)),
        "t0_ps": np.float64(t0_ps if t0_ps is not None else np.nan),
        "workdir": str(root),
    }
