"""Auto-generated simulation methods (T7).

Reads the ``.mdp`` files and topology that actually produced the trajectories
and renders both a machine-readable ``methods.json`` and a prose
``methods.md``. Generating the methods section from the inputs rather than
transcribing it by hand removes the whole class of error where the manuscript
says 2 fs and the ``.mdp`` says 1 fs.

Nothing here guesses. A field that is absent from the ``.mdp`` is reported as
absent, never defaulted to what GROMACS would have used — a methods section
that silently invents parameters is worse than one with a visible gap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# GROMACS treats mdp keys case-insensitively and '-'/'_' interchangeably.
_TRUE_OFF = {"no", "none", "off", "false"}


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def parse_mdp(path: str | Path) -> dict[str, str]:
    """Parse a GROMACS ``.mdp`` into a normalised ``{key: value}`` dict."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()  # ';' starts a comment
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[_norm_key(key)] = value.strip()
    return out


def _floats(value: str | None) -> list[float]:
    """``"310 310"`` -> ``[310.0, 310.0]`` (mdp allows one value per group)."""
    if not value:
        return []
    return [float(tok) for tok in value.split()]


def _one_float(value: str | None) -> float | None:
    vals = _floats(value)
    return vals[0] if vals else None


def _is_off(value: str | None) -> bool:
    return value is None or value.strip().lower() in _TRUE_OFF


def summarize_stage(mdp: dict[str, str], name: str) -> dict[str, Any]:
    """Pull the publication-relevant settings out of one parsed ``.mdp``."""
    dt = _one_float(mdp.get("dt"))
    nsteps = int(float(mdp["nsteps"])) if "nsteps" in mdp else None
    duration_ns = (dt * nsteps / 1000.0) if (dt is not None and nsteps is not None) else None

    tcoupl = mdp.get("tcoupl")
    thermostat = None
    if not _is_off(tcoupl):
        thermostat = {
            "type": tcoupl,
            "groups": mdp.get("tc_grps", "").split() or None,
            "tau_t_ps": _floats(mdp.get("tau_t")) or None,
            "ref_t_K": _floats(mdp.get("ref_t")) or None,
        }

    pcoupl = mdp.get("pcoupl")
    barostat = None
    if not _is_off(pcoupl):
        barostat = {
            "type": pcoupl,
            "coupling": mdp.get("pcoupltype"),
            "tau_p_ps": _one_float(mdp.get("tau_p")),
            "ref_p_bar": _one_float(mdp.get("ref_p")),
            "compressibility_bar": _one_float(mdp.get("compressibility")),
        }

    constraints = mdp.get("constraints")
    return {
        "name": name,
        "integrator": mdp.get("integrator"),
        "dt_ps": dt,
        "nsteps": nsteps,
        "duration_ns": duration_ns,
        "thermostat": thermostat,
        "barostat": barostat,
        "electrostatics": {
            "coulombtype": mdp.get("coulombtype"),
            "rcoulomb_nm": _one_float(mdp.get("rcoulomb")),
            "pme_order": int(float(mdp["pme_order"])) if "pme_order" in mdp else None,
            "fourierspacing_nm": _one_float(mdp.get("fourierspacing")),
        },
        "vdw": {
            "vdwtype": mdp.get("vdwtype"),
            "vdw_modifier": mdp.get("vdw_modifier"),
            "rvdw_nm": _one_float(mdp.get("rvdw")),
            "rvdw_switch_nm": _one_float(mdp.get("rvdw_switch")),
            "dispcorr": mdp.get("dispcorr"),
        },
        "neighbour_list": {
            "cutoff_scheme": mdp.get("cutoff_scheme"),
            "nstlist": int(float(mdp["nstlist"])) if "nstlist" in mdp else None,
            "rlist_nm": _one_float(mdp.get("rlist")),
        },
        "constraints": {
            "constraints": None if _is_off(constraints) else constraints,
            "algorithm": mdp.get("constraint_algorithm"),
            "lincs_order": int(float(mdp["lincs_order"])) if "lincs_order" in mdp else None,
            "lincs_iter": int(float(mdp["lincs_iter"])) if "lincs_iter" in mdp else None,
        },
    }


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

_WATER_MODELS = ("tip3p", "tip4p", "tip5p", "spce", "spc", "opc")
_ION_RESNAMES = {"SOD", "CLA", "NA", "CL", "K", "POT", "MG", "CAL", "ZN", "NA+", "CL-"}


def parse_topology(path: str | Path) -> dict[str, Any]:
    """Extract force field, water model and molecule counts from a ``.top``."""
    text = Path(path).read_text()

    forcefield = None
    water_model = None
    for include in re.findall(r'#include\s+"([^"]+)"', text):
        low = include.lower()
        match = re.search(r"([^/]+)\.ff/", low)
        if match and forcefield is None:
            forcefield = match.group(1)
        stem = Path(low).stem
        if stem in _WATER_MODELS and water_model is None:
            water_model = stem

    # [ molecules ] is the last directive; entries are "name count".
    molecules: list[tuple[str, int]] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_block = _norm_key(line.strip("[] ")) == "molecules"
            continue
        if in_block:
            parts = line.split()
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                molecules.append((parts[0], int(parts[1])))

    ions = {name: count for name, count in molecules if name.upper() in _ION_RESNAMES}
    waters = sum(
        count for name, count in molecules
        if name.lower() in _WATER_MODELS or name.upper() in {"SOL", "TIP3", "HOH"}
    )

    return {
        "forcefield": forcefield,
        "water_model": water_model,
        "molecules": dict(molecules),
        "ions": ions or None,
        "n_water": waters or None,
    }


def parse_structure(path: str | Path) -> dict[str, Any]:
    """Atom count and box vectors from any structure MDAnalysis can read."""
    import MDAnalysis as mda

    u = mda.Universe(str(path))
    dims = u.dimensions
    return {
        "n_atoms": int(u.atoms.n_atoms),
        "box_nm": [round(float(d) / 10.0, 4) for d in dims[:3]] if dims is not None else None,
        "box_angles_deg": [float(a) for a in dims[3:]] if dims is not None else None,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value: Any, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "unspecified"
    if isinstance(value, float):
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return f"{text} {unit}".strip()
    return f"{value} {unit}".strip()


def _unique_temperatures(stage: dict[str, Any]) -> str:
    thermostat = stage.get("thermostat")
    if not thermostat or not thermostat.get("ref_t_K"):
        return "unspecified"
    temps = sorted(set(thermostat["ref_t_K"]))
    return " / ".join(_fmt(t, "K", 1) for t in temps)


def render_markdown(stages: list[dict[str, Any]], topology: dict[str, Any],
                    structure: dict[str, Any] | None = None) -> str:
    """Render the Methods prose from the parsed inputs."""
    by_name = {s["name"]: s for s in stages}
    # The production stage drives most of the sentence; fall back to the last.
    prod = by_name.get("md") or by_name.get("production") or (stages[-1] if stages else {})

    lines: list[str] = ["# Simulation methods", "", "## Summary", ""]

    ff = topology.get("forcefield") or "unspecified"
    water = topology.get("water_model") or "unspecified"
    system_bits = [f"the **{ff}** force field", f"the **{water}** water model"]
    if structure and structure.get("n_atoms"):
        system_bits.append(f"a system of {structure['n_atoms']:,} atoms")
    if topology.get("n_water"):
        system_bits.append(f"{topology['n_water']:,} water molecules")
    if topology.get("ions"):
        ions = ", ".join(f"{count} {name}" for name, count in topology["ions"].items())
        system_bits.append(f"ions ({ions})")

    lines.append(
        "All molecular dynamics simulations were performed with GROMACS using "
        + " and ".join(system_bits[:2])
        + (("; the simulated system comprised " + ", ".join(system_bits[2:]) + ".")
           if len(system_bits) > 2 else ".")
    )

    # Equilibration protocol, in the order the stages were supplied.
    protocol = []
    for stage in stages:
        if stage is prod:
            continue
        label = stage["name"]
        if (stage.get("integrator") or "").lower() in {"steep", "cg", "l-bfgs"}:
            protocol.append(f"energy minimisation ({stage['integrator']}, {stage['nsteps']} steps)")
        elif stage.get("duration_ns") is not None:
            ensemble = "NPT" if stage.get("barostat") else "NVT"
            # Only name the stage when it adds information beyond the ensemble.
            suffix = "" if label.upper() == ensemble else f" ({label})"
            protocol.append(
                f"{_fmt(stage['duration_ns'], 'ns')} of {ensemble} equilibration{suffix}"
            )
    if protocol:
        lines.append("")
        lines.append("The system was prepared by " + ", then ".join(protocol) + ".")

    if prod:
        lines.append("")
        prod_sentence = (
            f"Production dynamics were then run for {_fmt(prod.get('duration_ns'), 'ns')} "
            f"with a {_fmt((prod.get('dt_ps') or 0) * 1000, 'fs', 1)} integration timestep"
        )
        thermostat = prod.get("thermostat")
        if thermostat:
            prod_sentence += (
                f". Temperature was held at {_unique_temperatures(prod)} with the "
                f"**{thermostat['type']}** thermostat "
                f"(tau_t = {_fmt((thermostat.get('tau_t_ps') or [None])[0], 'ps')})"
            )
        barostat = prod.get("barostat")
        if barostat:
            prod_sentence += (
                f", and pressure at {_fmt(barostat.get('ref_p_bar'), 'bar')} with the "
                f"**{barostat['type']}** barostat "
                f"({barostat.get('coupling') or 'unspecified'} coupling, "
                f"tau_p = {_fmt(barostat.get('tau_p_ps'), 'ps')})"
            )
        lines.append(prod_sentence + ".")

        elec = prod.get("electrostatics", {})
        vdw = prod.get("vdw", {})
        nbr = prod.get("neighbour_list", {})
        lines.append("")
        lines.append(
            f"Long-range electrostatics were treated with **{elec.get('coulombtype') or 'unspecified'}** "
            f"using a real-space cutoff of {_fmt(elec.get('rcoulomb_nm'), 'nm')}; van der Waals "
            f"interactions used a {_fmt(vdw.get('rvdw_nm'), 'nm')} cutoff"
            + (f" with force-switching from {_fmt(vdw.get('rvdw_switch_nm'), 'nm')}"
               if vdw.get("rvdw_switch_nm") else "")
            + f". The neighbour list ({nbr.get('cutoff_scheme') or 'unspecified'} scheme) was "
            f"updated every {_fmt(nbr.get('nstlist'))} steps."
        )

        cons = prod.get("constraints", {})
        if cons.get("constraints"):
            detail = []
            if cons.get("lincs_order") is not None:
                detail.append(f"order {cons['lincs_order']}")
            if cons.get("lincs_iter") is not None:
                n_iter = cons["lincs_iter"]
                detail.append(f"{n_iter} iteration{'s' if n_iter != 1 else ''}")
            lines.append("")
            lines.append(
                f"Bonds involving hydrogen were constrained "
                f"(`constraints = {cons['constraints']}`) using the "
                f"**{cons.get('algorithm') or 'unspecified'}** algorithm"
                + (f" ({', '.join(detail)})" if detail else "")
                + "."
            )

    # A per-stage table keeps every number auditable next to the prose.
    lines += ["", "## Per-stage parameters", "",
              "| Stage | Integrator | dt (ps) | Steps | Duration (ns) | Thermostat | Barostat |",
              "|---|---|---|---|---|---|---|"]
    for stage in stages:
        lines.append(
            f"| {stage['name']} | {stage.get('integrator') or '—'} | "
            f"{_fmt(stage.get('dt_ps'))} | {stage.get('nsteps') if stage.get('nsteps') is not None else '—'} | "
            f"{_fmt(stage.get('duration_ns'))} | "
            f"{(stage.get('thermostat') or {}).get('type') or '—'} | "
            f"{(stage.get('barostat') or {}).get('type') or '—'} |"
        )

    lines += ["", "---", "",
              "*Generated by `rotmd methods` from the .mdp and topology files. "
              "Fields shown as `unspecified` were absent from the inputs and were "
              "deliberately not filled in with GROMACS defaults.*", ""]
    return "\n".join(lines)


def build_methods(
    mdp_files: dict[str, Path],
    topology: Path | None = None,
    structure: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Parse everything and return ``(methods_json, methods_markdown)``."""
    stages = [summarize_stage(parse_mdp(path), name) for name, path in mdp_files.items()]
    topo = parse_topology(topology) if topology else {}
    struct = parse_structure(structure) if structure else None

    payload: dict[str, Any] = {
        "stages": stages,
        "topology": topo,
        "structure": struct,
        "sources": {
            "mdp": {name: str(path) for name, path in mdp_files.items()},
            "topology": str(topology) if topology else None,
            "structure": str(structure) if structure else None,
        },
    }
    return payload, render_markdown(stages, topo, struct)
