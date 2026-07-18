"""T7: auto-generated methods provenance.

The acceptance criterion is that every rendered value is cross-checked against
the ``.mdp`` fixture, so the fixtures below are realistic CHARMM36m inputs and
the assertions name the exact numbers that appear in them.
"""

from __future__ import annotations

import pytest

from rotmd.analysis.methods import (
    build_methods,
    parse_mdp,
    parse_topology,
    render_markdown,
    summarize_stage,
)

MD_MDP = """\
; Production MD — CHARMM36m
integrator              = md
dt                      = 0.002
nsteps                  = 50000000   ; 100 ns
nstxout-compressed      = 5000
; Neighbour searching
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
; Electrostatics
coulombtype             = PME
rcoulomb                = 1.2
pme_order               = 4
fourierspacing          = 0.12
; Van der Waals
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw_switch             = 1.0
rvdw                    = 1.2
DispCorr                = no
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = Protein Water_and_ions
tau_t                   = 0.1     0.1
ref_t                   = 310     310
; Pressure coupling
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 5.0
ref_p                   = 1.0
compressibility         = 4.5e-5
; Constraints
constraints             = h-bonds
constraint_algorithm    = LINCS
lincs_order             = 4
lincs_iter              = 1
"""

EM_MDP = """\
integrator  = steep
nsteps      = 50000
emtol       = 1000.0
tcoupl      = no
pcoupl      = no
"""

NVT_MDP = """\
integrator  = md
dt          = 0.001
nsteps      = 125000
tcoupl      = V-rescale
tc-grps     = Protein Water_and_ions
tau_t       = 0.1 0.1
ref_t       = 310 310
pcoupl      = no
constraints = h-bonds
"""

NPT_MDP = """\
integrator  = md
dt          = 0.002
nsteps      = 500000
tcoupl      = V-rescale
tc-grps     = Protein Water_and_ions
tau_t       = 0.1 0.1
ref_t       = 310 310
pcoupl      = C-rescale
pcoupltype  = isotropic
tau_p       = 5.0
ref_p       = 1.0
constraints = h-bonds
"""

TOPOL = """\
#include "charmm36m.ff/forcefield.itp"
#include "charmm36m.ff/tip3p.itp"
#include "charmm36m.ff/ions.itp"

[ system ]
HPCA in water

[ molecules ]
; Compound        #mols
Protein_chain_A         1
TIP3                12345
SOD                    12
CLA                    10
"""


@pytest.fixture
def inputs(tmp_path):
    paths = {}
    for name, text in [
        ("md", MD_MDP), ("em", EM_MDP), ("nvt", NVT_MDP), ("npt", NPT_MDP)
    ]:
        p = tmp_path / f"{name}.mdp"
        p.write_text(text)
        paths[name] = p
    top = tmp_path / "topol.top"
    top.write_text(TOPOL)
    paths["topol"] = top
    return paths


# ---------------------------------------------------------------------------
# parse_mdp
# ---------------------------------------------------------------------------

def test_parse_mdp_normalises_keys_and_strips_comments(inputs):
    mdp = parse_mdp(inputs["md"])
    # '-' and '_' are interchangeable in GROMACS; both normalise to '_'.
    assert mdp["nstxout_compressed"] == "5000"
    assert mdp["vdw_modifier"] == "force-switch"
    assert mdp["tc_grps"] == "Protein Water_and_ions"
    assert mdp["cutoff_scheme"] == "Verlet"
    # Case is folded, and the trailing "; 100 ns" comment is gone.
    assert mdp["dispcorr"] == "no"
    assert mdp["nsteps"] == "50000000"


def test_parse_mdp_ignores_blank_and_comment_only_lines(tmp_path):
    p = tmp_path / "x.mdp"
    p.write_text("; header only\n\n   \ndt = 0.002\n; trailing\n")
    assert parse_mdp(p) == {"dt": "0.002"}


# ---------------------------------------------------------------------------
# summarize_stage — every value cross-checked against MD_MDP above
# ---------------------------------------------------------------------------

def test_production_stage_values_match_the_mdp(inputs):
    s = summarize_stage(parse_mdp(inputs["md"]), "md")

    assert s["integrator"] == "md"
    assert s["dt_ps"] == 0.002
    assert s["nsteps"] == 50_000_000
    assert s["duration_ns"] == pytest.approx(100.0)

    assert s["thermostat"]["type"] == "V-rescale"
    assert s["thermostat"]["ref_t_K"] == [310.0, 310.0]
    assert s["thermostat"]["tau_t_ps"] == [0.1, 0.1]
    assert s["thermostat"]["groups"] == ["Protein", "Water_and_ions"]

    assert s["barostat"]["type"] == "Parrinello-Rahman"
    assert s["barostat"]["ref_p_bar"] == 1.0
    assert s["barostat"]["tau_p_ps"] == 5.0
    assert s["barostat"]["coupling"] == "isotropic"
    assert s["barostat"]["compressibility_bar"] == pytest.approx(4.5e-5)

    assert s["electrostatics"] == {
        "coulombtype": "PME", "rcoulomb_nm": 1.2,
        "pme_order": 4, "fourierspacing_nm": 0.12,
    }
    assert s["vdw"]["rvdw_nm"] == 1.2
    assert s["vdw"]["rvdw_switch_nm"] == 1.0
    assert s["vdw"]["vdw_modifier"] == "force-switch"
    assert s["neighbour_list"] == {"cutoff_scheme": "Verlet", "nstlist": 20, "rlist_nm": 1.2}
    assert s["constraints"] == {
        "constraints": "h-bonds", "algorithm": "LINCS", "lincs_order": 4, "lincs_iter": 1,
    }


def test_minimisation_stage_has_no_coupling_and_no_duration(inputs):
    s = summarize_stage(parse_mdp(inputs["em"]), "em")
    assert s["integrator"] == "steep"
    assert s["thermostat"] is None      # tcoupl = no
    assert s["barostat"] is None        # pcoupl = no
    assert s["dt_ps"] is None           # em.mdp sets no dt
    assert s["duration_ns"] is None     # so no duration can be claimed


def test_nvt_stage_has_thermostat_but_no_barostat(inputs):
    s = summarize_stage(parse_mdp(inputs["nvt"]), "nvt")
    assert s["thermostat"]["type"] == "V-rescale"
    assert s["barostat"] is None
    assert s["duration_ns"] == pytest.approx(0.125)


def test_absent_fields_are_none_not_gromacs_defaults():
    """A methods section must never invent parameters that weren't specified."""
    s = summarize_stage({"integrator": "md"}, "bare")
    assert s["dt_ps"] is None
    assert s["electrostatics"]["coulombtype"] is None
    assert s["neighbour_list"]["nstlist"] is None
    assert s["constraints"]["constraints"] is None


# ---------------------------------------------------------------------------
# parse_topology
# ---------------------------------------------------------------------------

def test_topology_extracts_forcefield_water_and_counts(inputs):
    topo = parse_topology(inputs["topol"])
    assert topo["forcefield"] == "charmm36m"
    assert topo["water_model"] == "tip3p"
    assert topo["n_water"] == 12345
    assert topo["ions"] == {"SOD": 12, "CLA": 10}
    assert topo["molecules"]["Protein_chain_A"] == 1


def test_topology_without_ions_reports_none(tmp_path):
    p = tmp_path / "t.top"
    p.write_text('#include "amber99sb.ff/forcefield.itp"\n[ molecules ]\nProtein 1\nSOL 100\n')
    topo = parse_topology(p)
    assert topo["forcefield"] == "amber99sb"
    assert topo["ions"] is None
    assert topo["n_water"] == 100


# ---------------------------------------------------------------------------
# Rendering — the acceptance criterion for the prose
# ---------------------------------------------------------------------------

def test_markdown_names_every_required_element(inputs):
    order = ["em", "nvt", "npt", "md"]
    stages = [summarize_stage(parse_mdp(inputs[n]), n) for n in order]
    text = render_markdown(stages, parse_topology(inputs["topol"]))

    for required in [
        "charmm36m",            # force field
        "tip3p",                # water model
        "V-rescale",            # thermostat
        "Parrinello-Rahman",    # barostat
        "PME",                  # electrostatics
        "1.2 nm",               # cutoffs
        "LINCS",                # constraints
        "h-bonds",
        "energy minimisation",  # equilibration protocol
        "NVT equilibration",
        "NPT equilibration",
        "100 ns",               # production length
        "2 fs",                 # timestep
        "310 K",                # temperature
    ]:
        assert required in text, f"methods.md never mentions {required!r}"


def test_markdown_flags_gaps_instead_of_inventing_values():
    stages = [summarize_stage({"integrator": "md", "dt": "0.002", "nsteps": "1000"}, "md")]
    text = render_markdown(stages, {})
    assert "unspecified" in text
    assert "deliberately not filled in with GROMACS defaults" in text


def test_markdown_includes_a_per_stage_table(inputs):
    stages = [summarize_stage(parse_mdp(inputs[n]), n) for n in ["em", "nvt", "npt", "md"]]
    text = render_markdown(stages, parse_topology(inputs["topol"]))
    assert "| Stage | Integrator |" in text
    for name in ("em", "nvt", "npt", "md"):
        assert f"| {name} |" in text


# ---------------------------------------------------------------------------
# build_methods + CLI
# ---------------------------------------------------------------------------

def test_build_methods_returns_payload_and_prose(inputs):
    payload, text = build_methods(
        {n: inputs[n] for n in ["em", "nvt", "npt", "md"]}, topology=inputs["topol"]
    )
    assert [s["name"] for s in payload["stages"]] == ["em", "nvt", "npt", "md"]
    assert payload["topology"]["forcefield"] == "charmm36m"
    assert payload["sources"]["topology"] == str(inputs["topol"])
    assert text.startswith("# Simulation methods")


def test_cli_methods_writes_both_artifacts(inputs, tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import read_json

    outdir = tmp_path / "methods"
    argv = ["methods", "-o", str(outdir), "--topology", str(inputs["topol"])]
    for name in ["em", "nvt", "npt", "md"]:
        argv += ["--mdp", f"{name}={inputs[name]}"]

    assert main(argv) == 0
    payload = read_json(outdir / "methods.json")
    assert [s["name"] for s in payload["stages"]] == ["em", "nvt", "npt", "md"]
    assert "charmm36m" in (outdir / "methods.md").read_text()


def test_cli_methods_labels_bare_paths_by_stem(inputs, tmp_path):
    from rotmd.cli import main
    from rotmd.io.meta import read_json

    outdir = tmp_path / "m2"
    assert main(["methods", "-o", str(outdir), "--mdp", str(inputs["md"])]) == 0
    assert [s["name"] for s in read_json(outdir / "methods.json")["stages"]] == ["md"]


def test_cli_methods_requires_an_mdp(tmp_path):
    from rotmd.cli import main

    with pytest.raises(SystemExit, match="at least one --mdp"):
        main(["methods", "-o", str(tmp_path / "out")])


def test_cli_methods_reports_missing_file(tmp_path):
    from rotmd.cli import main

    with pytest.raises(SystemExit, match="mdp file not found"):
        main(["methods", "-o", str(tmp_path / "out"), "--mdp", str(tmp_path / "ghost.mdp")])


def test_regenerating_methods_is_deterministic(inputs, tmp_path):
    """`methods.md` must regenerate identically — the plan's DoD."""
    from rotmd.cli import main

    files = {n: inputs[n] for n in ["em", "nvt", "npt", "md"]}
    first, _ = build_methods(files, topology=inputs["topol"])
    second, _ = build_methods(files, topology=inputs["topol"])
    assert first == second

    out = tmp_path / "m"
    argv = ["methods", "-o", str(out), "--topology", str(inputs["topol"])]
    for name, path in files.items():
        argv += ["--mdp", f"{name}={path}"]
    main(argv)
    once = (out / "methods.md").read_text()
    main([*argv, "--force"])
    assert (out / "methods.md").read_text() == once
