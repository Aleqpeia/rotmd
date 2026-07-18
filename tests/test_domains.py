"""Domain-spec parsing and mask construction (T1).

Pure numpy/stdlib — no MDAnalysis needed, so these run everywhere.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rotmd.analysis.domains import domain_masks, parse_domains


def test_parse_empty_spec_yields_no_domains():
    assert parse_domains(None) == {}
    assert parse_domains("") == {}


def test_parse_inline_single_and_multiple():
    assert parse_domains("EF1:20-35") == {"EF1": [(20, 35)]}
    assert parse_domains("EF1:20-35,EF2:56-70") == {
        "EF1": [(20, 35)],
        "EF2": [(56, 70)],
    }


def test_parse_inline_preserves_order():
    # rmsd_dom columns are positional, so spec order must survive parsing.
    spec = "C-lobe:100-191,EF1:20-35,N-lobe:1-90"
    assert list(parse_domains(spec)) == ["C-lobe", "EF1", "N-lobe"]


def test_parse_discontiguous_spans_with_plus():
    assert parse_domains("N-lobe:1-40+55-70") == {"N-lobe": [(1, 40), (55, 70)]}


def test_parse_tolerates_whitespace():
    assert parse_domains(" EF1 : 20-35 , EF2:56-70 ") == {
        "EF1": [(20, 35)],
        "EF2": [(56, 70)],
    }


def test_parse_json_file_accepts_flat_and_nested_ranges(tmp_path):
    path = tmp_path / "domains.json"
    path.write_text(json.dumps({"EF1": [20, 35], "N-lobe": [[1, 40], [55, 70]]}))
    assert parse_domains(str(path)) == {
        "EF1": [(20, 35)],
        "N-lobe": [(1, 40), (55, 70)],
    }


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("EF1", "expected 'name:start-end'"),
        ("EF1:2035", "expected 'start-end'"),
        ("EF1:a-b", "non-integer resid"),
        ("EF1:35-20", r"start 35 > end 20"),
    ],
)
def test_parse_rejects_malformed_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_domains(spec)


def test_domain_masks_are_inclusive_on_both_ends():
    resids = np.arange(1, 11)
    masks = domain_masks(resids, parse_domains("D:3-5"))
    assert list(resids[masks["D"]]) == [3, 4, 5]


def test_domain_masks_union_discontiguous_spans():
    resids = np.arange(1, 11)
    masks = domain_masks(resids, parse_domains("D:1-2+9-10"))
    assert list(resids[masks["D"]]) == [1, 2, 9, 10]


def test_domain_masks_handle_repeated_resids():
    # Multiple atoms per residue (a CA selection is 1:1, but callers may pass
    # a fuller atom set) must all be selected, not just the first.
    resids = np.array([1, 1, 2, 2, 3, 3])
    masks = domain_masks(resids, parse_domains("D:2-3"))
    assert masks["D"].sum() == 4


def test_domain_masks_reject_empty_domain():
    # An out-of-range domain would otherwise produce a silent all-NaN column.
    resids = np.arange(1, 11)
    with pytest.raises(ValueError, match="selects no residues"):
        domain_masks(resids, parse_domains("Ghost:500-600"))
