import MDAnalysis as mda
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytest


from twodanalysis import MembProp


from twodanalysis.data.files import MEMBRANE_TPR, MEMBRANE_XTC


tpr = MEMBRANE_TPR
traj = MEMBRANE_XTC

# Creating the class


@pytest.fixture
def universe():
    return mda.Universe(tpr, traj)


def test_import():
    assert MembProp is not None


def test_init():
    u = mda.Universe(MEMBRANE_TPR, MEMBRANE_XTC)
    obj = MembProp(u)
    assert obj is not None


def test_chain_length():
    u = mda.Universe(tpr, traj)
    membrane = MembProp(u)
    membrane.guess_chain_lenght()
    chains = membrane.chain_info
    print(membrane.chain_info)

    expected_dict = {
        "DODMA": [17, 17],
        "CHL1": [32],
        "DSPC": [17, 17],
        "POPE": [15, 17],
    }
    assert expected_dict == chains


@pytest.mark.parametrize(
    "connections",
    [
        {
            "DODMA": [("C21", "C22"), ("C31", "C32")],
            "DSPC": [("C21", "C22"), ("C31", "C32")],
            "POPE": [("C21", "C22"), ("C31", "C32")],
            "CHL1": [("O3", "C3")],
        },
        {
            "DODMA": [("C21", "C22"), ("C31", "C32")],
            "DSPC": [("C21", "C22"), ("C31", "C32")],
            "POPE": [("C21", "C22"), ("C31", "C32")],
        },
        {
            "DODMA": [("C21", "C22"), ("C31", "C32")],
            "DSPC": [("C21", "C22"), ("C31", "C32")],
        },
    ],
)
@pytest.mark.parametrize(
    "lipids", [["DODMA", "CHL1", "DSPC"], ["DODMA", "DSPC", "POPE"]]
)
@pytest.mark.parametrize("verbose", [True, False])
def test_last_carbon(lipids, connections, verbose):
    u = mda.Universe(tpr, traj)
    membrane = MembProp(
        u, lipid_list=lipids, connection_chains=connections, verbose=verbose
    )
    membrane.guess_last_cs()
    membrane.guess_chain_lenght()
    lipids = membrane.lipid_list
    last_cs = [
        membrane.working_lip[lipid]["last_c"] for lipid in lipids if lipid != "CHL1"
    ]
    chains = membrane.chain_info
    expected = [
        [f"C3{chains[lipid][0] + 1}", f"C2{chains[lipid][1] + 1}"]
        for lipid in lipids
        if lipid != "CHL1"
    ]
    assert last_cs == expected


@pytest.mark.parametrize(
    "connections",
    [
        {
            "DODMA": [("C21", "C22"), ("C31", "C32")],
            "DSPC": [("C21", "C22"), ("C31", "C32")],
            "POPE": [("C21", "C22"), ("C31", "C32")],
            "CHL1": [("O3", "C3")],
        },
        {
            "DODMA": [("C21", "C22"), ("C31", "C32")],
            "DSPC": [("C21", "C22"), ("C31", "C32")],
            "POPE": [("C21", "C22"), ("C31", "C32")],
        },
        {
            "DODMA": [("C21", "C22"), ("C31", "C32")],
            "DSPC": [("C21", "C22"), ("C31", "C32")],
        },
    ],
)
@pytest.mark.parametrize("lipids", [None, ["DODMA", "DSPC"]])
@pytest.mark.parametrize("verbose", [True, False])
def test_nx_polarity(lipids, connections, verbose):
    u = mda.Universe(tpr, traj)
    membrane = MembProp(
        u, lipid_list=lipids, connection_chains=connections, verbose=verbose
    )
    membrane.nx_polarity()
    polar_dict = membrane.polar_dict
    if lipids == None:
        assert len(polar_dict) == 4
    else:
        assert len(polar_dict) == len(lipids)
