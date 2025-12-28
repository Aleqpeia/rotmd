import MDAnalysis as mda
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytest


from twodanalysis import MembProp
from twodanalysis import Cumulative2D


from twodanalysis.data.files import MEMBRANE_TPR, MEMBRANE_XTC


tpr = MEMBRANE_TPR
traj = MEMBRANE_XTC


def test_import():
    assert MembProp is not None


def test_init():
    u = mda.Universe(MEMBRANE_TPR, MEMBRANE_XTC)
    obj = Cumulative2D(u)
    assert obj is not None


@pytest.mark.parametrize("lipids", [["DODMA", "CHL1", "POPE", "DSPC"]])
@pytest.mark.parametrize("nbins", [50])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [100])
def test_thickness(lipids, nbins, start, final):
    u = mda.Universe(
        MEMBRANE_TPR,
        MEMBRANE_XTC,
    )
    membrane = Cumulative2D(u, nbins=nbins)
    mat_thi, edges = membrane.thickness(start=start, final=final)

    assert mat_thi.shape == (nbins, nbins)
    assert len(edges) == 4


@pytest.mark.parametrize("nbins", [50])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [100])
@pytest.mark.parametrize("layer", ["top", "bot"])
def test_order(nbins, start, final, layer):
    u = mda.Universe(
        MEMBRANE_TPR,
        MEMBRANE_XTC,
    )
    membrane = Cumulative2D(u, nbins=nbins)
    scd, edges = membrane.all_lip_order(
        layer,
        nbins,
        start=start,
        final=final,
    )

    assert scd.shape == (nbins, nbins)
    assert len(edges) == 4


@pytest.mark.parametrize("lipids", [["DODMA", "POPE", "DSPC"]])
@pytest.mark.parametrize("nbins", [50])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [100])
@pytest.mark.parametrize("layer", ["top", "bot"])
def test_splay(nbins, start, final, layer, lipids):
    u = mda.Universe(
        MEMBRANE_TPR,
        MEMBRANE_XTC,
    )
    membrane = Cumulative2D(u, nbins=nbins)
    splay, edges = membrane.splay_matrix(
        layer=layer,
        nbins=nbins,
        lipid_list=lipids,
        start=start,
        final=final,
    )

    assert splay.shape == (nbins, nbins)
    assert len(edges) == 4
