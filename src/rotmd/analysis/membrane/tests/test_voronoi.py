import MDAnalysis as mda
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
import logging


from twodanalysis import MembProp
from twodanalysis import Voronoi2D


from twodanalysis.data.files import MEMBRANE_TPR, MEMBRANE_XTC


tpr = MEMBRANE_TPR
traj = MEMBRANE_XTC


def test_import():
    assert Voronoi2D is not None


def test_init():
    u = mda.Universe(MEMBRANE_TPR, MEMBRANE_XTC)
    obj = Voronoi2D(u)
    assert obj is not None


@pytest.mark.parametrize("lipids", [["DODMA", "CHL1", "POPE", "DSPC"]])
@pytest.mark.parametrize("nbins", [80])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [70])
def test_thickness(lipids, nbins, start, final):
    u = mda.Universe(
        MEMBRANE_TPR,
        MEMBRANE_XTC,
    )
    membrane = Voronoi2D(u, nbins=nbins)
    mat_thi, edges = membrane.voronoi_thickness(
        lipid_list=lipids, start=start, final=final
    )
    logging.info(f"{edges}")
    assert mat_thi.shape == (nbins, nbins)
    assert len(edges) == 4


@pytest.mark.parametrize("nbins", [100])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [70])
@pytest.mark.parametrize("layer", ["top", "bot"])
def test_apl(nbins, start, final, layer):
    u = mda.Universe(
        MEMBRANE_TPR,
        MEMBRANE_XTC,
    )
    membrane = Voronoi2D(u, nbins=nbins)
    apl, edges = membrane.voronoi_apl(
        layer=layer,
        nbins=nbins,
        start=start,
        final=final,
    )

    assert apl.shape == (nbins, nbins)
    assert len(edges) == 4


@pytest.mark.parametrize("lipids", [["DODMA", "POPE", "DSPC"]])
@pytest.mark.parametrize("nbins", [80])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [70])
@pytest.mark.parametrize("layer", ["top", "bot"])
def test_splay(nbins, start, final, layer, lipids):
    u = mda.Universe(
        MEMBRANE_TPR,
        MEMBRANE_XTC,
    )
    membrane = Voronoi2D(u, nbins=nbins)
    splay, edges = membrane.voronoi_splay(
        layer=layer,
        nbins=nbins,
        lipid_list=lipids,
        start=start,
        final=final,
    )

    assert splay.shape == (nbins, nbins)
    assert len(edges) == 4
