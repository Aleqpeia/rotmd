import MDAnalysis as mda
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
import logging


from twodanalysis import PackingDefects


from twodanalysis.data.files import MEMBRANE_TPR, MEMBRANE_XTC


tpr = MEMBRANE_TPR
traj = MEMBRANE_XTC


def test_import():
    assert PackingDefects is not None


def test_init():
    u = mda.Universe(MEMBRANE_TPR, MEMBRANE_XTC)
    obj = PackingDefects(u)
    assert obj is not None


@pytest.mark.parametrize("nbins", [100, 200])
@pytest.mark.parametrize("boolean", [True, False])
def test_single_frame(nbins, boolean):
    u = mda.Universe(MEMBRANE_TPR, MEMBRANE_XTC)
    membrane = PackingDefects(u)
    membrane.u.trajectory[60]  # Compute deffects for the 80 frame
    defects, defects_dict = membrane.packing_defects(
        layer="top",  # layer to compute packing defects
        periodic=boolean,  # edges for output
        nbins=nbins,  # number of bins
    )
    if boolean:
        assert defects.shape[0] < nbins
        assert defects.shape[1] < nbins
    else:
        assert defects.shape == (nbins, nbins)
    assert type(defects_dict) == dict
    assert len(defects_dict["edges"]) == 4
    assert defects_dict["edges"][1] > defects_dict["edges"][0]
    assert defects_dict["edges"][3] > defects_dict["edges"][2]


@pytest.mark.parametrize("nbins", [100, 200])
@pytest.mark.parametrize("start", [50])
@pytest.mark.parametrize("final", [60])
@pytest.mark.parametrize("layer", ["top", "bot"])
def test_multiple_frames(start, final, nbins, layer):
    u = mda.Universe(MEMBRANE_TPR, MEMBRANE_XTC)
    membrane = PackingDefects(u)
    data_df, numpy_sizes = membrane.packing_defects_stats(
        nbins=nbins, layer=layer, periodic=True, start=start, final=final, step=1
    )

    assert isinstance(data_df, pd.DataFrame)
    assert isinstance(numpy_sizes, np.ndarray)
