import MDAnalysis as mda
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
import time


# from twodanalysis import Memb2D
# import imageio
import os
import time
import networkx as nx

from twodanalysis import MembProp
from twodanalysis import Cumulative2D
from twodanalysis import Voronoi2D
from twodanalysis import analysis


from twodanalysis.data.files import MEMBRANE_TPR, MEMBRANE_XTC


tpr = MEMBRANE_TPR
traj = MEMBRANE_XTC

tpr = "reduced_traj.gro"
traj = "reduced_traj.xtc"

tpr = "membrane.gro"
traj = "membrane.xtc"


u = mda.Universe(
    tpr,
    traj,
)
import re


def natural_key(node_name):
    """
    Key function to sort strings like C1, C2, C10 numerically.
    Extracts the prefix and numeric part.
    """
    match = re.match(r"([A-Za-z]+)(\d+)", str(node_name))
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (node_name, 0)  # fallback


nbins = 50
start = 1000
final = 1400
layer = "top"  # or "bot", depending on the layer you want to analyze


connection_chains = connection_chains = {
    "PSM": [("C1F", "C2F"), ("C2S", "C3S")],
}

membrane = Voronoi2D(
    u,
    nbins=nbins,
    connection_chains=connection_chains,
)

lipid_list = membrane.lipid_list.copy()
lipid_list.remove("CHL1")
print(lipid_list)
print(len(u.trajectory))


scd, edges = membrane.voronoi_all_lip_order(
    lipid_list=lipid_list, layer=layer, start=start, final=final, chain="sn1"
)
plt.imshow(scd, cmap="Spectral", extent=edges)
plt.colorbar(label="SCD Order Parameter")
plt.title(f"SCD Order Parameter for {layer} layer")
plt.xlabel("X Position (nm)")
plt.ylabel("Y Position (nm)")
plt.show()

"""
for lipid in lipid_list:
    carbons, scd, error = analysis.OrderParameters.window_scd(u,
                                                          selection=f"resname {lipid}",
                                                            start=0,
                                                              final=4000,
                                                                window = 400,
                                                                  chain ="sn1",
                                                                    step = 2,
                                                                    connection_chains=connection_chains)
    carbons = list(range(len(scd)))
    print(len(carbons), len(scd), len(error))
    plt.errorbar(carbons, scd, yerr=error, fmt='o-', label=f"{lipid}")
plt.xticks(rotation=45)
plt.legend()
plt.show()

print(scd)

"""
# membrane._store_fist_lipids()
# lipids_ids = membrane.first_lipids
"""
for key in lipid_list:
    print(membrane.connection_chains[key])
    _, chain0 = membrane.cut_structure(
                    membrane.memb.select_atoms(f"resid {lipids_ids[key]}"),
                    membrane.connection_chains[key][0],
                    )
    _, chain1 =     membrane.cut_structure(
                    membrane.memb.select_atoms(f"resid {lipids_ids[key]}"),
                    membrane.connection_chains[key][0],
                    )


    c_in_node = [n for n in chain0.nodes if "C" in str(n)]
    c_in_node = sorted(c_in_node, key=natural_key)
    neighbors = [list(chain0.neighbors(n)) for n in c_in_node]
    neighbors = [[c_in_node[i]] + [item for item in neighbors[i] if "C" not in item] for i in range(len(c_in_node))]

"""


# temp_chain0 = chain0.copy()
# temp_chain0.remove_nodes_from(nodes_to_remove)

# pos = nx.spring_layout(chain0)
# nx.draw(temp_chain0, pos, with_labels=True, node_size=50, font_size=8)
# nodes = temp_chain0.nodes
# sorted_nodes = sorted(nodes, key=natural_key)

#    print(neighbors)
# plt.show()


# scd, edges = membrane.voronoi_all_lip_order(
#                                            lipid_list=lipid_list,
#                                            layer = layer,
#                                            start = start,
#                                            final=final,
#                                            )

# plt.imshow(scd, cmap='viridis',extent =edges)
# plt.colorbar(label='SCD Order Parameter')
# plt.show()
