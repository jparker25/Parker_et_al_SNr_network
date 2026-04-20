import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import networkx as nx
from matplotlib import pyplot as plt
import sys


from helpers import *


def calc_gfr(df, parameter, choice):
    mask = True
    for i, ch in enumerate(choice):
        mask &= df[parameter[i]] == ch
    mask &= df["Good_Fit"] == 1
    return len(df[mask]) / (len(df) * (1 / 3 ** len(choice)))


FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"

# ---- STEP 1: LOAD DATA ----
df = pd.read_csv(
    "/Users/johnparker/Desktop/results_df_tree.csv"
)  # replace with your filename

parameter_columns = [
    "kcc2_S_change",
    "kcc2_D_change",
    "trpc3_change",
    "tonic_CL_S_change",
    "tonic_CL_D_change",
    "stn_change",
    "noise_change",
]

baseline_gfr = len(df[df["Good_Fit"] == 1]) / len(df)
baseline_length = len(df)


paths = []

# Find initial paths that are better than baseline
current_paths = []
for col in parameter_columns:
    for uv in pd.unique(df[col]):
        if uv != 1.0:
            cgfr = calc_gfr(df, [col], [uv])
            if cgfr > baseline_gfr:
                current_paths.append([(col, uv, cgfr)])

# add to paths until no longer improving
max_depth = 5
for depth in range(1, max_depth):
    new_path = []
    for path in current_paths:
        params = [p[0] for p in path]  # used params
        last_cgfr = path[-1][2]  # latest cgfr

        params_to_check = [p for p in parameter_columns if p not in params]
        if not params_to_check:
            paths.append(path)
            continue

        for next_param in params_to_check:
            for uv in pd.unique(df[next_param]):
                if uv != 1.0:
                    new_params = [p[0] for p in path] + [next_param]
                    new_choices = [p[1] for p in path] + [uv]
                    cgfr = calc_gfr(df, new_params, new_choices)

                    if cgfr > last_cgfr:
                        new_path.append(path + [(next_param, uv, cgfr)])
                    else:
                        paths.append(path)
    current_paths = new_path


paths.extend(current_paths)

paths_above_threshold = []

threshold = 0.4
max_threshold = 0
for path in paths:
    if path[-1][2] > threshold:
        paths_above_threshold.append(path)
    if path[-1][2] > max_threshold:
        max_threshold = path[-1][2]

count = 0

trpc3_66 = 0
layer_to_check = 5
partial_threshold = 0.0
partial_path_above_threshold = []
all_changes = {}
for path in paths_above_threshold:
    if path[layer_to_check - 1][2] > partial_threshold:
        path_to_add = [path[i][0] for i in range(layer_to_check)]
        partial_path_above_threshold.append(path_to_add)
        for p in path_to_add:
            if p[0] == "trpc3_change" and p[1] == 0.66:
                trpc3_66 += 1
        for p in path_to_add:
            if p not in all_changes.keys():
                all_changes[p] = 0

print("trpc3 66: ", trpc3_66)

for partial_path in partial_path_above_threshold:
    for p in partial_path:
        all_changes[p] += 1
"""
print(all_changes, len(partial_path_above_threshold))
sys.exit()
print(count)

# print(paths_above_threshold)
print(len(paths), len(paths_above_threshold), max_threshold)
sys.exit()
"""

param_map = {
    "trpc3_change": "$k_{g_{TRPC3}}$",
    "tonic_CL_S_change": "$k_{g_{GABA}^{Tonic,S}}$",
    "tonic_CL_D_change": "$k_{g_{GABA}^{Tonic,D}}$",
    "kcc2_S_change": "$k_{g_{KCC2}^S}$",
    "kcc2_D_change": "$k_{g_{KCC2}^D}$",
    "noise_change": "$k_{\\sigma_{\\eta_i}}$",
    "stn_change": "$k_{g_{STN}}$",
}

G = nx.DiGraph()
for path in paths_above_threshold:
    prev_node = "GFR = 8.5%"
    G.add_node(prev_node, subset=0)
    for i, (param, val, cgfr) in enumerate(path):
        node = f"{param_map[param]}={val}\ncGFR:{cgfr:0.2f}"
        depth = i + 1
        G.add_node(node, subset=depth)
        G.add_edge(prev_node, node)
        prev_node = node

pos = nx.multipartite_layout(G, subset_key="subset")

node_colors = []
for node in G.nodes:
    if node == "GFR = 8.5%":
        node_colors.append(0.0)
    else:
        node_colors.append(float(node.split(":")[1]))

labels = {}
for n in G.nodes:
    if n == "GFR = 8.5%":
        labels[n] = "GFR\n8.5%"  # prettier root label (optional)
    else:
        # node looks like "<param>=<value>\ncGFR:<num>"
        head = n.split("\n", 1)[0]  # "<param>=<value>"
        if "=" in head:
            p, v = head.split("=", 1)
            labels[n] = f"{p}:\n{v}"  # two-line label
        else:
            labels[n] = head


fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300, tight_layout=True)
nx.draw(
    G,
    pos,
    with_labels=True,
    labels=labels,
    edge_color="gray",
    font_size=8,
    node_shape="s",
    node_size=1000,
    node_color=node_colors,
    cmap="RdYlGn",
    edgecolors="k",
    ax=ax,
)
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.RdYlGn,
    norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),
)
sm.set_array([])
cbar = fig.colorbar(
    sm, shrink=0.3, ax=ax, orientation="horizontal", fraction=0.046, pad=0.05
)
cbar.set_label("Conditional Good Fit Rate (cGFR)", fontsize=8)
cbar.ax.tick_params(labelsize=8)
plt.savefig(f"{FIG_DIR}/choice_decision_tree.pdf")
plt.close()

run_cmd(f"open {FIG_DIR}/choice_decision_tree.pdf")
