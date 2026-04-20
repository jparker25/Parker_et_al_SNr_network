import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import networkx as nx
from matplotlib import pyplot as plt
import sys

from helpers import *

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


# ---- STEP 2: CATEGORIZE PARAMETERS ----
def categorize_change(value):
    if value < 1.0:
        return "Decrease"
    elif value == 1.0:
        return "Same"
    else:
        return "Increase"


df_categorical = df.copy()
for col in parameter_columns:
    df_categorical[col] = df_categorical[col].apply(categorize_change)

# ---- STEP 3: ONE-HOT ENCODE PARAMETERS ----
ohe = OneHotEncoder(sparse_output=False)
X_ohe = ohe.fit_transform(df_categorical[parameter_columns])
ohe_feature_names = ohe.get_feature_names_out(parameter_columns)


# ---- STEP 4: TRAIN DECISION TREE ----
y = df_categorical["Good_Fit"]
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_ohe, y)

"""plt.figure()
tree.plot_tree(clf)
plt.show()
sys.exit()"""


# ----------------------------
# STEP 2: HELPER FUNCTIONS
# ----------------------------
def get_node_good_fit_rate(tree_model, node_id):
    value = tree_model.tree_.value[node_id][0]
    if value.sum() == 0:
        return 0.0
    return value[1] / value.sum()


def build_networkx_tree(
    tree_model, feature_names, node_id=0, G=None, parent=None, edge_label=""
):
    categroy_map = {
        "Increase": ">1",
        "Decrease": "<1",
        "Same": "=1",
    }

    param_map = {
        "trpc3_change": "$k_{g_{TRPC3}}$",
        "tonic_CL_S_change": "$k_{g_{GABA}^{Tonic,S}}$",
        "tonic_CL_D_change": "$k_{g_{GABA}^{Tonic,D}}$",
        "kcc2_S_change": "$k_{g_{KCC2}^S}$",
        "kcc2_D_change": "$k_{g_{KCC2}^D}$",
        "noise_change": "$k_{\\sigma_{\\eta_i}}$",
        "stn_change": "$k_{g_{STN}}$",
    }
    if G is None:
        G = nx.DiGraph()

    rate = get_node_good_fit_rate(tree_model, node_id)
    rate_str = f"cGFR:\n{rate:.1%}"

    if tree_model.tree_.feature[node_id] != tree._tree.TREE_UNDEFINED:
        name = feature_names[tree_model.tree_.feature[node_id]]
        if "_Same" in name or "_Increase" in name or "_Decrease" in name:
            category = name.split("_")[-1]
            param = name[: name.rfind("_")]
            decision = f"{param_map[param]} {categroy_map[category]}"
        else:
            decision = name
        label = f"{rate_str}\n\nChoice:\n{decision}"
    else:
        label = f"{rate_str}"

    G.add_node(node_id, label=label, rate=rate)

    if parent is not None:
        G.add_edge(parent, node_id, label=edge_label)

    if tree_model.tree_.feature[node_id] != tree._tree.TREE_UNDEFINED:
        build_networkx_tree(
            tree_model,
            feature_names,
            tree_model.tree_.children_left[node_id],
            G,
            node_id,
            "False",
        )
        build_networkx_tree(
            tree_model,
            feature_names,
            tree_model.tree_.children_right[node_id],
            G,
            node_id,
            "True",
        )

    return G


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.1, vert_loc=0, xcenter=0.5):
    pos = {}

    def _hierarchy_pos(
        G, root, leftmost, width, vert_gap, vert_loc, pos, parent=None, parsed=[]
    ):
        if root not in parsed:
            parsed.append(root)
            pos[root] = (leftmost + width / 2, vert_loc)
            neighbors = list(G.neighbors(root))
            if len(neighbors) != 0:
                dx = width / len(neighbors)
                nextx = leftmost
                for neighbor in neighbors:
                    nextx = _hierarchy_pos(
                        G,
                        neighbor,
                        nextx,
                        dx,
                        vert_gap,
                        vert_loc - vert_gap,
                        pos,
                        root,
                        parsed,
                    )
                    nextx += dx
        return leftmost

    _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, pos)
    return pos


# ----------------------------
# STEP 3: PLOT
# ----------------------------
G = build_networkx_tree(clf, ohe_feature_names)
pos = hierarchy_pos(G, root=0)

plt.figure(figsize=(14, 8), dpi=300, tight_layout=True)

# Node colors based on good fit rate
node_colors = [G.nodes[n]["rate"] for n in G.nodes]
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    cmap="RdYlGn",
    node_size=2500,
    node_shape="s",
    edgecolors="k",
)
nx.draw_networkx_edges(G, pos, edge_color="gray")
nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), font_size=7)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=9
)

plt.colorbar(nodes, label="Conditional Good Fit Rate")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sklearn_dec_tree_networkx.pdf")
plt.close()

run_cmd(f"open {FIG_DIR}/sklearn_dec_tree_networkx.pdf")
