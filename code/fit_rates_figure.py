"""
histogram_shifts_figure.py

Author: John E. Parker
"""

# import python modules
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import random
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import networkx as nx
from scipy import optimize
from scipy import stats
import matplotlib.patches as patches
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from collections import Counter


# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import run_model
from distribution import Distribution as Dist

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8


# Classify parameter changes
def categorize_change(val):
    if val < 1.0:
        return "Decrease"
    elif val > 1.0:
        return "Increase"
    else:
        return "No Change"


# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"


hues = [
    "kcc2_S_change",
    "kcc2_D_change",
    "trpc3_change",
    "tonic_CL_S_change",
    "tonic_CL_D_change",
    "stn_change",
    "noise_change",
]

all_seeds = []  # all DD network seed
naive_all_seeds = []  # all naive network seeds
all_dd_hetero = []  # all neurons in matching DD network seeds
kept_all_dd_seeds = []  # best matching DD network seeds
plotting_matches = []
N_vivo_sims = 3
N_dd_sims = 3
in_vivo_seed_counter = 1
slice_results = pd.read_csv(f"data/param_seed_finder_slice_correlated/results.csv")

slice_results = slice_results[
    (slice_results["KS_FR_pval"] >= 0.05)
    & (slice_results["Ttest_FR_pval"] >= 0.05)
    & (slice_results["Ttest_CV_pval"] >= 0.05)
]
avg_err_sorted = slice_results.sort_values(by="Avg. Error")
slice_seeds = [int(row["Seed"]) for index, row in avg_err_sorted.iterrows()]
for slice_seed in slice_seeds:
    vivo_data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"

    results = pd.read_csv(f"{vivo_data_dir}/results.csv")
    results = results[
        (results["KS_FR_pval"] >= 0.05)
        & (results["Ttest_FR_pval"] >= 0.05)
        & (results["Ttest_CV_pval"] >= 0.05)
        & (results["KS_CV_pval"] >= 0.05)
    ]
    sorted_results = results.sort_values(by="Avg. Error")

    sorted_results = sorted_results.iloc[:N_vivo_sims]
    for index, row in sorted_results.iterrows():
        dd_dir_res = pd.read_csv(
            f"{vivo_data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/results.csv"
        )

        dd_dir_res["Slice Seed"] = slice_seed
        dd_dir_res["in_vivo_seed"] = in_vivo_seed_counter

        all_seeds.append(dd_dir_res)
        in_vivo_seed_counter += 1

results_df = pd.concat(all_seeds, ignore_index=True)

matching_results_mask = (
    (results_df["KS_FR_pval"] > 0.05)
    & (results_df["KS_CV_pval"] > 0.05)
    & (results_df["Ttest_FR_pval"] > 0.05)
    & (results_df["Ttest_CV_pval"] > 0.05)
)

results_df["fit_type"] = [
    "Good Fit" if val else "Bad Fit" for val in matching_results_mask
]

# Get the change parameters
change_cols = [col for col in results_df.columns if "change" in col]

fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
sim_changes = np.zeros((in_vivo_seed_counter - 1, 8))
for count in range(1, in_vivo_seed_counter):
    tmp_df = results_df[results_df["in_vivo_seed"] == count]

    for index, row in tmp_df[tmp_df["fit_type"] == "Good Fit"].iterrows():
        changes = [1 if row[x] == 1.0 else 0 for x in change_cols]
        sim_changes[count - 1, 7 - np.sum(changes)] += 1
    ax.plot(
        np.arange(sim_changes.shape[1]),
        sim_changes[count - 1, :] / np.sum(sim_changes[count - 1, :]),
        color="gray",
        marker="o",
        lw=0.5,
        markersize=2,
        alpha=0.2,
    )

denom = np.asarray([3**k for k in range(1, 8)])

sim_changes = sim_changes / np.sum(sim_changes[0, :])

ax.plot(
    np.arange(sim_changes.shape[1]),
    np.mean(sim_changes, axis=0),
    color="blue",
    label="Mean success rate",
    linewidth=1,
)
ax.fill_between(
    np.arange(sim_changes.shape[1]),
    np.mean(sim_changes, axis=0) - np.std(sim_changes, axis=0),
    np.mean(sim_changes, axis=0) + np.std(sim_changes, axis=0),
    color="blue",
    alpha=0.2,
    label="±1 SD",
)
ax.set_xlabel("# Parameter Changes")
ax.set_ylabel("Good Fit Rate (by model)")
# makeNice([ax])
add_fig_labels([ax])
fig.savefig(f"{FIG_DIR}/num_changes_by_vivo.pdf", bbox_inches="tight")
plt.close()
run_cmd(f"open {FIG_DIR}/num_changes_by_vivo.pdf")

fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
sim_changes = np.zeros(8)
for index, row in results_df[results_df["fit_type"] == "Good Fit"].iterrows():
    changes = [1 if row[x] == 1.0 else 0 for x in change_cols]
    sim_changes[7 - np.sum(changes)] += 1
ax.bar(np.arange(8), sim_changes / np.sum(sim_changes), color="k")
print(np.sum(sim_changes))
ax.set_xlabel("# Parameter Changes")
ax.set_ylabel("Good Fit Rate (all)")
# makeNice([ax])
add_fig_labels([ax])
fig.savefig(f"{FIG_DIR}/num_changes_all.pdf", bbox_inches="tight")
plt.close()


run_cmd(f"open {FIG_DIR}/num_changes_all.pdf")


# Prepare plotting
label_mapping = {
    "trpc3_change": "$k_{g_{TRPC3}}$",
    "tonic_CL_S_change": "$k_{g_{GABA}^{Tonic,S}}$",
    "tonic_CL_D_change": "$k_{g_{GABA}^{Tonic,D}}$",
    "kcc2_S_change": "$k_{g_{KCC2}^S}$",
    "kcc2_D_change": "$k_{g_{KCC2}^D}$",
    "noise_change": "$k_{\\sigma_{\\eta_i}}$",
    "stn_change": "$k_{g_{STN}}$",
}

# fig, ax = plt.subplots(3, 3, figsize=(12, 6), dpi=300, tight_layout=True)
# axes = [ax[i, j] for i in range(3) for j in range(3)]
fig = plt.figure(figsize=(10, 6), dpi=300, tight_layout=True)
gs = gridspec.GridSpec(3, 3)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
axes.extend([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1:])])


decrease_palette = ["b", "skyblue", "gray"]
increase_palette = ["gray", "salmon", "r"]
for i, param in enumerate(change_cols):

    # Compute good fit rate

    count_data = results_df.groupby([param, "fit_type"]).size().unstack().fillna(0)
    total = count_data.sum(axis=1)
    good_counts = count_data.get("Good Fit", pd.Series(0, index=total.index))
    prop = good_counts / good_counts.sum()

    # Plot barplot with error bars
    sns.barplot(
        x=prop.index,
        y=prop.values,
        ax=axes[i],
        palette=decrease_palette if i < 5 else increase_palette,
        edgecolor="w",
    )

    # Format axes
    axes[i].set_ylabel("% Good Fits", fontsize=8)
    axes[i].set_ylim(0, 0.7)
    axes[i].set_xlabel(f"{label_mapping[param]}", fontsize=8)

results_df = pd.read_csv("/Users/johnparker/Desktop/results_df_tree.csv")

decrease_color_map = {0.33: "b", 0.66: "skyblue", 1.0: "gray"}
stn_color_map = {1.0: "gray", 1.25: "salmon", 1.5: "r"}
noise_color_map = {1.0: "gray", 1.5: "salmon", 2: "r"}

# Filter for good fits and extract parameter combinations
good_fit_params = results_df[results_df["Good_Fit"] == 1][hues]
combos = [tuple(row) for row in good_fit_params.values]
combo_counts = Counter(combos)

# Get top N combinations
top_combos = combo_counts.most_common(10)
labels = [c[0] for c in top_combos]
counts = [c[1] for c in top_combos]

coverage_score = []
for c in top_combos:
    n_changes = sum(1 for val in c[0] if val != 1.0)
    coverage_score.append(c[1] / 30)

# axes[7].barh(labels, coverage_score)
for i, label in enumerate(labels):
    for j, val in enumerate(label):
        bar_color = "gray"
        if j < 5:
            bar_color = decrease_color_map[val]
        elif j == 5:
            bar_color = stn_color_map[val]
        else:
            bar_color = noise_color_map[val]
        axes[7].barh(
            y=len(labels) - i,
            width=coverage_score[i] / 7,
            height=0.8,
            left=coverage_score[i] * (j) / 7,
            color=bar_color,
            edgecolor="w",
        )

axes[7].set_yticks([])
axes[7].set_xlabel("Coverage Score", fontsize=8)
axes[7].set_ylabel("Parameter\nCombinations", fontsize=8)

for i, plot in enumerate(axes):
    plot.grid(False)
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)

# makeNice(axes)
add_fig_labels(axes)
fig.savefig(f"{FIG_DIR}/good_fit_rate_bars.pdf", bbox_inches="tight")
plt.close()

run_cmd(f"open {FIG_DIR}/good_fit_rate_bars.pdf")
