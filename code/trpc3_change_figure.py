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


# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"

stats_file = open(f"{FIG_DIR}/trpc3_change_figure.txt", "w")


size = 100
T0 = 3
T = 5
all_seeds = []  # all DD network seed
naive_all_seeds = []  # all naive network seeds
all_dd_hetero = []  # all neurons in matching DD network seeds
kept_all_dd_seeds = []  # best matching DD network seeds
plotting_matches = []
N_vivo_sims = 3
N_dd_sims = 3
in_vivo_seed_counter = 1

trpc3_change_paths = []
naive_vivo_paths = []

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

        no_change_results_sim = dd_dir_res[
            (dd_dir_res["kcc2_S_change"] == 1)
            & (dd_dir_res["kcc2_D_change"] == 1)
            & (dd_dir_res["trpc3_change"] == 1)
            & (dd_dir_res["tonic_CL_S_change"] == 1)
            & (dd_dir_res["tonic_CL_D_change"] == 1)
            & (dd_dir_res["stn_change"] == 1)
            & (dd_dir_res["noise_change"] == 1)
        ]["Sim"].values

        trpc3_change_results_sim = dd_dir_res[
            (dd_dir_res["kcc2_S_change"] == 1)
            & (dd_dir_res["kcc2_D_change"] == 1)
            & (dd_dir_res["trpc3_change"] == 0.33)
            & (dd_dir_res["tonic_CL_S_change"] == 1)
            & (dd_dir_res["tonic_CL_D_change"] == 1)
            & (dd_dir_res["stn_change"] == 1)
            & (dd_dir_res["noise_change"] == 1)
        ]["Sim"].values

        naive_vivo_paths.append(
            f"{vivo_data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/sim_{no_change_results_sim[0]:06d}"
        )

        trpc3_change_paths.append(
            f"{vivo_data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/sim_{trpc3_change_results_sim[0]:06d}"
        )

        dd_dir_res["Slice Seed"] = slice_seed
        dd_dir_res["in_vivo_seed"] = in_vivo_seed_counter

        all_seeds.append(dd_dir_res)
        in_vivo_seed_counter += 1

results_df = pd.concat(all_seeds, ignore_index=True)


fig, ax = plt.subplots(1, 3, figsize=(8, 3), dpi=300, tight_layout=True)
axes = [ax[i] for i in range(3)]

no_change_data_fr = []
no_change_data_cv = []
for path in naive_vivo_paths:
    sim_spikes, sim_df = pmr.gather_sim_data(path, T, T0, size=size)
    sim_data = pd.DataFrame(
        {"pre_exp_freq": sim_df["pre_exp_freq"], "pre_exp_cv": sim_df["pre_exp_cv"]}
    )
    sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        sim_data
    )
    no_change_data_fr.extend(sim_data["pre_exp_freq"].values)
    no_change_data_cv.extend(sim_data["pre_exp_cv"].values)

trpc3_change_data_fr = []
trpc3_change_data_cv = []
for path in trpc3_change_paths:
    sim_spikes, sim_df = pmr.gather_sim_data(path, T, T0, size=size)
    sim_data = pd.DataFrame(
        {"pre_exp_freq": sim_df["pre_exp_freq"], "pre_exp_cv": sim_df["pre_exp_cv"]}
    )
    sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        sim_data
    )
    trpc3_change_data_fr.extend(sim_data["pre_exp_freq"].values)
    trpc3_change_data_cv.extend(sim_data["pre_exp_cv"].values)

bw_fr = 10
bins_fr = np.arange(
    0,
    np.max(
        [
            np.max(trpc3_change_data_fr),
            np.max(no_change_data_fr),
        ]
    )
    + bw_fr,
    bw_fr,
)

sns.histplot(
    no_change_data_fr,
    bins=bins_fr,
    ax=axes[0],
    kde=True,
    color="gray",
    edgecolor="w",
    stat="probability",
    label=f"naive ($n={len(no_change_data_fr)}$)",
)

sns.histplot(
    trpc3_change_data_fr,
    bins=bins_fr,
    ax=axes[0],
    kde=True,
    color="green",
    edgecolor="w",
    stat="probability",
    label=f"trpc3=0.33 ($n={len(trpc3_change_data_fr)}$)",
)

axes[0].legend(fancybox=False, frameon=False, fontsize=6)

# perfrom statistical tests and plot on histograms
ylims = axes[0].get_ylim()
ks_pval = ks_2samp(no_change_data_fr, trpc3_change_data_fr).pvalue
pmr.plot_bracket(
    axes[0],
    np.percentile(no_change_data_fr, 25),
    np.percentile(no_change_data_fr, 75),
    ylims[0] + 1.15 * (ylims[1] - ylims[0]),
    ylims[0] + 1.18 * (ylims[1] - ylims[0]),
    f"KS $p=${ks_pval:.3f}",
)

ttest_pval = ttest_ind(no_change_data_fr, trpc3_change_data_fr).pvalue
pmr.plot_bracket(
    axes[0],
    np.mean(no_change_data_fr),
    np.mean(trpc3_change_data_fr),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

axes[0].vlines(np.mean(no_change_data_fr), 0, ylims[1], color="gray", ls="dashed")
axes[0].vlines(np.mean(trpc3_change_data_fr), 0, ylims[1], color="green", ls="dashed")

# Print statistical tests
stats_file.write("### PANEL A ###\n")
stats_file.write("FR Comparison, naive vs trpc3=0.33 Experimental:\n")
stats_file.write(
    f"\tFR, naive: {np.mean(no_change_data_fr):.3f} +/- {np.std(no_change_data_fr):.3f}\n",
)
stats_file.write(
    f"\tFR, dd: {np.mean(trpc3_change_data_fr):.3f} +/- {np.std(trpc3_change_data_fr):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

bw_cv = 0.2
bins_cv = np.arange(
    0,
    np.max(
        [
            np.max(trpc3_change_data_cv),
            np.max(no_change_data_cv),
        ]
    )
    + bw_cv,
    bw_cv,
)

sns.histplot(
    no_change_data_cv,
    bins=bins_cv,
    ax=axes[1],
    kde=True,
    color="gray",
    edgecolor="w",
    stat="probability",
    label=f"naive ($n={len(no_change_data_cv)}$)",
)

sns.histplot(
    trpc3_change_data_cv,
    bins=bins_cv,
    ax=axes[1],
    kde=True,
    color="green",
    edgecolor="w",
    stat="probability",
    label=f"trpc3=0.33 ($n={len(trpc3_change_data_cv)}$)",
)

# axes[1].legend(fancybox=False, frameon=False, fontsize=6)

# perfrom statistical tests and plot on histograms
ylims = axes[1].get_ylim()
ks_pval = ks_2samp(no_change_data_cv, trpc3_change_data_cv).pvalue
pmr.plot_bracket(
    axes[1],
    np.percentile(no_change_data_cv, 25),
    np.percentile(no_change_data_cv, 75),
    ylims[0] + 1.15 * (ylims[1] - ylims[0]),
    ylims[0] + 1.18 * (ylims[1] - ylims[0]),
    f"KS $p=${ks_pval:.3f}",
)

ttest_pval = ttest_ind(no_change_data_cv, trpc3_change_data_cv).pvalue
pmr.plot_bracket(
    axes[1],
    np.mean(no_change_data_cv),
    np.mean(trpc3_change_data_cv),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

axes[1].vlines(np.mean(no_change_data_cv), 0, ylims[1], color="gray", ls="dashed")
axes[1].vlines(np.mean(trpc3_change_data_cv), 0, ylims[1], color="green", ls="dashed")

# Print statistical tests
stats_file.write("### PANEL B ###\n")
stats_file.write("CV Comparison, naive vs trpc3=0.33 Experimental:\n")
stats_file.write(
    f"\tCV, naive: {np.mean(no_change_data_cv):.3f} +/- {np.std(no_change_data_cv):.3f}\n",
)
stats_file.write(
    f"\tCV, dd: {np.mean(trpc3_change_data_cv):.3f} +/- {np.std(trpc3_change_data_cv):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")


pmr.plot_linear_fit(no_change_data_fr, no_change_data_cv, axes[2], color="gray")
pmr.plot_linear_fit(trpc3_change_data_fr, trpc3_change_data_cv, axes[2], color="green")

slope, intercept, r, p, std_err = stats.linregress(no_change_data_fr, no_change_data_cv)
stats_file.write("### PANEL C ###\n")
stats_file.write("naive Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")

slope, intercept, r, p, std_err = stats.linregress(
    trpc3_change_data_fr, trpc3_change_data_cv
)
stats_file.write("trpc3=0.33 Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")

axes[0].set_xlabel("Firing Rate (Hz)")
axes[0].set_ylabel("Probability")

axes[1].set_xlabel("CV")
axes[1].set_ylabel("Probability")

axes[2].set_xlabel("Firing Rate (Hz)")
axes[2].set_ylabel("CV")

for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)

# add labels and match axes
add_fig_labels(axes)

# save, close, and open pdf of figure
fig.savefig(
    f"{FIG_DIR}/trpc3_change_figure.pdf",
    bbox_inches="tight",
)

plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/trpc3_change_figure.pdf")
