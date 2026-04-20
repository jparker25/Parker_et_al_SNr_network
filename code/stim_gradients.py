"""
naive_stim_figure.py

Creates figure for in-vivo naive stim analysis, comparing experimental and simulation results for publication.

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
from scipy.stats import chi2_contingency

# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import run_model

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8

# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"
vivo = True
dopamine_depletion = False
model_color = "#165b05"

T0 = 3  # End of transient period
T1 = 5  # End of baseline period
T2 = 35  # End of stim period
seeds = 20  # seeds for simulation
size = 100  # units in network

# graphing colors and types
color_dict = {
    "complete inhibition": "navy",
    "adapting inhibition": "cornflowerblue",
    "partial inhibition": "blue",
    "no effect": "slategrey",
    "excitation": "lightcoral",
    "biphasic IE": "blueviolet",
    "biphasic EI": "orchid",
}


types = [
    "no effect",
    "complete inhibition",
    "partial inhibition",
    "adapting inhibition",
    "excitation",
    "biphasic IE",
    "biphasic EI",
]

# intialiaze MF variables
all_gpe_mfs = []  # np.zeros((seeds, size))
all_str_mfs = []  # = np.zeros((seeds, size))
gpe_streac = []
str_streac = []

# set up params for analysis
params = {
    "str_start_time": np.ones(size) * T1,
    "str_base_length": np.ones(size),
    "str_stim_length": np.ones(size),
    "str_post_length": np.ones(size),
    "gpe_start_time": np.ones(size) * T1,
    "gpe_base_length": np.ones(size),
    "gpe_stim_length": np.ones(size),
    "gpe_post_length": np.ones(size),
}

N_stim_seeds = 3
N_vivo_sims = 3
slice_results = pd.read_csv(f"data/param_seed_finder_slice_correlated/results.csv")

slice_results = slice_results[
    (slice_results["KS_FR_pval"] >= 0.05)
    & (slice_results["Ttest_FR_pval"] >= 0.05)
    & (slice_results["Ttest_CV_pval"] >= 0.05)
]
avg_err_sorted = slice_results.sort_values(by="Avg. Error")

slice_seeds = [int(row["Seed"]) for index, row in avg_err_sorted.iterrows()]

d1_mfs = {}
gpe_mfs = {}

vivo_models = {}

count = 0
for slice_seed in slice_seeds:
    data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
    results = pd.read_csv(f"{data_dir}/results.csv")
    results = results[
        (results["KS_FR_pval"] >= 0.05)
        & (results["Ttest_FR_pval"] >= 0.05)
        & (results["Ttest_CV_pval"] >= 0.05)
        & (results["KS_CV_pval"] >= 0.05)
    ]
    sorted_results = results.sort_values(by="Avg. Error")
    sorted_results = sorted_results.iloc[:N_vivo_sims]

    stim_dir = f"{data_dir}_naive_d1_stim"
    for index, row in sorted_results.iterrows():
        for stim_seeds in np.arange(1, N_stim_seeds + 1):
            str_seed_dir = (
                f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/sim_{stim_seeds:04d}"
            )

            str_streac.append(pd.read_csv(f"{str_seed_dir}/processed/all_data.csv"))

            all_str_mfs.append(
                pmr.generate_modulation_factors(
                    str_seed_dir, params, T1, T2, False, True, size=size
                )[:, 1]
            )
    stim_dir = f"{data_dir}_naive_gpe_stim"
    for index, row in sorted_results.iterrows():
        for stim_seeds in np.arange(1, N_stim_seeds + 1):
            gpe_seed_dir = (
                f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/sim_{stim_seeds:04d}"
            )

            gpe_streac.append(pd.read_csv(f"{gpe_seed_dir}/processed/all_data.csv"))

            all_gpe_mfs.append(
                pmr.generate_modulation_factors(
                    gpe_seed_dir, params, T1, T2, False, True, size=size
                )[:, 1]
            )

            count += 1

all_gpe_mfs = np.asarray(all_gpe_mfs)  # np.zeros((seeds, size))
all_str_mfs = np.asarray(all_str_mfs)


########################################################################################
fig, ax = plt.subplots(2, 2, figsize=(6, 3), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(2)]

# W vs tau scatter plot for gpe
all_taus = []
all_ws = []
all_responses = []
for i in range(len(gpe_streac)):
    all_taus.extend(gpe_streac[i]["tausyn"])
    all_ws.extend(gpe_streac[i]["W_gpe"])
    all_responses.extend(gpe_streac[i]["neural_response"])

##### Gradient Analysis #####
grad_bins = 20
grad_bin_edges = np.linspace(min(all_ws), max(all_ws), grad_bins + 1)
grad_df = pd.DataFrame(
    {"W_gpe": all_ws, "tausyn": all_taus, "neural_response": all_responses}
)

grad_df = grad_df.replace(
    {
        "neural_response": {
            # "partial inhibition": "inhibition",
            # "adapting inhibition": "inhibition",
            "biphasic IE": "inhibition",
            # "complete inhibition": "inhibition",
            "biphasic EI": "excitation",
        }
    }
)

bars = np.zeros((grad_bins, 5))
for i in range(grad_bins):
    gbes_l, gbes_r = grad_bin_edges[i], grad_bin_edges[i + 1]
    partial_grad_df = grad_df[
        (grad_df["W_gpe"] >= gbes_l) & (grad_df["W_gpe"] < gbes_r)
    ]
    props = partial_grad_df.value_counts("neural_response") / len(partial_grad_df)
    for j, type in enumerate(types[:-2]):
        bars[i, j] = len(
            partial_grad_df[partial_grad_df["neural_response"] == type]
        ) / len(partial_grad_df)

for bi, bar in enumerate(bars):
    for k, h in enumerate(bar):
        axes[0].bar(
            bi,
            h,
            width=1,
            edgecolor="w",
            bottom=np.sum(bar[:k]),
            color=color_dict[types[k]],
            align="edge",
        )
axes[0].set_xticks(np.arange(0, len(grad_bin_edges), 2))
axes[0].set_xticklabels(labels=[f"{x:.3f}" for x in grad_bin_edges[::2]], rotation=25)


all_taus = []
all_ws = []
all_responses = []
for i in range(len(str_streac)):
    all_taus.extend(str_streac[i]["tausyn_dend"])
    all_ws.extend(str_streac[i]["W_str"])
    all_responses.extend(str_streac[i]["neural_response"])

##### Gradient Analysis #####
grad_bins = 20
grad_bin_edges = np.linspace(min(all_ws), max(all_ws), grad_bins + 1)
grad_df = pd.DataFrame(
    {"W_str": all_ws, "tausyn_dend": all_taus, "neural_response": all_responses}
)

grad_df = grad_df.replace(
    {
        "neural_response": {
            # "partial inhibition": "inhibition",
            # "adapting inhibition": "inhibition",
            "biphasic IE": "inhibition",
            # "complete inhibition": "inhibition",
            "biphasic EI": "excitation",
        }
    }
)

bars = np.zeros((grad_bins, 5))
for i in range(grad_bins):
    gbes_l, gbes_r = grad_bin_edges[i], grad_bin_edges[i + 1]
    partial_grad_df = grad_df[
        (grad_df["W_str"] >= gbes_l) & (grad_df["W_str"] < gbes_r)
    ]
    props = partial_grad_df.value_counts("neural_response") / len(partial_grad_df)
    for j, type in enumerate(types[:-2]):
        bars[i, j] = len(
            partial_grad_df[partial_grad_df["neural_response"] == type]
        ) / len(partial_grad_df)

for bi, bar in enumerate(bars):
    for k, h in enumerate(bar):
        axes[1].bar(
            bi,
            h,
            width=1,
            edgecolor="w",
            bottom=np.sum(bar[:k]),
            color=color_dict[types[k]],
            align="edge",
        )
axes[1].set_xticks(np.arange(0, len(grad_bin_edges), 2))
axes[1].set_xticklabels(labels=[f"{x:.3f}" for x in grad_bin_edges[::2]], rotation=25)


# grab MF data for gpe and str analysis
# intialiaze MF variables
all_gpe_mfs = []  # np.zeros((seeds, size))
all_str_mfs = []  # = np.zeros((seeds, size))
gpe_streac = []
str_streac = []

# set up params for analysis
params = {
    "str_start_time": np.ones(size) * T1,
    "str_base_length": np.ones(size),
    "str_stim_length": np.ones(size),
    "str_post_length": np.ones(size),
    "gpe_start_time": np.ones(size) * T1,
    "gpe_base_length": np.ones(size),
    "gpe_stim_length": np.ones(size),
    "gpe_post_length": np.ones(size),
}

N_stim_seeds = 3
N_vivo_sims = 3
slice_results = pd.read_csv(f"data/param_seed_finder_slice_correlated/results.csv")

slice_results = slice_results[
    (slice_results["KS_FR_pval"] >= 0.05)
    & (slice_results["Ttest_FR_pval"] >= 0.05)
    & (slice_results["Ttest_CV_pval"] >= 0.05)
]
avg_err_sorted = slice_results.sort_values(by="Avg. Error")

slice_seeds = [int(row["Seed"]) for index, row in avg_err_sorted.iterrows()]

d1_mfs = {}
gpe_mfs = {}

vivo_models = {}

count = 0
for slice_seed in slice_seeds:
    data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
    results = pd.read_csv(f"{data_dir}/results.csv")
    results = results[
        (results["KS_FR_pval"] >= 0.05)
        & (results["Ttest_FR_pval"] >= 0.05)
        & (results["Ttest_CV_pval"] >= 0.05)
        & (results["KS_CV_pval"] >= 0.05)
    ]
    sorted_results = results.sort_values(by="Avg. Error")
    sorted_results = sorted_results.iloc[:N_vivo_sims]

    stim_dir = f"{data_dir}_dd_d1_stim"
    for index, row in sorted_results.iterrows():
        # sim_to_pull_from = f"{data_dir}/sim_{int(row["Sim"]):04d}"

        dd_search_results = pd.read_csv(
            f"{data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/results.csv"
        )
        dd_search_results = dd_search_results[
            (dd_search_results["KS_FR_pval"] >= 0.05)
            & (dd_search_results["Ttest_FR_pval"] >= 0.05)
            & (dd_search_results["Ttest_CV_pval"] >= 0.05)
            & (dd_search_results["KS_CV_pval"] >= 0.05)
        ]
        sorted_dd_search_results = dd_search_results.sort_values(by="Avg. Error")
        sorted_dd_search_results = sorted_dd_search_results.iloc[:N_vivo_sims]

        for dd_index, dd_row in sorted_dd_search_results.iterrows():
            # sim_to_pull_from = f"{data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/sim_{int(dd_row["Sim"]):06d}"
            for stim_seeds in np.arange(1, N_stim_seeds + 1):
                str_seed_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/dd_sim_{int(dd_row["Sim"]):06d}/sim_{stim_seeds:04d}"
                str_streac.append(pd.read_csv(f"{str_seed_dir}/processed/all_data.csv"))
                all_str_mfs.append(
                    pmr.generate_modulation_factors(
                        str_seed_dir, params, T1, T2, False, True, size=size
                    )[:, 1]
                )
    stim_dir = f"{data_dir}_dd_gpe_stim"
    for index, row in sorted_results.iterrows():
        # sim_to_pull_from = f"{data_dir}/sim_{int(row["Sim"]):04d}"

        dd_search_results = pd.read_csv(
            f"{data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/results.csv"
        )
        dd_search_results = dd_search_results[
            (dd_search_results["KS_FR_pval"] >= 0.05)
            & (dd_search_results["Ttest_FR_pval"] >= 0.05)
            & (dd_search_results["Ttest_CV_pval"] >= 0.05)
            & (dd_search_results["KS_CV_pval"] >= 0.05)
        ]
        sorted_dd_search_results = dd_search_results.sort_values(by="Avg. Error")
        sorted_dd_search_results = sorted_dd_search_results.iloc[:N_vivo_sims]

        for dd_index, dd_row in sorted_dd_search_results.iterrows():
            # sim_to_pull_from = f"{data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/sim_{int(dd_row["Sim"]):06d}"
            for stim_seeds in np.arange(1, N_stim_seeds + 1):
                gpe_seed_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/dd_sim_{int(dd_row["Sim"]):06d}/sim_{stim_seeds:04d}"
                gpe_streac.append(pd.read_csv(f"{gpe_seed_dir}/processed/all_data.csv"))
                all_gpe_mfs.append(
                    pmr.generate_modulation_factors(
                        gpe_seed_dir, params, T1, T2, False, True, size=size
                    )[:, 1]
                )

            count += 1


all_gpe_mfs = np.asarray(all_gpe_mfs)  # np.zeros((seeds, size))
all_str_mfs = np.asarray(all_str_mfs)


# W vs tau scatter plot for gpe
all_taus = []
all_ws = []
all_responses = []
for i in range(len(gpe_streac)):
    all_taus.extend(gpe_streac[i]["tausyn"])
    all_ws.extend(gpe_streac[i]["W_gpe"])
    all_responses.extend(gpe_streac[i]["neural_response"])


##### Gradient Analysis #####
grad_bins = 20
grad_bin_edges = np.linspace(min(all_ws), max(all_ws), grad_bins + 1)
grad_df = pd.DataFrame(
    {"W_gpe": all_ws, "tausyn": all_taus, "neural_response": all_responses}
)

grad_df = grad_df.replace(
    {
        "neural_response": {
            # "partial inhibition": "inhibition",
            # "adapting inhibition": "inhibition",
            "biphasic IE": "inhibition",
            # "complete inhibition": "inhibition",
            "biphasic EI": "excitation",
        }
    }
)

bars = np.zeros((grad_bins, 5))
for i in range(grad_bins):
    gbes_l, gbes_r = grad_bin_edges[i], grad_bin_edges[i + 1]
    partial_grad_df = grad_df[
        (grad_df["W_gpe"] >= gbes_l) & (grad_df["W_gpe"] < gbes_r)
    ]
    props = partial_grad_df.value_counts("neural_response") / len(partial_grad_df)
    for j, type in enumerate(types[:-2]):
        bars[i, j] = len(
            partial_grad_df[partial_grad_df["neural_response"] == type]
        ) / len(partial_grad_df)

for bi, bar in enumerate(bars):
    for k, h in enumerate(bar):
        axes[2].bar(
            bi,
            h,
            width=1,
            edgecolor="w",
            bottom=np.sum(bar[:k]),
            color=color_dict[types[k]],
            align="edge",
        )
axes[2].set_xticks(np.arange(0, len(grad_bin_edges), 2))
axes[2].set_xticklabels(labels=[f"{x:.3f}" for x in grad_bin_edges[::2]], rotation=25)


# W vs tau scatter plot for STR
all_taus = []
all_ws = []
all_responses = []
for i in range(len(str_streac)):
    all_taus.extend(str_streac[i]["tausyn_dend"])
    all_ws.extend(str_streac[i]["W_str"])
    all_responses.extend(str_streac[i]["neural_response"])

##### Gradient Analysis #####
grad_bins = 20
grad_bin_edges = np.linspace(min(all_ws), max(all_ws), grad_bins + 1)
grad_df = pd.DataFrame(
    {"W_str": all_ws, "tausyn_dend": all_taus, "neural_response": all_responses}
)

grad_df = grad_df.replace(
    {
        "neural_response": {
            # "partial inhibition": "inhibition",
            # "adapting inhibition": "inhibition",
            "biphasic IE": "inhibition",
            # "complete inhibition": "inhibition",
            "biphasic EI": "excitation",
        }
    }
)

bars = np.zeros((grad_bins, 5))
for i in range(grad_bins):
    gbes_l, gbes_r = grad_bin_edges[i], grad_bin_edges[i + 1]
    partial_grad_df = grad_df[
        (grad_df["W_str"] >= gbes_l) & (grad_df["W_str"] < gbes_r)
    ]
    props = partial_grad_df.value_counts("neural_response") / len(partial_grad_df)
    for j, type in enumerate(types[:-2]):
        bars[i, j] = len(
            partial_grad_df[partial_grad_df["neural_response"] == type]
        ) / len(partial_grad_df)

for bi, bar in enumerate(bars):
    for k, h in enumerate(bar):
        axes[3].bar(
            bi,
            h,
            width=1,
            edgecolor="w",
            bottom=np.sum(bar[:k]),
            color=color_dict[types[k]],
            align="edge",
        )
axes[3].set_xticks(np.arange(0, len(grad_bin_edges), 2))
axes[3].set_xticklabels(labels=[f"{x:.3f}" for x in grad_bin_edges[::2]], rotation=25)

add_fig_labels(axes)

titles = [
    "$W_{GPe}$ (nS/pF)",
    "$W_{STR}$ (nS/pF)",
    "$W_{GPe}$ (nS/pF)",
    "$W_{STR}$ (nS/pF)",
]
for i, plot in enumerate(axes):
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
    plot.set_xlabel(titles[i])
    plot.set_ylim([0, 1])
    plot.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plot.set_yticklabels([0, 25, 50, 75, 100])
    plot.set_ylabel("Percentage")


fig.savefig(f"{FIG_DIR}/stim_gradients.pdf", bbox_inches="tight")
plt.close()

run_cmd(f"open {FIG_DIR}/stim_gradients.pdf")
