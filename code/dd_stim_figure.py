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

model_color = "#165b05"


def run_chi_sq_contingency(data):
    # Find columns where sum is NOT zero
    cols_to_keep = np.where(data.sum(axis=0) != 0)[0]

    # Keep only those columns
    matrix_clean = data[:, cols_to_keep]
    return chi2_contingency(matrix_clean).pvalue


# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"
vivo = True
dopamine_depletion = True

T0 = 3  # End of transient period
T1 = 5  # End of baseline period
T2 = 35  # End of stim period
seeds = 20  # seeds for simulation
size = 100  # units in network

# get experimental data
exp_data = (
    expan.get_slice_baseline_data()
    if not vivo
    else expan.get_in_vivo_baseline_data_short(segment=2, DD=dopamine_depletion)
)
exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]
exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(exp_data)
exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]
exp_fr_outliers_removed, exp_cv_outliers_removed = (
    exp_data["pre_exp_freq"],
    exp_data["pre_exp_cv"],
)

# get streac experimental results
_, gpe_pulse_exp, _, d1_pulse_exp = expan.get_dd_classification_results()
exp_gpe_mfs = expan.get_modulation_factors(gpe_pulse_exp)
exp_str_mfs = expan.get_modulation_factors(d1_pulse_exp)

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

# grab MF data for gpe and str analysis

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

# open figure to store stats
stats_file = open(f"{FIG_DIR}/dd_stim_figure_stats.txt", "w")

# set up figure
fig, ax = plt.subplots(2, 4, figsize=(8, 5), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(4)]

#### GPE ANALYSIS ####

# plot gpe modulation factors
plot_gpe_mfs = all_gpe_mfs.flatten()
axes[0].plot(
    np.arange(len(plot_gpe_mfs)) / len(plot_gpe_mfs),
    sorted(plot_gpe_mfs),
    color=model_color,
    label="sim",
)

axes[0].plot(
    np.arange(len(exp_gpe_mfs)) / len(exp_gpe_mfs),
    sorted(exp_gpe_mfs),
    color="gray",
    label="exp",
)
axes[0].set_ylim([-1, 1])

# plot histogram of gpe modulation factors
sns.histplot(
    plot_gpe_mfs,
    bins=np.arange(-1, 1, 0.1),
    color=model_color,
    edgecolor="w",
    kde=True,
    stat="probability",
    ax=axes[1],
)

sns.histplot(
    exp_gpe_mfs,
    bins=np.arange(-1, 1, 0.1),
    color="gray",
    edgecolor="w",
    kde=True,
    stat="probability",
    ax=axes[1],
)

# run and plot statistical tests
ks_pval = ks_2samp(plot_gpe_mfs, exp_gpe_mfs)[1]
ylims = axes[1].get_ylim()
pmr.plot_bracket(
    axes[1],
    np.percentile(exp_gpe_mfs, 25),
    np.percentile(exp_gpe_mfs, 75),
    ylims[0] + 1.15 * (ylims[1] - ylims[0]),
    ylims[0] + 1.18 * (ylims[1] - ylims[0]),
    f"KS $p=${ks_pval:.3f}",
)

ttest_pval = ttest_ind(plot_gpe_mfs, exp_gpe_mfs)[1]
pmr.plot_bracket(
    axes[1],
    np.mean(exp_gpe_mfs),
    np.mean(plot_gpe_mfs),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

# Print statistical tests
stats_file.write("### PANEL B ###\n")
stats_file.write("GPe MF Comparison, in-vivo Model vs Experimental:\n")
stats_file.write(
    f"\tMF, Model: {np.mean(plot_gpe_mfs):.3f} +/- {np.std(plot_gpe_mfs):.3f}\n",
)
stats_file.write(
    f"\tMF, Exp: {np.mean(exp_gpe_mfs):.3f} +/- {np.std(exp_gpe_mfs):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

axes[1].vlines(
    np.mean(plot_gpe_mfs),
    0,
    0.5,
    color=model_color,
    lw=1,
    ls="dashed",
)
axes[1].vlines(
    np.mean(exp_gpe_mfs),
    0,
    0.5,
    color="k",
    lw=1,
    ls="dashed",
)


# plot all sims for streac responses
gpe_streac_responses = np.zeros((len(gpe_streac) + 2, len(types) - 2))
for i in range(len(gpe_streac)):
    for j in range(len(types)):
        if types[j] == "biphasic IE":
            gpe_streac_responses[i + 2, 2] += np.sum(
                gpe_streac[i]["neural_response"] == types[5]
            )
        elif types[j] == "biphasic EI":
            gpe_streac_responses[i + 2, 4] += np.sum(
                gpe_streac[i]["neural_response"] == types[6]
            )
        else:
            gpe_streac_responses[i + 2, j] += np.sum(
                gpe_streac[i]["neural_response"] == types[j]
            )
gpe_streac_responses[1, :] = np.sum(gpe_streac_responses[2:], axis=0)
gpe_streac_responses[0, :] = [
    np.sum(gpe_pulse_exp["neural_response"] == x) for x in types[0:-2]
]

gpe_streac_responses_plot = (
    gpe_streac_responses * 100 / gpe_streac_responses.sum(axis=1)[:, None]
)

n_sample = 10
random.seed(10)
naive_random_choices = random.sample(
    list(range(2, gpe_streac_responses_plot.shape[0])), k=n_sample
)
naive_random_choices.insert(0, 0)
naive_random_choices.insert(1, 1)
"""for i in range(n_sample + 2):
    for j in range(gpe_streac_responses_plot.shape[1]):
        axes[2].bar(
            i,
            gpe_streac_responses_plot[naive_random_choices[i], j],
            bottom=np.sum(gpe_streac_responses_plot[naive_random_choices[i], :j]),
            color=color_dict[types[j]],
        )
"""

positions = [1, 3, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
for i in range(n_sample + 2):
    for j in range(gpe_streac_responses_plot.shape[1]):
        axes[2].bar(
            positions[i],
            gpe_streac_responses_plot[naive_random_choices[i], j],
            bottom=np.sum(gpe_streac_responses_plot[naive_random_choices[i], :j]),
            color=color_dict[types[j]],
            width=1.8 if i < 2 else 0.8,
        )

type_abbrevs = ["NE", "CI", "PI", "AI", "EX"]
for i, ta in enumerate(type_abbrevs):
    axes[2].annotate(
        ta,
        xy=(1, 0.1 + i * 0.1),
        xytext=(1, 0.1 + i * 0.1),
        xycoords="axes fraction",
        color=color_dict[types[i]],
        fontsize=6,
    )

# W vs tau scatter plot for gpe
all_taus = []
all_ws = []
all_responses = []
for i in range(len(gpe_streac)):
    sns.scatterplot(
        x=gpe_streac[i]["W_gpe"],
        y=gpe_streac[i]["tausyn"],
        color=[color_dict[x] for x in gpe_streac[i]["neural_response"]],
        alpha=[
            0.1 if x == "no effect" else 1 for x in gpe_streac[i]["neural_response"]
        ],
        ax=axes[3],
        s=2,
    )
    all_taus.extend(gpe_streac[i]["tausyn"])
    all_ws.extend(gpe_streac[i]["W_gpe"])
    all_responses.extend(gpe_streac[i]["neural_response"])
ylims = axes[3].get_ylim()
xlims = axes[3].get_xlim()
axes[3].hlines(
    np.mean(all_taus), xlims[0], xlims[1], linestyle="dashed", lw=0.5, color="k"
)
axes[3].vlines(
    np.mean(all_ws), ylims[0], ylims[1], linestyle="dashed", lw=0.5, color="k"
)
axes[3].set_ylim(ylims)
axes[3].set_xlim(xlims)

stats_file.write("### PANEL D ###\n")
stats_file.write("GPe connections stats:\n")
stats_file.write(f"\ttausyn: {np.mean(all_taus):.3f} +/- {np.std(all_taus):.3f}\n")
stats_file.write(f"\tWgpe: {np.mean(all_ws):.3f} +/- {np.std(all_ws):.3f}\n\n")


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

fig_gpe_grad, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
for bi, bar in enumerate(bars):
    for k, h in enumerate(bar):
        ax.bar(
            bi,
            h,
            width=1,
            edgecolor="w",
            bottom=np.sum(bar[:k]),
            color=color_dict[types[k]],
            align="edge",
        )
ax.set_xticks(np.arange(0, len(grad_bin_edges), 2))
ax.set_xticklabels(labels=[f"{x:.3f}" for x in grad_bin_edges[::2]], rotation=25)
fig_gpe_grad.savefig("/Users/johnparker/Desktop/fig_gpe_grad_dd.pdf")
plt.close()
run_cmd("open /Users/johnparker/Desktop/fig_gpe_grad_dd.pdf")


#### STR ANALYSIS #####
# plot str modulation factors
plot_str_mfs = all_str_mfs.flatten()
axes[4].plot(
    np.arange(len(plot_str_mfs)) / len(plot_str_mfs),
    sorted(plot_str_mfs),
    color=model_color,
    label="sim",
)
axes[4].plot(
    np.arange(len(exp_str_mfs)) / len(exp_str_mfs),
    sorted(exp_str_mfs),
    color="gray",
    label="exp",
)
axes[4].set_ylim([-1, 1])

# plot histogram of str modulation factors
sns.histplot(
    plot_str_mfs,
    bins=np.arange(-1, 1, 0.1),
    color=model_color,
    edgecolor="w",
    kde=True,
    stat="probability",
    ax=axes[5],
)

sns.histplot(
    exp_str_mfs,
    bins=np.arange(-1, 1, 0.1),
    color="gray",
    edgecolor="w",
    kde=True,
    stat="probability",
    ax=axes[5],
)

ks_pval = ks_2samp(plot_str_mfs, exp_str_mfs)[1]
ylims = axes[5].get_ylim()
pmr.plot_bracket(
    axes[5],
    np.percentile(exp_str_mfs, 25),
    np.percentile(exp_str_mfs, 75),
    ylims[0] + 1.15 * (ylims[1] - ylims[0]),
    ylims[0] + 1.18 * (ylims[1] - ylims[0]),
    f"KS $p=${ks_pval:.3f}",
)

ttest_pval = ttest_ind(plot_str_mfs, exp_str_mfs)[1]
pmr.plot_bracket(
    axes[5],
    np.mean(exp_str_mfs),
    np.mean(plot_str_mfs),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)


# Print statistical tests
stats_file.write("### PANEL F ###\n")
stats_file.write("STR MF Comparison, in-vivo Model vs Experimental:\n")
stats_file.write(
    f"\tMF, Model: {np.mean(plot_str_mfs):.3f} +/- {np.std(plot_str_mfs):.3f}\n",
)
stats_file.write(
    f"\tMF, Exp: {np.mean(exp_str_mfs):.3f} +/- {np.std(exp_str_mfs):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

axes[5].vlines(
    np.mean(plot_str_mfs),
    0,
    0.15,
    color=model_color,
    lw=1,
    ls="dashed",
)
axes[5].vlines(
    np.mean(exp_str_mfs),
    0,
    0.15,
    color="k",
    lw=1,
    ls="dashed",
)

# plot all sims for streac responses
str_streac_responses = np.zeros((len(str_streac) + 2, len(types) - 2))
for i in range(len(str_streac)):
    for j in range(len(types)):
        if types[j] == "biphasic IE":
            str_streac_responses[i + 2, 2] += np.sum(
                str_streac[i]["neural_response"] == types[5]
            )
        elif types[j] == "biphasic EI":
            str_streac_responses[i + 2, 4] += np.sum(
                str_streac[i]["neural_response"] == types[6]
            )
        else:
            str_streac_responses[i + 2, j] += np.sum(
                str_streac[i]["neural_response"] == types[j]
            )
str_streac_responses[1, :] = np.sum(str_streac_responses[2:], axis=0)
str_streac_responses[0, :] = [
    np.sum(d1_pulse_exp["neural_response"] == x) for x in types[0:-2]
]
str_streac_responses_plot = (
    str_streac_responses * 100 / str_streac_responses.sum(axis=1)[:, None]
)

"""for i in range(n_sample + 2):
    for j in range(str_streac_responses_plot.shape[1]):
        axes[6].bar(
            i,
            str_streac_responses_plot[naive_random_choices[i], j],
            bottom=np.sum(str_streac_responses_plot[naive_random_choices[i], :j]),
            color=color_dict[types[j]],
        )"""

positions = [1, 3, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
for i in range(n_sample + 2):
    for j in range(str_streac_responses_plot.shape[1]):
        axes[6].bar(
            positions[i],
            str_streac_responses_plot[naive_random_choices[i], j],
            bottom=np.sum(str_streac_responses_plot[naive_random_choices[i], :j]),
            color=color_dict[types[j]],
            width=1.8 if i < 2 else 0.8,
        )

type_abbrevs = ["NE", "CI", "PI", "AI", "EX"]
for i, ta in enumerate(type_abbrevs):
    axes[6].annotate(
        ta,
        xy=(1, 0.1 + i * 0.1),
        xytext=(1, 0.1 + i * 0.1),
        xycoords="axes fraction",
        color=color_dict[types[i]],
        fontsize=6,
    )


# W vs tau scatter plot for STR
all_taus = []
all_ws = []
all_responses = []
for i in range(len(str_streac)):
    sns.scatterplot(
        x=str_streac[i]["W_str"],
        y=str_streac[i]["tausyn_dend"],
        color=[color_dict[x] for x in str_streac[i]["neural_response"]],
        alpha=[
            0.1 if x == "no effect" else 1 for x in str_streac[i]["neural_response"]
        ],
        ax=axes[7],
        s=2,
    )
    all_taus.extend(str_streac[i]["tausyn_dend"])
    all_ws.extend(str_streac[i]["W_str"])
    all_responses.extend(str_streac[i]["neural_response"])
ylims = axes[7].get_ylim()
xlims = axes[7].get_xlim()
axes[7].hlines(
    np.mean(all_taus), xlims[0], xlims[1], linestyle="dashed", lw=0.5, color="k"
)
axes[7].vlines(
    np.mean(all_ws), ylims[0], ylims[1], linestyle="dashed", lw=0.5, color="k"
)
axes[7].set_ylim(ylims)
axes[7].set_xlim(xlims)

stats_file.write("### PANEL H ###\n")
stats_file.write("STR connections stats:\n")
stats_file.write(f"\ttausyn_dend: {np.mean(all_taus):.3f} +/- {np.std(all_taus):.3f}\n")
stats_file.write(f"\tWstr: {np.mean(all_ws):.3f} +/- {np.std(all_ws):.3f}\n\n")

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

fig_str_grad, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
for bi, bar in enumerate(bars):
    for k, h in enumerate(bar):
        ax.bar(
            bi,
            h,
            width=1,
            edgecolor="w",
            bottom=np.sum(bar[:k]),
            color=color_dict[types[k]],
            align="edge",
        )
ax.set_xticks(np.arange(0, len(grad_bin_edges), 2))
ax.set_xticklabels(labels=[f"{x:.3f}" for x in grad_bin_edges[::2]], rotation=25)
fig_str_grad.savefig("/Users/johnparker/Desktop/fig_str_grad_dd.pdf")
plt.close()
run_cmd("open /Users/johnparker/Desktop/fig_str_grad_dd.pdf")


# add labels and match axes
add_fig_labels(axes)

for i in [0, 4]:
    axes[i].set_xlabel("Normalized Unit", fontsize=8)
    axes[i].set_ylabel("Modulation Factor", fontsize=8)
    axes[i].hlines(0, 0, 1, linestyle="dashed", color="k", lw=0.5)

for i in [1, 5]:
    axes[i].set_xlim([-1, 1])
    axes[i].set_xlabel("Modulation Factor", fontsize=8)


# adjust labels
for i in [3, 7]:
    axes[i].set_xlabel(
        "$W_{GPe}$ (nS/pF)" if i == 3 else "$W_{Str}$ (nS/pF)", fontsize=8
    )
    axes[i].set_ylabel(
        "$\\tau_{GPe}^{syn}$ (ms)" if i == 3 else "$\\tau_{Str}^{syn}$ (ms)", fontsize=8
    )

# clean up plots
axes[2].set_ylabel("Percentage", fontsize=8)
axes[6].set_ylabel("Percentage", fontsize=8)
axes[2].set_ylim([60, 100])
axes[6].set_ylim([30, 100])
# axes[3].set_xticks(np.arange(gpe_streac_responses.shape[0]))
# axes[7].set_xticks(np.arange(gpe_streac_responses.shape[0]))

gpe_stat_sig = [
    1 if run_chi_sq_contingency(gpe_streac_responses[[0, i], :]) < 0.05 else 0
    for i in range(2, gpe_streac_responses.shape[0])
]
str_stat_sig = [
    1 if run_chi_sq_contingency(str_streac_responses[[0, i], :]) < 0.05 else 0
    for i in range(2, str_streac_responses.shape[0])
]

stats_file.write(f"### Panel C ###\n")
stats_file.write(
    f"\tGPe All Sims: {run_chi_sq_contingency(gpe_streac_responses[[0, 1], :])}\n"
)
stats_file.write(
    f"\tGPe Individual Sig: {np.sum(gpe_stat_sig)}/{gpe_streac_responses.shape[0]-2}\n\n"
)


stats_file.write(f"### Panel G ###\n")
stats_file.write(
    f"\tSTR All Sims: {run_chi_sq_contingency(str_streac_responses[[0, 1], :])}\n"
)
stats_file.write(
    f"\tSTR Individual Sig: {np.sum(str_stat_sig)}/{str_streac_responses.shape[0]-2}\n"
)


str_streac_responses_labels = [
    "Exp",
    f"All\nSims{ "*" if run_chi_sq_contingency(str_streac_responses[[0, 1], :]) < 0.05 else ""}",
    # "Example Sims",
]

gpe_streac_responses_labels = [
    "Exp",
    f"All\nSims{ "*" if run_chi_sq_contingency(gpe_streac_responses[[0, 1], :]) < 0.05 else ""}",
    # "Example Sims",
]

axes[2].set_xticks([1, 3])
axes[2].set_xticklabels(gpe_streac_responses_labels, fontsize=5, rotation=0)
axes[2].annotate(
    "Example Sims",
    xy=(0.625, 0),
    xytext=(0.625, -0.1),
    xycoords="axes fraction",
    fontsize=6,
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="-[, widthB=5.1, lengthB=0.4", lw=0.5, color="k"),
)


axes[6].set_xticks([1, 3])
axes[6].set_xticklabels(str_streac_responses_labels, fontsize=5, rotation=0)

axes[6].annotate(
    "Example Sims",
    xy=(0.625, 0),
    xytext=(0.625, -0.1),
    xycoords="axes fraction",
    fontsize=6,
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="-[, widthB=5.1, lengthB=0.4", lw=0.5, color="k"),
)

for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)

# save, close, and open pdf of figure
fig.savefig(
    f"{FIG_DIR}/dd_stim_figure.pdf",
    bbox_inches="tight",
)
plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/dd_stim_figure.pdf")
