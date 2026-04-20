"""
in_vivo_figure.py

Creates figure for in-vivo data analysis, comparing experimental and simulation results for publication.

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
import os
from matplotlib.gridspec import GridSpec


# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import run_model

import matplotlib as mpl

DPI_SIZE = 600
mpl.rcParams["savefig.dpi"] = DPI_SIZE  # or whatever you need
mpl.rcParams["figure.dpi"] = DPI_SIZE
# mpl.use("pdf")

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8

colors_cb_friendly = {
    "naive_baseline": "#b00096",  # soft green
    "naive_gpe": "#00BFC4",  # deep green
    "naive_str": "#08306B",  # teal-green
    "dd_baseline": "#F03B20",  # mustard yellow
    "dd_gpe": "#FC8D62",  # soft red
    "dd_str": "#67000D",  # burgundy
}

model_color = "#165b05"


def get_exp_spikes(exp_data):
    """
    Reads in dataframe for experimental data and returns corresponding spike trains.

    Parameters:
    exp_data (dataframe): Dataframe of experimental data.

    Returns:
    list: List of lists containing all spike trains.
    """

    # List to contain all spike trains
    all_spikes = []
    # iterate through dataframe and record all spike trains
    for x in exp_data["src"]:
        with open(f"{x}/spikes.txt") as file:
            lines = file.readlines()
        header = [line.strip() for line in lines if line.startswith("#")]
        data_lines = [line for line in lines if not line.startswith("#")]
        if data_lines:
            spikes = np.loadtxt(f"{x}/spikes.txt", comments="#")
            light_on = np.loadtxt(f"{x}/light_on.txt")
            all_spikes.append(
                spikes[(spikes >= light_on[0] - 2) & (spikes < light_on[0])]
                - (light_on[0] - 2)
            )
        else:
            all_spikes.append([])
    return all_spikes


def slice_seed_data(sim_path):
    """
    Reads in simulation data from the specified path and calculates firing rates and coefficients of variation (CV) for each neuron.

    Parameters:
    sim_path (str): Path to the simulation data directory.

    Returns:
    tuple: Two numpy arrays containing firing rates and CVs for each neuron.
    """
    # Create empty array to hold data
    rates = np.zeros(100)
    cvs = np.zeros(100)
    # Loop through all neurons in sim_path
    for i in range(100):
        if os.path.getsize(f"{sim_path}/Neuron_{i}/spike_times.txt") > 0:
            spikes = np.loadtxt(f"{sim_path}/Neuron_{i}/spike_times.txt")
            spikes = spikes[(spikes >= 3) & (spikes <= 5)]  # only keep final 2 seconds
            rates[i] = len(spikes) / 2
            cvs[i] = (
                np.std(np.diff(spikes)) / np.mean(np.diff(spikes))
                if len(spikes) > 1
                else 0
            )
        else:
            rates[i] = 0
            cvs[i] = 0
    return rates, cvs


# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"

# get and process experimental data for comparison
exp_data = expan.get_in_vivo_baseline_data_short(segment=2, DD=False)
all_exp_spikes = get_exp_spikes(exp_data)
exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]
exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(exp_data)
exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

# get and process model data
save_dir = "data/default_network/vivo"
save_dir = "data/param_search_vivo_from_slice/slice_seed_105/sim_0353"
size = 100
T0 = 3
T = 5
sim_spikes, sim_df = pmr.gather_sim_data(save_dir, T, T0, size=size)
sim_data = pd.DataFrame(
    {"pre_exp_freq": sim_df["pre_exp_freq"], "pre_exp_cv": sim_df["pre_exp_cv"]}
)
sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(sim_data)
sim_fr_low_outliers, sim_cv_low_outliers = low_outliers[0], low_outliers[1]
sim_fr_high_outliers, sim_cv_high_outliers = high_outliers[0], high_outliers[1]

# set up FR and CV histogram bins
bw_fr = 10
bw_cv = 0.2
bins_fr = np.arange(
    0,
    np.max([np.max(exp_data["pre_exp_freq"]), np.max(sim_data["pre_exp_freq"])])
    + bw_fr,
    bw_fr,
)
bins_cv = np.arange(
    0,
    np.max([np.max(exp_data["pre_exp_cv"]), np.max(sim_data["pre_exp_cv"])]) + bw_cv,
    bw_cv,
)

# open figure to store stats
stats_file = open(f"{FIG_DIR}/in_vivo_figure_stats.txt", "w")

# set up figure
# fig, ax = plt.subplots(3, 3, figsize=(8, 6), dpi=600, tight_layout=True)
# fig, ax = plt.subplots(2, 3, figsize=(8, 4), dpi=600, tight_layout=True)
# axes = [ax[i, j] for i in range(2) for j in range(3)]

fig = plt.figure(figsize=(7, 4), dpi=DPI_SIZE, tight_layout=True)
print(fig.get_dpi())
gs = GridSpec(3, 3, figure=fig)
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1:]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    fig.add_subplot(gs[2, 2]),
]

N_vivo_sims = 3
all_seeds = []
slice_results = pd.read_csv(f"data/param_seed_finder_slice_correlated/results.csv")

slice_results = slice_results[
    (slice_results["KS_FR_pval"] >= 0.05)
    & (slice_results["Ttest_FR_pval"] >= 0.05)
    & (slice_results["Ttest_CV_pval"] >= 0.05)
]
avg_err_sorted = slice_results.sort_values(by="Avg. Error")
slice_seeds = [int(row["Seed"]) for index, row in avg_err_sorted.iterrows()][:1]
for slice_seed in slice_seeds:
    data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
    results = pd.read_csv(f"{data_dir}/results.csv")
    """results = results[
        (results["KS_FR_pval"] >= 0.05)
        & (results["Ttest_FR_pval"] >= 0.05)
        & (results["Ttest_CV_pval"] >= 0.05)
        & (results["KS_CV_pval"] >= 0.05)
    ]
    sorted_results = results.sort_values(by="Avg. Error")
    sorted_results = sorted_results.iloc[:N_vivo_sims]"""
    all_seeds.append(results)

all_sorted_results = pd.concat(all_seeds, ignore_index=True)

filtered_results = all_sorted_results[
    (all_sorted_results["KS_FR_pval"] >= 0.05)
    & (all_sorted_results["Ttest_FR_pval"] >= 0.05)
    & (all_sorted_results["Ttest_CV_pval"] >= 0.05)
    & (all_sorted_results["KS_CV_pval"] >= 0.05)
]


sns.scatterplot(
    data=all_sorted_results,
    x="Rel. Error FR",
    y="Rel. Error CV",
    color="gray",
    ax=axes[0],
    s=10,
    legend=False,
    alpha=0.25,
)

sns.scatterplot(
    data=filtered_results,
    x="Rel. Error FR",
    y="Rel. Error CV",
    color=model_color,
    ax=axes[0],
    s=10,
    legend=False,
)

sns.scatterplot(
    data=filtered_results[filtered_results["Sim"] == 353],
    x="Rel. Error FR",
    y="Rel. Error CV",
    color=model_color,
    ax=axes[0],
    marker="*",
    s=50,
    legend=False,
)


####################################################################################
####### RESUBMISSION ADDITIONS ####################
# get and process experimental data for comparison
exp_data = expan.get_in_vivo_baseline_data_short(segment=2, DD=False)
all_exp_spikes = get_exp_spikes(exp_data)
exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]
exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(exp_data)
exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

# get and process model data
save_dir = "data/default_network/vivo"
save_dir = "data/param_search_vivo_from_slice/slice_seed_105/sim_0353"
size = 100
T0 = 3
T = 5
sim_spikes, sim_df = pmr.gather_sim_data(save_dir, T, T0, size=size)
sim_data = pd.DataFrame(
    {"pre_exp_freq": sim_df["pre_exp_freq"], "pre_exp_cv": sim_df["pre_exp_cv"]}
)
sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(sim_data)
sim_fr_low_outliers, sim_cv_low_outliers = low_outliers[0], low_outliers[1]
sim_fr_high_outliers, sim_cv_high_outliers = high_outliers[0], high_outliers[1]

sim_rates, sim_cvs = slice_seed_data(f"data/default_network/vivo")
sim_data = pd.DataFrame({"pre_exp_freq": sim_rates, "pre_exp_cv": sim_cvs})
sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(sim_data)
example_spikes = [
    (
        np.loadtxt(f"data/default_network/vivo/Neuron_{i}/spike_times.txt")
        if os.path.getsize(f"data/default_network/vivo/Neuron_{i}/spike_times.txt") > 0
        else np.array([])
    )
    for i in range(100)
]
example_spikes = [x[x >= 3] - 1 for x in example_spikes]
example_frs = np.asarray([len(x) / 2 for x in example_spikes])
sorted_frs_inds = example_frs.argsort()
example_spikes_sorted = [
    example_spikes[x] for x in sorted_frs_inds if example_frs[x] > 0
]

exp_slice_spikes = [all_exp_spikes[i] for i in exp_data.index]


exp_slice_frs = np.asarray([len(x) / 2 for x in exp_slice_spikes])
bins = np.arange(0, np.max(exp_data["pre_exp_freq"].values) + 5, 5)
heights, _ = np.histogram(
    exp_data["pre_exp_freq"].values,
    bins=bins,
)
heights = heights / len(exp_data["pre_exp_freq"].values)
heights = np.rint(heights * 100).astype(int)
exp_sample_spikes = []
for bi in range(len(bins) - 1):
    fr_indices = list(
        np.where((exp_slice_frs >= bins[bi]) & (exp_slice_frs < bins[bi + 1]))[0]
    )
    exp_sample_spikes.extend(random.sample(fr_indices, heights[bi]))

exp_raster = [exp_slice_spikes[x] for x in exp_sample_spikes]
samp_exp_raster = random.sample(exp_raster, k=len(example_spikes_sorted))
frs_exp_raster = np.asarray([len(x) / 2 for x in samp_exp_raster])
samp_exp_raster = [samp_exp_raster[x] for x in frs_exp_raster.argsort()]

axes[1].eventplot(example_spikes_sorted, color=model_color, linelengths=1)
axes[1].eventplot(samp_exp_raster, colors="gray", linelengths=1)


random.seed(10)
samp_neurons = random.sample(
    sim_data[
        (sim_data["pre_exp_freq"] > 10) & (sim_data["pre_exp_freq"] < 30)
    ].index.tolist(),
    15,
)

samp_neurons = [73, 51, 98, 12, 60, 80, 40, 4]
count = 0
for i in samp_neurons:
    t, vs, vd = np.loadtxt(
        f"data/default_network/vivo/Neuron_{i}/cell_dynamics.txt",
        usecols=(0, 1, 2),
        unpack=True,
    )
    vs = vs[t >= 3]
    vd = vd[t >= 3]
    t = t[t >= 3]
    axes[2].plot(
        t,
        (vs - np.max(vs)) / (np.max(vs) - np.min(vs)) + count * 1.5,
        color=model_color,
        lw=0.5,
    )
    count += 1

for i in [1]:
    axes[i].set_yticks([0, 50, 100])
    axes[i].set_xticks([0, 1, 2, 3, 4])
    axes[i].set_xticklabels([0, 1, 2, 3, 4])
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Neuron ID")

for i in [2]:
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Normalized $V_S$")
    axes[i].set_xticks([3, 4, 5])
    axes[i].set_xticklabels([0, 1, 2])
    axes[i].set_yticks([])


####################################################################################


# plot connections histogram
adjacency, connections, total_weight = pmr.find_connections(100, f"{save_dir}")

# plot connections histogram
bins = np.arange(0, np.max(connections) + 1)
sns.histplot(
    connections,
    kde=True,
    stat="proportion",
    bins=bins,
    color=model_color,
    edgecolor="w",
    ax=axes[3],
)

# plot connections stats
"""ylims = axes[3].get_ylim()
axes[3].vlines(
    np.mean(connections),
    ylims[0],
    ylims[1],
    linestyle="dashed",
    color=model_color,
)
axes[3].vlines(
    np.median(connections),
    ylims[0],
    ylims[1],
    linestyle="dotted",
    color=model_color,
)"""

print(np.mean(connections), np.median(connections))

# update axes labels
axes[3].set_xticks(bins + 0.5)
axes[3].set_xticklabels(bins.astype(int))
axes[3].set_xlabel("# SNr Connections", fontsize=8)
axes[3].set_ylabel("Proportion", fontsize=8)
axes[3].set_xlim([bins[0], bins[-1]])

# report connection stats
stats_file.write("### PANEL B ###\n")
stats_file.write("# SNr Connections, Model:\n")
stats_file.write(f"\tMean: {np.mean(connections):.3f}\n")
stats_file.write(f"\tMedian: {np.median(connections):.3f}\n\n")

# perform and plot linear regression on experimental and example model
pmr.plot_linear_fit(
    sim_data["pre_exp_freq"],
    sim_data["pre_exp_cv"],
    axes[4],
    color=model_color,
)
pmr.plot_linear_fit(
    exp_data["pre_exp_freq"], exp_data["pre_exp_cv"], axes[4], color="black"
)


# report linear regression values
slope, intercept, r, p, std_err = stats.linregress(
    sim_data["pre_exp_freq"], sim_data["pre_exp_cv"]
)

stats_file.write("### PANEL C ###\n")
stats_file.write("Model Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")

slope, intercept, r, p, std_err = stats.linregress(
    exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]
)
stats_file.write("Experimental Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")


# compare experimental and simulated FR
sns.histplot(
    sim_data["pre_exp_freq"].values,
    bins=bins_fr,
    ax=axes[5],
    kde=True,
    color=model_color,
    edgecolor="w",
    stat="probability",
    label=f"Model ($n={len(sim_data)}$)",
)

sns.histplot(
    exp_data["pre_exp_freq"].values,
    bins=bins_fr,
    ax=axes[5],
    kde=True,
    color="gray",
    edgecolor="w",
    stat="probability",
    label=f"Exp ($n={len(exp_data)}$)",
)

axes[5].legend(fancybox=False, frameon=False, fontsize=6)

# perform statistical tests and plot on histograms
ylims = axes[5].get_ylim()
ks_pval = ks_2samp(
    sim_data["pre_exp_freq"].values, exp_data["pre_exp_freq"].values
).pvalue
pmr.plot_bracket(
    axes[5],
    np.percentile(exp_data["pre_exp_freq"].values, 25),
    np.percentile(exp_data["pre_exp_freq"].values, 75),
    ylims[0] + 1.15 * (ylims[1] - ylims[0]),
    ylims[0] + 1.18 * (ylims[1] - ylims[0]),
    f"KS $p=${ks_pval:.3f}",
)

ttest_pval = ttest_ind(
    sim_data["pre_exp_freq"].values, exp_data["pre_exp_freq"].values
).pvalue
pmr.plot_bracket(
    axes[5],
    np.mean(exp_data["pre_exp_freq"].values),
    np.mean(sim_data["pre_exp_freq"].values),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

# Print statistical tests
stats_file.write("### PANEL D ###\n")
stats_file.write("FR Comparison, in-vivo Model vs Experimental:\n")
stats_file.write(
    f"\tFR, Model: {np.mean(sim_data["pre_exp_freq"].values):.3f} +/- {np.std(sim_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(
    f"\tFR, Exp: {np.mean(exp_data["pre_exp_freq"].values):.3f} +/- {np.std(exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

axes[5].vlines(
    np.mean(sim_data["pre_exp_freq"].values),
    0,
    ylims[1],
    color=model_color,
    ls="dashed",
)
axes[5].vlines(
    np.mean(exp_data["pre_exp_freq"].values), 0, ylims[1], color="k", ls="dashed"
)

# plot model simulated and experimental CV
sns.histplot(
    sim_data["pre_exp_cv"].values,
    bins=bins_cv,
    ax=axes[6],
    kde=True,
    color=model_color,
    edgecolor="w",
    stat="probability",
    label=f"Model ($n={len(sim_data)}$)",
)

sns.histplot(
    exp_data["pre_exp_cv"].values,
    bins=bins_cv,
    ax=axes[6],
    kde=True,
    color="gray",
    edgecolor="w",
    stat="probability",
    label=f"Exp ($n={len(exp_data)}$)",
)

# axes[4].legend(fancybox=False, frameon=False, fontsize=6)

# perform statistical tests and plot on figure
ylims = axes[6].get_ylim()
ks_pval = ks_2samp(sim_data["pre_exp_cv"].values, exp_data["pre_exp_cv"].values).pvalue
pmr.plot_bracket(
    axes[6],
    np.percentile(exp_data["pre_exp_cv"].values, 25),
    np.percentile(exp_data["pre_exp_cv"].values, 75),
    ylims[0] + 1.15 * (ylims[1] - ylims[0]),
    ylims[0] + 1.18 * (ylims[1] - ylims[0]),
    f"KS $p=${ks_pval:.3f}",
)

ttest_pval = ttest_ind(
    sim_data["pre_exp_cv"].values, exp_data["pre_exp_cv"].values
).pvalue
pmr.plot_bracket(
    axes[6],
    np.mean(exp_data["pre_exp_cv"].values),
    np.mean(sim_data["pre_exp_cv"].values),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

# Print statistical tests
stats_file.write("### PANEL E ###\n")
stats_file.write("CV Comparison, in-vivo Model vs Experimental:\n")
stats_file.write(
    f"\tCV, Model: {np.mean(sim_data["pre_exp_cv"].values):.3f} +/- {np.std(sim_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(
    f"\tCV, Exp: {np.mean(exp_data["pre_exp_cv"].values):.3f} +/- {np.std(exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")

stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

axes[6].vlines(
    np.mean(sim_data["pre_exp_cv"].values),
    0,
    ylims[1],
    color=model_color,
    ls="dashed",
)
axes[6].vlines(
    np.mean(exp_data["pre_exp_cv"].values), 0, ylims[1], color="k", ls="dashed"
)

# find EGABA values from steady state
v = -55
egaba_avg = np.zeros(size)
meta_data = pmr.get_neuron_meta_data(save_dir, size=size)
for i in range(size):
    egaba_avg[i] = run_model.chloride_to_egaba(
        optimize.newton(
            run_model.chloride_dynamics_ss,
            10,
            args=(
                meta_data["gKCC2_S_nS_pF"][i],
                meta_data["gTON_CL_S_MEAN_nS_pF"][i],
                v,
            ),
        )
    )

# print EGABA_S values
stats_file.write("### PANEL F ###\n")
stats_file.write("EGABA_S Model Statistics:\n")
stats_file.write(f"\tMean: {np.mean(egaba_avg):.3f}\n")
stats_file.write(f"\tMin: {np.min(egaba_avg):.3f}\n")
stats_file.write(f"\tMax: {np.max(egaba_avg):.3f}\n\n")

# Create grid for EGABA heatmap
N = 500
gtonvals = np.linspace(1e-5, np.max(meta_data["gTON_CL_S_MEAN_nS_pF"]) * 1.1, N)
gkcc2vals = np.linspace(1e-5, np.max(meta_data["gKCC2_S_nS_pF"]) * 1.1, N)
egabas = np.zeros((N, N))

# Fill EGABA heatmap
i = 0
for gkcc2 in gkcc2vals:
    j = 0
    for gtonic in gtonvals:
        root = optimize.newton(
            run_model.chloride_dynamics_ss, 10, args=(gkcc2, gtonic, v)
        )
        egabas[i, j] = run_model.chloride_to_egaba(root)
        j += 1
    i += 1

# set up labels for EGABA heatmap
idx = np.round(np.linspace(0, N - 1, 5)).astype(int)
labels = [f"{gkcc2vals[x]:.03f}" for x in idx]
gtonlabels = [f"{gtonvals[x]:.03f}" for x in idx]

# plot EGABA heatmap
hm2 = axes[7].imshow(egabas, cmap="gnuplot", aspect="auto")
cs = axes[7].contour(
    egabas, levels=[-90, -85, -80, -75, -70, -65, -60, -55, -50, -45], colors="w"
)

cs_labels = axes[7].clabel(cs, inline=1, fontsize=8, inline_spacing=50)
for label in cs_labels:
    label.set_rotation(0)
cbar = fig.colorbar(hm2, ax=axes[7], location="right", fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

# plot GKCC2 and gTONIC values on heatmap
sns.scatterplot(
    y=meta_data["gKCC2_S_nS_pF"] / (gkcc2vals[-1] - gkcc2vals[0]) * N,
    x=meta_data["gTON_CL_S_MEAN_nS_pF"] / (gtonvals[-1] - gtonvals[0]) * N,
    ax=axes[7],
    color=model_color,
    legend=False,
    size=1,
)

# set up labels for gkcc2 and gtonic
idx = np.round(np.linspace(0, N - 1, 5)).astype(int)
labels = [f"{gkcc2vals[x]:.02e}" for x in idx]
gtonlabels = [f"{gtonvals[x]:.02e}" for x in idx]
axes[7].set_xlabel("$g_{tonic}^S$ (nS/pF)")
axes[7].set_ylabel("$g_{KCC2}^S$ (nS/pF)")
axes[7].set_xticks(idx)
axes[7].set_yticks(idx)
axes[7].set_yticklabels(labels, rotation=0)
axes[7].set_xticklabels(gtonlabels, rotation=0)
axes[7].invert_yaxis()


axes[5].set_xlabel("Rel. Error FR")
axes[5].set_ylabel("Rel. Error CV")

for i in [5, 6]:
    axes[i].set_xlabel("Firing Rate (Hz)" if i == 3 else "CV")
    axes[i].set_ylabel("Probability")

axes[4].set_xlabel("Firing Rate (Hz)")
axes[4].set_ylabel("CV")

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
    f"{FIG_DIR}/in_vivo_figure.pdf",
    bbox_inches="tight",
    dpi=DPI_SIZE,
)
plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/in_vivo_figure.pdf")
