"""
slice_figure.py

Creates figure for slice data analysis, comparing experimental and simulation results for publication.

Author: John E. Parker
"""

# import python modules
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import networkx as nx
from scipy import stats
import sys
import os
from matplotlib.gridspec import GridSpec

# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8

# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"


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


# Grab experimental data
exp_data = expan.get_slice_baseline_data()
slice_spikes = expan.get_slice_baseline_spikes_short()

# redefine variables
exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]

# find experimental outliers
exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(exp_data)
exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

# remove outliers from experimental spikes
exp_slice_spikes = [slice_spikes[i] for i in exp_data.index]

# read in slice directory and results
slice_data = "data/param_seed_finder_slice_correlated"
slice_results = pd.read_csv(f"{slice_data}/results.csv")

# find sims that matched characteristics
sims_to_keep = slice_results[
    (slice_results["KS_FR_pval"] >= 0.05)
    & (slice_results["Ttest_FR_pval"] >= 0.05)
    & (slice_results["Ttest_CV_pval"] >= 0.05)
]

best_seed = sims_to_keep.sort_values(by="Avg. Error")["Seed"].values[0]
print(slice_data, best_seed)

# Instantiate arrays and loop variables
rel_error = np.zeros((len(slice_results), 2))  # 0: FR, 1: CV
all_rates = []
all_cvs = []
count = 0

# bin widths
bw_fr = 10
bw_cv = 0.2

# loop through and calcuate rates, cvs, and relative erros
for sim in slice_results["Seed"]:
    sim_rates, sim_cvs = slice_seed_data(f"{slice_data}/sim_{sim:04d}")
    sim_data = pd.DataFrame({"pre_exp_freq": sim_rates, "pre_exp_cv": sim_cvs})
    sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        sim_data
    )
    all_rates.extend(sim_data["pre_exp_freq"])
    all_cvs.extend(sim_data["pre_exp_cv"])
    rel_error[count, 0] = na.percent_error_fr(
        sim_data["pre_exp_freq"], exp_data["pre_exp_freq"], bin_width=bw_fr
    )
    rel_error[count, 1] = na.percent_error_fr(
        sim_data["pre_exp_cv"], exp_data["pre_exp_cv"], bin_width=bw_cv
    )
    count += 1


# open figure to store stats
stats_file = open(f"{FIG_DIR}/slice_figure_stats.txt", "w")

model_color = "#b00096"
model_color = "#165b05"

# Set up figure
# fig, ax = plt.subplots(2, 3, figsize=(8, 4), dpi=300, tight_layout=True)
# axes = [ax[i, j] for i in range(2) for j in range(3)]

fig = plt.figure(figsize=(8, 4), dpi=300, tight_layout=True)
gs = GridSpec(2, 6, figure=fig)
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1:4]),
    fig.add_subplot(gs[0, 4:]),
]
remaining_sub_plots = [fig.add_subplot(gs[1, j : j + 2]) for j in range(0, 6, 2)]
axes.extend(remaining_sub_plots)

# relative error of FR and CV, color coded by matching simulations

sns.scatterplot(
    data=slice_results,
    x="Rel. Error FR",
    y="Rel. Error CV",
    color="gray",
    ax=axes[0],
    size=5,
    legend=False,
    alpha=0.25,
)

sns.scatterplot(
    data=sims_to_keep,
    x="Rel. Error FR",
    y="Rel. Error CV",
    color=model_color,
    ax=axes[0],
    size=5,
    legend=False,
)

sns.scatterplot(
    data=sims_to_keep[sims_to_keep["Seed"] == best_seed],
    x="Rel. Error FR",
    y="Rel. Error CV",
    color=model_color,
    ax=axes[0],
    # size=500,
    legend=False,
    marker="*",
    s=50,
)
axes[0].set_xlabel("Firing Rate Relative Error")
axes[0].set_ylabel("CV Relative Error")

stats_file.write("### PANEL A ###\n")
stats_file.write(
    f"\tDisplay Seed Rel. Error FR: {sims_to_keep[sims_to_keep["Seed"] == best_seed]["Rel. Error FR"]}"
)
stats_file.write(
    f"\tDisplay Seed Rel. Error CV: {sims_to_keep[sims_to_keep["Seed"] == best_seed]["Rel. Error CV"]}"
)

# grab key example data
sim_example = best_seed
print(best_seed)
sys.exit()
sim_rates, sim_cvs = slice_seed_data(f"{slice_data}/sim_{sim_example:04d}")
sim_data = pd.DataFrame({"pre_exp_freq": sim_rates, "pre_exp_cv": sim_cvs})
sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(sim_data)
example_spikes = [
    (
        np.loadtxt(f"{slice_data}/sim_{sim_example:04d}/Neuron_{i}/spike_times.txt")
        if os.path.getsize(
            f"{slice_data}/sim_{sim_example:04d}/Neuron_{i}/spike_times.txt"
        )
        > 0
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

# raster plot of experimental data
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


axes[1].eventplot(
    samp_exp_raster,
    colors="gray",
    linelengths=1,
)


# plot model example raster
axes[1].eventplot(
    example_spikes_sorted,
    colors=model_color,
    linelengths=1,
)

# grab handful of neurons and plot normalized traces from model
random.seed(10)
samp_neurons = random.sample(
    sim_data[
        (sim_data["pre_exp_freq"] > 10) & (sim_data["pre_exp_freq"] < 30)
    ].index.tolist(),
    15,
)
# print(samp_neurons)
# samp_neurons = [37, 48, 73, 91, 69]
samp_neurons = [92, 23, 31, 37, 48, 73, 91, 63]

count = 0
for i in samp_neurons:
    t, vs, vd = np.loadtxt(
        f"data/default_network/slice/Neuron_{i}/cell_dynamics.txt",
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

# define bins for histograms
bins_fr = np.arange(
    0,
    np.max(
        [
            np.max(exp_data["pre_exp_freq"].values),
            np.max(sim_data["pre_exp_freq"].values),
        ]
    )
    + bw_fr,
    bw_fr,
)
bins_cv = np.arange(
    0,
    np.max(
        [np.max(exp_data["pre_exp_cv"].values), np.max(sim_data["pre_exp_cv"].values)]
    )
    + bw_cv,
    bw_cv,
)


# compare experimental and simulated FR
sns.histplot(
    sim_data["pre_exp_freq"].values,
    bins=bins_fr,
    ax=axes[3],
    kde=True,
    color=model_color,
    edgecolor="w",
    stat="probability",
    label=f"Model ($n={len(sim_data)}$)",
)

sns.histplot(
    exp_data["pre_exp_freq"].values,
    bins=bins_fr,
    ax=axes[3],
    kde=True,
    color="gray",
    edgecolor="w",
    stat="probability",
    label=f"Exp ($n={len(exp_data)}$)",
)

axes[3].legend(fancybox=False, frameon=False, fontsize=6)

# perfrom statistical tests and plot on histograms
ylims = axes[3].get_ylim()
ks_pval = ks_2samp(
    sim_data["pre_exp_freq"].values, exp_data["pre_exp_freq"].values
).pvalue
pmr.plot_bracket(
    axes[3],
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
    axes[3],
    np.mean(exp_data["pre_exp_freq"].values),
    np.mean(sim_data["pre_exp_freq"].values),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

# Print statistical tests
stats_file.write("### PANEL D ###\n")
stats_file.write("FR Comparison, Slice Model vs Experimental:\n")
stats_file.write(
    f"\tFR, Model: {np.mean(sim_data["pre_exp_freq"].values):.3f} +/- {np.std(sim_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(
    f"\tFR, Exp: {np.mean(exp_data["pre_exp_freq"].values):.3f} +/- {np.std(exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")

stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

axes[3].vlines(
    np.mean(sim_data["pre_exp_freq"].values),
    0,
    ylims[1],
    color=model_color,
    ls="dashed",
)
axes[3].vlines(
    np.mean(exp_data["pre_exp_freq"].values), 0, ylims[1], color="k", ls="dashed"
)

# plot model simulated and experimental CV
sns.histplot(
    sim_data["pre_exp_cv"].values,
    bins=bins_cv,
    ax=axes[4],
    kde=False,
    color=model_color,
    edgecolor="w",
    stat="probability",
    label=f"Model ($n={len(sim_data)}$)",
)

sns.histplot(
    exp_data["pre_exp_cv"].values,
    bins=bins_cv,
    ax=axes[4],
    kde=False,
    color="gray",
    edgecolor="w",
    stat="probability",
    label=f"Exp ($n={len(exp_data)}$)",
)

# axes[4].legend(fancybox=False, frameon=False, fontsize=6)

# perform statistical tests and plot on figure
ylims = axes[4].get_ylim()
ks_pval = ks_2samp(sim_data["pre_exp_cv"].values, exp_data["pre_exp_cv"].values).pvalue
pmr.plot_bracket(
    axes[4],
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
    axes[4],
    np.mean(exp_data["pre_exp_cv"].values),
    np.mean(sim_data["pre_exp_cv"].values),
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${ttest_pval:.3f}",
)

# Print statistical tests
stats_file.write("### PANEL E ###\n")
stats_file.write("CV Comparison, Slice Model vs Experimental:\n")
stats_file.write(
    f"\tCV, Model: {np.mean(sim_data["pre_exp_cv"].values):.3f} +/- {np.std(sim_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(
    f"\tCV, Exp: {np.mean(exp_data["pre_exp_cv"].values):.3f} +/- {np.std(exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")

stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

axes[4].vlines(
    np.mean(sim_data["pre_exp_cv"].values), 0, ylims[1], color=model_color, ls="dashed"
)
axes[4].vlines(
    np.mean(exp_data["pre_exp_cv"].values), 0, ylims[1], color="k", ls="dashed"
)

# perform and plot linear regression on experimental and example model
pmr.plot_linear_fit(
    sim_data["pre_exp_freq"], sim_data["pre_exp_cv"], axes[5], color=model_color
)
pmr.plot_linear_fit(
    exp_data["pre_exp_freq"], exp_data["pre_exp_cv"], axes[5], color="black"
)

slope, intercept, r, p, std_err = stats.linregress(
    sim_data["pre_exp_freq"], sim_data["pre_exp_cv"]
)
stats_file.write("### PANEL F ###\n")
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

# Clean up figures and label appropriately
for i in [1]:
    axes[i].set_yticks([0, 50, 100])
    axes[i].set_xticks([0, 1, 2, 3, 4])
    axes[i].set_xticklabels([0, 1, 2, 3, 4])
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Neuron ID")

for i in [3, 4]:
    axes[i].set_xlabel("Firing Rate (Hz)" if i == 3 else "CV")
    axes[i].set_ylabel("Probability")

axes[5].set_xlabel("Firing Rate (Hz)")
axes[5].set_ylabel("CV")

for i in [2]:
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Normalized $V_S$")
    axes[i].set_xticks([3, 4, 5])
    axes[i].set_xticklabels([0, 1, 2])
    axes[i].set_yticks([])


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
    f"{FIG_DIR}/slice_figure.pdf",
    bbox_inches="tight",
)
plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/slice_figure.pdf")
