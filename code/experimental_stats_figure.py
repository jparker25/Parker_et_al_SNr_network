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


def plot_compare_dists(
    x,
    y,
    ax,
    bw,
    data="pre_exp_freq",
    color1="gray",
    color2="green",
    label1="",
    label2="",
    legend=True,
):

    bins = np.arange(
        0,
        np.max(
            [
                np.max(x[data].values),
                np.max(y[data].values),
            ]
        )
        + bw,
        bw,
    )

    # compare experimental and simulated FR
    sns.histplot(
        x[data].values,
        bins=bins,
        ax=ax,
        kde=True,
        color=color1,
        edgecolor="w",
        stat="probability",
        label=label1,
    )

    sns.histplot(
        y[data].values,
        bins=bins,
        ax=ax,
        kde=True,
        color=color2,
        edgecolor="w",
        stat="probability",
        label=label2,
    )

    if legend:
        ax.legend(fancybox=False, frameon=False, fontsize=6)

    # perfrom statistical tests and plot on histograms
    ylims = ax.get_ylim()
    ks_pval = ks_2samp(x[data].values, y[data].values).pvalue
    pmr.plot_bracket(
        ax,
        np.percentile(x[data], 25),
        np.percentile(x[data], 75),
        ylims[0] + 1.15 * (ylims[1] - ylims[0]),
        ylims[0] + 1.18 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = ttest_ind(x[data].values, y[data].values).pvalue
    pmr.plot_bracket(
        ax,
        np.mean(x[data].values),
        np.mean(y[data].values),
        ylims[1],
        ylims[0] + 1.03 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )

    ax.vlines(np.mean(x[data].values), 0, ylims[1], color=color1, ls="dashed")
    ax.vlines(np.mean(y[data].values), 0, ylims[1], color=color2, ls="dashed")

    return ks_pval, ttest_pval


# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8

# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"


# naive slice data
slice_exp_data = expan.get_slice_baseline_data()
print(len(slice_exp_data))
slice_exp_data, slice_high_outliers, slice_low_outliers = (
    expan.find_and_remove_outliers_as_df(slice_exp_data)
)
print(len(slice_exp_data))
# sys.exit()

# naive in-vivo data
naive_exp_data = expan.get_in_vivo_baseline_data_short(segment=2, DD=False)
naive_exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
    naive_exp_data
)

# DD in-vivo data
dd_exp_data = expan.get_in_vivo_baseline_data_short(segment=2, DD=True)
dd_exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
    dd_exp_data
)


# open figure to store stats
stats_file = open(f"{FIG_DIR}/supplemental_experiemental_stats.txt", "w")

fig, ax = plt.subplots(2, 3, figsize=(8, 4), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(3)]


ks_pval, ttest_pval = plot_compare_dists(
    slice_exp_data,
    naive_exp_data,
    axes[0],
    10,
    data="pre_exp_freq",
    color1="gray",
    color2="green",
    label1=f"Slice ($n={len(slice_exp_data)}$)",
    label2="In vivo" + f" ($n={len(naive_exp_data)}$)",
)

# Print statistical tests
stats_file.write("### PANEL A ###\n")
stats_file.write("FR Comparison, Slice vs in-vivo Experimental:\n")
stats_file.write(
    f"\tFR, Slice: {np.mean(slice_exp_data["pre_exp_freq"].values):.3f} +/- {np.std(slice_exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(
    f"\tFR, in-vivo: {np.mean(naive_exp_data["pre_exp_freq"].values):.3f} +/- {np.std(naive_exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")


ks_pval, ttest_pval = plot_compare_dists(
    slice_exp_data,
    naive_exp_data,
    axes[1],
    0.2,
    data="pre_exp_cv",
    color1="gray",
    color2="green",
    label1=f"Slice ($n={len(slice_exp_data)}$)",
    label2=f"Naive ($n={len(naive_exp_data)}$)",
    legend=False,
)

# Print statistical tests
stats_file.write("### PANEL B ###\n")
stats_file.write("CV Comparison, Slice vs in-vivo Experimental:\n")
stats_file.write(
    f"\tCV, Slice: {np.mean(slice_exp_data["pre_exp_cv"].values):.3f} +/- {np.std(slice_exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(
    f"\tCV, in-vivo: {np.mean(naive_exp_data["pre_exp_cv"].values):.3f} +/- {np.std(naive_exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

pmr.plot_linear_fit(
    slice_exp_data["pre_exp_freq"], slice_exp_data["pre_exp_cv"], axes[2], color="gray"
)

# perform and plot linear regression on experimental and example model
pmr.plot_linear_fit(
    naive_exp_data["pre_exp_freq"], naive_exp_data["pre_exp_cv"], axes[2], color="green"
)

slope, intercept, r, p, std_err = stats.linregress(
    slice_exp_data["pre_exp_freq"], slice_exp_data["pre_exp_cv"]
)
stats_file.write("### PANEL C ###\n")
stats_file.write("Slice Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")

slope, intercept, r, p, std_err = stats.linregress(
    naive_exp_data["pre_exp_freq"], naive_exp_data["pre_exp_cv"]
)
stats_file.write("in-vivo Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")


ks_pval, ttest_pval = plot_compare_dists(
    naive_exp_data,
    dd_exp_data,
    axes[3],
    10,
    data="pre_exp_freq",
    color1="gray",
    color2="green",
    label1="Naive " + f"($n={len(naive_exp_data)}$)",
    label2=f"DD ($n={len(dd_exp_data)}$)",
)

# Print statistical tests
stats_file.write("### PANEL D ###\n")
stats_file.write("FR Comparison, naive vs dd Experimental:\n")
stats_file.write(
    f"\tFR, naive: {np.mean(naive_exp_data["pre_exp_freq"].values):.3f} +/- {np.std(naive_exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(
    f"\tFR, dd: {np.mean(dd_exp_data["pre_exp_freq"].values):.3f} +/- {np.std(dd_exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

ks_pval, ttest_pval = plot_compare_dists(
    naive_exp_data,
    dd_exp_data,
    axes[4],
    0.2,
    data="pre_exp_cv",
    color1="gray",
    color2="green",
    label1=f" ($n={len(naive_exp_data)}$)",
    label2=f"DD ($n={len(dd_exp_data)}$)",
    legend=False,
)

# Print statistical tests
stats_file.write("### PANEL E ###\n")
stats_file.write("CV Comparison, naive vs dd Experimental:\n")
stats_file.write(
    f"\tCV, naive: {np.mean(naive_exp_data["pre_exp_cv"].values):.3f} +/- {np.std(naive_exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(
    f"\tCV, dd: {np.mean(dd_exp_data["pre_exp_cv"].values):.3f} +/- {np.std(dd_exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")
stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")


pmr.plot_linear_fit(
    naive_exp_data["pre_exp_freq"], naive_exp_data["pre_exp_cv"], axes[5], color="gray"
)

# perform and plot linear regression on experimental and example model
pmr.plot_linear_fit(
    dd_exp_data["pre_exp_freq"], dd_exp_data["pre_exp_cv"], axes[5], color="green"
)

slope, intercept, r, p, std_err = stats.linregress(
    naive_exp_data["pre_exp_freq"], naive_exp_data["pre_exp_cv"]
)
stats_file.write("### PANEL F ###\n")
stats_file.write("naive Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")

slope, intercept, r, p, std_err = stats.linregress(
    dd_exp_data["pre_exp_freq"], dd_exp_data["pre_exp_cv"]
)
stats_file.write("dd Linear Regression:\n")
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


axes[3].set_xlabel("Firing Rate (Hz)")
axes[3].set_ylabel("Probability")

axes[4].set_xlabel("CV")
axes[4].set_ylabel("Probability")

axes[5].set_xlabel("Firing Rate (Hz)")
axes[5].set_ylabel("CV")


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
    f"{FIG_DIR}/supplementary_experimental.pdf",
    bbox_inches="tight",
)
plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/supplementary_experimental.pdf")
