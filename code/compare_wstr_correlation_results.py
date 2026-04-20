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
plt.rcParams["axes.labelsize"] = 6
model_color = "#165b05"


def run_chi_sq_contingency(data):
    # Find columns where sum is NOT zero
    cols_to_keep = np.where(data.sum(axis=0) != 0)[0]

    # Keep only those columns
    matrix_clean = data[:, cols_to_keep]
    return (
        chi2_contingency(matrix_clean).pvalue,
        chi2_contingency(matrix_clean).statistic,
    )


FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"

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


src_dir = "data/param_search_vivo_from_slice"
dirs_to_check = [
    f"data/example_sims_Wstr_anticorrelations/{src_dir}",
    f"data/example_sims_Wstr_correlations/{src_dir}",
    f"data/example_sims_Wstr_random/{src_dir}",
]
titles = ["Anti-correlated", "Correlated", "Random", "Paired Result"]

size = 100
T1 = 5
T2 = 35

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

_, gpe_pulse_exp, _, d1_pulse_exp = expan.get_classification_results()
exp_gpe_mfs = expan.get_modulation_factors(gpe_pulse_exp)
exp_str_mfs = expan.get_modulation_factors(d1_pulse_exp)

mfs = []
for dirs in dirs_to_check:
    mf_dir = []
    if dirs != ".DS_Store":
        sim_count = 0
        all_responses = np.zeros(len(types) - 2)
        for subdir in os.listdir(f"{dirs}"):
            if subdir != ".DS_Store":
                for sim in os.listdir(f"{dirs}/{subdir}"):
                    if sim != ".DS_Store":
                        mf_dir.append(
                            pmr.generate_modulation_factors(
                                f"{dirs}/{subdir}/{sim}/sim_0001",
                                params,
                                T1,
                                T2,
                                False,
                                True,
                                size=size,
                            )[:, 1]
                        )
    mfs.append(mf_dir)

fig, ax = plt.subplots(2, 3, figsize=(6, 3), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(3)]

plot_exp_str_mfs = exp_str_mfs.flatten()

for j in range(3):

    axes[j].plot(
        np.arange(len(plot_exp_str_mfs)) / len(plot_exp_str_mfs),
        sorted(plot_exp_str_mfs),
        color="gray",
        label="sim",
    )

    model_mfs = mf_dir[j].flatten()
    axes[j].plot(
        np.arange(len(model_mfs)) / len(model_mfs),
        sorted(model_mfs),
        color=model_color,
        label="exp",
    )

    print(f"Wstr {titles[j]}", np.mean(model_mfs), np.std(model_mfs))
    print(f"Wstr exp", np.mean(plot_exp_str_mfs), np.std(plot_exp_str_mfs))

    axes[j].set_ylim([-1, 1])
    xlims = axes[j].get_xlim()
    axes[j].hlines(0, xlims[0], xlims[1], color="k", lw=0.5, ls="--")
    axes[j].set_xlim(xlims)
    axes[j].set_xlabel("Normalized Unit", fontsize=8)
    axes[j].set_ylabel("Modulation\nFactor", fontsize=8)

for j in range(3):
    sns.histplot(
        exp_str_mfs,
        bins=np.arange(-1, 1, 0.1),
        stat="probability",
        kde=True,
        color="gray",
        ax=axes[j + 3],
    )
    sns.histplot(
        mf_dir[j],
        bins=np.arange(-1, 1, 0.1),
        stat="probability",
        kde=True,
        ax=axes[j + 3],
        color=model_color,
    )

    # run and plot statistical tests
    ks_pval = ks_2samp(mf_dir[j].flatten(), exp_str_mfs)[1]
    ylims = axes[j + 3].get_ylim()
    pmr.plot_bracket(
        axes[j + 3],
        np.percentile(exp_str_mfs, 25),
        np.percentile(exp_str_mfs, 75),
        ylims[0] + 1.19 * (ylims[1] - ylims[0]),
        ylims[0] + 1.21 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = ttest_ind(mf_dir[j].flatten(), exp_str_mfs)[1]
    pmr.plot_bracket(
        axes[j + 3],
        # np.mean(exp_str_mfs),
        # np.mean(mf_dir[j].flatten()),
        -0.3,
        -0.1,
        # ylims[1],
        ylims[0] + 0.98 * (ylims[1] - ylims[0]),
        ylims[0] + 1.0 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )
    axes[j].set_title(titles[j], fontsize=8)
    axes[j + 3].set_xlim([-1, 1])
    axes[j + 3].set_xlabel("Modulation Factor", fontsize=8)
    axes[j + 3].set_ylabel("Probability", fontsize=8)


for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)

add_fig_labels(axes)

fig.savefig(f"{FIG_DIR}/wstr_correlation_comparisons.pdf", bbox_inches="tight")
plt.close()
# stats_file.close()

run_cmd(f"open {FIG_DIR}/wstr_correlation_comparisons.pdf")
sys.exit()


#### classifrication #######
_, gpe_pulse_exp, _, d1_pulse_exp = expan.get_classification_results()
d1_pulse_exp = d1_pulse_exp.replace(
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

# open figure to store stats
stats_file = open(f"{FIG_DIR}/wstr_correlation_comparisons_stats.txt", "w")


exp_results = np.asarray(
    [np.sum(d1_pulse_exp["neural_response"] == x) for x in types[:-2]]
)
exp_results = exp_results * 100 / np.sum(exp_results)

all_case_responses = np.zeros((12, len(types) - 2))
all_case_responses[0, :] = exp_results
num_significant = np.zeros(3)

all_case_chi2_stats = np.zeros((10, 3))  # networks x cases

positions = [1, 3, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
case_count = 0
for dirs in dirs_to_check:
    if dirs != ".DS_Store":
        stats_file.write(f"### {titles[case_count]} ###\n")

        sim_count = 0
        all_responses = np.zeros(len(types) - 2)
        for subdir in os.listdir(f"{dirs}"):
            if subdir != ".DS_Store":
                for sim in os.listdir(f"{dirs}/{subdir}"):
                    if sim != ".DS_Store":
                        sim_results = np.zeros(len(types) - 2)
                        df = pd.read_csv(
                            f"{dirs}/{subdir}/{sim}/sim_0001/processed/all_data.csv"
                        )
                        df.loc[
                            df["neural_response"] == "biphasic IE", "neural_response"
                        ] = "partial inhibition"
                        df.loc[
                            df["neural_response"] == "biphasic EI", "neural_response"
                        ] = "excitation"

                        for i, resp in enumerate(types[:-2]):
                            sim_results[i] = (df["neural_response"] == resp).sum()
                            all_responses[i] += (df["neural_response"] == resp).sum()
                            axes[case_count + 6].bar(
                                positions[sim_count + 2],
                                sim_results[i],
                                bottom=np.sum(sim_results[:i]),
                                color=color_dict[resp],
                                edgecolor="w",
                                linewidth=0.25,
                                width=1,
                            )
                all_case_responses[sim_count + 2, :] = sim_results
                sim_count += 1
        all_case_responses[1, :] = all_responses
        all_case_chi2_stats[:, case_count] = [
            run_chi_sq_contingency(all_case_responses[[0, i], :])[1]
            for i in range(2, all_case_responses.shape[0])
        ]
        str_stat_sig = [
            1 if run_chi_sq_contingency(all_case_responses[[0, i], :])[0] < 0.05 else 0
            for i in range(2, all_case_responses.shape[0])
        ]
        num_significant[case_count] = np.sum(str_stat_sig)

        stats_file.write(
            f"\tSTR All Sims: {run_chi_sq_contingency(all_case_responses[[0, 1], :])[0]}\n"
        )
        stats_file.write(
            f"\tSTR Individual Sig: {np.sum(str_stat_sig)}/{all_case_responses.shape[0]-2}\n\n"
        )

        for i, resp in enumerate(types[:-2]):
            axes[case_count + 6].bar(
                positions[1],
                all_responses[i] * 100 / np.sum(all_responses),
                bottom=np.sum(all_responses[:i]) * 100 / (np.sum(all_responses)),
                color=color_dict[resp],
                edgecolor="w",
                linewidth=0.25,
                width=2,
            )
            axes[case_count + 6].bar(
                positions[0],
                exp_results[i] * 100 / np.sum(exp_results),
                bottom=np.sum(exp_results[:i]) * 100 / np.sum(exp_results),
                color=color_dict[resp],
                edgecolor="w",
                linewidth=0.25,
                width=2,
            )
            axes[case_count + 6].set_xticks([1, 3])
            axes[case_count + 6].set_xticklabels(
                [
                    "Exp",
                    f"All\nSims{ "*" if run_chi_sq_contingency(all_case_responses[[0, 1], :])[0] < 0.05 else ""}",
                ]
            )

        case_count += 1


"""all_case_chi2_stats[:, [1, 2]] = all_case_chi2_stats[:, [2, 1]]


axes[3].plot(all_case_chi2_stats.T, "--", color="gray", lw=0.5)
axes[3].scatter(
    np.zeros(10), all_case_chi2_stats[:, 0], color="red", s=10, zorder=10, alpha=0.5
)
axes[3].scatter(
    np.ones(10), all_case_chi2_stats[:, 1], color="orange", s=10, zorder=10, alpha=0.5
)
axes[3].scatter(
    np.ones(10) * 2,
    all_case_chi2_stats[:, 2],
    color="green",
    s=10,
    zorder=10,
    alpha=0.5,
)

yerrs = np.zeros(3)
resamples = 10000
confidence_level = 0.95
bootstrap_labels = ["Anticorrelated", "Random", "Correlated"]


stats_file.write("\n### Chi^2 Bootstrap Results ###\n")
for i in range(3):
    d = all_case_chi2_stats[:, i]
    bs = stats.bootstrap(
        (d,), np.mean, n_resamples=resamples, confidence_level=confidence_level
    )
    yerrs[i] = bs.confidence_interval.high - np.mean(d)
    stats_file.write(
        f"\t{bootstrap_labels[i]}: \n\t\tMean: {np.mean(d)}\n\t\tCI:({bs.confidence_interval.low}, {bs.confidence_interval.high})\n\t\tResamples: {resamples}\n"
    )


axes[3].errorbar(
    0,
    np.mean(all_case_chi2_stats[:, 0]),
    yerr=yerrs[i],
    marker="_",
    color="k",
    zorder=15,
    capsize=2,
)
axes[3].errorbar(
    1,
    np.mean(all_case_chi2_stats[:, 1]),
    yerr=yerrs[i],
    marker="_",
    color="k",
    zorder=15,
    capsize=2,
)
axes[3].errorbar(
    2,
    np.mean(all_case_chi2_stats[:, 2]),
    yerr=yerrs[i],
    marker="_",
    color="k",
    zorder=15,
    capsize=2,
)

axes[3].spines["top"].set_visible(False)
axes[3].spines["right"].set_visible(False)
axes[3].spines["left"].set_linewidth(0.5)
axes[3].spines["bottom"].set_linewidth(0.5)
axes[3].tick_params(width=0.5, labelsize=6)
axes[3].set_xticks([0, 1, 2])
axes[3].set_xticklabels(bootstrap_labels)
axes[3].set_ylabel("$\\chi^2$ statistics (lower $=$ better)", fontsize=6)"""

for i, plot in enumerate(axes[6:]):
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
    plot.set_ylabel("Percentage", fontsize=8)
    # plot.set_title(titles[i], fontsize=8)
    plot.set_ylim([0, 101])
    # plot.set_xticks([])
    # plot.set_xlabel("Example Sims", fontsize=6)
    plot.annotate(
        f"Example Sims\n",
        xy=(0.63, 0),
        xytext=(0.63, -0.4),
        xycoords="axes fraction",
        fontsize=6,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-[, widthB=7.6, lengthB=0.5", lw=0.5, color="k"),
    )

add_fig_labels(axes)

fig.savefig(f"{FIG_DIR}/wstr_correlation_comparisons.pdf", bbox_inches="tight")
plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/wstr_correlation_comparisons.pdf")


sys.exit()


fig, ax = plt.subplots(2, 2, figsize=(5, 4), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(2)]

_, gpe_pulse_exp, _, d1_pulse_exp = expan.get_classification_results()
d1_pulse_exp = d1_pulse_exp.replace(
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

# open figure to store stats
stats_file = open(f"{FIG_DIR}/wstr_correlation_comparisons_stats.txt", "w")


exp_results = np.asarray(
    [np.sum(d1_pulse_exp["neural_response"] == x) for x in types[:-2]]
)
exp_results = exp_results * 100 / np.sum(exp_results)

all_case_responses = np.zeros((12, len(types) - 2))
all_case_responses[0, :] = exp_results
num_significant = np.zeros(3)

all_case_chi2_stats = np.zeros((10, 3))  # networks x cases

positions = [1, 3, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
case_count = 0
for dirs in dirs_to_check:
    if dirs != ".DS_Store":
        stats_file.write(f"### {titles[case_count]} ###\n")

        sim_count = 0
        all_responses = np.zeros(len(types) - 2)
        for subdir in os.listdir(f"{dirs}"):
            if subdir != ".DS_Store":
                for sim in os.listdir(f"{dirs}/{subdir}"):
                    if sim != ".DS_Store":
                        sim_results = np.zeros(len(types) - 2)
                        df = pd.read_csv(
                            f"{dirs}/{subdir}/{sim}/sim_0001/processed/all_data.csv"
                        )
                        df.loc[
                            df["neural_response"] == "biphasic IE", "neural_response"
                        ] = "partial inhibition"
                        df.loc[
                            df["neural_response"] == "biphasic EI", "neural_response"
                        ] = "excitation"

                        for i, resp in enumerate(types[:-2]):
                            sim_results[i] = (df["neural_response"] == resp).sum()
                            all_responses[i] += (df["neural_response"] == resp).sum()
                            axes[case_count].bar(
                                positions[sim_count + 2],
                                sim_results[i],
                                bottom=np.sum(sim_results[:i]),
                                color=color_dict[resp],
                                edgecolor="w",
                                linewidth=0.25,
                                width=0.8,
                            )
                all_case_responses[sim_count + 2, :] = sim_results
                sim_count += 1
        all_case_responses[1, :] = all_responses
        all_case_chi2_stats[:, case_count] = [
            run_chi_sq_contingency(all_case_responses[[0, i], :])[1]
            for i in range(2, all_case_responses.shape[0])
        ]
        str_stat_sig = [
            1 if run_chi_sq_contingency(all_case_responses[[0, i], :])[0] < 0.05 else 0
            for i in range(2, all_case_responses.shape[0])
        ]
        num_significant[case_count] = np.sum(str_stat_sig)

        stats_file.write(
            f"\tSTR All Sims: {run_chi_sq_contingency(all_case_responses[[0, 1], :])[0]}\n"
        )
        stats_file.write(
            f"\tSTR Individual Sig: {np.sum(str_stat_sig)}/{all_case_responses.shape[0]-2}\n\n"
        )

        for i, resp in enumerate(types[:-2]):
            axes[case_count].bar(
                positions[1],
                all_responses[i] * 100 / np.sum(all_responses),
                bottom=np.sum(all_responses[:i]) * 100 / (np.sum(all_responses)),
                color=color_dict[resp],
                edgecolor="w",
                linewidth=0.25,
                width=1.8,
            )
            axes[case_count].bar(
                positions[0],
                exp_results[i] * 100 / np.sum(exp_results),
                bottom=np.sum(exp_results[:i]) * 100 / np.sum(exp_results),
                color=color_dict[resp],
                edgecolor="w",
                linewidth=0.25,
                width=1.8,
            )
            axes[case_count].set_xticks([1, 3])
            axes[case_count].set_xticklabels(
                [
                    "Exp",
                    f"All\nSims{ "*" if run_chi_sq_contingency(all_case_responses[[0, 1], :])[0] < 0.05 else ""}",
                ]
            )

        case_count += 1


all_case_chi2_stats[:, [1, 2]] = all_case_chi2_stats[:, [2, 1]]


axes[3].plot(all_case_chi2_stats.T, "--", color="gray", lw=0.5)
axes[3].scatter(
    np.zeros(10), all_case_chi2_stats[:, 0], color="red", s=10, zorder=10, alpha=0.5
)
axes[3].scatter(
    np.ones(10), all_case_chi2_stats[:, 1], color="orange", s=10, zorder=10, alpha=0.5
)
axes[3].scatter(
    np.ones(10) * 2,
    all_case_chi2_stats[:, 2],
    color="green",
    s=10,
    zorder=10,
    alpha=0.5,
)

yerrs = np.zeros(3)
resamples = 10000
confidence_level = 0.95
bootstrap_labels = ["Anticorrelated", "Random", "Correlated"]


stats_file.write("\n### Chi^2 Bootstrap Results ###\n")
for i in range(3):
    d = all_case_chi2_stats[:, i]
    bs = stats.bootstrap(
        (d,), np.mean, n_resamples=resamples, confidence_level=confidence_level
    )
    yerrs[i] = bs.confidence_interval.high - np.mean(d)
    stats_file.write(
        f"\t{bootstrap_labels[i]}: \n\t\tMean: {np.mean(d)}\n\t\tCI:({bs.confidence_interval.low}, {bs.confidence_interval.high})\n\t\tResamples: {resamples}\n"
    )


axes[3].errorbar(
    0,
    np.mean(all_case_chi2_stats[:, 0]),
    yerr=yerrs[i],
    marker="_",
    color="k",
    zorder=15,
    capsize=2,
)
axes[3].errorbar(
    1,
    np.mean(all_case_chi2_stats[:, 1]),
    yerr=yerrs[i],
    marker="_",
    color="k",
    zorder=15,
    capsize=2,
)
axes[3].errorbar(
    2,
    np.mean(all_case_chi2_stats[:, 2]),
    yerr=yerrs[i],
    marker="_",
    color="k",
    zorder=15,
    capsize=2,
)

axes[3].spines["top"].set_visible(False)
axes[3].spines["right"].set_visible(False)
axes[3].spines["left"].set_linewidth(0.5)
axes[3].spines["bottom"].set_linewidth(0.5)
axes[3].tick_params(width=0.5, labelsize=6)
axes[3].set_xticks([0, 1, 2])
axes[3].set_xticklabels(bootstrap_labels)
axes[3].set_ylabel("$\\chi^2$ statistics (lower $=$ better)", fontsize=6)

for i, plot in enumerate(axes[:3]):
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
    plot.set_ylabel("Percentage", fontsize=8)
    plot.set_title(titles[i], fontsize=8)
    plot.set_ylim([0, 101])
    # plot.set_xticks([])
    # plot.set_xlabel("Example Sims", fontsize=6)
    plot.annotate(
        f"Example Sims\n(p<0.05: {int(num_significant[i])}/10)",
        xy=(0.63, 0),
        xytext=(0.63, -0.2),
        xycoords="axes fraction",
        fontsize=6,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-[, widthB=7.1, lengthB=0.4", lw=0.5, color="k"),
    )

add_fig_labels(axes)

fig.savefig(f"{FIG_DIR}/wstr_correlation_comparisons.pdf", bbox_inches="tight")
plt.close()
stats_file.close()


# print(all_case_chi2_stats)

run_cmd(f"open {FIG_DIR}/wstr_correlation_comparisons.pdf")
