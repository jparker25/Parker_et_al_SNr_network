"""
responses_violin_figure.py

Creates figure analyzing STReaC respones based on model SNr network paramters.

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
from scipy import optimize
from scipy import stats
import os


# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import run_model
from distribution import Distribution as Dist


def sig_lvl(p_t, test):
    if p_t < 0.001:
        return "***" if "MWU" in test else "+++"
    elif p_t < 0.01:
        return "**" if "MWU" in test else "++"
    else:
        return "*" if "MWU" in test else "+"


def gather_sim_meta_data(stim_dir, gpestim):
    sim_meta_data = pd.read_csv(f"{stim_dir}/processed/all_data.csv")
    sim_meta_data = remap_responses(sim_meta_data)
    sim_meta_data = sim_meta_data.sort_values(by="cell_num")

    _, sim_meta_data["connections"], sim_meta_data["total_weight"] = (
        pmr.find_connections(size, stim_dir)
    )

    mfs, mfs_inputs = modulation_factors_input_change(T1, 1, 1, 1, T1, T2, stim_dir)

    sim_meta_data["input_change"] = mfs_inputs
    sim_meta_data["mfs"] = mfs

    sim_meta_data["input_change"] = mfs_inputs
    sim_meta_data["mfs"] = mfs

    v = -55
    egaba_avg = np.zeros(size)
    for i in range(size):
        egaba_avg[i] = run_model.chloride_to_egaba(
            optimize.newton(
                run_model.chloride_dynamics_ss,
                10,
                args=(
                    sim_meta_data.iloc[
                        i,
                        sim_meta_data.columns.get_loc(
                            "gKCC2_S_nS_pF" if gpestim else "gKCC2_D_nS_pF"
                        ),
                    ],
                    sim_meta_data.iloc[
                        i,
                        sim_meta_data.columns.get_loc(
                            "gTON_CL_S_MEAN_nS_pF"
                            if gpestim
                            else "gTON_CL_D_MEAN_nS_pF"
                        ),
                    ],
                    v,
                ),
            )
        )
    sim_meta_data["EGABA_S" if gpestim else "EGABA_D"] = egaba_avg
    return sim_meta_data


def compare_distributions(x, y, alpha=0.05):
    x_d = Dist(x)
    y_d = Dist(y)

    # remove outliers
    if x_d.values_norm_dist and y_d.values_norm_dist:
        x_d.remove_outliers(x, method="zscore")
        y_d.remove_outliers(y, method="zscore")
    else:
        x_d.remove_outliers(x, method="iqr")
        y_d.remove_outliers(y, method="iqr")

    # one or both not normal ,run MannWhitneyU
    if not x_d.isNormal(x_d.fliers_removed) or not y_d.isNormal(y_d.fliers_removed):
        _, p = stats.mannwhitneyu(x_d.fliers_removed, y_d.fliers_removed)
        return (
            p,
            "MWU Test",
            ks_2samp(x_d.fliers_removed, y_d.fliers_removed).pvalue,
            x_d.fliers_removed,
            y_d.fliers_removed,
        )
    else:
        # check for variance for Welch t-test or not
        equal_var = stats.levene(x_d.fliers_removed, y_d.fliers_removed).pvalue > 0.05
        _, p = stats.ttest_ind(
            x_d.fliers_removed, y_d.fliers_removed, equal_var=equal_var
        )
        return (
            p,
            "t-Test",
            ks_2samp(x_d.fliers_removed, y_d.fliers_removed).pvalue,
            x_d.fliers_removed,
            y_d.fliers_removed,
        )


def remap_responses(df):
    df.loc[df.neural_response == "biphasic IE", "neural_response"] = "inhibition"
    df.loc[df.neural_response == "biphasic EI", "neural_response"] = "excitation"
    df.loc[df.neural_response.str.contains("inhibition"), "neural_response"] = (
        "inhibition"
    )
    return df


def modulation_factors_input_change(
    start_time,
    base_length,
    stim_length,
    post_length,
    T1,
    T2,
    save_dir,
    size=100,
):
    stim_times = np.arange(
        start_time + base_length, T2, base_length + stim_length + post_length
    )
    modulation_factors = np.zeros(size)
    input_mf = np.zeros(size)
    for i in range(size):
        if os.path.getsize(f"{save_dir}/Neuron_{i}/spike_times.txt") > 0:
            spikes = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt")
            base = len(spikes[(spikes >= T1) & (spikes < stim_times[0])]) / (
                stim_times[0] - T1
            )
            for trial in range(len(stim_times)):
                pre_stim = (
                    len(
                        spikes[
                            (spikes >= stim_times[trial] - stim_length)
                            & (spikes < stim_times[trial])
                        ]
                    )
                    / base_length
                )
                stim = (
                    len(
                        spikes[
                            (spikes >= stim_times[trial])
                            & (spikes < stim_times[trial] + stim_length)
                        ]
                    )
                    / base_length
                )

                modulation_factors[i] += [
                    (
                        ((stim - pre_stim) / (stim + pre_stim)) / len(stim_times)
                        if (stim + pre_stim) > 0
                        else 0
                    )
                ]
        else:
            modulation_factors[i] = 0

    for neuron_cell in range(size):
        connections = np.loadtxt(f"{save_dir}/weights.txt")
        connections = connections[:, neuron_cell]
        connect_ids = [i for i in range(size) if connections[i] > 0]
        input_mf[neuron_cell] = (
            np.mean(modulation_factors[connect_ids])
            if np.abs(np.mean(modulation_factors[connect_ids])) > 0
            else 0
        )
    return modulation_factors, input_mf


# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8

# colors for responses (remapped)
color_dict = {
    "inhibition": "blue",
    "no effect": "slategrey",
    "excitation": "lightcoral",
}

types = [
    "no effect",
    "inhibition",
    "excitation",
]

# global variable for to save figures
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"

# figure labels
params_gpe = ["EGABA_S", "tausyn", "W_gpe", "total_weight", "input_change"]
params_gpe_labels = [
    "$E_{GABA}^S$",
    "$\\tau_{GPe}^{syn}$",
    "$W_{GPe}$",
    "$W_{SNr}^{Total}$",
    "$MF_{input}$",
]
params_str_labels = [
    "$E_{GABA}^D$",
    "$\\tau_{Str}^{syn}$",
    "$W_{Str}$",
    "$W_{SNr}^{Total}$",
    "$MF_{input}$",
]
params_str = ["EGABA_D", "tausyn_dend", "W_str", "total_weight", "input_change"]

# common network settings
seeds = 20
size = 100
T0 = 3
T1 = 5
T2 = 35

all_data = []

# open figure to store stats
stats_file = open(f"{FIG_DIR}/responses_figure_violin_stats.txt", "w")

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


d1stim = False
gpestim = True
all_case_data = None
sim_count = 0
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

    stim_dir = f"{data_dir}_naive_gpe_stim"
    for index, row in sorted_results.iterrows():
        for stim_seeds in np.arange(1, N_stim_seeds + 1):
            gpe_seed_dir = (
                f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/sim_{stim_seeds:04d}"
            )
            sim_meta_data = gather_sim_meta_data(gpe_seed_dir, True)
            all_case_data = (
                sim_meta_data.copy()
                if sim_count == 0
                else pd.concat([all_case_data, sim_meta_data])
            )
            sim_count += 1

all_case_data["Case"] = "Naive"
all_case_data["Stim"] = "GPe"
all_data.append(all_case_data)

d1stim = True
gpestim = False
all_case_data = None
sim_count = 0
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
            sim_meta_data = gather_sim_meta_data(str_seed_dir, False)
            all_case_data = (
                sim_meta_data.copy()
                if sim_count == 0
                else pd.concat([all_case_data, sim_meta_data])
            )
            sim_count += 1

all_case_data["Case"] = "Naive"
all_case_data["Stim"] = "STR"
all_data.append(all_case_data)

# DD GPe responses
d1stim = False
gpestim = True
all_case_data = None
sim_count = 0
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

    stim_dir = f"{data_dir}_dd_gpe_stim"
    for index, row in sorted_results.iterrows():
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
            for stim_seeds in np.arange(1, N_stim_seeds + 1):
                gpe_dd_stim_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/dd_sim_{int(dd_row["Sim"]):06d}/sim_{stim_seeds:04d}"
                sim_meta_data = gather_sim_meta_data(gpe_dd_stim_dir, True)
                all_case_data = (
                    sim_meta_data.copy()
                    if sim_count == 0
                    else pd.concat([all_case_data, sim_meta_data])
                )
                sim_count += 1

all_case_data["Case"] = "DD"
all_case_data["Stim"] = "GPe"
all_data.append(all_case_data)

# naive D1 responses
dd_dir = "data/blended_dopamine_depletion_stim_results/best"

d1stim = True
gpestim = False
all_case_data = None
sim_count = 0
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
            for stim_seeds in np.arange(1, N_stim_seeds + 1):
                str_dd_stim_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/dd_sim_{int(dd_row["Sim"]):06d}/sim_{stim_seeds:04d}"
                sim_meta_data = gather_sim_meta_data(str_dd_stim_dir, False)
                all_case_data = (
                    sim_meta_data.copy()
                    if sim_count == 0
                    else pd.concat([all_case_data, sim_meta_data])
                )
                sim_count += 1

all_case_data["Case"] = "DD"
all_case_data["Stim"] = "STR"
all_data.append(all_case_data)

all_data_df = pd.concat(all_data)

all_data_df.to_csv("/Users/johnparker/Desktop/neurons_response.csv")

responses = ["no effect", "inhibition", "excitation"]
color_palette = {
    "Naive no effect": "#a8a8a8",
    "DD no effect": "#4f5050",  # no effect
    "Naive inhibition": "#546cf4",
    "DD inhibition": "#122afa",  # inhibition
    "Naive excitation": "#fc58aa",
    "DD excitation": "#fa1186",  # excitation
}
hue_color_palette = {
    "no effect": ["#a8a8a8", "#4f5050"],
    "inhibition": ["#546cf4", "#0014c5"],
    # "excitation": ["#fa3a9a", "#ad0056"],
    "excitation": ["#FF7E70", "#F13D29"],
}
pos_mapping = {
    "Naive no effect": 0,
    "DD no effect": 1,  # no effect
    "Naive inhibition": 2,
    "DD inhibition": 3,  # inhibition
    "Naive excitation": 4,
    "DD excitation": 5,  # excitation
}

fig, ax = plt.subplots(5, 2, figsize=(6, 9), dpi=300)
axes = [ax[i, j] for j in range(2) for i in range(5)]
for axis_plot, gpe_p in enumerate(params_gpe):
    pos_count = 0
    ttest_vals_gpe = np.zeros(3)
    kstest_vals_gpe = np.zeros(3)
    ttest_vals_str = np.zeros(3)
    kstest_vals_str = np.zeros(3)
    test_type_gpe = []
    test_type_str = []
    for i, response in enumerate(["no effect", "inhibition", "excitation"]):
        sns.violinplot(
            data=all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "GPe")
            ],
            x=pos_count,
            y=gpe_p,
            hue="Case",
            hue_order=["Naive", "DD"],
            palette=hue_color_palette[response],
            edgecolor="w",
            ax=axes[axis_plot],
            split=True,
            inner=None,
            gap=0.35,
            fill=True,
            legend=False,
        )

        sns.boxplot(
            data=all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "GPe")
            ],
            x=pos_count,
            y=gpe_p,
            hue="Case",
            hue_order=["Naive", "DD"],
            palette=hue_color_palette[response],
            ax=axes[axis_plot],
            fill=False,
            legend=False,
            width=0.12,
            gap=0.12,
            linewidth=0.5,
            fliersize=3,
            flierprops={"markeredgewidth": 0.5, "marker": "."},
        )

        sns.violinplot(
            data=all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "STR")
            ],
            x=pos_count,
            y=params_str[axis_plot],
            hue="Case",
            hue_order=["Naive", "DD"],
            palette=hue_color_palette[response],
            edgecolor="w",
            ax=axes[axis_plot + 5],
            split=True,
            inner=None,
            gap=0.35,
            fill=True,
            legend=False,
        )
        sns.boxplot(
            data=all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "STR")
            ],
            x=pos_count,
            y=params_str[axis_plot],
            hue="Case",
            hue_order=["Naive", "DD"],
            palette=hue_color_palette[response],
            ax=axes[axis_plot + 5],
            fill=False,
            legend=False,
            width=0.12,
            gap=0.12,
            linewidth=0.5,
            fliersize=5,
            flierprops={"markeredgewidth": 0.5, "marker": "."},
        )

        stat_res_gpe = compare_distributions(
            all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "GPe")
                & (all_data_df["Case"] == "Naive")
            ][gpe_p],
            all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "GPe")
                & (all_data_df["Case"] == "DD")
            ][gpe_p],
        )

        ttest_vals_gpe[i] = stat_res_gpe[0]
        kstest_vals_gpe[i] = stat_res_gpe[2]
        test_type_gpe.append(stat_res_gpe[1])

        stat_res_str = compare_distributions(
            all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "STR")
                & (all_data_df["Case"] == "Naive")
            ][params_str[axis_plot]],
            all_data_df[
                (all_data_df["neural_response"] == response)
                & (all_data_df["Stim"] == "STR")
                & (all_data_df["Case"] == "DD")
            ][params_str[axis_plot]],
        )
        ttest_vals_str[i] = stat_res_str[0]
        kstest_vals_str[i] = stat_res_str[2]
        test_type_str.append(stat_res_str[1])

        # print(f"GPe {response}: {stat_res_gpe[:3]}")
        # print(f"STR {response}: {stat_res_str[:3]}")

        pos_count += 2
        axes[axis_plot].set_ylabel(params_gpe_labels[axis_plot])
        axes[axis_plot + 5].set_ylabel(params_str_labels[axis_plot])

        stats_file.write(f"GPe {response}, {params_gpe[axis_plot]}:\n")
        stats_file.write(f"\t{stat_res_gpe[1]}: {stat_res_gpe[0]:.3f}\n")
        stats_file.write(f"\tK-S Test {stat_res_gpe[2]:.3f}:\n")
        stats_file.write(f"\tNaive Mean: {np.mean(stat_res_gpe[3]):.3f}\n")
        stats_file.write(f"\tNaive Median: {np.median(stat_res_gpe[3]):.3f}\n")
        stats_file.write(f"\tNaive STD: {np.std(stat_res_gpe[3]):.3f}\n")
        stats_file.write(f"\tDD Mean: {np.mean(stat_res_gpe[4]):.3f}\n")
        stats_file.write(f"\tDD Median: {np.median(stat_res_gpe[4]):.3f}\n")
        stats_file.write(f"\tDD STD: {np.std(stat_res_gpe[4]):.3f}\n\n")

        stats_file.write(f"STR {response}, {params_str[axis_plot]}:\n")
        stats_file.write(f"\t{stat_res_str[1]}: {stat_res_str[0]:.3f}\n")
        stats_file.write(f"\tK-S Test {stat_res_str[2]:.3f}:\n")
        stats_file.write(f"\tNaive Mean: {np.mean(stat_res_str[3]):.3f}\n")
        stats_file.write(f"\tNaive Median: {np.median(stat_res_str[3]):.3f}\n")
        stats_file.write(f"\tNaive STD: {np.std(stat_res_str[3]):.3f}\n")
        stats_file.write(f"\tDD Mean: {np.mean(stat_res_str[4]):.3f}\n")
        stats_file.write(f"\tDD Median: {np.median(stat_res_str[4]):.2f}\n")
        stats_file.write(f"\tDD STD: {np.std(stat_res_str[4]):.3f}\n\n")

    axes[axis_plot].set_xticks([0, 1, 2])
    axes[axis_plot].set_xticklabels(
        [
            f"No Eff.{sig_lvl(ttest_vals_gpe[0],test_type_gpe[0])}",
            f"Inh{sig_lvl(ttest_vals_gpe[1],test_type_gpe[1])}",
            f"Exc{sig_lvl(ttest_vals_gpe[2],test_type_gpe[2])}",
        ]
    )

    axes[axis_plot + 5].set_xticks([0, 1, 2])
    axes[axis_plot + 5].set_xticklabels(
        [
            f"No Eff.{sig_lvl(ttest_vals_str[0],test_type_str[0])}",
            f"Inh{sig_lvl(ttest_vals_str[1],test_type_str[1])}",
            f"Exc{sig_lvl(ttest_vals_str[2],test_type_str[2])}",
        ]
    )


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
    f"{FIG_DIR}/responses_violin.pdf",
    bbox_inches="tight",
)
plt.close()
run_cmd(f"open {FIG_DIR}/responses_violin.pdf")
sys.exit()

# set up figure
fig, ax = plt.subplots(5, 2, figsize=(6, 9), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(5) for j in range(2)]
for i in range(len(params_gpe)):
    pos_count = 0
    for type in types:

        # naive GPe
        sns.violinplot(
            data=all_data[0][all_data[0]["neural_response"] == type],
            x=pos_count,
            y=params_gpe[i],
            ax=axes[2 * i],
            split=False,
            inner="box",
            color=color_dict[type],
            orient="v",
            fill=False,
        )

        # naive D1
        sns.violinplot(
            data=all_data[1][all_data[1]["neural_response"] == type],
            x=pos_count,
            y=params_str[i],
            ax=axes[2 * i + 1],
            split=False,
            inner="box",
            color=color_dict[type],
            orient="v",
            fill=False,
        )

        pos_count += 1

        # DD GPe
        sns.violinplot(
            data=all_data[2][all_data[2]["neural_response"] == type],
            x=pos_count,
            y=params_gpe[i],
            ax=axes[2 * i],
            split=False,
            inner="box",
            color=color_dict[type],
            orient="v",
            fill=False,
        )

        # DD D1
        sns.violinplot(
            data=all_data[3][all_data[3]["neural_response"] == type],
            x=pos_count,
            y=params_str[i],
            ax=axes[2 * i + 1],
            split=False,
            inner="box",
            color=color_dict[type],
            orient="v",
            fill=False,
        )
        xlims = axes[2 * i].get_xlim()
        ylims = axes[2 * i].get_ylim()
        if (
            len(all_data[0][all_data[0]["neural_response"] == type][params_gpe[i]]) > 3
            and len(all_data[2][all_data[2]["neural_response"] == type][params_gpe[i]])
            > 3
        ):
            stat_res = compare_distributions(
                all_data[0][all_data[0]["neural_response"] == type][params_gpe[i]],
                all_data[2][all_data[2]["neural_response"] == type][params_gpe[i]],
            )
            if stat_res[0] < 0.05:
                pmr.plot_bracket(
                    axes[2 * i],
                    pos_count,
                    pos_count - 1,
                    ylims[1],
                    ylims[1],
                    f"{'*' if stat_res[0] < 0.05 else ''}{'+' if stat_res[2] < 0.05 else ''}",
                )
            stats_file.write(f"GPe {type}, {params_gpe[i]}:\n")
            stats_file.write(f"\t{stat_res[1]}: {stat_res[0]:.3f}\n")
            stats_file.write(f"\tK-S Test {stat_res[2]:.3f}:\n")
            stats_file.write(f"\tNaive Mean: {np.mean(stat_res[3]):.3f}\n")
            stats_file.write(f"\tNaive Median: {np.median(stat_res[3]):.3f}\n")
            stats_file.write(f"\tNaive STD: {np.std(stat_res[3]):.3f}\n")
            stats_file.write(f"\tDD Mean: {np.mean(stat_res[4]):.3f}\n")
            stats_file.write(f"\tDD Median: {np.median(stat_res[4]):.3f}\n")
            stats_file.write(f"\tDD STD: {np.std(stat_res[4]):.3f}\n\n")

        xlims = axes[2 * i + 1].get_xlim()
        ylims = axes[2 * i + 1].get_ylim()
        stat_res = compare_distributions(
            all_data[1][all_data[1]["neural_response"] == type][params_str[i]],
            all_data[3][all_data[3]["neural_response"] == type][params_str[i]],
        )
        if stat_res[0] < 0.05:
            pmr.plot_bracket(
                axes[2 * i + 1],
                pos_count,
                pos_count - 1,
                ylims[1],
                ylims[1],
                f"{'*' if stat_res[0] < 0.05 else ''}{'+' if stat_res[2] < 0.05 else ''}",
            )
        stats_file.write(f"STR {type}, {params_str[i]}:\n")
        stats_file.write(f"\t{stat_res[1]}: {stat_res[0]:.3f}\n")
        stats_file.write(f"\tK-S Test {stat_res[2]:.3f}:\n")
        stats_file.write(f"\tNaive Mean: {np.mean(stat_res[3]):.3f}\n")
        stats_file.write(f"\tNaive Median: {np.median(stat_res[3]):.3f}\n")
        stats_file.write(f"\tNaive STD: {np.std(stat_res[3]):.3f}\n")
        stats_file.write(f"\tDD Mean: {np.mean(stat_res[4]):.3f}\n")
        stats_file.write(f"\tDD Median: {np.median(stat_res[4]):.2f}\n")
        stats_file.write(f"\tDD STD: {np.std(stat_res[4]):.3f}\n\n")
        pos_count += 1

for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
    plot.set_xticks(np.arange(0.5, 5.5, 2))
    plot.set_xticklabels(["No Eff.", "Inh", "Exc"])

for plot in range(0, len(axes), 2):
    axes[plot].set_ylabel(params_gpe_labels[plot // 2])
    axes[plot + 1].set_ylabel(params_str_labels[plot // 2])

# add labels and match axes
add_fig_labels(axes)

# save, close, and open pdf of figure
fig.savefig(
    f"{FIG_DIR}/responses_violin.pdf",
    bbox_inches="tight",
)
plt.close()
stats_file.close()

run_cmd(f"open {FIG_DIR}/responses_violin.pdf")
