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
from matplotlib.gridspec import GridSpec


# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import run_model
from distribution import Distribution as Dist


def norm_dists(dist1, dist2, method="zscore"):
    mu = np.mean(dist1)
    std = np.std(dist1)
    return (dist1 - mu) / std, (dist2 - mu) / std


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
    "$\\tau_{syn}$",
    "$W_{GPe}$",
    "$W_{SNr}^{Total}$",
    "$MF_{input}$",
]
params_str_labels = [
    "$E_{GABA}^D$",
    "$\\tau_{syn}^{dend}$",
    "$W_{STR}$",
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
run = False
read = True


if run:
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
                gpe_seed_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/sim_{stim_seeds:04d}"
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
                str_seed_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/sim_{stim_seeds:04d}"
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

if read:
    all_data_df = pd.read_csv("/Users/johnparker/Desktop/neurons_response.csv")


hue_color_palette = {
    "no effect": ["#a8a8a8", "#4f5050"],
    "inhibition": ["#546cf4", "#0014c5"],
    # "excitation": ["#fa3a9a", "#ad0056"],
    "excitation": ["#FF7E70", "#F13D29"],
}

outlier_removal = "iqr"
norm_method = "zscore"

str_results = np.zeros((5, 2, 2))  # param, naive/dd,inh/exc
gpe_results = np.zeros((5, 2, 2))


fig, ax = plt.subplots(5, 2, figsize=(6, 9), dpi=300, tight_layout=True)
axes = [ax[i, j] for j in range(2) for i in range(5)]
for axis_plot, gpe_p in enumerate(params_gpe):
    for resp in ["inhibition", "excitation"]:
        ne_dd_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "DD")
                & (all_data_df["Stim"] == "GPe")
                & (all_data_df["neural_response"] == "no effect")
            ][gpe_p]
        )
        ne_dd_dist.remove_outliers(ne_dd_dist.values, method=outlier_removal)
        ne_dd_dist = ne_dd_dist.fliers_removed

        resp_dd_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "DD")
                & (all_data_df["Stim"] == "GPe")
                & (all_data_df["neural_response"] == resp)
            ][gpe_p]
        )
        resp_dd_dist.remove_outliers(resp_dd_dist.values, method=outlier_removal)
        resp_dd_dist = resp_dd_dist.fliers_removed

        ne_naive_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "Naive")
                & (all_data_df["Stim"] == "GPe")
                & (all_data_df["neural_response"] == "no effect")
            ][gpe_p]
        )
        ne_naive_dist.remove_outliers(ne_naive_dist.values, method=outlier_removal)
        ne_naive_dist = ne_naive_dist.fliers_removed

        resp_naive_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "Naive")
                & (all_data_df["Stim"] == "GPe")
                & (all_data_df["neural_response"] == resp)
            ][gpe_p]
        )
        resp_naive_dist.remove_outliers(resp_naive_dist.values, method=outlier_removal)
        resp_naive_dist = resp_naive_dist.fliers_removed

        norm_resp_naive_dist, norm_resp_dd_dist = norm_dists(
            resp_naive_dist, resp_dd_dist
        )
        w_dd = stats.wasserstein_distance(norm_resp_naive_dist, norm_resp_dd_dist)

        norm_ne_naive_dist, norm_resp_naive_dist = norm_dists(
            ne_naive_dist,
            resp_naive_dist,
        )
        w_ne_naive = stats.wasserstein_distance(
            norm_ne_naive_dist, norm_resp_naive_dist
        )

        norm_ne_dd_dist, norm_resp_dd_dist = norm_dists(
            ne_dd_dist,
            resp_dd_dist,
        )
        w_ne_dd = stats.wasserstein_distance(norm_ne_dd_dist, norm_resp_dd_dist)

        if resp == "excitation":
            gpe_results[axis_plot, 0, 1] = w_ne_naive
            gpe_results[axis_plot, 1, 1] = w_ne_dd
        else:
            gpe_results[axis_plot, 0, 0] = w_ne_naive
            gpe_results[axis_plot, 1, 0] = w_ne_dd

        axes[axis_plot].scatter(w_ne_naive, w_dd, c=hue_color_palette[resp][0])
        axes[axis_plot].scatter(w_ne_dd, w_dd, c=hue_color_palette[resp][1], marker="s")

        norm_ne_naive_dist, norm_ne_dd_dist = norm_dists(ne_naive_dist, ne_dd_dist)
        w_ne_ne = stats.wasserstein_distance(norm_ne_naive_dist, norm_ne_dd_dist)

        if resp == "excitation":
            xlims = axes[axis_plot].get_xlim()
            axes[axis_plot].hlines(
                w_ne_ne, xlims[0], xlims[1], color="gray", ls="dashed"
            )
            axes[axis_plot].set_xlim(xlims)

        axes[axis_plot].set_ylabel("DD Effect")
        axes[axis_plot].set_xlabel("Modulatory Effect (Resp-to-NE)")
        axes[axis_plot].set_title(params_gpe_labels[axis_plot])

        ne_dd_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "DD")
                & (all_data_df["Stim"] == "STR")
                & (all_data_df["neural_response"] == "no effect")
            ][params_str[axis_plot]]
        )
        ne_dd_dist.remove_outliers(ne_dd_dist.values, method=outlier_removal)
        ne_dd_dist = ne_dd_dist.fliers_removed

        resp_dd_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "DD")
                & (all_data_df["Stim"] == "STR")
                & (all_data_df["neural_response"] == resp)
            ][params_str[axis_plot]]
        )
        resp_dd_dist.remove_outliers(resp_dd_dist.values, method=outlier_removal)
        resp_dd_dist = resp_dd_dist.fliers_removed

        ne_naive_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "Naive")
                & (all_data_df["Stim"] == "STR")
                & (all_data_df["neural_response"] == "no effect")
            ][params_str[axis_plot]]
        )
        ne_naive_dist.remove_outliers(ne_naive_dist.values, method=outlier_removal)
        ne_naive_dist = ne_naive_dist.fliers_removed

        resp_naive_dist = Dist(
            all_data_df[
                (all_data_df["Case"] == "Naive")
                & (all_data_df["Stim"] == "STR")
                & (all_data_df["neural_response"] == resp)
            ][params_str[axis_plot]]
        )
        resp_naive_dist.remove_outliers(resp_naive_dist.values, method=outlier_removal)
        resp_naive_dist = resp_naive_dist.fliers_removed

        norm_resp_naive_dist, norm_resp_dd_dist = norm_dists(
            resp_naive_dist, resp_dd_dist
        )
        w_dd = stats.wasserstein_distance(norm_resp_naive_dist, norm_resp_dd_dist)

        norm_ne_naive_dist, norm_resp_naive_dist = norm_dists(
            ne_naive_dist,
            resp_naive_dist,
        )
        w_ne_naive = stats.wasserstein_distance(
            norm_ne_naive_dist, norm_resp_naive_dist
        )

        norm_ne_dd_dist, norm_resp_dd_dist = norm_dists(
            ne_dd_dist,
            resp_dd_dist,
        )
        w_ne_dd = stats.wasserstein_distance(norm_ne_dd_dist, norm_resp_dd_dist)

        axes[axis_plot + 5].scatter(w_ne_naive, w_dd, c=hue_color_palette[resp][0])
        axes[axis_plot + 5].scatter(
            w_ne_dd, w_dd, c=hue_color_palette[resp][1], marker="s"
        )

        if resp == "excitation":
            str_results[axis_plot, 0, 1] = w_ne_naive
            str_results[axis_plot, 1, 1] = w_ne_dd
        else:
            str_results[axis_plot, 0, 0] = w_ne_naive
            str_results[axis_plot, 1, 0] = w_ne_dd

        norm_ne_naive_dist, norm_ne_dd_dist = norm_dists(ne_naive_dist, ne_dd_dist)
        w_ne_ne = stats.wasserstein_distance(norm_ne_naive_dist, norm_ne_dd_dist)
        if resp == "excitation":
            xlims = axes[axis_plot + 5].get_xlim()
            axes[axis_plot + 5].hlines(
                w_ne_ne, xlims[0], xlims[1], color="gray", ls="dashed"
            )
            axes[axis_plot + 5].set_xlim(xlims)

        axes[axis_plot + 5].set_ylabel("Norm. DD Effect")
        axes[axis_plot + 5].set_xlabel("Norm. Modulatory Effect (Resp-to-NE)")
        axes[axis_plot + 5].set_title(params_str_labels[axis_plot])
for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)

# add labels and match axes
add_fig_labels(axes)
# match_axis(axes)

# save, close, and open pdf of figure
fig.savefig(
    f"{FIG_DIR}/responses_WD.pdf",
    bbox_inches="tight",
)
plt.close()
run_cmd(f"open {FIG_DIR}/responses_WD.pdf")

stats_file = open(f"{FIG_DIR}/responses_figure_WD_stats.txt", "w")
### PANEL A ####
stats_file.write(f"PANEL A\n")
stats_file.write(f"GPe Naive Inh {params_gpe[0]} WD:\t{gpe_results[0,0,0]}\n\n")
stats_file.write(f"GPe Naive Inh {params_gpe[1]} WD:\t{gpe_results[1,0,0]}\n\n")
stats_file.write(f"GPe Naive Inh {params_gpe[2]} WD:\t{gpe_results[2,0,0]}\n\n")
stats_file.write(f"GPe Naive Inh {params_gpe[3]} WD:\t{gpe_results[3,0,0]}\n\n")
stats_file.write(f"GPe Naive Inh {params_gpe[4]} WD:\t{gpe_results[4,0,0]}\n\n")

stats_file.write(f"GPe DD Inh {params_gpe[0]} WD:\t{gpe_results[0,1,0]}\n\n")
stats_file.write(f"GPe DD Inh {params_gpe[1]} WD:\t{gpe_results[1,1,0]}\n\n")
stats_file.write(f"GPe DD Inh {params_gpe[2]} WD:\t{gpe_results[2,1,0]}\n\n")
stats_file.write(f"GPe DD Inh {params_gpe[3]} WD:\t{gpe_results[3,1,0]}\n\n")
stats_file.write(f"GPe DD Inh {params_gpe[4]} WD:\t{gpe_results[4,1,0]}\n\n")

stats_file.write(f"GPe Naive Exc {params_gpe[0]} WD:\t{gpe_results[0,0,1]}\n\n")
stats_file.write(f"GPe Naive Exc {params_gpe[1]} WD:\t{gpe_results[1,0,1]}\n\n")
stats_file.write(f"GPe Naive Exc {params_gpe[2]} WD:\t{gpe_results[2,0,1]}\n\n")
stats_file.write(f"GPe Naive Exc {params_gpe[3]} WD:\t{gpe_results[3,0,1]}\n\n")
stats_file.write(f"GPe Naive Exc {params_gpe[4]} WD:\t{gpe_results[4,0,1]}\n\n")

stats_file.write(f"GPe DD Exc {params_gpe[0]} WD:\t{gpe_results[0,1,1]}\n\n")
stats_file.write(f"GPe DD Exc {params_gpe[1]} WD:\t{gpe_results[1,1,1]}\n\n")
stats_file.write(f"GPe DD Exc {params_gpe[2]} WD:\t{gpe_results[2,1,1]}\n\n")
stats_file.write(f"GPe DD Exc {params_gpe[3]} WD:\t{gpe_results[3,1,1]}\n\n")
stats_file.write(f"GPe DD Exc {params_gpe[4]} WD:\t{gpe_results[4,1,1]}\n\n")

### PANEL B ####
stats_file.write(f"PANEL B\n")
stats_file.write(f"STR Naive Inh {params_str[0]} WD:\t{str_results[0,0,0]}\n\n")
stats_file.write(f"STR Naive Inh {params_str[1]} WD:\t{str_results[1,0,0]}\n\n")
stats_file.write(f"STR Naive Inh {params_str[2]} WD:\t{str_results[2,0,0]}\n\n")
stats_file.write(f"STR Naive Inh {params_str[3]} WD:\t{str_results[3,0,0]}\n\n")
stats_file.write(f"STR Naive Inh {params_str[4]} WD:\t{str_results[4,0,0]}\n\n")

stats_file.write(f"STR DD Inh {params_str[0]} WD:\t{str_results[0,1,0]}\n\n")
stats_file.write(f"STR DD Inh {params_str[1]} WD:\t{str_results[1,1,0]}\n\n")
stats_file.write(f"STR DD Inh {params_str[2]} WD:\t{str_results[2,1,0]}\n\n")
stats_file.write(f"STR DD Inh {params_str[3]} WD:\t{str_results[3,1,0]}\n\n")
stats_file.write(f"STR DD Inh {params_str[4]} WD:\t{str_results[4,1,0]}\n\n")

stats_file.write(f"STR Naive Exc {params_str[0]} WD:\t{str_results[0,0,1]}\n\n")
stats_file.write(f"STR Naive Exc {params_str[1]} WD:\t{str_results[1,0,1]}\n\n")
stats_file.write(f"STR Naive Exc {params_str[2]} WD:\t{str_results[2,0,1]}\n\n")
stats_file.write(f"STR Naive Exc {params_str[3]} WD:\t{str_results[3,0,1]}\n\n")
stats_file.write(f"STR Naive Exc {params_str[4]} WD:\t{str_results[4,0,1]}\n\n")

stats_file.write(f"STR DD Exc {params_str[0]} WD:\t{str_results[0,1,1]}\n\n")
stats_file.write(f"STR DD Exc {params_str[1]} WD:\t{str_results[1,1,1]}\n\n")
stats_file.write(f"STR DD Exc {params_str[2]} WD:\t{str_results[2,1,1]}\n\n")
stats_file.write(f"STR DD Exc {params_str[3]} WD:\t{str_results[3,1,1]}\n\n")
stats_file.write(f"STR DD Exc {params_str[4]} WD:\t{str_results[4,1,1]}\n\n")
stats_file.close()

fig, ax = plt.subplots(2, 1, figsize=(6, 4), dpi=300, tight_layout=True)
axes = [ax[i] for i in range(2)]
# results shape (5,2,2) # param, naive/dd,inh/exc
count = 0
for i in range(str_results.shape[0]):
    text_bar = axes[0].bar(
        count,
        gpe_results[i, 0, 0],
        color=hue_color_palette["inhibition"][0],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[0].text(
        count,
        gpe_results[i, 0, 0],
        f"{gpe_results[i, 0, 0]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[0].bar(
        count + 6,
        gpe_results[i, 1, 0],
        color=hue_color_palette["inhibition"][1],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[0].text(
        count + 6,
        gpe_results[i, 1, 0],
        f"{gpe_results[i, 1, 0]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[0].bar(
        count + 12,
        gpe_results[i, 0, 1],
        color=hue_color_palette["excitation"][0],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[0].text(
        count + 12,
        gpe_results[i, 0, 1],
        f"{gpe_results[i, 0, 1]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[0].bar(
        count + 18,
        gpe_results[i, 1, 1],
        color=hue_color_palette["excitation"][1],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[0].text(
        count + 18,
        gpe_results[i, 1, 1],
        f"{gpe_results[i, 1, 1]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[1].bar(
        count,
        str_results[i, 0, 0],
        color=hue_color_palette["inhibition"][0],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[1].text(
        count,
        str_results[i, 0, 0],
        f"{str_results[i, 0, 0]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[1].bar(
        count + 6,
        str_results[i, 1, 0],
        color=hue_color_palette["inhibition"][1],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[1].text(
        count + 6,
        str_results[i, 1, 0],
        f"{str_results[i, 1, 0]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[1].bar(
        count + 12,
        str_results[i, 0, 1],
        color=hue_color_palette["excitation"][0],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[1].text(
        count + 12,
        str_results[i, 0, 1],
        f"{str_results[i, 0, 1]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    axes[1].bar(
        count + 18,
        str_results[i, 1, 1],
        color=hue_color_palette["excitation"][1],
        width=1,
        edgecolor="k",
        zorder=15,
    )
    axes[1].text(
        count + 18,
        str_results[i, 1, 1],
        f"{str_results[i, 1, 1]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    count += 1

for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
axes[0].set_xlim([-0.6, 22.6])
axes[1].set_xlim([-0.6, 22.6])

axes[0].grid("on", color="gray", lw=0.5, axis="y", zorder=5, alpha=0.25)
axes[1].grid("on", color="gray", lw=0.5, axis="y", zorder=5, alpha=0.25)

axes[0].set_xticks(
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
)
axes[0].set_xticklabels(params_gpe_labels * 4, rotation=25)

axes[1].set_xticks(
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
)
axes[1].set_xticklabels(params_str_labels * 4, rotation=25)

axes[0].set_ylabel("Norm. Modulatory Effect\n(Resp.-to-NE)")
axes[1].set_ylabel("Norm. Modulatory Effect\n(Resp.-to-NE)")

add_fig_labels(axes)
match_axis(axes)
# save, close, and open pdf of figure
fig.savefig(
    f"{FIG_DIR}/responses_WD_bar.pdf",
    bbox_inches="tight",
)
plt.close()
run_cmd(f"open {FIG_DIR}/responses_WD_bar.pdf")


# str_results = np.zeros((5, 2, 2))  # param, naive/dd,inh/exc
# gpe_results = np.zeros((5, 2, 2))

bar_height = 0.9
bar_edgecolor = "k"
# fig, ax = plt.subplots(2, 2, figsize=(8, 4), dpi=300, tight_layout=True)
# axes = [ax[i, j] for i in range(2) for j in range(2)]

fig = plt.figure(figsize=(8, 4), dpi=300, tight_layout=True)
gs = GridSpec(2, 3, figure=fig)
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[:, 2]),
]


for i in range(len(params_gpe)):
    axes[0].barh(
        i,
        -gpe_results[i, 0, 0],
        color=hue_color_palette["inhibition"][0],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[0].barh(
        i,
        gpe_results[i, 1, 0],
        color=hue_color_palette["inhibition"][1],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[0].annotate(
        f"{gpe_results[i, 0, 0]:.2f} ",
        xytext=(-gpe_results[i, 0, 0], i),
        xy=(-gpe_results[i, 0, 0], i),
        ha="right",
        va="center",
        fontsize=7,
    )
    axes[0].annotate(
        f" {gpe_results[i, 1, 0]:.2f}",
        xytext=(gpe_results[i, 1, 0], i),
        xy=(gpe_results[i, 1, 0], i),
        ha="left",
        va="center",
        fontsize=7,
    )

    axes[1].barh(
        i,
        -gpe_results[i, 0, 1],
        color=hue_color_palette["excitation"][0],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[1].barh(
        i,
        gpe_results[i, 1, 1],
        color=hue_color_palette["excitation"][1],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[1].annotate(
        f"{gpe_results[i, 0, 1]:.2f} ",
        xytext=(-gpe_results[i, 0, 1], i),
        xy=(-gpe_results[i, 0, 1], i),
        ha="right",
        va="center",
        fontsize=7,
    )
    axes[1].annotate(
        f" {gpe_results[i, 1, 1]:.2f}",
        xytext=(gpe_results[i, 1, 1], i),
        xy=(gpe_results[i, 1, 1], i),
        ha="left",
        va="center",
        fontsize=7,
    )

    axes[2].barh(
        i,
        -str_results[i, 0, 0],
        color=hue_color_palette["inhibition"][0],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[2].barh(
        i,
        str_results[i, 1, 0],
        color=hue_color_palette["inhibition"][1],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[2].annotate(
        f"{str_results[i, 0, 0]:.2f} ",
        xytext=(-str_results[i, 0, 0], i),
        xy=(-str_results[i, 0, 0], i),
        ha="right",
        va="center",
        fontsize=7,
    )
    axes[2].annotate(
        f" {str_results[i, 1, 0]:.2f}",
        xytext=(str_results[i, 1, 0], i),
        xy=(str_results[i, 1, 0], i),
        ha="left",
        va="center",
        fontsize=7,
    )

    axes[3].barh(
        i,
        -str_results[i, 0, 1],
        color=hue_color_palette["excitation"][0],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[3].barh(
        i,
        str_results[i, 1, 1],
        color=hue_color_palette["excitation"][1],
        edgecolor=bar_edgecolor,
        height=bar_height,
    )
    axes[3].annotate(
        f"{str_results[i, 0, 1]:.2f} ",
        xytext=(-str_results[i, 0, 1], i),
        xy=(-str_results[i, 0, 1], i),
        ha="right",
        va="center",
        fontsize=7,
    )
    axes[3].annotate(
        f" {str_results[i, 1, 1]:.2f}",
        xytext=(str_results[i, 1, 1], i),
        xy=(str_results[i, 1, 1], i),
        ha="left",
        va="center",
        fontsize=7,
    )


for plot in [axes[i] for i in range(4)]:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
    plot.set_xlabel("Norm. Modulatory Effect (WD Resp.-to-NE)")
    plot.set_ylim([-0.5, 5])
    plot.vlines(0, -0.5, 5, color="gray", ls="dashed", lw=1.5)

for i in [0, 1]:
    axes[i].set_yticks(range(5))
    axes[i].set_yticklabels(params_gpe_labels)

for i in [2, 3]:
    axes[i].set_yticks(range(5))
    axes[i].set_yticklabels(params_str_labels)

for i in [0, 2]:
    axes[i].set_xlim([-2.8, 1.8])

for i in [1, 3]:
    axes[i].set_xlim([-1.3, 1.3])


for i in [0, 2]:
    ticks = axes[i].get_xticks()
    axes[i].set_xticks(ticks)
    axes[i].set_xticklabels([x if x >= 0 else -x for x in ticks])

for i in [1, 3]:
    ticks = axes[i].get_xticks()
    axes[i].set_xticks(ticks)
    axes[i].set_xticklabels([x if x >= 0 else -x for x in ticks])

for i in [0, 2]:
    axes[i].annotate(
        "GPe-Naive" if i == 0 else "Str-Naive",
        xy=(-1.5, 4.75),
        xytext=(-1.5, 4.75),
        fontsize=8,
        ha="center",
        va="center",
    )
    axes[i].annotate(
        "GPe-DD" if i == 0 else "Str-DD",
        xy=(1.0, 4.75),
        xytext=(1.0, 4.75),
        fontsize=8,
        ha="center",
        va="center",
    )

for i in [1, 3]:
    axes[i].annotate(
        "GPe-Naive" if i == 1 else "Str-Naive",
        xy=(-0.75, 4.75),
        xytext=(-0.75, 4.75),
        fontsize=8,
        ha="center",
        va="center",
    )
    axes[i].annotate(
        "GPe-DD" if i == 1 else "Str-DD",
        xy=(0.75, 4.75),
        xytext=(0.75, 4.75),
        fontsize=8,
        ha="center",
        va="center",
    )

trials_to_average = 10
data_dir = "data/param_search_vivo_from_slice"
save_dir = "data/example_sims_dynamics"

slice_seeds = []
for subdir in os.listdir(data_dir):
    if subdir != ".DS_Store":
        slice_seeds.append(eval(subdir.split("_")[2]))
slice_seeds = np.unique(slice_seeds)

sim_case = ""
dopamine_depletion = False
d1 = True
gpe = False
status = True
dynamics = True
show = False
size = 100
T0 = 3  # End of transient period
T1 = 5  # End of baseline period
T2 = 35 if d1 or gpe else 5  # End of stim period


dt = 0.05
stim_times = np.arange(5, 35, 3)

stim_indices = np.rint(stim_times * 1000 / dt).astype(int)
stim_len = int(np.rint(2 * 1000 / dt))  # do no include post
num_exc = 0
counts = np.zeros(2)


variable_of_interest = "SNr"
trials_to_average = 10
resp_ratio_results = []
resp_to_examine = {
    "no effect": [],
    "inhibition": [],
    "excitation": [],
}

resp_types_to_examine = [
    "no effect",
    "complete inhibition",
    "partial inhibition",
    "adapting inhibition",
    "excitation",
    "biphasic IE",
    "biphasic EI",
]


cell_nums = []


# for key in resp_to_examine:
for response_type in resp_types_to_examine:
    resp_ratios = []
    cell_count = 0

    seeds_to_sim = slice_seeds[:5]
    for seed in seeds_to_sim:
        sim_case = f"slice_seed_{seed}"
        if dopamine_depletion:
            if d1:
                sim_case = f"{sim_case}_dd_stim"
            elif gpe:
                sim_case = f"{sim_case}_gpe_stim"
            else:
                sim_case = f"{sim_case}_dd_search"
        else:
            if d1:
                sim_case = f"{sim_case}_naive_d1_stim"
            elif gpe:
                sim_case = f"{sim_case}_naive_gpe_stim"

        sims_to_copy = [
            x for x in os.listdir(f"{data_dir}/{sim_case}") if x != ".DS_Store"
        ]
        sims_to_copy = sims_to_copy[:1]
        for sim_dir in sims_to_copy:
            path = f"{save_dir}/{data_dir}/{sim_case}/{sim_dir}/sim_0001"
            responses = pd.read_csv(f"{path}/processed/all_data.csv")

            for index, row in responses[
                responses["neural_response"] == response_type
            ].iterrows():

                t, v = np.loadtxt(
                    f"{path}/Neuron_{row["cell_num"]-1}/cell_dynamics.txt",
                    usecols=(0, 1 if variable_of_interest == "Vm" else 5),
                    unpack=True,
                )

                t, v, v_dend, dv, dv_dend, gGABA_snr, gGABA_gpe, gGABA_str = np.loadtxt(
                    f"{path}/Neuron_{row["cell_num"]-1}/cell_dynamics.txt",
                    usecols=(0, 1, 2, 3, 4, 5, 6, 7),
                    unpack=True,
                )

                v_trial = np.zeros(stim_len, dtype=float)

                for trials in range(trials_to_average):
                    start = stim_indices[trials]
                    end = start + stim_len
                    v_trial += v[start:end] / trials_to_average
                """resp_ratios[key].append(
                    np.mean(v_trial[len(v_trial) // 2 :])
                    / np.mean(v_trial[: len(v_trial) // 2])
                )"""
                if "inhibition" in response_type:
                    resp_to_examine["inhibition"].append(
                        np.mean(v_trial[len(v_trial) // 2 :])
                        / np.mean(v_trial[: len(v_trial) // 2])
                    )
                elif "excitation" in response_type:
                    resp_to_examine["excitation"].append(
                        np.mean(v_trial[len(v_trial) // 2 :])
                        / np.mean(v_trial[: len(v_trial) // 2])
                    )
                elif "biphasic" in response_type:
                    resp_to_examine[
                        "inhibition" if response_type == "biphasic IE" else "excitation"
                    ].append(
                        np.mean(v_trial[len(v_trial) // 2 :])
                        / np.mean(v_trial[: len(v_trial) // 2])
                    )
                else:
                    resp_to_examine["no effect"].append(
                        np.mean(v_trial[len(v_trial) // 2 :])
                        / np.mean(v_trial[: len(v_trial) // 2])
                    )

                cell_count += 1
    cell_nums.append(cell_count)
    # print(key, cell_nums, len(resp_ratios))
    # resp_ratio_results.append(resp_ratios)


axes[4].boxplot([resp_to_examine[key] for key in resp_to_examine])
axes[4].set_xticks(np.arange(1, len(resp_to_examine) + 1))
axes[4].set_xticklabels(
    [f"{key}\n n={cell_nums[i]}" for i, key in enumerate(resp_to_examine)],
    rotation=15,
)

axes[4].set_ylabel("Average $g_{GABA}^{SNr}$ Change\n(stim/baseline)", fontsize=6)

axes[4].spines["top"].set_visible(False)
axes[4].spines["right"].set_visible(False)
axes[4].spines["left"].set_linewidth(0.5)
axes[4].spines["bottom"].set_linewidth(0.5)
axes[4].tick_params(width=0.5, labelsize=6)

inh_vs_exc_pval = ttest_ind(
    resp_to_examine["inhibition"],
    resp_to_examine["excitation"],
    equal_var=False,
).pvalue

ne_vs_ex_pval = ttest_ind(
    resp_to_examine["no effect"],
    resp_to_examine["excitation"],
    equal_var=False,
).pvalue

inh_vs_ne_pval = ttest_ind(
    resp_to_examine["inhibition"],
    resp_to_examine["no effect"],
    equal_var=False,
).pvalue

# open figure to store stats
stats_file = open(f"{FIG_DIR}/responses_WD_barh_stats.txt", "w")
for key in resp_to_examine:
    stats_file.write(f"{key}:\n")
    stats_file.write(f"\t Mean: {np.mean(resp_to_examine[key])}\n")
    stats_file.write(f"\t Median: {np.median(resp_to_examine[key])}\n")
    stats_file.write(f"\t n={len(resp_to_examine[key])}\n\n")

stats_file.write(f"INH vs EXC t-Test: p={inh_vs_exc_pval}\n")
stats_file.write(f"NE vs EXC t-Test: p={ne_vs_ex_pval}\n")
stats_file.write(f"INH vs NE t-Test: p={inh_vs_ne_pval}\n")

stats_file.close()

ylims = axes[4].get_ylim()
pmr.plot_bracket(
    axes[4],
    2,
    3,
    ylims[1],
    ylims[0] + 1.03 * (ylims[1] - ylims[0]),
    f"T-test $p=${inh_vs_exc_pval:.3f}",
)


add_fig_labels(axes)
# match_axis([axes[0], axes[2]])
# match_axis([axes[1], axes[3]])
# save, close, and open pdf of figure
fig.savefig(
    f"{FIG_DIR}/responses_WD_barh.pdf",
    bbox_inches="tight",
)
plt.close()
run_cmd(f"open {FIG_DIR}/responses_WD_barh.pdf")
