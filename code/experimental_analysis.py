# import python modules
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
import pandas as pd
import networkx as nx
from scipy.stats import ks_2samp
from scipy import stats
import pickle
import matplotlib as mpl

# import user modules
sys.path.append("/Users/johnparker/streac")
sys.path.append("../")
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import poisson_surprise

# change to location where expirmental data resides
EXPERIMENTAL_DATA_DIREC = "/Users/johnparker/UPitt_Data"

def ephys_exp_data():
    xcel_file = pd.ExcelFile(
        f"{EXPERIMENTAL_DATA_DIREC}/Aristieta2024_GaoRecordings/Ephys_overall_analysis.xlsx"
    )
    gpe = pd.read_excel(xcel_file, "GPe")
    gpe_dd = pd.read_excel(xcel_file, "GPe_DD")
    d1 = pd.read_excel(xcel_file, "D1")
    d1_dd = pd.read_excel(xcel_file, "D1_DD")
    return gpe, gpe_dd, d1, d1_dd


def ephys_sample(x, y, samp_size=100, conf=0.99):
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    # Compute predicted values
    x_sample = np.random.uniform(np.min(x), np.max(x), samp_size)
    Y_pred = slope * x_sample + intercept

    data = np.array(y)
    n = len(y)
    mean = np.mean(y)
    sem = stats.sem(y)  # Standard error of the mean
    interval = sem * stats.t.ppf((1 + conf) / 2.0, n - 1)

    # Compute confidence bounds
    lower_bound = Y_pred - interval
    upper_bound = Y_pred + interval

    y_samp = np.asarray(
        [np.random.uniform(lower_bound[i], upper_bound[i]) for i in range(samp_size)]
    )

    return x_sample, Y_pred, lower_bound, upper_bound, y_samp


def get_modulation_factors(exp_df):
    exp_mfs = np.zeros(len(exp_df))
    count = 0
    for _, row in exp_df.iterrows():
        exp_mfs[count] = (
            (row["stim_freq"] - row["baseline_freq"])
            / (row["stim_freq"] + row["baseline_freq"])
            if (row["stim_freq"] + row["baseline_freq"]) > 0
            else 0
        )
        count += 1
    return exp_mfs


def get_classification_results():
    results_dir = (
        f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/gpe_pv_hsyn/processed_data_short"
    )
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    gpe_pv_cont_soma = df[
        (df["group"].str.contains("PV")) & (df["mouse"].str.contains("Naive"))
    ]

    results_dir = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/gpe_pv/processed_data_short"
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    gpe_pv_pulsed20Hz = df[
        (df["group"].str.contains("PV")) & (df["mouse"].str.contains("Naive"))
    ]

    results_dir = (
        f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/d1_msns/processed_data_short"
    )
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    d1_pv_cont_soma = df[
        (df["group"].str.contains("MSNs")) & (df["mouse"].str.contains("Naive"))
    ]

    results_dir = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/d1_msns/processed_data_short"
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    d1_pv_pulsed20Hz = df[
        (df["group"].str.contains("MSNs")) & (df["mouse"].str.contains("Naive"))
    ]

    return gpe_pv_cont_soma, gpe_pv_pulsed20Hz, d1_pv_cont_soma, d1_pv_pulsed20Hz


def get_dd_classification_results():
    results_dir = (
        f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/gpe_pv_hsyn/processed_data_short"
    )
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    gpe_pv_cont_soma = df[
        (df["group"].str.contains("PV")) & (df["mouse"].str.contains("6-OHDA"))
    ]

    results_dir = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/gpe_pv/processed_data_short"
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    gpe_pv_pulsed20Hz = df[
        (df["group"].str.contains("PV")) & (df["mouse"].str.contains("6-OHDA"))
    ]

    results_dir = (
        f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/d1_msns/processed_data_short"
    )
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    d1_pv_cont_soma = df[
        (df["group"].str.contains("MSNs")) & (df["mouse"].str.contains("6-OHDA"))
    ]

    results_dir = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/d1_msns/processed_data_short"
    csv = f"{results_dir}/all_data.csv"
    df = pd.read_csv(csv)
    d1_pv_pulsed20Hz = df[
        (df["group"].str.contains("MSNs")) & (df["mouse"].str.contains("6-OHDA"))
    ]

    return gpe_pv_cont_soma, gpe_pv_pulsed20Hz, d1_pv_cont_soma, d1_pv_pulsed20Hz


def get_naive_continuous_stim_data():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/gpe_pv_hsyn/processed_data_short/all_data.csv"
    cont_gpe = pd.read_csv(csv)
    cont_gpe = cont_gpe[
        (cont_gpe["mouse"] == "Naive mice")
        & (cont_gpe["group"] == "Naive_mice_PV-DIO-ChR2_in_GPe")
    ]

    csv = f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/d1_msns/processed_data_short/all_data.csv"
    cont_d1 = pd.read_csv(csv)
    cont_d1 = cont_d1[(cont_d1["mouse"] == "Naive")]
    return cont_gpe, cont_d1


def get_dd_continuous_stim_data():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/gpe_pv_hsyn/processed_data_short/all_data.csv"
    cont_gpe = pd.read_csv(csv)
    cont_gpe = cont_gpe[
        (cont_gpe["mouse"] == "6-OHDA mice")
        & (cont_gpe["group"] == "6-OHDA_mice_PV-DIO-ChR2_in_GPe")
    ]

    csv = f"{EXPERIMENTAL_DATA_DIREC}/continuous_light/d1_msns/processed_data_short/all_data.csv"
    cont_d1 = pd.read_csv(csv)
    cont_d1 = cont_d1[(cont_d1["mouse"] == "6-OHDA")]
    return cont_gpe, cont_d1


def get_naive_pulsed_stim_data():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/gpe_pv/processed_data_short/all_data.csv"
    df = pd.read_csv(csv)
    df_gpe = df[(df["mouse"] == "Naive")]

    csv = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/d1_msns/processed_data_short/all_data.csv"
    df = pd.read_csv(csv)
    df_d1 = df[(df["mouse"] == "Naive")]
    return df_gpe, df_d1


def get_dd_pulsed_stim_data():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/gpe_pv/processed_data_short/all_data.csv"
    df = pd.read_csv(csv)
    df_gpe = df[(df["mouse"] == "6-OHDA")]

    csv = f"{EXPERIMENTAL_DATA_DIREC}/pulsed_light_terminal_20Hz/d1_msns/processed_data_short/all_data.csv"
    df = pd.read_csv(csv)
    df_d1 = df[(df["mouse"] == "6-OHDA")]
    return df_gpe, df_d1


def get_slice_data():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/slice/gpe_pv/processed_data_short/all_data.csv"
    slice_gpe = pd.read_csv(csv)
    slice_gpe = slice_gpe[slice_gpe["group"] != "stim_20Hz_ZD_True"]
    csv = f"{EXPERIMENTAL_DATA_DIREC}/slice/d1_msns/processed_data_short/all_data.csv"
    slice_d1 = pd.read_csv(csv)
    slice_d1 = slice_d1[slice_d1["group"] != "stim_20Hz_ZD_True"]
    return slice_gpe, slice_d1


def get_slice_baseline_spikes_short():
    all_spikes = []
    gpe_slice_path = f"{EXPERIMENTAL_DATA_DIREC}/slice/gpe_pv/processed_data_short"
    for dir in os.listdir(gpe_slice_path):
        if "False" in dir:
            for neuron in os.listdir(f"{gpe_slice_path}/{dir}"):
                if neuron != ".DS_Store":
                    spikes = np.loadtxt(f"{gpe_slice_path}/{dir}/{neuron}/spikes.txt")
                    spikes = spikes[spikes <= 2]
                    all_spikes.append(spikes)
    d1_slice_path = f"{EXPERIMENTAL_DATA_DIREC}/slice/d1_msns/processed_data_short"
    for dir in os.listdir(d1_slice_path):
        if "False" in dir:
            for neuron in os.listdir(f"{d1_slice_path}/{dir}"):
                if neuron != ".DS_Store":
                    spikes = np.loadtxt(f"{d1_slice_path}/{dir}/{neuron}/spikes.txt")
                    spikes = spikes[spikes <= 2]
                    all_spikes.append(spikes)

    data = pd.read_csv(
        f"{EXPERIMENTAL_DATA_DIREC}/slice/SNr cell firing baseline.csv"
    )

    # Convert each column to a list of spike times, dropping NaN values
    spike_times = [
        np.asarray(data[column].dropna().tolist()) for column in data.columns
    ]
    spike_times = [x[x >= 13] - 13 for x in spike_times]
    all_spikes.extend(spike_times)
    return all_spikes


def get_baseline_slice_data():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/slice/baseline_15s_freq_cv.csv"
    slice_baseline = pd.read_csv(csv)
    return slice_baseline


def get_baseline_slice_data_short():
    csv = f"{EXPERIMENTAL_DATA_DIREC}/slice/baseline_15s_freq_cv_last2s.csv"
    slice_baseline = pd.read_csv(csv)
    return slice_baseline


def get_in_vivo_baseline_data_short(segment=2, DD=False):
    cont_gpe, cont_d1 = (
        get_naive_continuous_stim_data() if not DD else get_dd_continuous_stim_data()
    )
    pulse_gpe, pulse_d1 = (
        get_naive_pulsed_stim_data() if not DD else get_dd_pulsed_stim_data()
    )
    in_vivo = pd.concat([cont_gpe, cont_d1, pulse_gpe, pulse_d1], ignore_index=True)

    in_vivo_update = np.zeros((len(in_vivo), 10))
    count = 0
    for _, row in in_vivo.iterrows():
        # neuron = pickle.load(open(f"{row['src']}/neuron.obj", "rb"))
        neuron_spike = np.loadtxt(f"{row["src"]}/spikes.txt")
        light_on = np.loadtxt(f"{row['src']}/light_on.txt")

        spikes_final = neuron_spike[
            (neuron_spike >= light_on[0] - segment) & (neuron_spike < light_on[0])
        ]
        if len(spikes_final) > 0:
            freq_final = spikes_final.shape[0] / segment
            cv_final = (
                np.std(np.diff(spikes_final)) / np.mean(np.diff(spikes_final))
                if len(spikes_final) > 2
                else 0
            )

            bursts = poisson_surprise.run_poisson_surprise(
                spikes_final, surprise_threshold=3
            )
            if len(bursts) > 0:
                (
                    n_bursts,
                    avg_burst_firing_rate,
                    percent_time_bursting,
                    percent_spike_bursting,
                    avg_burst_duration,
                    avg_inter_burst_interval,
                    cv_inter_burst_interval,
                    avg_surprise,
                    non_bursting_firing_rate,
                    burst_firing_rate_increase,
                ) = poisson_surprise.burst_statistics(spikes_final, bursts, segment)
            else:
                n_bursts = 0
                avg_burst_firing_rate = 0
                percent_time_bursting = 0
                percent_spike_bursting = 0
                avg_burst_duration = 0
                avg_inter_burst_interval = 0
                cv_inter_burst_interval = 0
                avg_surprise = 0
                non_bursting_firing_rate = 0
                burst_firing_rate_increase = 0
        else:
            freq_final = 0
            cv_final = 0

        # in_vivo_update[count] = [freq_final, cv_final]
        in_vivo_update[count] = [
            freq_final,
            cv_final,
            n_bursts if len(bursts) > 0 else 0,
            avg_burst_firing_rate if len(bursts) > 0 else 0,
            percent_time_bursting if len(bursts) > 0 else 0,
            percent_spike_bursting if len(bursts) > 0 else 0,
            avg_burst_duration if len(bursts) > 0 else 0,
            avg_inter_burst_interval if len(bursts) > 0 else 0,
            non_bursting_firing_rate if len(bursts) > 0 else 0,
            burst_firing_rate_increase if len(bursts) > 0 else 0,
        ]
        count += 1
    in_vivo["pre_exp_freq"] = in_vivo_update[:, 0]
    in_vivo["pre_exp_cv"] = in_vivo_update[:, 1]
    in_vivo["pre_exp_num_bursts"] = in_vivo_update[:, 2]
    in_vivo["pre_exp_avg_burst_firing_rate"] = in_vivo_update[:, 3]
    in_vivo["pre_exp_percent_time_bursting"] = in_vivo_update[:, 4]
    in_vivo["pre_exp_percent_spike_bursting"] = in_vivo_update[:, 5]
    in_vivo["pre_exp_avg_burst_duration"] = in_vivo_update[:, 6]
    in_vivo["pre_exp_avg_inter_burst_interval"] = in_vivo_update[:, 7]
    in_vivo["pre_exp_non_bursting_firing_rate"] = in_vivo_update[:, 8]
    in_vivo["pre_exp_burst_firing_rate_increase"] = in_vivo_update[:, 9]
    return in_vivo


def get_in_vivo_baseline_spikes_short():
    return 0


def get_in_vivo_baseline_data():
    cont_gpe, cont_d1 = get_naive_continuous_stim_data()
    pulse_gpe, pulse_d1 = get_naive_pulsed_stim_data()
    in_vivo = pd.concat([cont_gpe, cont_d1, pulse_gpe, pulse_d1], ignore_index=True)
    return in_vivo


def get_slice_baseline_data():
    slice_gpe, slice_d1 = get_slice_data()
    slice_baseline = get_baseline_slice_data_short()
    slice = pd.concat(
        [
            slice_gpe[["pre_exp_freq", "pre_exp_cv"]],
            slice_d1[["pre_exp_freq", "pre_exp_cv"]],
            slice_baseline[["pre_exp_freq", "pre_exp_cv"]],
        ],
        ignore_index=True,
    )
    return slice


def plot_pval_compare(
    dfs,
    labels,
    pval=0.05,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    fig, ax = plt.subplots(2, 2, figsize=(8, 4), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    fr_tests = np.zeros((6, 6))
    cv_tests = np.zeros((6, 6))
    t_tests_fr = np.zeros((6, 6))
    t_tests_cv = np.zeros((6, 6))
    for i in range(6):
        for k in range(6):
            fr_tests[i, k] = ks_2samp(
                dfs[i]["pre_exp_freq"], dfs[k]["pre_exp_freq"]
            ).pvalue  # not distributions_match(dfs[i]["pre_exp_freq"].values,dfs[k]["pre_exp_freq"].values,alpha=0.05)
            t_tests_fr[i, k] = stats.ttest_ind(
                dfs[i]["pre_exp_freq"], dfs[k]["pre_exp_freq"]
            ).pvalue
    for i in range(6):
        for k in range(6):
            cv_tests[i, k] = ks_2samp(
                dfs[i]["pre_exp_cv"], dfs[k]["pre_exp_cv"]
            ).pvalue  # not distributions_match(dfs[i]["pre_exp_cv"].values,dfs[k]["pre_exp_cv"].values,alpha=0.05)
            t_tests_cv[i, k] = stats.ttest_ind(
                dfs[i]["pre_exp_cv"], dfs[k]["pre_exp_cv"]
            ).pvalue

    cmap = mpl.colormaps.get_cmap("Greens")
    cmap.set_bad("r")

    ax1 = sns.heatmap(
        fr_tests,
        annot=True,
        fmt=".02f",
        ax=axes[0],
        cmap=cmap,
        mask=fr_tests < pval,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 6},
    )
    ax2 = sns.heatmap(
        cv_tests,
        annot=True,
        fmt=".02f",
        ax=axes[1],
        cmap=cmap,
        mask=cv_tests < pval,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 6},
    )

    ax3 = sns.heatmap(
        t_tests_fr,
        annot=True,
        fmt=".02f",
        ax=axes[2],
        cmap=cmap,
        mask=t_tests_fr < pval,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 6},
    )

    ax4 = sns.heatmap(
        t_tests_cv,
        annot=True,
        fmt=".02f",
        ax=axes[3],
        cmap=cmap,
        mask=t_tests_cv < pval,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 6},
    )

    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    cbar = ax3.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    cbar = ax4.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    for i in range(4):
        axes[i].set_yticklabels(labels, rotation=0, fontsize=6)
        axes[i].set_xticklabels(labels, rotation=0, fontsize=6)

    axes[0].set_title("Firing Rate KS Test", fontsize=8)
    axes[1].set_title("CV KS Test", fontsize=8)
    axes[2].set_title("Firing Rate T-Test", fontsize=8)
    axes[3].set_title("CV T-Test", fontsize=8)
    fig.savefig(f"{save_dir}/pval_comparisons.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/pval_comparisons.pdf")


def plot_exp_results(
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
    remove_outliers=True,
):
    cont_gpe, cont_d1 = get_naive_continuous_stim_data()
    pulse_gpe, pulse_d1 = get_naive_pulsed_stim_data()
    slice_gpe, slice_d1 = get_slice_data()
    if remove_outliers:
        cont_gpe, _, _, _, _ = find_and_remove_outliers_as_df(cont_gpe)
        cont_d1, _, _, _, _ = find_and_remove_outliers_as_df(cont_d1)
        pulse_gpe, _, _, _, _ = find_and_remove_outliers_as_df(pulse_gpe)
        pulse_d1, _, _, _, _ = find_and_remove_outliers_as_df(pulse_d1)
        slice_gpe, _, _, _, _ = find_and_remove_outliers_as_df(slice_gpe)
        slice_d1, _, _, _, _ = find_and_remove_outliers_as_df(slice_d1)

    dfs = [cont_gpe, cont_d1, pulse_gpe, pulse_d1, slice_gpe, slice_d1]

    labels = ["Cont. GPe", "Cont. D1", "Pulse GPe", "Pulse D1", "Slice GPe", "Slice D1"]

    plot_pval_compare(dfs, labels, save_dir=save_dir, show=show)

    plot_all_conds_fr(dfs, labels, save_dir=save_dir, show=show)
    plot_all_conds_cv(dfs, labels, save_dir=save_dir, show=show)

    plot_heat_maps(dfs, labels, save_dir=save_dir, show=show)
    plot_regressions(dfs, labels, save_dir=save_dir, show=show)


def plot_all_conds_cv(
    dfs,
    labels,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    fig, ax2 = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax2[i, j] for i in range(3) for j in range(2)]
    max_cv = np.max([np.max(x["pre_exp_cv"]) for x in dfs])
    bins_cv = np.arange(0, max_cv + 0.1, 0.1)
    for i in range(len(dfs)):
        sns.histplot(
            dfs[i]["pre_exp_cv"].values,
            kde=True,
            stat="probability",
            bins=bins_cv,
            edgecolor="w",
            alpha=0.5,
            label=labels[i],
            ax=axes[i],
            legend=True,
            color="blue",
        )

        axes[i].vlines(
            np.mean(dfs[i]["pre_exp_cv"].values),
            0,
            axes[i].get_ylim()[1],
            ls="dashed",
            color="k",
        )

        axes[i].legend(fontsize=6, frameon=False, fancybox=False)
        axes[i].set_xticks(bins_cv[::2])
        axes[i].set_xticklabels([f"{x:.1f}" for x in bins_cv[::2]], fontsize=6)
        axes[i].set_xlabel("Pre-Stim CV", fontsize=8)
        axes[i].set_ylabel("Probability", fontsize=8)

    match_axis([axes[k] for k in range(6)], type="both")

    makeNice(axes, labelsize=6)
    add_fig_labels(axes)

    fig.savefig(f"{save_dir}/all_cv.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/all_cv.pdf")


def plot_all_conds_fr(
    dfs,
    labels,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]
    max_fr = np.max([np.max(x["pre_exp_freq"]) for x in dfs])
    bins_fr = np.arange(0, max_fr + 10, 10)

    for i in range(len(dfs)):
        sns.histplot(
            dfs[i]["pre_exp_freq"].values,
            kde=True,
            stat="probability",
            bins=bins_fr,
            edgecolor="w",
            alpha=0.5,
            label=labels[i],
            ax=axes[i],
            legend=True,
            color="blue",
        )
        axes[i].vlines(
            np.mean(dfs[i]["pre_exp_freq"].values),
            0,
            axes[i].get_ylim()[1],
            ls="dashed",
            color="k",
        )

        axes[i].legend(fontsize=6, frameon=False, fancybox=False)
        axes[i].set_xticks(bins_fr)
        axes[i].set_xticklabels(bins_fr, fontsize=6)
        axes[i].set_xlabel("Pre-Stim Firing Rate (Hz)", fontsize=8)
        axes[i].set_ylabel("Probability", fontsize=8)

    match_axis([axes[k] for k in range(6)], type="both")

    makeNice(axes, labelsize=6)
    add_fig_labels(axes)

    fig.savefig(f"{save_dir}/all_fr.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/all_fr.pdf")


def plot_regressions(
    dfs,
    labels,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]

    for i in range(len(dfs)):
        pmr.plot_linear_fit(
            dfs[i]["pre_exp_freq"], dfs[i]["pre_exp_cv"], axes[i], color="blue"
        )
        axes[i].set_title(labels[i], fontsize=8)
        axes[i].set_xlabel("Pre-Stim Firing Rate (Hz)", fontsize=8)
        axes[i].set_ylabel("Pre-Stim CV", fontsize=8)

    match_axis([axes[k] for k in range(6)], type="both")

    makeNice(axes, labelsize=6)
    add_fig_labels(axes)

    fig.savefig(f"{save_dir}/all_regressions.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/all_regressions.pdf")


def plot_heat_maps(
    dfs,
    labels,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    max_fr = np.max([np.max(x["pre_exp_freq"]) for x in dfs])
    bins_fr = np.arange(0, max_fr, 10)
    max_cv = np.max([np.max(x["pre_exp_cv"]) for x in dfs])
    bins_cv = np.arange(0, max_cv, 0.1)

    xlabels = [f"{x:.1f}" for x in bins_cv]
    ylabels = [int(x) for x in bins_fr]

    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]

    for i in range(len(dfs)):
        data = np.zeros((len(bins_fr), len(bins_cv)))
        for index, row in dfs[i].iterrows():
            k = np.searchsorted(bins_fr, row["pre_exp_freq"], side="right")
            j = np.searchsorted(bins_cv, row["pre_exp_cv"], side="right")

            data[k - 1, j - 1] += 1
        data = data / np.sum(data)
        ax1 = sns.heatmap(
            data=data,
            ax=axes[i],
            cmap="jet",
            xticklabels=xlabels,
            yticklabels=ylabels,
            mask=data == 0,
            annot=True,
            annot_kws={"fontsize": 4},
            fmt=".2f",
            vmin=0,
            vmax=0.35,
            linecolor="gray",
            linewidths=0.5,
        )
        axes[i].set_title(labels[i], fontsize=8)

        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)

        axes[i].set_xticks(list(range(0, len(xlabels), 2)))
        axes[i].set_xticklabels([x for x in xlabels[::2]], rotation=0)
        axes[i].set_yticklabels(ylabels, rotation=0)
        axes[i].set_xlabel("Pre-Stim CV", fontsize=8)
        axes[i].set_ylabel("Pre-Stim Firing Rate (Hz)", fontsize=8)
        axes[i].invert_yaxis()

    makeNice(axes, labelsize=6)
    add_fig_labels(axes)

    fig.savefig(f"{save_dir}/all_heatmaps.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/all_heatmaps.pdf")


def find_and_remove_outliers_as_df(
    df, columns=["pre_exp_freq", "pre_exp_cv"], threshold=[0, -1], zvals=[3, 3]
):
    """(
        df_pre_freq,
        df_pre_cv,
        fr_low_outliers,
        cv_low_outliers,
        fr_high_outliers,
        cv_high_outliers,
    ) = na.find_and_remove_outliers_fr_cv(
        df[["pre_exp_freq", "pre_exp_cv"]].values, fr_threshold=fr_threshold, zval=zval
    )
    return (
        pd.DataFrame({"pre_exp_freq": df_pre_freq, "pre_exp_cv": df_pre_cv}),
        fr_low_outliers,
        cv_low_outliers,
        fr_high_outliers,
        cv_high_outliers,
    )
    """
    df_removed, high_outliers, low_outliers = na.find_and_remove_outliers_columns(
        df, columns=columns, threshold=threshold, zvals=zvals
    )
    return df_removed, high_outliers, low_outliers


def find_and_remove_outliers_as_df_return_df(df, zval=3):
    df["pre_exp_freq_zscores"] = stats.zscore(df["pre_exp_freq"])
    df["pre_exp_cv_zscores"] = stats.zscore(df["pre_exp_cv"])
    return df[
        (np.abs(df["pre_exp_freq_zscores"]) <= zval)
        & (np.abs(df["pre_exp_cv_zscores"]) <= zval)
    ]


def plot_vivo_vs_slice_fr_cv(
    vivo,
    slice,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    bins_fr = np.arange(
        0,
        np.max([np.max(slice["pre_exp_freq"]), np.max(vivo["pre_exp_freq"])]),
        10,
    )
    bins_cv = np.arange(
        0,
        np.max([np.max(slice["pre_exp_cv"]), np.max(vivo["pre_exp_cv"])]),
        0.2,
    )

    fig = plt.figure(figsize=(8, 6), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0], width_ratios=[2, 2, 3], wspace=0.38
    )
    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], width_ratios=[1, 1], wspace=0.38
    )
    axes = [
        fig.add_subplot(gs1[0]),
        fig.add_subplot(gs1[1]),
        fig.add_subplot(gs1[2]),
        fig.add_subplot(gs2[0]),
        fig.add_subplot(gs2[1]),
    ]

    sns.histplot(
        vivo["pre_exp_freq"].values,
        kde=True,
        stat="probability",
        bins=bins_fr,
        edgecolor="w",
        alpha=0.5,
        ax=axes[0],
        label=f"in-vivo (n={len(vivo)})",
        legend=True,
        color="blue",
    )
    sns.histplot(
        slice["pre_exp_freq"].values,
        kde=True,
        stat="probability",
        bins=bins_fr,
        edgecolor="w",
        alpha=0.5,
        ax=axes[0],
        label=f"slice (n={len(slice)})",
        legend=True,
        color="gray",
    )

    ylims = axes[0].get_ylim()
    axes[0].vlines(
        np.mean(vivo["pre_exp_freq"].values), 0, ylims[1], color="blue", ls="dashed"
    )
    axes[0].vlines(
        np.mean(slice["pre_exp_freq"].values), 0, ylims[1], color="k", ls="dashed"
    )

    ks_pval = ks_2samp(vivo["pre_exp_freq"], slice["pre_exp_freq"]).pvalue
    pmr.plot_bracket(
        axes[0],
        np.percentile(vivo["pre_exp_freq"], 25),
        np.percentile(vivo["pre_exp_freq"], 75),
        ylims[0] + 1.15 * (ylims[1] - ylims[0]),
        ylims[0] + 1.18 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = stats.ttest_ind(vivo["pre_exp_freq"], slice["pre_exp_freq"]).pvalue
    pmr.plot_bracket(
        axes[0],
        np.mean(vivo["pre_exp_freq"]),
        np.mean(slice["pre_exp_freq"]),
        ylims[1],
        ylims[0] + 1.03 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )

    axes[0].legend(fontsize="xx-small", fancybox=False, frameon=False)

    sns.histplot(
        vivo["pre_exp_cv"].values,
        kde=True,
        stat="probability",
        bins=bins_cv,
        edgecolor="w",
        alpha=0.5,
        ax=axes[1],
        legend=True,
        color="blue",
    )
    sns.histplot(
        slice["pre_exp_cv"].values,
        kde=True,
        stat="probability",
        bins=bins_cv,
        edgecolor="w",
        alpha=0.5,
        ax=axes[1],
        legend=True,
        color="gray",
    )
    ylims = axes[1].get_ylim()
    axes[1].vlines(
        np.mean(vivo["pre_exp_cv"].values), 0, ylims[1], color="blue", ls="dashed"
    )
    axes[1].vlines(
        np.mean(slice["pre_exp_cv"].values), 0, ylims[1], color="k", ls="dashed"
    )

    ks_pval = ks_2samp(vivo["pre_exp_cv"], slice["pre_exp_cv"]).pvalue
    pmr.plot_bracket(
        axes[1],
        np.percentile(vivo["pre_exp_cv"], 25),
        np.percentile(vivo["pre_exp_cv"], 75),
        ylims[0] + 1.15 * (ylims[1] - ylims[0]),
        ylims[0] + 1.18 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = stats.ttest_ind(vivo["pre_exp_cv"], slice["pre_exp_cv"]).pvalue
    pmr.plot_bracket(
        axes[1],
        np.mean(vivo["pre_exp_cv"]),
        np.mean(slice["pre_exp_cv"]),
        ylims[1],
        ylims[0] + 1.03 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )

    axes[0].set_xlabel("Pre-Stim Firing Rate", fontsize=8)
    axes[1].set_xlabel("Pre-Stim CV", fontsize=8)
    axes[0].set_ylabel("Probability", fontsize=8)
    axes[1].set_ylabel("Probability", fontsize=8)

    sns.scatterplot(
        vivo, x="pre_exp_freq", y="pre_exp_cv", ax=axes[2], color="blue", s=4, alpha=0.5
    )
    sns.scatterplot(
        slice,
        x="pre_exp_freq",
        y="pre_exp_cv",
        ax=axes[2],
        color="gray",
        s=4,
        alpha=0.5,
    )

    pmr.plot_linear_fit(vivo["pre_exp_freq"], vivo["pre_exp_cv"], axes[2], color="blue")
    pmr.plot_linear_fit(slice["pre_exp_freq"], slice["pre_exp_cv"], axes[2], color="k")

    axes[2].set_xlabel("Pre-Stim Firing Rate (Hz)", fontsize=8)
    axes[2].set_ylabel("Pre-Stim CV", fontsize=8)

    xlabels = [f"{x:.1f}" for x in bins_cv]
    ylabels = [int(x) for x in bins_fr]

    data = np.zeros((len(bins_fr), len(bins_cv)))
    for index, row in vivo.iterrows():
        i = np.searchsorted(bins_fr, row["pre_exp_freq"], side="right")
        j = np.searchsorted(bins_cv, row["pre_exp_cv"], side="right")

        data[i - 1, j - 1] += 1
    data = data / np.sum(data)
    ax1 = sns.heatmap(
        data=data,
        ax=axes[3],
        cmap="jet",
        xticklabels=xlabels,
        yticklabels=ylabels,
        mask=data == 0,
        annot=True,
        annot_kws={"fontsize": 4},
        fmt=".2f",
        vmin=0,
        vmax=0.25,
        linecolor="gray",
        linewidths=0.5,
    )

    data = np.zeros((len(bins_fr), len(bins_cv)))
    for index, row in slice.iterrows():
        i = np.searchsorted(bins_fr, row["pre_exp_freq"], side="right")
        j = np.searchsorted(bins_cv, row["pre_exp_cv"], side="right")

        data[i - 1, j - 1] += 1
    data = data / np.sum(data)
    ax2 = sns.heatmap(
        data=data,
        ax=axes[4],
        cmap="jet",
        xticklabels=xlabels,
        yticklabels=ylabels,
        mask=data == 0,
        annot=True,
        annot_kws={"fontsize": 4},
        fmt=".2f",
        vmin=0,
        vmax=0.25,
        linecolor="gray",
        linewidths=0.5,
    )

    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    axes[3].set_xticks(list(range(0, len(xlabels), 2)))
    axes[3].set_xticklabels([x for x in xlabels[::2]], rotation=0)
    axes[3].set_yticklabels(ylabels, rotation=0)
    axes[4].set_xticks(list(range(0, len(xlabels), 2)))
    axes[4].set_xticklabels([x for x in xlabels[::2]], rotation=0)
    axes[4].set_yticklabels(ylabels, rotation=0)
    axes[3].set_xlabel("Pre-Stim CV", fontsize=8)
    axes[3].set_ylabel("Pre-Stim Firing Rate (Hz)", fontsize=8)
    axes[4].set_xlabel("Pre-Stim CV", fontsize=8)
    axes[4].set_ylabel("Pre-Stim Firing Rate (Hz)", fontsize=8)
    axes[3].tick_params(axis="both", which="major", labelsize=2)
    axes[4].tick_params(axis="both", which="major", labelsize=2)

    axes[3].invert_yaxis()
    axes[4].invert_yaxis()

    makeNice(axes, labelsize=6)
    fig.savefig(f"{save_dir}/exp_fr_cv.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/exp_fr_cv.pdf")


def get_exp_results(vivo):
    if vivo:
        pulse_gpe, pulse_d1 = get_naive_pulsed_stim_data()
        cont_gpe, cont_d1 = get_naive_continuous_stim_data()
        return pd.concat([cont_gpe, cont_d1, pulse_gpe, pulse_d1], ignore_index=True)
    else:
        slice_gpe, slice_d1 = get_slice_data()
        return pd.concat([slice_gpe, slice_d1], ignore_index=True)


def compare_baseline_data_lengths(
    vivo,
    slice,
    baseline_slice,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/experimental_results",
    show=False,
):
    segment = 2
    vivo_recordings = np.zeros((len(vivo), 3))
    vivo_comparisons = np.zeros((len(vivo), 6))
    count = 0
    for index, row in vivo.iterrows():
        neuron = pickle.load(open(f"{row['cell_dir']}/neuron.obj", "rb"))
        light_on = np.loadtxt(f"{row['cell_dir']}/light_on.txt")

        spikes = neuron.spikes[neuron.spikes < light_on[0]]
        spikes_init = neuron.spikes[neuron.spikes < segment]
        spikes_final = neuron.spikes[
            (neuron.spikes >= light_on[0] - segment) & (neuron.spikes < light_on[0])
        ]
        freq_init = spikes_init.shape[0] / segment
        freq_final = spikes_final.shape[0] / segment
        cv_init = (
            np.std(np.diff(spikes_init)) / np.mean(np.diff(spikes_init))
            if np.mean(np.diff(spikes_init)) > 0
            else 0
        )
        cv_final = (
            np.std(np.diff(spikes_final)) / np.mean(np.diff(spikes_final))
            if np.mean(np.diff(spikes_final)) > 0
            else 0
        )
        vivo_comparisons[count] = [
            row["pre_exp_freq"],
            freq_init,
            freq_final,
            row["pre_exp_cv"],
            cv_init,
            cv_final,
        ]
        vivo_recordings[count, :] = [
            light_on[0],
            row["pre_exp_freq"],
            row["pre_exp_cv"],
        ]

        count += 1

    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]
    sns.scatterplot(
        x=np.ones(len(slice)) * 2,
        y=slice["pre_exp_freq"],
        label=f"slice 2s and 15s (n={len(slice)})",
        ax=axes[0],
        s=10,
    )
    """
    sns.scatterplot(
        x=np.ones(len(baseline_slice)) * 15,
        y=baseline_slice["pre_exp_freq"],
        label=f"slice 15s (n={len(baseline_slice)})",
        ax=axes[0],
        s=10,
    )
    """
    sns.scatterplot(
        x=vivo_recordings[:, 0],
        y=vivo_recordings[:, 1],
        label=f"in-vivo (n={len(vivo)})",
        ax=axes[0],
        s=10,
    )
    pmr.plot_linear_fit(
        vivo_recordings[:, 0], vivo_recordings[:, 1], axes[0], color="orange"
    )

    sns.scatterplot(
        x=np.ones(len(slice)) * 2,
        y=slice["pre_exp_cv"],
        label=f"slice 2s and 15s (n={len(slice)})",
        ax=axes[1],
        s=10,
    )
    """
    sns.scatterplot(
        x=np.ones(len(baseline_slice)) * 15,
        y=baseline_slice["pre_exp_cv"],
        label=f"slice 15s (n={len(baseline_slice)})",
        ax=axes[1],
        s=10,
    )
    """
    sns.scatterplot(
        x=vivo_recordings[:, 0],
        y=vivo_recordings[:, 2],
        label=f"in-vivo (n={len(vivo)})",
        ax=axes[1],
        s=10,
    )
    axes[1].set_ylim([0, 2])
    pmr.plot_linear_fit(
        vivo_recordings[:, 0], vivo_recordings[:, 2], axes[1], color="orange"
    )

    axes[0].legend(fancybox=False, frameon=False, fontsize=8)
    axes[1].legend(fancybox=False, frameon=False, fontsize=8)

    bins_fr = np.arange(0, 150, 10)
    bins_cv = np.arange(0, 2, 0.2)

    sns.histplot(
        slice["pre_exp_freq"],
        bins=bins_fr,
        ax=axes[2],
        kde=True,
        stat="probability",
        label=f"slice 2s and 15s (n={len(slice)})",
        edgecolor="w",
    )
    """
    sns.histplot(
        baseline_slice["pre_exp_freq"],
        bins=bins_fr,
        ax=axes[2],
        kde=True,
        stat="probability",
        label=f"slice 15s (n={len(baseline_slice)})",
        edgecolor="w",
    )
    """
    sns.histplot(
        vivo["pre_exp_freq"],
        bins=bins_fr,
        ax=axes[2],
        kde=True,
        stat="probability",
        label=f"in-vivo (n={len(vivo)})",
        edgecolor="w",
    )

    sns.histplot(
        slice["pre_exp_cv"],
        bins=bins_cv,
        ax=axes[3],
        kde=True,
        stat="probability",
        label=f"slice 2s and 15s (n={len(slice)})",
        edgecolor="w",
    )
    """
    sns.histplot(
        baseline_slice["pre_exp_cv"],
        bins=bins_cv,
        ax=axes[3],
        kde=True,
        stat="probability",
        label=f"slice 15s (n={len(baseline_slice)})",
        edgecolor="w",
    )
    """
    sns.histplot(
        vivo["pre_exp_cv"],
        bins=bins_cv,
        ax=axes[3],
        kde=True,
        stat="probability",
        label=f"in-vivo (n={len(vivo)})",
        edgecolor="w",
    )
    axes[3].set_xlim([0, 2])
    axes[2].legend(fancybox=False, frameon=False, fontsize=8)
    axes[3].legend(fancybox=False, frameon=False, fontsize=8)

    sns.histplot(
        vivo_comparisons[:, 0],
        bins=bins_fr,
        kde=True,
        stat="probability",
        ax=axes[4],
        label=f"in-vivo (n={len(vivo)})",
        edgecolor="w",
    )
    sns.histplot(
        vivo_comparisons[:, 1],
        bins=bins_fr,
        kde=True,
        stat="probability",
        ax=axes[4],
        label=f"in-vivo first {segment}s (n={len(vivo)})",
        edgecolor="w",
    )
    sns.histplot(
        vivo_comparisons[:, 2],
        bins=bins_fr,
        kde=True,
        stat="probability",
        ax=axes[4],
        label=f"in-vivo last {segment}s (n={len(vivo)})",
        edgecolor="w",
    )

    sns.histplot(
        vivo_comparisons[:, 3],
        bins=bins_cv,
        kde=True,
        stat="probability",
        ax=axes[5],
        label=f"in-vivo (n={len(vivo)})",
        edgecolor="w",
    )
    sns.histplot(
        vivo_comparisons[:, 4],
        bins=bins_cv,
        kde=True,
        stat="probability",
        ax=axes[5],
        label=f"in-vivo first {segment}s (n={len(vivo)})",
        edgecolor="w",
    )
    sns.histplot(
        vivo_comparisons[:, 5],
        bins=bins_cv,
        kde=True,
        stat="probability",
        ax=axes[5],
        label=f"in-vivo last {segment}s (n={len(vivo)})",
        edgecolor="w",
    )
    axes[5].set_xlim([0, 2])
    axes[4].legend(fancybox=False, frameon=False, fontsize=8)
    axes[5].legend(fancybox=False, frameon=False, fontsize=8)

    axes[0].set_xlabel("Baseline Recording Time (s)", fontsize=8)
    axes[1].set_xlabel("Baseline Recording Time (s)", fontsize=8)
    axes[2].set_xlabel("Baseline Firing Rate (Hz)", fontsize=8)
    axes[3].set_xlabel("Baseline CV", fontsize=8)
    axes[4].set_xlabel("Baseline Recording Time (s)", fontsize=8)
    axes[5].set_xlabel("Baseline Recording Time (s)", fontsize=8)

    axes[0].set_ylabel("Baselien Firing Rate (Hz)", fontsize=8)
    axes[1].set_ylabel("Baselien CV", fontsize=8)

    for i in range(2, 6):
        axes[i].set_ylabel("Probability", fontsize=8)

    makeNice(axes, labelsize=6)
    add_fig_labels(axes)

    """
    print(ks_2samp(slice["pre_exp_cv"], baseline_slice["pre_exp_cv"]).pvalue)
    print(ks_2samp(vivo_comparisons[:, 0], vivo_comparisons[:, 1]).pvalue)
    print(ks_2samp(vivo_comparisons[:, 0], vivo_comparisons[:, 2]).pvalue)
    print(ks_2samp(vivo_comparisons[:, 1], vivo_comparisons[:, 2]).pvalue)

    print(ks_2samp(vivo_comparisons[:, 3], vivo_comparisons[:, 4]).pvalue)
    print(ks_2samp(vivo_comparisons[:, 3], vivo_comparisons[:, 5]).pvalue)
    print(ks_2samp(vivo_comparisons[:, 4], vivo_comparisons[:, 5]).pvalue)
    """

    fig.savefig(f"{save_dir}/stats_recording_length.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open {save_dir}/stats_recording_length.pdf")
