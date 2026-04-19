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
from scipy import optimize
from scipy import signal
import math

# import user modules
sys.path.append("../")
from helpers import *
import experimental_analysis as expan
import network_analysis as na
import run_model
import poisson_surprise

################################################################################
#########   SIMULATION METHODS    ##############################################
################################################################################

types = [
    "no effect",
    "complete inhibition",
    "partial inhibition",
    "adapting inhibition",
    "excitation",
    "biphasic IE",
    "biphasic EI",
]

types_abbrev = ["NE", "CI", "PI", "AI", "EX", "BPIE", "BPEI"]

color_dict = {
    "complete inhibition": "navy",
    "adapting inhibition": "cornflowerblue",
    "partial inhibition": "blue",
    "no effect": "slategrey",
    "excitation": "lightcoral",
    "biphasic IE": "blueviolet",
    "biphasic EI": "orchid",
}


def response_squared_error(df1, df2, remap=True):
    df1_results = get_streac_responses(df1)
    df2_results = get_streac_responses(df2)
    if remap:
        df1_results[2] += df1_results[-2]
        df1_results[4] += df1_results[-1]
        df1_results = df1_results[:5]
        df2_results[2] += df2_results[-2]
        df2_results[4] += df2_results[-1]
        df2_results = df2_results[:5]
    response_errors = [
        (np.abs(df1_results[i] - df2_results[i]) / df1_results[i]) ** 2
        for i in range(df1_results.shape[0])
    ]
    excite_error = response_errors[4]
    inhibit_error = np.sum(
        response_errors[1:4]
    )  # (np.sum(df1_results[1:4]) - np.sum(df2_results[1:4])) ** 2
    total_error = np.sum(
        response_errors
    )  # (np.sum(df1_results[1:4]) - np.sum(df2_results[1:4])) ** 2
    return response_errors, inhibit_error, excite_error, total_error


def contingency_table(x1, x2):
    table = np.asarray([x1, x2])
    cols_to_keep = []
    for i in range(len(x1)):
        if x1[i] + x2[i] != 0:
            cols_to_keep.append(i)
    return table[:, cols_to_keep]


def run_chi_squared(df1, df2):
    df1_results = get_streac_responses(df1, percent=False)
    df2_results = get_streac_responses(df2, percent=False)
    chi2, pval, _, _ = stats.chi2_contingency(
        contingency_table(df1_results, df2_results)
    )
    return chi2, pval


def get_streac_responses(df, percent=True, remap=True):
    df_results = np.zeros(len(types))
    count = 0
    for type in types:
        df_results[count] = len(df[df["neural_response"] == type])
        if percent:
            df_results[count] /= len(df)
        count += 1
    if remap:
        df_results[2] += df_results[-2]
        df_results[4] += df_results[-1]
        df_results = df_results[:5]
    return df_results


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def plot_bar_responses(df, axes, placement):
    count = 0
    responses = np.zeros(len(types))
    for type in types:
        responses[count] = len(df[df["neural_response"] == type]) * 100 / len(df)
        axes.bar(
            placement,
            responses[count],
            bottom=np.sum(responses[0:count]) if count > 0 else 0,
            color=color_dict[type],
        )
        count += 1
    return responses


def generate_modulation_factors(save_dir, params, T0, T, d1stim, gpestim, size=100):
    stim_times = (
        np.arange(
            params["str_start_time"][0] + params["str_base_length"][0],
            T,
            params["str_base_length"][0]
            + params["str_stim_length"][0]
            + params["str_post_length"][0],
        )
        if d1stim
        else np.arange(
            params["gpe_start_time"][0] + params["gpe_base_length"][0],
            T,
            params["gpe_base_length"][0]
            + params["gpe_stim_length"][0]
            + params["gpe_post_length"][0],
        )
    )

    """stim_times = (
        np.arange(
            params["str_start_time"] + params["str_base_length"],
            T,
            params["str_base_length"]
            + params["str_stim_length"]
            + params["str_post_length"],
        )
        if d1stim
        else np.arange(
            params["gpe_start_time"] + params["gpe_base_length"],
            T,
            params["gpe_base_length"]
            + params["gpe_stim_length"]
            + params["gpe_post_length"],
        )
    )"""

    modulation_factors = np.zeros((size, 2))
    for i in range(size):
        if os.path.getsize(f"{save_dir}/Neuron_{i}/spike_times.txt") > 0:
            spikes = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt")
            base = len(spikes[(spikes >= T0) & (spikes < stim_times[0])]) / (
                stim_times[0] - T0
            )
            for trial in range(len(stim_times)):
                pre_stim = (
                    len(
                        spikes[
                            (
                                spikes
                                >= stim_times[trial]
                                - params[
                                    "str_stim_length" if d1stim else "gpe_stim_length"
                                ][0]
                            )
                            & (spikes < stim_times[trial])
                        ]
                    )
                    / params["str_base_length" if d1stim else "gpe_base_length"][0]
                )
                stim = (
                    len(
                        spikes[
                            (spikes >= stim_times[trial])
                            & (
                                spikes
                                < stim_times[trial]
                                + params[
                                    "str_stim_length" if d1stim else "gpe_stim_length"
                                ][0]
                            )
                        ]
                    )
                    / params["str_base_length" if d1stim else "gpe_base_length"][0]
                )

                modulation_factors[i] += [
                    base / len(stim_times),
                    (
                        ((stim - pre_stim) / (stim + pre_stim)) / len(stim_times)
                        if (stim + pre_stim) > 0
                        else 0
                    ),
                ]
        else:
            modulation_factors[i] = [0, 0]
    return modulation_factors


def plot_modulation_factors(
    save_dir, exp_df, params, T0, T, d1stim, gpestim, size=100, show=True
):
    modulation_factors = generate_modulation_factors(
        save_dir, params, T0, T, d1stim, gpestim, size=size
    )
    exp_mfs = expan.get_modulation_factors(exp_df)

    excitation_threshold = np.percentile(
        exp_mfs[exp_df["neural_response"] == "excitation"], 25
    )
    inhibition_threshold = np.percentile(
        exp_mfs[exp_df["neural_response"].str.contains("inhibition")], 75
    )

    """excitation_threshold = np.mean(exp_mfs[exp_df["neural_response"] == "excitation"])
    inhibition_threshold = np.mean(
        exp_mfs[exp_df["neural_response"].str.contains("inhibition")]
    )

    plt.figure()
    for i in range(len(exp_mfs)):
        plt.scatter(
            i / len(exp_mfs),
            exp_mfs[i],
            color=color_dict[exp_df.iloc[i, exp_df.columns.get_loc("neural_response")]],
        )"""

    exp_area = np.trapz(sorted(exp_mfs), x=np.arange(len(exp_mfs)) / len(exp_mfs))
    sim_area = np.trapz(sorted(modulation_factors[:, 1]), x=np.arange(size) / size)
    # print(exp_area, sim_area, exp_area - sim_area)

    x_exp = np.arange(len(exp_mfs)) / len(exp_mfs)
    x_sim = np.arange(size) / size
    sorted_exp_mfs = sorted(exp_mfs)
    sorted_sim_mfs = sorted(modulation_factors[:, 1])

    mf_excite_diff = (
        np.abs(
            np.min(x_exp[sorted_exp_mfs >= excitation_threshold])
            - np.min(x_sim[sorted_sim_mfs >= excitation_threshold])
        )
        if len(x_sim[sorted_sim_mfs >= excitation_threshold]) > 0
        else sorted_sim_mfs[-1]
    )

    mf_inhibit_diff = (
        np.abs(
            np.min(x_exp[sorted_exp_mfs >= inhibition_threshold])
            - np.min(x_sim[sorted_sim_mfs >= inhibition_threshold])
        )
        if len(x_sim[sorted_sim_mfs >= inhibition_threshold]) > 0
        else sorted_sim_mfs[0]
    )

    """plt.hlines(excitation_threshold, 0, 1, color="k", ls="dotted", lw=1)
    plt.hlines(inhibition_threshold, 0, 1, color="k", ls="dashed", lw=1)
    plt.ylim([-1, 1])
    plt.xlim([0, 1])
    plt.xlabel("Cell Number")
    plt.ylabel("Modulation Factor")
    print(excitation_threshold, inhibition_threshold)
    plt.show()
    sys.exit()"""
    index_switches = [[0, "exp" if sorted_exp_mfs[0] > sorted_sim_mfs[0] else "sim"]]
    exp_indices = [0]

    for i in range(1, size):
        jexp = np.where(x_exp < x_sim[i])[0][-1]
        # print(x_exp[jexp], x_exp[jexp + 1], x_sim)
        # print(sorted_sim_mfs[i], sorted_exp_mfs[jexp])
        index_switches.append(
            [i, "exp" if sorted_exp_mfs[jexp] > sorted_sim_mfs[i] else "sim"]
        )
        exp_indices.append(jexp)

    area = 0
    end = 0
    start = 0
    larger = index_switches[0][1]
    inf_norm = np.max(
        [
            np.abs(sorted_sim_mfs[i] - sorted_exp_mfs[exp_indices[i]])
            for i in range(size)
        ]
    )
    while end < size - 1:
        switch = False
        while not switch and end < size - 1:
            if index_switches[end][1] != larger:
                switch = True
            elif end == size - 1:
                switch = True
            else:
                end += 1

        area_sim = np.trapz(sorted_sim_mfs[start:end], x=x_sim[start:end])
        area_exp = np.trapz(
            sorted_exp_mfs[exp_indices[start] : exp_indices[end]],
            x=x_exp[exp_indices[start] : exp_indices[end]],
        )
        area += (
            area_sim - area_exp
            if index_switches[start][1] == "sim"
            else area_exp - area_sim
        )
        if end >= 100:
            break
        larger = index_switches[end][1]
        start = end

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    axes[0].plot(
        np.arange(size) / size, sorted(modulation_factors[:, 1]), color="b", label="sim"
    )
    axes[0].plot(
        np.arange(exp_mfs.shape[0]) / exp_mfs.shape[0],
        sorted(exp_mfs),
        color="k",
        label="exp",
    )
    axes[0].hlines(0, 0, 1, lw=0.5, ls="dashed", color="gray")
    axes[0].set_xlabel("Cell Num")
    axes[0].set_ylabel("Modulation Factor")
    axes[0].set_ylim([-1, 1])
    axes[0].legend(fancybox=False, frameon=False)

    sns.histplot(
        exp_mfs,
        bins=np.arange(-1, 1, 0.1),
        color="black",
        edgecolor="w",
        kde=True,
        stat="probability",
        ax=axes[1],
    )
    sns.histplot(
        modulation_factors[:, 1],
        bins=np.arange(-1, 1, 0.1),
        color="b",
        edgecolor="w",
        kde=True,
        stat="probability",
        ax=axes[1],
    )
    ylims = axes[1].get_ylim()
    ks_pval = ks_2samp(modulation_factors[:, 1], exp_mfs).pvalue
    plot_bracket(
        axes[1],
        np.percentile(exp_mfs, 25),
        np.percentile(exp_mfs, 75),
        ylims[0] + 1.15 * (ylims[1] - ylims[0]),
        ylims[0] + 1.18 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = stats.ttest_ind(modulation_factors[:, 1], exp_mfs).pvalue
    plot_bracket(
        axes[1],
        np.mean(modulation_factors[:, 1]),
        np.mean(exp_mfs),
        ylims[1],
        ylims[0] + 1.03 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )
    axes[1].set_xlim([-1, 1])
    axes[1].set_xlabel("Modulation Factor")

    sns.histplot(
        params["W_str"] if d1stim else params["W_gpe"],
        color="k",
        edgecolor="w",
        kde=True,
        stat="probability",
        ax=axes[2],
    )
    axes[2].set_xlabel("$W_{STR}$" if d1stim else "$W_{GPe}$")
    axes[2].vlines(
        np.mean(params["W_str"] if d1stim else params["W_gpe"]),
        axes[2].get_ylim()[0],
        axes[2].get_ylim()[1],
        ls="dashed",
        color="k",
    )

    sns.scatterplot(
        x=params["W_str"] if d1stim else params["W_gpe"],
        y=params["tausyn_dend"] if d1stim else params["tausyn"],
        hue=modulation_factors[:, 1],
        hue_norm=(-1, 1),
        size=modulation_factors[:, 1],
        palette="coolwarm",
        legend=False,
        ax=axes[3],
    )
    axes[3].set_xlabel("$W_{STR}$" if d1stim else "$W_{GPe}$")
    axes[3].set_ylabel("$\\tau^D_{syn}$" if d1stim else "$\\tau^S_{syn}$")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1, 1))
    cbar = axes[3].figure.colorbar(sm, ax=axes[3])
    cbar.ax.tick_params(labelsize=6)

    makeNice(axes)
    fig.savefig(f"{save_dir}/modulation_factors.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open {save_dir}/modulation_factors.pdf")
    return ks_pval, ttest_pval, area, inf_norm, mf_excite_diff, mf_inhibit_diff


def get_neuron_meta_data(save_dir, size=100):
    meta_data = {}
    for i in range(size):
        with open(f"{save_dir}/Neuron_{i}/meta_data.txt") as meta_file:
            for line in meta_file.readlines():
                key, val = line.split(":")
                # print(i, key, val)
                if key not in meta_data.keys():

                    meta_data[key] = [float(val)] if val.lower() != "nan" else 0
                else:
                    meta_data[key].append(float(val) if val.lower() != "nan" else 0)
    return meta_data


def plot_egaba_with_stim(save_dir, T, T0, size=100, show=True):
    v = -55
    egaba_avg = np.zeros(size)
    meta_data = get_neuron_meta_data(save_dir, size=size)

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

    N = 100
    gtonvals = np.linspace(1e-5, np.max(meta_data["gTON_CL_S_MEAN_nS_pF"]) * 1.1, N)
    gkcc2vals = np.linspace(1e-5, np.max(meta_data["gKCC2_S_nS_pF"]) * 1.1, N)
    egabas = np.zeros((N, N))

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

    idx = np.round(np.linspace(0, N - 1, 5)).astype(int)

    labels = [f"{gkcc2vals[x]:.03f}" for x in idx]
    gtonlabels = [f"{gtonvals[x]:.03f}" for x in idx]

    df = pd.read_csv(f"{save_dir}/processed/all_data.csv")
    scatter_colors = [color_dict[x] for x in df["neural_response"]]

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    sns.scatterplot(x=np.arange(size), y=egaba_avg, ax=axes[0], color=scatter_colors)
    axes[0].set_ylim([-90, -40])
    hm2 = axes[1].imshow(egabas, cmap="gnuplot", aspect="auto")
    cs = axes[1].contour(
        egabas, levels=[-90, -85, -80, -75, -70, -65, -60, -55, -50, -45], colors="w"
    )

    cs_labels = axes[1].clabel(cs, inline=1, fontsize=8, inline_spacing=50)
    for label in cs_labels:
        label.set_rotation(0)
    cbar = fig.colorbar(hm2, ax=axes[1], location="right", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    sns.scatterplot(
        y=meta_data["gKCC2_S_nS_pF"] / (gkcc2vals[-1] - gkcc2vals[0]) * 100,
        x=meta_data["gTON_CL_S_MEAN_nS_pF"] / (gtonvals[-1] - gtonvals[0]) * 100,
        ax=axes[1],
        color=scatter_colors,
        legend=False,
    )

    idx = np.round(np.linspace(0, N - 1, 5)).astype(int)
    labels = [f"{gkcc2vals[x]:.03f}" for x in idx]
    gtonlabels = [f"{gtonvals[x]:.03f}" for x in idx]

    axes[0].set_xlabel("Cell ID")
    axes[0].set_ylabel("Baseline $E_{GABA}$ (mV)")
    axes[1].set_xlabel("$g_{tonic}$ (nS/pF)")
    axes[1].set_ylabel("$g_{KCC2}$ (nS/pF)")
    axes[1].set_xticks(idx)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(labels, rotation=0)
    axes[1].set_xticklabels(gtonlabels, rotation=0)
    axes[1].invert_yaxis()

    adjacency, connections, total_weight = find_connections(size, save_dir)
    input_deltaF = np.zeros(size)
    for i in range(size):
        connections_to_i = adjacency[:, i]
        connect_ids = [k for k in range(size) if connections_to_i[k] > 0]
        input_deltaF[i] = (
            np.mean(df["stim_freq"][connect_ids] / df["baseline_freq"][connect_ids])
            if len(connect_ids) > 0
            else 0
        )

    scatter_colors = [color_dict[x] for x in df["neural_response"]]
    scatter_colors = [
        scatter_colors[i] for i in range(size) if not np.isnan(input_deltaF[i])
    ]

    count = 0
    for type in types:
        data = input_deltaF[df[df["neural_response"] == type]["cell_num"] - 1]
        # axes[2].scatter(count * np.ones(len(data)), data, color="gray", s=10)
        axes[2].scatter(
            count, np.mean(data), color=color_dict[type], edgecolor="k", zorder=15
        )
        _, caplines, _ = axes[2].errorbar(
            count, np.mean(data), yerr=stats.sem(data), color="k", capsize=10
        )
        # caplines[0].set_marker("_")
        # caplines[0].set_markersize(10)
        count += 1
    axes[2].set_xticks(list(range(count)))
    axes[2].set_xticklabels(types_abbrev)
    axes[2].set_ylabel("Input $\\Delta$ Norm FR")
    axes[2].hlines(1, 0, count, color="gray", linestyle="dashed", lw=0.5)

    count = 0
    for type in types:
        data = connections[df[df["neural_response"] == type]["cell_num"] - 1]

        axes[3].bar(
            count,
            np.mean(data),
            color=color_dict[type],
            edgecolor="k",
        )
        _, caplines, _ = axes[3].errorbar(
            count,
            np.mean(data),
            yerr=stats.sem(data),
            lolims=True,
            uplims=False,
            color="k",
        )
        caplines[0].set_marker("_")
        caplines[0].set_markersize(10)
        count += 1
    axes[3].set_xticks(list(range(count)))
    axes[3].set_xticklabels(types_abbrev)
    axes[3].set_ylabel("Connections")

    makeNice(axes)
    fig.savefig(f"{save_dir}/egaba_results.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open {save_dir}/egaba_results.pdf")


def plot_egaba(save_dir, T, T0, size=100, show=True):
    v = -55
    egaba_avg = np.zeros(size)
    meta_data = get_neuron_meta_data(save_dir, size=size)
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

    N = 100
    gtonvals = np.linspace(1e-5, np.max(meta_data["gTON_CL_S_MEAN_nS_pF"]) * 1.1, N)
    gkcc2vals = np.linspace(1e-5, np.max(meta_data["gKCC2_S_nS_pF"]) * 1.1, N)
    egabas = np.zeros((N, N))

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

    idx = np.round(np.linspace(0, N - 1, 5)).astype(int)

    labels = [f"{gkcc2vals[x]:.03f}" for x in idx]
    gtonlabels = [f"{gtonvals[x]:.03f}" for x in idx]

    adjacency, connections, total_weight = find_connections(size, save_dir)

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    sns.scatterplot(x=meta_data["W_gpe"], y=egaba_avg, ax=axes[0], color="k")
    ax1 = axes[0].twinx()
    sns.scatterplot(x=meta_data["W_gpe"], y=total_weight, ax=ax1, color="gray")
    ax1.set_xlabel("$W_{SNr}^{Total}$", color="gray")
    axes[0].set_xlabel("$W_{GPe}$")
    axes[0].set_ylabel("$E_{GABA}^S$ (mV)")
    hm2 = axes[1].imshow(egabas, cmap="gnuplot", aspect="auto")
    cs = axes[1].contour(
        egabas, levels=[-90, -85, -80, -75, -70, -65, -60, -55, -50, -45], colors="w"
    )

    cs_labels = axes[1].clabel(cs, inline=1, fontsize=8, inline_spacing=50)
    for label in cs_labels:
        label.set_rotation(0)
    cbar = fig.colorbar(hm2, ax=axes[1], location="right", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    sns.scatterplot(
        y=meta_data["gKCC2_S_nS_pF"] / (gkcc2vals[-1] - gkcc2vals[0]) * 100,
        x=meta_data["gTON_CL_S_MEAN_nS_pF"] / (gtonvals[-1] - gtonvals[0]) * 100,
        ax=axes[1],
        color="k",
        legend=False,
    )

    idx = np.round(np.linspace(0, N - 1, 5)).astype(int)
    labels = [f"{gkcc2vals[x]:.02e}" for x in idx]
    gtonlabels = [f"{gtonvals[x]:.02e}" for x in idx]
    axes[1].set_xlabel("$g_{tonic}^S$ (nS/pF)")
    axes[1].set_ylabel("$g_{KCC2}^S$ (nS/pF)")
    axes[1].set_xticks(idx)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(labels, rotation=0)
    axes[1].set_xticklabels(gtonlabels, rotation=0)
    axes[1].invert_yaxis()

    sns.scatterplot(
        x=connections,
        y=egaba_avg,
        hue=total_weight,
        size=total_weight,
        ax=axes[2],
        legend=False,
    )
    plot_linear_fit(connections, egaba_avg, axes=axes[2], color="k", scatter=False)
    axes[2].set_xlabel("Connections")
    axes[2].set_ylabel("$E_{GABA}^S$ (mV)")

    sns.scatterplot(
        x=total_weight,
        y=egaba_avg,
        hue=connections,
        size=connections,
        ax=axes[3],
        legend=False,
    )
    plot_linear_fit(total_weight, egaba_avg, axes=axes[3], color="k", scatter=False)
    axes[3].set_xlabel("$W_{SNr}^{total}$")
    axes[3].set_ylabel("$E_{GABA}^S$ (mV)")

    makeNice(axes[1:])
    fig.savefig(f"{save_dir}/egaba_results_soma.pdf", bbox_inches="tight")
    plt.close()

    v = -55
    egaba_avg = np.zeros(size)
    meta_data = get_neuron_meta_data(save_dir, size=size)
    for i in range(size):
        egaba_avg[i] = run_model.chloride_to_egaba(
            optimize.newton(
                run_model.chloride_dynamics_ss,
                10,
                args=(
                    meta_data["gKCC2_D_nS_pF"][i],
                    meta_data["gTON_CL_D_MEAN_nS_pF"][i],
                    v,
                ),
            )
        )

    N = 100
    gtonvals = np.linspace(1e-5, np.max(meta_data["gTON_CL_D_MEAN_nS_pF"]) * 1.1, N)
    gkcc2vals = np.linspace(1e-5, np.max(meta_data["gKCC2_D_nS_pF"]) * 1.1, N)
    egabas = np.zeros((N, N))

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

    idx = np.round(np.linspace(0, N - 1, 5)).astype(int)

    labels = [f"{gkcc2vals[x]:.03f}" for x in idx]
    gtonlabels = [f"{gtonvals[x]:.03f}" for x in idx]

    adjacency, connections, total_weight = find_connections(size, save_dir)

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    sns.scatterplot(x=meta_data["W_str"], y=egaba_avg, ax=axes[0], color="k")
    ax1 = axes[0].twinx()
    sns.scatterplot(x=meta_data["W_str"], y=total_weight, ax=ax1, color="gray")
    ax1.set_xlabel("$W_{SNr}^{Total}$")
    axes[0].set_xlabel("$W_{STR}$")
    axes[0].set_ylabel("$E_{GABA}^D$ (mV)")
    hm2 = axes[1].imshow(egabas, cmap="gnuplot", aspect="auto")
    cs = axes[1].contour(
        egabas, levels=[-90, -85, -80, -75, -70, -65, -60, -55, -50, -45], colors="w"
    )

    cs_labels = axes[1].clabel(cs, inline=1, fontsize=8, inline_spacing=50)
    for label in cs_labels:
        label.set_rotation(0)
    cbar = fig.colorbar(hm2, ax=axes[1], location="right", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    sns.scatterplot(
        y=meta_data["gKCC2_D_nS_pF"] / (gkcc2vals[-1] - gkcc2vals[0]) * 100,
        x=meta_data["gTON_CL_D_MEAN_nS_pF"] / (gtonvals[-1] - gtonvals[0]) * 100,
        ax=axes[1],
        color="k",
        legend=False,
    )

    idx = np.round(np.linspace(0, N - 1, 5)).astype(int)
    labels = [f"{gkcc2vals[x]:.02e}" for x in idx]
    gtonlabels = [f"{gtonvals[x]:.02e}" for x in idx]
    axes[1].set_xlabel("$g_{tonic}^D$ (nS/pF)")
    axes[1].set_ylabel("$g_{KCC2}^D$ (nS/pF)")
    axes[1].set_xticks(idx)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(labels, rotation=0)
    axes[1].set_xticklabels(gtonlabels, rotation=0)
    axes[1].invert_yaxis()

    sns.scatterplot(
        x=connections,
        y=egaba_avg,
        hue=total_weight,
        size=total_weight,
        ax=axes[2],
        legend=False,
    )
    plot_linear_fit(connections, egaba_avg, axes=axes[2], color="k", scatter=False)
    axes[2].set_xlabel("Connections")
    axes[2].set_ylabel("$E_{GABA}^D$ (mV)")

    sns.scatterplot(
        x=total_weight,
        y=egaba_avg,
        hue=connections,
        size=connections,
        ax=axes[3],
        legend=False,
    )
    plot_linear_fit(total_weight, egaba_avg, axes=axes[3], color="k", scatter=False)
    axes[3].set_xlabel("$W_{SNr}^{total}$")
    axes[3].set_ylabel("$E_{GABA}^D$ (mV)")

    makeNice(axes[1:])
    fig.savefig(f"{save_dir}/egaba_results_dendrite.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open {save_dir}/egaba_results_dendrite.pdf")
        run_cmd(f"open {save_dir}/egaba_results_soma.pdf")


def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def plot_xcorr(save_dir, T, T0, size=100):
    for neuron_cell in range(size):  # 21
        connections = np.loadtxt(f"{save_dir}/weights.txt")
        connections = connections[:, neuron_cell]
        connect_ids = [i for i in range(size) if connections[i] > 0]
        x = np.loadtxt(f"{save_dir}/Neuron_{neuron_cell}/cell_dynamics.txt")
        fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
        x = x[(x[:, 0] >= T0) & (x[:, 0] < T), :]
        axes = [ax[i, j] for i in range(2) for j in range(2)]
        axes[0].plot(
            x[:, 0], normalize_signal(x[:, 1]), color="b", lw=0.5, label="$V_S$ (mV)"
        )
        yy = []
        inputs = np.zeros((len(connect_ids), x.shape[0]))
        input_count = 0
        for i in connect_ids:
            y = np.loadtxt(f"{save_dir}/Neuron_{i}/cell_dynamics.txt")
            ys = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt")
            y = y[(y[:, 0] >= T0) & (y[:, 0] < T), :]
            inputs[input_count, :] = y[:, 1]
            ys = ys[(ys >= T0) & (ys < T)]
            yy.extend(ys)
            input_count += 1
            axes[2].plot(y[:, 0], normalize_signal(y[:, 1]), lw=0.5)
            xcor = signal.correlate(
                normalize_signal(x[:, 1]), normalize_signal(y[:, 1])
            )
            xcor /= np.max(xcor)
            pks, _ = signal.find_peaks(xcor)
            xcor_max = np.max(xcor[pks])
            mx_pks = pks[xcor[pks] == xcor_max]
            lags = signal.correlation_lags(len(x[:, 1]), len(y[:, 1])) / (1000 / 0.05)
            axes[1].plot(lags, xcor, lw=0.5)
            axes[1].scatter(lags[mx_pks], xcor_max, marker="x")
            axes[3].scatter(lags[mx_pks], xcor_max, marker="x")
        axes[0].set_ylabel("Norm $V_S$", fontsize=8)
        axes[1].set_ylabel("Cross Correlation", fontsize=8)
        axes[2].set_ylabel("Norm $V_i$", fontsize=8)
        axes[3].set_ylabel("Peak Cross Corr", fontsize=8)

        makeNice(axes, labelsize=8)
        add_fig_labels(axes)
        fig.savefig(f"{save_dir}/Neuron_{neuron_cell}/xcorr.pdf", bbox_inches="tight")
        plt.close()


def plot_stim_trials(save_dir, T, T0, size=100):
    for neuron_cell in range(size):  # [21]:  # range(size):
        connections = np.loadtxt(f"{save_dir}/weights.txt")
        connections = connections[:, neuron_cell]
        connect_ids = [i for i in range(size) if connections[i] > 0]
        x = np.loadtxt(f"{save_dir}/Neuron_{neuron_cell}/cell_dynamics.txt")
        fig, ax = plt.subplots(4, 1, figsize=(12, 6), dpi=300, tight_layout=True)
        x = x[(x[:, 0] >= T0) & (x[:, 0] < T), :]
        axes = [ax[i] for i in range(4)]
        axes[0].plot(x[:, 0], x[:, 1], color="b", lw=0.5, label="$V_S$ (mV)")
        axes[2].plot(x[:, 0], x[:, 6], color="b", lw=0.5, label="$g_{GABA}^{GPe}$")
        axes[2].plot(x[:, 0], x[:, 7], color="r", lw=0.5, label="$g_{GABA}^{STR}$")
        axes[3].plot(x[:, 0], x[:, 11], lw=0.5, color="b", label="$E_{GABA}^S$")
        axes[3].plot(x[:, 0], x[:, 12], lw=0.5, color="r", label="$E_{GABA}^D$")
        axes[3].set_ylim([-80, -40])
        yy = []
        inputs = np.zeros((len(connect_ids), x.shape[0]))
        input_count = 0
        for i in connect_ids:
            y = np.loadtxt(f"{save_dir}/Neuron_{i}/cell_dynamics.txt")
            ys = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt")
            y = y[(y[:, 0] >= T0) & (y[:, 0] < T), :]
            inputs[input_count, :] = y[:, 1]
            ys = ys[(ys >= T0) & (ys < T)]
            yy.extend(ys)
            input_count += 1
            axes[1].plot(y[:, 0], y[:, 1], lw=0.5)
        for i in range(len(axes)):
            if i != 1:
                axes[i].legend(
                    fancybox=False,
                    frameon=True,
                    # mode="expand",
                    ncol=3,
                    fontsize=6,
                    bbox_to_anchor=(0, 1.02, 1, 0.2),
                    loc="lower left",
                    edgecolor="k",
                )
        axes[0].set_ylabel("$V_S$", fontsize=8)
        axes[1].set_ylabel("$V_i$", fontsize=8)
        axes[2].set_ylabel("$g_{GABA}$", fontsize=8)
        axes[3].set_ylabel("$E_{GABA}$", fontsize=8)
        axes[3].set_xlabel("Time (s)", fontsize=8)

        makeNice(axes, labelsize=8)
        fig.savefig(
            f"{save_dir}/Neuron_{neuron_cell}/stim_zoom.pdf", bbox_inches="tight"
        )
        plt.close()

    """
    dynamics = np.zeros((size, int((T - T0) * 1000 / 0.05) - 1))
    for i in range(size):
        dynamics[i, :] = np.loadtxt(
            f"{save_dir}/Neuron_{i}/cell_dynamics.txt", unpack=True, usecols=1
        )
    plt.figure(figsize=(8, 6))
    plt.imshow(
        dynamics,
        cmap="jet",
        aspect="auto",
        vmin=-55,
        vmax=0,
    )
    plt.show()
    """


def plot_feature_comparison(
    sim_df,
    vivo,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/default_network/",
    show=False,
    bw_fr=10,
    bw_cv=0.2,
    DD=False,
):

    exp_data = (
        expan.get_slice_baseline_data()
        if not vivo
        else expan.get_in_vivo_baseline_data_short(segment=2, DD=DD)
    )

    exp_data, _, _ = expan.find_and_remove_outliers_as_df(exp_data)

    sim_df, _, _ = expan.find_and_remove_outliers_as_df(sim_df)

    cols = [x for x in sim_df.columns if "pre_exp" in x]
    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    bins_fr = np.arange(
        0,
        np.max([np.max(exp_data["pre_exp_freq"]), np.max(sim_df["pre_exp_freq"])])
        + bw_fr,
        bw_fr,
    )
    count = 0
    labels = [
        "Freq",
        "CV",
        "Num Bursts",
        "Avg Burst FR",
        "Time Bursting",
        "Spikes in Burst",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting FR",
        "Burst FR Increase",
    ]
    for col in cols:
        sim_vals = (
            sim_df[col] if count < 2 else sim_df[sim_df["pre_exp_num_bursts"] > 0][col]
        )
        exp_vals = (
            exp_data[col]
            if count < 2
            else exp_data[exp_data["pre_exp_num_bursts"] > 0][col]
        )
        bins = np.linspace(
            0,
            np.max([np.max(exp_data[col]), np.max(sim_df[col])]),
            20,
        )
        sns.histplot(
            x=(exp_vals),
            ax=axes[count],
            kde=True,
            stat="probability",
            bins=bins,
            edgecolor="w",
            alpha=0.5,
            color="gray",
        )
        sns.histplot(
            x=(sim_vals),
            ax=axes[count],
            kde=True,
            stat="probability",
            bins=bins,
            edgecolor="w",
            alpha=0.5,
            color="blue",
        )
        ylims = axes[count].get_ylim()
        axes[count].vlines(np.mean(sim_vals), 0, ylims[1], color="blue", ls="dashed")
        axes[count].vlines(np.mean(exp_vals), 0, ylims[1], color="k", ls="dashed")

        ks_pval = ks_2samp(sim_vals, exp_vals).pvalue
        plot_bracket(
            axes[count],
            np.percentile(sim_vals, 25),
            np.percentile(sim_vals, 75),
            ylims[0] + 1.15 * (ylims[1] - ylims[0]),
            ylims[0] + 1.18 * (ylims[1] - ylims[0]),
            f"KS $p=${ks_pval:.3f}",
        )

        ttest_pval = stats.ttest_ind(sim_vals, exp_vals).pvalue
        plot_bracket(
            axes[count],
            np.mean(sim_vals),
            np.mean(exp_vals),
            ylims[1],
            ylims[0] + 1.03 * (ylims[1] - ylims[0]),
            f"T-test $p=${ttest_pval:.3f}",
        )
        axes[count].set_xlabel(labels[count], fontsize=6)
        count += 1

        # axes[count].legend(fontsize="xx-small", fancybox=False, frameon=False)
    makeNice(axes, labelsize=6)
    fig.savefig(f"{save_dir}/feature_comparison.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open {save_dir}/feature_comparison.pdf")


def plot_vivo_vs_slice_fr_cv(
    sim_spikes,
    sim_fr_tmp,
    sim_cv_tmp,
    vivo,
    save_dir="/Users/johnparker/snr_dynamics/python_code/network/default_network/",
    show=False,
    bw_fr=10,
    bw_cv=0.2,
    DD=False,
):
    percent_firing = len(sim_fr_tmp[sim_fr_tmp > 0]) / len(sim_fr_tmp)
    sim_data = pd.DataFrame({"pre_exp_freq": sim_fr_tmp, "pre_exp_cv": sim_cv_tmp})

    sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        sim_data
    )
    sim_fr_low_outliers, sim_cv_low_outliers = low_outliers[0], low_outliers[1]
    sim_fr_high_outliers, sim_cv_high_outliers = high_outliers[0], high_outliers[1]

    exp_data = (
        expan.get_slice_baseline_data()
        if not vivo
        else expan.get_in_vivo_baseline_data_short(segment=2, DD=DD)
    )

    exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]

    exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        exp_data
    )
    exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
    exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

    bins_fr = np.arange(
        0,
        np.max(
            [np.max(exp_data["pre_exp_freq"]), np.max(sim_data["pre_exp_freq"].values)]
        )
        + bw_fr,
        bw_fr,
    )
    bins_cv = np.arange(
        0,
        np.max([np.max(exp_data["pre_exp_cv"]), np.max(sim_data["pre_exp_cv"].values)])
        + bw_cv,
        bw_cv,
    )

    fig = plt.figure(figsize=(10, 6), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[0],
        width_ratios=[2, 4, 3],
    )
    gs2 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs1[1], height_ratios=[4, 1], hspace=0.3
    )
    gs3 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs1[2], height_ratios=[4, 1], hspace=0.3
    )
    gs4 = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[1],
        width_ratios=[2, 2, 3],
    )
    axes = [
        fig.add_subplot(gs1[0]),
        fig.add_subplot(gs2[0]),
        fig.add_subplot(gs2[1]),
        fig.add_subplot(gs3[0]),
        fig.add_subplot(gs3[1]),
        fig.add_subplot(gs4[0]),
        fig.add_subplot(gs4[1]),
        fig.add_subplot(gs4[2]),
    ]

    axes[0].eventplot(sim_spikes, color="k", linewidths=1)
    axes[0].set_xlabel("Time (s)", fontsize=6)
    axes[0].set_ylabel("Neuron ID", fontsize=6)
    axes[0].set_ylim([0, len(sim_spikes)])

    sns.histplot(
        sim_data["pre_exp_freq"].values,
        kde=True,
        stat="probability",
        bins=bins_fr,
        edgecolor="w",
        alpha=0.5,
        ax=axes[1],
        label=f"Model (n = {len(sim_data)})",
        legend=True,
        color="blue",
    )
    sns.histplot(
        exp_data["pre_exp_freq"].values,
        kde=True,
        stat="probability",
        bins=bins_fr,
        edgecolor="w",
        alpha=0.5,
        ax=axes[1],
        label=f"Exp Data (n = {len(exp_data)})",
        legend=True,
        color="gray",
    )

    ylims = axes[1].get_ylim()

    axes[1].vlines(
        np.mean(sim_data["pre_exp_freq"].values), 0, ylims[1], color="blue", ls="dashed"
    )
    axes[1].vlines(
        np.mean(exp_data["pre_exp_freq"].values), 0, ylims[1], color="k", ls="dashed"
    )
    ks_pval = ks_2samp(sim_data["pre_exp_freq"], exp_data["pre_exp_freq"]).pvalue
    plot_bracket(
        axes[1],
        np.percentile(sim_data["pre_exp_freq"], 25),
        np.percentile(sim_data["pre_exp_freq"], 75),
        ylims[0] + 1.15 * (ylims[1] - ylims[0]),
        ylims[0] + 1.18 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = stats.ttest_ind(
        sim_data["pre_exp_freq"], exp_data["pre_exp_freq"]
    ).pvalue
    plot_bracket(
        axes[1],
        np.mean(sim_data["pre_exp_freq"]),
        np.mean(exp_data["pre_exp_freq"]),
        ylims[1],
        ylims[0] + 1.03 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )

    axes[1].legend(fontsize="xx-small", fancybox=False, frameon=False)

    exp_fr_err = 3 * np.std(exp_fr)
    sim_fr_err = 3 * np.std(sim_fr_tmp)
    axes[2].errorbar(
        np.mean(exp_fr),
        1,
        xerr=exp_fr_err,
        color="gray",
        marker="o",
        capsize=5,
        markersize=4,
    )
    axes[2].errorbar(
        np.mean(sim_fr_tmp),
        0,
        xerr=sim_fr_err,
        color="blue",
        marker="o",
        capsize=5,
        markersize=4,
    )
    axes[2].scatter(
        exp_fr_high_outliers,
        np.ones(exp_fr_high_outliers.shape[0]),
        color="gray",
        marker="D",
        s=4,
    )
    axes[2].scatter(
        exp_fr_low_outliers,
        np.ones(exp_fr_low_outliers.shape[0]),
        color="gray",
        marker="D",
        s=4,
    )
    axes[2].scatter(
        sim_fr_high_outliers,
        np.zeros(sim_fr_high_outliers.shape[0]),
        color="blue",
        marker="D",
        s=4,
    )
    axes[2].scatter(
        sim_fr_low_outliers,
        np.zeros(sim_fr_low_outliers.shape[0]),
        color="blue",
        marker="D",
        s=4,
    )
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Sim", "Exp"], fontsize=6)
    axes[2].set_xlabel(f"Pre-Stim Firing Rate (Hz)", fontsize=6)
    axes[2].set_ylim([-1, 2])
    axes[2].set_xlim([0, axes[2].get_xlim()[1]])

    sns.histplot(
        sim_data["pre_exp_cv"].values,
        kde=True,
        stat="probability",
        bins=bins_cv,
        edgecolor="w",
        alpha=0.5,
        ax=axes[3],
        legend=True,
        color="blue",
    )
    sns.histplot(
        exp_data["pre_exp_cv"].values,
        kde=True,
        stat="probability",
        bins=bins_cv,
        edgecolor="w",
        alpha=0.5,
        ax=axes[3],
        legend=True,
        color="gray",
    )
    ylims = axes[3].get_ylim()
    axes[3].vlines(
        np.mean(sim_data["pre_exp_cv"].values), 0, ylims[1], color="blue", ls="dashed"
    )
    axes[3].vlines(
        np.mean(exp_data["pre_exp_cv"].values), 0, ylims[1], color="k", ls="dashed"
    )

    ks_pval = ks_2samp(sim_data["pre_exp_cv"], exp_data["pre_exp_cv"]).pvalue
    plot_bracket(
        axes[3],
        np.percentile(sim_data["pre_exp_cv"], 25),
        np.percentile(sim_data["pre_exp_cv"], 75),
        ylims[0] + 1.15 * (ylims[1] - ylims[0]),
        ylims[0] + 1.18 * (ylims[1] - ylims[0]),
        f"KS $p=${ks_pval:.3f}",
    )

    ttest_pval = stats.ttest_ind(sim_data["pre_exp_cv"], exp_data["pre_exp_cv"]).pvalue
    plot_bracket(
        axes[3],
        np.mean(sim_data["pre_exp_cv"]),
        np.mean(exp_data["pre_exp_cv"]),
        ylims[1],
        ylims[0] + 1.03 * (ylims[1] - ylims[0]),
        f"T-test $p=${ttest_pval:.3f}",
    )

    axes[1].set_xlabel("Pre-Stim Firing Rate", fontsize=6)
    axes[3].set_xlabel("Pre-Stim CV", fontsize=6)
    axes[1].set_ylabel("Probability", fontsize=6)
    axes[3].set_ylabel("Probability", fontsize=6)

    exp_cv_err = 3 * np.std(exp_cv)
    sim_cv_err = 3 * np.std(sim_cv_tmp)
    axes[4].errorbar(
        np.mean(exp_cv),
        1,
        xerr=exp_cv_err,
        color="gray",
        marker="o",
        capsize=5,
        markersize=4,
    )
    axes[4].errorbar(
        np.mean(sim_cv_tmp),
        0,
        xerr=sim_cv_err,
        color="blue",
        marker="o",
        capsize=5,
        markersize=4,
    )
    axes[4].scatter(
        exp_cv_high_outliers,
        np.ones(exp_cv_high_outliers.shape[0]),
        color="gray",
        marker="D",
        s=4,
    )
    axes[4].scatter(
        exp_cv_low_outliers,
        np.ones(exp_cv_low_outliers.shape[0]),
        color="gray",
        marker="D",
        s=4,
    )
    axes[4].scatter(
        sim_cv_high_outliers,
        np.zeros(sim_cv_high_outliers.shape[0]),
        color="blue",
        marker="D",
        s=4,
    )
    axes[4].scatter(
        sim_cv_low_outliers,
        np.zeros(sim_cv_low_outliers.shape[0]),
        color="blue",
        marker="D",
        s=4,
    )
    axes[4].set_yticks([0, 1])
    axes[4].set_yticklabels(["Sim", "Exp"], fontsize=6)
    axes[4].set_xlabel(f"Pre-Stim CV", fontsize=6)
    axes[4].set_ylim([-1, 2])
    axes[4].set_xlim([0, axes[4].get_xlim()[1]])

    xlabels = [f"{x:.1f}" for x in bins_cv]
    ylabels = [int(x) for x in bins_fr]

    data = np.zeros((len(bins_fr), len(bins_cv)))
    for index, row in sim_data.iterrows():
        i = np.searchsorted(bins_fr, row["pre_exp_freq"], side="right")
        j = np.searchsorted(bins_cv, row["pre_exp_cv"], side="right")

        data[i - 1, j - 1] += 1
    data = data / np.sum(data)
    ax1 = sns.heatmap(
        data=data,
        ax=axes[5],
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
    for index, row in exp_data.iterrows():
        i = np.searchsorted(bins_fr, row["pre_exp_freq"], side="right")
        j = np.searchsorted(bins_cv, row["pre_exp_cv"], side="right")

        data[i - 1, j - 1] += 1
    data = data / np.sum(data)
    ax2 = sns.heatmap(
        data=data,
        ax=axes[6],
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

    axes[5].set_xticks(list(range(0, len(xlabels), 2)))
    axes[5].set_xticklabels([x for x in xlabels[::2]], rotation=0)
    axes[5].set_yticklabels(ylabels, rotation=0)
    axes[6].set_xticks(list(range(0, len(xlabels), 2)))
    axes[6].set_xticklabels([x for x in xlabels[::2]], rotation=0)
    axes[6].set_yticklabels(ylabels, rotation=0)
    axes[5].set_xlabel("Pre-Stim CV", fontsize=6)
    axes[5].set_ylabel("Pre-Stim Firing Rate (Hz)", fontsize=6)
    axes[6].set_xlabel("Pre-Stim CV", fontsize=6)
    axes[6].set_ylabel("Pre-Stim Firing Rate (Hz)", fontsize=6)
    axes[5].tick_params(axis="both", which="major", labelsize=2)
    axes[6].tick_params(axis="both", which="major", labelsize=2)

    axes[5].invert_yaxis()
    axes[6].invert_yaxis()

    plot_linear_fit(
        sim_data["pre_exp_freq"], sim_data["pre_exp_cv"], axes[7], color="blue"
    )
    plot_linear_fit(
        exp_data["pre_exp_freq"], exp_data["pre_exp_cv"], axes[7], color="k"
    )

    axes[7].set_xlabel("Pre-Stim Firing Rate (Hz)", fontsize=6)
    axes[7].set_ylabel("Pre-Stim CV", fontsize=6)

    rel_error_fr = na.percent_error_fr(
        sim_data["pre_exp_freq"], exp_data["pre_exp_freq"], bin_width=bw_fr
    )
    rel_error_cv = na.percent_error_cv(
        sim_data["pre_exp_cv"], exp_data["pre_exp_cv"], bin_width=bw_cv
    )

    plt.suptitle(
        f"Firing: {percent_firing*100:.1f}%; Rel. Error. FR: {rel_error_fr*100:.2f}%; Rel. Error. CV: {rel_error_cv*100:.2f}%",
        fontsize=6,
    )

    makeNice(axes, labelsize=6)
    fig.savefig(f"{save_dir}/model_vs_exp_fr_cv.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/model_vs_exp_fr_cv.pdf")


def gather_sim_data(save_dir, T, T0, size=100):
    sim_spikes = []
    rates = np.zeros(size)
    cvs = np.zeros(size)
    n_bursts = np.zeros(size)
    n_bursts = np.zeros(size)
    avg_burst_firing_rate = np.zeros(size)
    percent_time_bursting = np.zeros(size)
    percent_spike_bursting = np.zeros(size)
    avg_burst_duration = np.zeros(size)
    avg_inter_burst_interval = np.zeros(size)
    non_bursting_firing_rate = np.zeros(size)
    burst_firing_rate_increase = np.zeros(size)
    for i in range(size):
        if os.path.getsize(f"{save_dir}/Neuron_{i}/spike_times.txt") > 0:
            spikes = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt")
            spikes = spikes[(spikes >= T0) & (spikes <= T)]  # only keep final 2 seconds
            rates[i] = len(spikes) / (T - T0)
            cvs[i] = (
                np.std(np.diff(spikes)) / np.mean(np.diff(spikes))
                if len(spikes) > 2
                else 0
            )
        else:
            spikes = []
            rates[i] = 0
            cvs[i] = 0
        # commented out to account for if no spikes in file
        """spikes = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt", ndmin=2)
        spikes = spikes[(spikes >= T0) & (spikes < T)]
        rates[i] = spikes.shape[0] / (T - T0)
        cvs[i] = (
            np.std(np.diff(spikes)) / np.mean(np.diff(spikes)) if len(spikes) > 1 else 0
        )
        if np.isnan(cvs[i]):
            cvs[i] = 0"""
        sim_spikes.append(spikes)

        bursts = poisson_surprise.run_poisson_surprise(spikes, surprise_threshold=3)
        if len(bursts) > 0:
            (
                n_bursts[i],
                avg_burst_firing_rate[i],
                percent_time_bursting[i],
                percent_spike_bursting[i],
                avg_burst_duration[i],
                avg_inter_burst_interval[i],
                _,  # cv burst interval, don't care about
                _,  # avg surprise, don't care about
                non_bursting_firing_rate[i],
                burst_firing_rate_increase[i],
            ) = poisson_surprise.burst_statistics(spikes, bursts, T - T0)

    # return sim_spikes, rates, cvs
    sim_df = pd.DataFrame(
        {
            "pre_exp_freq": rates,
            "pre_exp_cv": cvs,
            "pre_exp_num_bursts": n_bursts,
            "pre_exp_avg_burst_firing_rate": avg_burst_firing_rate,
            "pre_exp_percent_time_bursting": percent_time_bursting,
            "pre_exp_percent_spike_bursting": percent_spike_bursting,
            "pre_exp_avg_burst_duration": avg_burst_duration,
            "pre_exp_avg_inter_burst_interval": avg_inter_burst_interval,
            "pre_exp_non_bursting_firing_rate": non_bursting_firing_rate,
            "pre_exp_burst_firing_rate_increase": burst_firing_rate_increase,
        }
    )
    return sim_spikes, sim_df


################################################################################
#########   PLOTTING    ########################################################
################################################################################


def plot_comparison_results(
    save_dir,
    T,
    T0,
    vivo,
    DD=False,
    size=100,
    show=True,
    bw_fr=10,
    bw_cv=0.1,
    all_features=True,
):
    # sim_spikes, sim_fr_tmp, sim_cv_tmp = gather_sim_data(save_dir, T, T0, size=size)
    sim_spikes, sim_df = gather_sim_data(save_dir, T, T0, size=size)

    plot_vivo_vs_slice_fr_cv(
        sim_spikes,
        sim_df["pre_exp_freq"],
        sim_df["pre_exp_cv"],
        vivo,
        save_dir=save_dir,
        show=show,
        bw_fr=bw_fr,
        bw_cv=bw_cv,
        DD=DD,
    )

    if all_features:
        plot_feature_comparison(
            sim_df, vivo, save_dir=save_dir, show=show, bw_fr=bw_fr, bw_cv=bw_cv, DD=DD
        )


def plot_linear_fit(x, y, axes, ci=99, color="black", scatter=True):
    sns.regplot(
        x=x,
        y=y,
        ax=axes,
        color=color,
        scatter_kws={"s": 4, "color": color, "alpha": 0.5},
        scatter=scatter,
        ci=ci,
        marker=".",
        line_kws=dict(linestyle="dashed", linewidth=1),
    )
    plt.setp(axes.collections[-1], alpha=0.3)


def plot_bracket(ax, x1, x2, y1, y2, text, color="k", lw=0.5):
    ax.hlines(y2, x1, x2, color=color, lw=lw, zorder=15)
    ax.vlines([x1, x2], y1, y2, color=color, lw=lw, zorder=15)
    ax.annotate(
        text,
        xycoords="data",
        xy=((x1 + x2) / 2, y2 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01),
        fontsize=6,
        zorder=15,
        ha="center",
    )


def plot_vertical_bracket(ax, x1, x2, y1, y2, text, color="k", lw=0.5):
    ax.hlines([y1, y2], x1, x2, color=color, lw=lw, zorder=15)
    ax.vlines(x2, y1, y2, color=color, lw=lw, zorder=15)
    ax.annotate(
        text,
        xycoords="data",
        xy=(x2 + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, (y1 + y2) / 2),
        fontsize=6,
        zorder=15,
        va="center",
    )


def find_connections(size, save_dir):
    adjacency = np.loadtxt(f"{save_dir}/weights.txt")
    num_connections = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if adjacency[i, j] != 0:
                num_connections[i, j] = 1

    connections = np.sum(num_connections, axis=0)
    total_weight = np.sum(adjacency, axis=0)
    return adjacency, connections, total_weight


def plot_connectivity_statistics(
    adjacency, connections, total_weight, sim_fr, sim_cv, size, save_dir, show=True
):
    graph = nx.Graph()
    for i in range(size):
        for j in range(size):
            graph.add_edge(i, j, weight=adjacency[i, j])
    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]

    bins = np.arange(0, np.max(connections) + 1)

    sns.histplot(
        connections,
        kde=True,
        stat="proportion",
        bins=bins,
        color="blue",
        edgecolor="w",
        ax=axes[0],
    )
    axes[0].vlines(
        np.mean(connections),
        axes[0].get_ylim()[0],
        axes[0].get_ylim()[1],
        linestyle="dashed",
        color="k",
    )
    axes[0].set_xticks(bins + 0.5)
    axes[0].set_xticklabels(bins.astype(int))
    axes[0].set_xlabel("# SNr Connections", fontsize=8)
    axes[0].set_ylabel("Proportion", fontsize=8)
    axes[0].set_xlim([bins[0], bins[-1]])
    sns.despine(ax=axes[0])

    W_bins = np.arange(np.min(total_weight), np.max(total_weight) + 0.2, 0.2)
    sns.histplot(
        total_weight,
        kde=True,
        stat="proportion",
        bins=W_bins,
        color="blue",
        edgecolor="w",
        ax=axes[1],
    )
    axes[1].vlines(
        np.mean(total_weight),
        axes[1].get_ylim()[0],
        axes[1].get_ylim()[1],
        linestyle="dashed",
        color="k",
    )
    axes[1].set_xticks(W_bins)
    axes[1].set_xlabel("$W_{SNr}$", fontsize=8)
    axes[1].set_ylabel("Proportion", fontsize=8)
    sns.despine(ax=axes[1])

    widths = nx.get_edge_attributes(graph, "weight")
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos=pos, ax=axes[2], node_color="blue", node_size=5)
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=axes[2],
        edge_color="black",
        edgelist=widths.keys(),
        width=list(widths.values()),
    )
    # nx.draw(graph,pos=nx.fruchterman_reingold_layout(graph),ax=axes[1],node_color="blue",node_size=5)
    for edg in ["top", "bottom", "left", "right"]:
        axes[2].spines[edg].set_visible(False)

    mshow = axes[3].matshow(
        adjacency, cmap="jet", vmin=0, vmax=np.max(adjacency), aspect="auto"
    )
    axes[3].set_ylabel("Source", fontsize=8)
    axes[3].set_xlabel("Target", fontsize=8)
    fig.colorbar(mshow, fraction=0.046, pad=0.04)
    axes[3].tick_params(
        axis="x", which="both", top=False, bottom=True, labelbottom=True, labeltop=False
    )

    sns.scatterplot(y=sim_fr, x=total_weight, ax=axes[4], color="blue", s=10)
    xlims = axes[4].get_xlim()
    bins = np.arange(0, np.max(sim_fr) + 10, 10)
    for k in range(len(bins) - 1):
        axes[4].hlines(
            bins[k],
            xlims[0],
            xlims[1],
            linestyle="solid",
            color="gray",
            lw=1,
            alpha=0.5,
        )
        fr_bin = total_weight[(sim_fr >= bins[k]) & (sim_fr < bins[k + 1])]
        axes[4].vlines(
            np.mean(fr_bin),
            bins[k],
            bins[k + 1],
            linestyle="solid",
            color="gray",
            lw=1,
            alpha=0.5,
        )

    plot_linear_fit(total_weight, sim_fr, axes[4], color="blue")
    axes[4].set_xlabel("$W_{SNr}$", fontsize=8)
    axes[4].set_ylabel("Firing Rate (Hz)", fontsize=8)

    sns.scatterplot(y=sim_cv, x=total_weight, ax=axes[5], color="blue", s=10)
    xlims = axes[5].get_xlim()
    bins = np.arange(0, np.max(sim_cv) + 0.1, 0.1)
    for k in range(len(bins) - 1):
        axes[5].hlines(
            bins[k],
            xlims[0],
            xlims[1],
            linestyle="solid",
            color="gray",
            lw=1,
            alpha=0.5,
        )
        cv_bin = total_weight[(sim_cv >= bins[k]) & (sim_cv < bins[k + 1])]
        axes[5].vlines(
            np.mean(cv_bin),
            bins[k],
            bins[k + 1],
            linestyle="solid",
            color="gray",
            lw=1,
            alpha=0.5,
        )

    plot_linear_fit(total_weight, sim_cv, axes[5], color="blue")
    axes[5].set_xlabel("$W_{SNr}$", fontsize=8)
    axes[5].set_ylabel("CV", fontsize=8)

    makeNice(axes, labelsize=6)
    plt.savefig(f"{save_dir}/connections.pdf")
    plt.close()

    if show:
        run_cmd(f"open {save_dir}/connections.pdf")

    return connections, total_weight


def plot_heterogeneity_with_mf(
    save_dir, T, T0, params, d1stim, gpestim, size, show=True, vivo=False
):
    """sim_spikes, sim_fr, sim_cv = gather_sim_data(save_dir, T, T0, size=size)"""
    sim_spikes, sim_df = gather_sim_data(save_dir, T, T0, size=size)
    sim_fr = sim_df["pre_exp_freq"]
    sim_cv = sim_df["pre_exp_cv"]
    meta_data = get_neuron_meta_data(save_dir, size=size)

    meta_data["pre_exp_fr"] = sim_fr
    meta_data["pre_exp_cv"] = sim_cv

    meta_data["pre_exp_fr_zscore"] = stats.zscore(meta_data["pre_exp_fr"])
    meta_data["pre_exp_cv_zscore"] = stats.zscore(meta_data["pre_exp_cv"])

    adjacency, connections, total_weight = find_connections(size, save_dir)

    meta_data["connections"] = connections
    meta_data["total_weight"] = total_weight

    meta_data["MF"] = generate_modulation_factors(
        save_dir, params, T0, T, d1stim, gpestim, size=size
    )[:, 1]

    meta_data = pd.DataFrame(meta_data)

    meta_data = meta_data[
        (np.abs(meta_data["pre_exp_fr_zscore"]) <= 3)
        & (np.abs(meta_data["pre_exp_cv_zscore"]) <= 3)
        & (np.abs(meta_data["pre_exp_fr"]) > 0)
    ]

    sim_fr = meta_data["pre_exp_fr"]
    sim_cv = meta_data["pre_exp_cv"]
    connections = meta_data["connections"]
    total_weight = meta_data["total_weight"]

    label_key = {
        "gTRPC3_nS_pF": "$g_{TRPC3}$",
        "gHCN_nS_pF": "$g_{HCN}$",
        "gCA_nS_pF": "$g_{Ca}$",
        "gL_nS_pF": "$g_{Leak}$",
        "gSK_nS_pF": "$g_{SK}$",
        "gNAP_nS_pF": "$g_{NaP}$",
        "gNAF_nS_pF": "$g_{NaF}$",
        "gKDR_nS_pF": "$g_{K}$",
        "Eleak_mV": "$E_{Leak}$",
        "gKCC2_S_nS_pF": "$g_{KCC2}^S$",
        "gTON_CL_S_MEAN_nS_pF": "$g_{GABA_S}^{Tonic}$",
        "gTON_STN_MEAN_nS_pF": "$g_{STN}^{Tonic}$",
        "soma_noise_intensity": "$\\sigma_{S}$",
    }

    keys = list(meta_data.keys())
    conductances = [keys[x] for x in [0, 3, 9, 11, 13, 14, 15, 16, 17, 18, 20, 24]]

    fig, ax = plt.subplots(3, 4, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(4)]

    for i in range(len(axes)):
        sns.scatterplot(
            x=meta_data[conductances[i]],
            y=meta_data["MF"],
            ax=axes[i],
            color="blue",
            zorder=15,
            s=10,
        )

        if np.std(meta_data[conductances[i]]) > 0:
            plot_linear_fit(
                meta_data[conductances[i]], meta_data["MF"], axes[i], color="blue"
            )
        axes[i].set_xlabel(label_key[conductances[i]], fontsize=8)
        axes[i].set_ylabel("MF", fontsize=8)
        axes[i].tick_params("both", labelsize=6)
    makeNice(axes, labelsize=6)
    add_fig_labels(axes)
    match_axis(axes, type="y")
    fig.savefig(f"{save_dir}/conductance_MF.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open {save_dir}/conductance_MF.pdf")


def plot_network_statistics(save_dir, T, T0, size, show=True, vivo=False):
    # sim_spikes, sim_fr, sim_cv = gather_sim_data(save_dir, T, T0, size=size)
    sim_spikes, sim_df = gather_sim_data(save_dir, T, T0, size=size)
    sim_fr = sim_df["pre_exp_freq"]
    sim_cv = sim_df["pre_exp_cv"]
    meta_data = get_neuron_meta_data(save_dir, size=size)

    meta_data["pre_exp_fr"] = sim_fr
    meta_data["pre_exp_cv"] = sim_cv

    meta_data["pre_exp_fr_zscore"] = stats.zscore(meta_data["pre_exp_fr"])
    meta_data["pre_exp_cv_zscore"] = stats.zscore(meta_data["pre_exp_cv"])

    adjacency, connections, total_weight = find_connections(size, save_dir)

    meta_data["connections"] = connections
    meta_data["total_weight"] = total_weight

    meta_data = pd.DataFrame(meta_data)

    meta_data = meta_data[
        (np.abs(meta_data["pre_exp_fr_zscore"]) <= 3)
        & (np.abs(meta_data["pre_exp_cv_zscore"]) <= 3)
        & (np.abs(meta_data["pre_exp_fr"]) > 0)
    ]

    sim_fr = meta_data["pre_exp_fr"]
    sim_cv = meta_data["pre_exp_cv"]
    connections = meta_data["connections"]
    total_weight = meta_data["total_weight"]

    plot_connectivity_statistics(
        adjacency, connections, total_weight, sim_fr, sim_cv, size, save_dir, show=show
    )

    """exp_data = expan.get_exp_results(vivo)

    (
        exp_data,
        exp_fr_low_outliers,
        exp_cv_low_outliers,
        exp_fr_high_outliers,
        exp_cv_high_outliers,
    ) = expan.find_and_remove_outliers_as_df(exp_data)

    exp_fr_outliers_removed, exp_cv_outliers_removed = (
        exp_data["pre_exp_freq"],
        exp_data["pre_exp_cv"],
    )"""
    exp_data = expan.get_exp_results(vivo)

    exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        exp_data
    )
    exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
    exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]
    exp_fr_outliers_removed, exp_cv_outliers_removed = (
        exp_data["pre_exp_freq"],
        exp_data["pre_exp_cv"],
    )

    label_key = {
        "gTRPC3_nS_pF": "$g_{TRPC3}$",
        "gHCN_nS_pF": "$g_{HCN}$",
        "gCA_nS_pF": "$g_{Ca}$",
        "gL_nS_pF": "$g_{Leak}$",
        "gSK_nS_pF": "$g_{SK}$",
        "gNAP_nS_pF": "$g_{NaP}$",
        "gNAF_nS_pF": "$g_{NaF}$",
        "gKDR_nS_pF": "$g_{K}$",
        "Eleak_mV": "$E_{Leak}$",
        "gKCC2_S_nS_pF": "$g_{KCC2}^S$",
        "gTON_CL_S_MEAN_nS_pF": "$g_{GABA_S}^{Tonic}$",
        "gTON_STN_MEAN_nS_pF": "$g_{STN}^{Tonic}$",
        "soma_noise_intensity": "$\\sigma_{S}$",
    }

    fig, ax = plt.subplots(3, 4, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(4)]

    fig2, ax = plt.subplots(3, 4, figsize=(8, 6), dpi=300, tight_layout=True)
    axes2 = [ax[i, j] for i in range(3) for j in range(4)]

    fig3, ax = plt.subplots(3, 4, figsize=(8, 6), dpi=300, tight_layout=True)
    axes3 = [ax[i, j] for i in range(3) for j in range(4)]

    keys = list(meta_data.keys())
    conductances = [keys[x] for x in [0, 3, 9, 11, 13, 14, 15, 16, 17, 18, 20, 24]]

    for i in range(len(axes)):
        sns.scatterplot(
            x=meta_data[conductances[i]],
            y=sim_fr,
            ax=axes[i],
            color="blue",
            zorder=15,
            s=10,
        )

        sns.scatterplot(
            x=sim_fr,
            y=sim_cv,
            hue=meta_data[conductances[i]],
            palette="coolwarm",
            ax=axes3[i],
            zorder=15,
            s=10,
            legend=False,
        )

        if np.std(meta_data[conductances[i]]) > 0:
            plot_linear_fit(meta_data[conductances[i]], sim_fr, axes[i], color="blue")

        xlims = axes[i].get_xlim()
        bins = np.arange(
            0, np.max([np.max(sim_fr), np.max(exp_fr_outliers_removed)]) + 10, 10
        )
        for k in range(len(bins) - 1):
            axes[i].hlines(
                bins[k],
                xlims[0],
                xlims[1],
                linestyle="solid",
                color="gray",
                lw=1,
                alpha=0.5,
            )
            samps = np.asarray(meta_data[conductances[i]])
            fr_bin = samps[(sim_fr >= bins[k]) & (sim_fr < bins[k + 1])]
            if len(fr_bin) > 0:
                axes[i].vlines(
                    np.mean(fr_bin),
                    bins[k],
                    bins[k + 1],
                    linestyle="solid",
                    color="gray",
                    lw=1,
                    alpha=0.5,
                )
        axes[i].hlines(
            bins[-1],
            xlims[0],
            xlims[1],
            linestyle="solid",
            color="gray",
            lw=1,
            alpha=0.5,
        )

        axes[i].set_xlabel(label_key[conductances[i]], fontsize=8)
        axes[i].set_ylabel("FR (Hz)", fontsize=8)
        axes[i].tick_params("both", labelsize=6)
        axes3[i].set_title(label_key[conductances[i]], fontsize=8)
        axes3[i].set_xlabel("FR (Hz)", fontsize=8)
        axes3[i].set_ylabel("CV", fontsize=8)
        axes3[i].tick_params("both", labelsize=6)

        sns.scatterplot(
            x=meta_data[conductances[i]], y=sim_cv, ax=axes2[i], color="blue", s=10
        )
        xlims = axes2[i].get_xlim()
        bins = np.arange(
            0, np.max([np.max(sim_cv), np.max(exp_cv_outliers_removed)]) + 0.1, 0.1
        )
        for k in range(len(bins) - 1):
            axes2[i].hlines(
                bins[k],
                xlims[0],
                xlims[1],
                linestyle="solid",
                color="gray",
                lw=1,
                alpha=0.5,
            )
            samps = np.asarray(meta_data[conductances[i]])
            cv_bin = samps[(sim_cv >= bins[k]) & (sim_cv < bins[k + 1])]
            if len(cv_bin) > 0:
                axes2[i].vlines(
                    np.mean(cv_bin),
                    bins[k],
                    bins[k + 1],
                    linestyle="solid",
                    color="gray",
                    lw=1,
                    alpha=0.5,
                )
        axes2[i].hlines(
            bins[-1],
            xlims[0],
            xlims[1],
            linestyle="solid",
            color="gray",
            lw=1,
            alpha=0.5,
        )

        if np.std(meta_data[conductances[i]]) > 0:
            plot_linear_fit(meta_data[conductances[i]], sim_cv, axes2[i], color="blue")
        axes2[i].set_xlabel(label_key[conductances[i]], fontsize=8)
        axes2[i].set_ylabel("CV", fontsize=8)

    makeNice(axes, labelsize=6)
    add_fig_labels(axes)
    match_axis(axes, type="y")
    fig.savefig(f"{save_dir}/conductances_fr.pdf", bbox_inches="tight")

    makeNice(axes2, labelsize=6)
    add_fig_labels(axes2)
    match_axis(axes2, type="y")
    fig2.savefig(f"{save_dir}/conductances_cv.pdf", bbox_inches="tight")

    makeNice(axes3, labelsize=6)
    add_fig_labels(axes3)
    match_axis(axes3, type="y")
    fig3.savefig(f"{save_dir}/conductances_fr_cv.pdf")

    plt.close(fig="all")

    if show:
        run_cmd(f"open {save_dir}/conductances_fr.pdf")
        run_cmd(f"open {save_dir}/conductances_cv.pdf")
        run_cmd(f"open {save_dir}/conductances_fr_cv.pdf")


### not finished ###
def get_input_change(save_dir, params, T0, T, d1stim, gpestim, size=100):
    input_mf = np.zeros(size)
    mfs = generate_modulation_factors(
        save_dir, params, T0, T, d1stim, gpestim, size=size
    )
    mfs = mfs[:, 1]  # only grab mfs and ignore baseline FR
    for neuron_cell in range(size):
        connections = np.loadtxt(f"{save_dir}/weights.txt")
        connections = connections[:, neuron_cell]
        connect_ids = [i for i in range(size) if connections[i] > 0]
        input_mf[neuron_cell] = (
            np.mean(mfs[connect_ids]) if np.abs(np.mean(mfs[connect_ids])) > 0 else 0
        )
    return input_mf


def plot_example_dynamics(save_dir, T, T0, size):
    for neuron_cell in range(size):
        connections = np.loadtxt(f"{save_dir}/weights.txt")
        connections = connections[:, neuron_cell]
        connect_ids = [i for i in range(size) if connections[i] > 0]
        x = np.loadtxt(f"{save_dir}/Neuron_{neuron_cell}/cell_dynamics.txt")
        fig, ax = plt.subplots(4, 2, figsize=(12, 6), dpi=300, tight_layout=True)
        x = x[(x[:, 0] >= T0) & (x[:, 0] < T), :]
        axes = [ax[i, j] for i in range(4) for j in range(2)]
        axes[0].plot(x[:, 0], x[:, 1], color="b", lw=0.5, label="$V_S$ (mV)")
        axes[0].plot(x[:, 0], x[:, 2], color="r", lw=0.5, label="$V_D$ (mV)")

        axes[1].plot(x[:, 0], x[:, 6], color="b", lw=0.5, label="$g_{GABA}^{GPe}$")
        axes[1].plot(x[:, 0], x[:, 7], color="r", lw=0.5, label="$g_{GABA}^{STR}$")

        yy = []
        for i in connect_ids:
            y = np.loadtxt(f"{save_dir}/Neuron_{i}/cell_dynamics.txt")
            ys = np.loadtxt(f"{save_dir}/Neuron_{i}/spike_times.txt")
            y = y[(y[:, 0] >= T0) & (y[:, 0] < T), :]
            ys = ys[(ys >= T0) & (ys < T)]
            yy.extend(ys)
            axes[2].plot(y[:, 0], y[:, 1], lw=0.5)
        axes[4].eventplot(yy, lw=0.5, color="k")

        # axes[2].plot(x[:, 0], x[:, 5], color="b", lw=0.5)

        axes[3].plot(x[:, 0], x[:, 5], color="b", lw=0.5, label="$g_{GABA}^{SNr}$")

        axes[5].plot(x[:, 0], x[:, 9], color="b", lw=0.5, label="$g_{Tonic}^S$")
        axes[5].plot(x[:, 0], x[:, 10], color="r", lw=0.5, label="$g_{Tonic}^D$")
        axes[5].plot(x[:, 0], x[:, 8], color="g", lw=0.5, label="$g_{Tonic}^{STN}$")

        axes[6].plot(x[:, 0], x[:, -2], lw=0.5, color="b", label="$\\eta_S(t)$")
        axes[6].plot(x[:, 0], x[:, -1], lw=0.5, color="r", label="$\\eta_D(t)$")

        axes[7].plot(x[:, 0], x[:, 11], lw=0.5, color="b", label="$E_{GABA}^S$")
        axes[7].plot(x[:, 0], x[:, 12], lw=0.5, color="r", label="$E_{GABA}^D$")

        axes[7].set_ylim([-90, -40])

        for i in range(len(axes)):
            axes[i].set_xlabel("Time (s)", fontsize=8)
            if i != 2 and i != 4:
                axes[i].legend(
                    fancybox=False,
                    frameon=True,
                    # mode="expand",
                    ncol=3,
                    fontsize=6,
                    bbox_to_anchor=(0, 1.02, 1, 0.2),
                    loc="lower left",
                    edgecolor="k",
                )

        axes[0].set_ylabel("$V$ (mV)", fontsize=8)
        axes[1].set_ylabel("$g_{GABA}$ (nS/pF)", fontsize=8)
        axes[2].set_ylabel("$V^i$ (mV)", fontsize=8)
        axes[3].set_ylabel("$g_{GABA}$ (nS/pF)", fontsize=8)
        axes[4].set_ylabel("SNr Input Spikes", fontsize=8)
        axes[5].set_ylabel("$g_{GABA}^{Tonic}$ (nS/pF)", fontsize=8)
        axes[6].set_ylabel("Current Noise", fontsize=8)
        axes[7].set_ylabel("$E_{GABA}$ (mV)", fontsize=8)

        makeNice(axes, labelsize=8)
        fig.savefig(
            f"{save_dir}/Neuron_{neuron_cell}/example_dynamics.pdf", bbox_inches="tight"
        )
        plt.close()
