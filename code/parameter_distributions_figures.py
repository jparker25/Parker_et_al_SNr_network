import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import seaborn as sns

from helpers import *
from run_model import chloride_to_egaba

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8

figure_direc = "/Users/johnparker/paper_repos/Parker_et_al_SNr_Network/figures"

# create df for all 252 matching parameter disributions
figure_name = "matching_slice_neuron_distributions"
matched_slice_neurons = pd.read_csv(
    "data/param_search_slice_correlated/match_neurons.csv"
)
matched_slice_neurons = matched_slice_neurons.drop(
    columns=[
        "Unnamed: 0.1",
        "Unnamed: 0",
        "sim",
        "gTON_CL_S_SIGMA",
        "gTON_CL_S_THETA",
        "gTON_CL_D_THETA",
        "gTON_CL_D_SIGMA",
        "dend_noise_intensity",
        "soma_noise_intensity_theta",
        "dend_noise_intensity_theta",
        "tausyn_dend",
        "gTON_CL_D_MEAN_nS_pF",
        "gKCC2_D_nS_pF",
        "tausyn",
        "tauexc",
        "CL_in_D",
        "gSD_nS",
    ]
)

matched_slice_neurons = matched_slice_neurons.loc[
    :, (matched_slice_neurons != 0).any(axis=0)
]

# Columns
# 'gTON_CL_S_MEAN_nS_pF', 'gTON_CL_D_MEAN_nS_pF', 'gKCC2_S_nS_pF',
# 'gKCC2_D_nS_pF', 'gTRPC3_nS_pF', 'gCA_nS_pF', 'gL_nS_pF', 'gSK_nS_pF',
# 'gNAP_nS_pF', 'gNAF_nS_pF', 'gKDR_nS_pF', 'gSD_nS',
# 'soma_noise_intensity', 'dend_noise_intensity', 'Eleak_mV', 'CL_in_S',
# 'CL_in_D', 'tausyn', 'tauexc', 'pre_exp_fr', 'pre_exp_cv',
# 'pre_exp_fr_zscore', 'pre_exp_cv_zscore', 'connections', 'total_weight',
# 'rel_error'


cols = matched_slice_neurons.columns
run_cmd(f"mkdir {figure_direc}/{figure_name}")
for col in cols:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
    sns.histplot(
        data=matched_slice_neurons,
        x=col,
        stat="probability",
        edgecolor="w",
        color="gray",
        kde=True,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5, labelsize=6)
    plt.savefig(
        f"{figure_direc}/{figure_name}/{col}_histogram.pdf", bbox_inches="tight"
    )
    plt.close()

cols_together = [
    "gTON_CL_S_MEAN_nS_pF",
    "gKCC2_S_nS_pF",
    "gTRPC3_nS_pF",
    "gCA_nS_pF",
    "gL_nS_pF",
    "gSK_nS_pF",
    "gNAP_nS_pF",
    "gNAF_nS_pF",
    "gKDR_nS_pF",
    "soma_noise_intensity",
    "Eleak_mV",
    "CL_in_S",
    "connections",
    "total_weight",
    "rel_error",
]

cols_together = [
    "gTON_CL_S_MEAN_nS_pF",
    "gKCC2_S_nS_pF",
    "gTRPC3_nS_pF",
    "gCA_nS_pF",
    "gL_nS_pF",
    "gSK_nS_pF",
    "gNAP_nS_pF",
    "gNAF_nS_pF",
    "gKDR_nS_pF",
    "Eleak_mV",
    "E_GABA",
    "rel_error",
]
cols_labels = [
    "$g^{tonic,S}_{GABA}$ (nS/pF)",
    "$g_{KCC2}^S$ (nS/pF)",
    "$g_{TRPC3}$ (nS/pF)",
    "$g_{CA}$ (nS/pF)",
    "$g_{L}$ (nS/pF)",
    "$g_{SK}$ (nS/pF)",
    "$g_{NaP}$ (nS/pF)",
    "$g_{NaF}$ (nS/pF)",
    "$g_{KDR}$ (nS/pF)",
    "$E_{leak}$ (mV)",
    "$E_{GABA}$ (mV)",
    "Rel. Error",
]

matched_slice_neurons["E_GABA"] = chloride_to_egaba(matched_slice_neurons["CL_in_S"])

fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(4) for j in range(3)]
for i, col in enumerate(cols_together):
    if col == "connections":
        sns.histplot(
            data=matched_slice_neurons,
            x=col,
            stat="probability",
            edgecolor="w",
            color="gray",
            kde=True,
            ax=axes[i],
            log_scale=col == "rel_error",
            binwidth=1,
        )
    else:
        sns.histplot(
            data=matched_slice_neurons,
            x=col,
            stat="probability",
            edgecolor="w",
            color="gray",
            kde=True,
            ax=axes[i],
            log_scale=col == "rel_error",
        )
    axes[i].set_xlabel(cols_labels[i])
for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
add_fig_labels(axes)
plt.savefig(f"{figure_direc}/{figure_name}/all_histogram.pdf", bbox_inches="tight")
plt.close()
run_cmd(f"open {figure_direc}/{figure_name}/all_histogram.pdf")


# create df for all matching slice network parameters


# create df matching networks of in-vivo
slice_results = pd.read_csv(f"data/param_seed_finder_slice_correlated/results.csv")

slice_results = slice_results[
    (slice_results["KS_FR_pval"] >= 0.05)
    & (slice_results["Ttest_FR_pval"] >= 0.05)
    & (slice_results["Ttest_CV_pval"] >= 0.05)
]
avg_err_sorted = slice_results.sort_values(by="Avg. Error")

slice_seeds = [int(row["Seed"]) for index, row in avg_err_sorted.iterrows()]

vivo_data_frames = []
vivo_sims = 0
for slice_seed in slice_seeds:
    data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
    neurons = pd.read_csv(f"{data_dir}/neurons.csv")
    neurons["slice_seed"] = slice_seed
    results = pd.read_csv(f"{data_dir}/results.csv")
    results = results[
        (results["KS_FR_pval"] >= 0.05)
        & (results["Ttest_FR_pval"] >= 0.05)
        & (results["Ttest_CV_pval"] >= 0.05)
        & (results["KS_CV_pval"] >= 0.05)
    ]
    sims_to_grab = results["Sim"]
    vivo_data_frames.append(neurons[neurons["sim"].isin(results["Sim"])])

vivo_df = pd.concat(vivo_data_frames)

vivo_df = vivo_df.loc[:, (vivo_df != 0).any(axis=0)]
vivo_df = vivo_df.drop(
    columns=[
        "Unnamed: 0",
        "sim",
        "gSD_nS",
        "tausyn",
        "tauexc",
        "tausyn_dend",
        "dend_noise_intensity",
        "gTON_CL_S_SIGMA",
        "gTON_CL_S_THETA",
        "gTON_CL_D_THETA",
        "gTON_CL_D_SIGMA",
        "dend_noise_intensity",
        "soma_noise_intensity_theta",
        "dend_noise_intensity_theta",
        "slice_seed",
        "gTON_STN_SIGMA",
        "gTON_STN_THETA",
        "gTON_CL_D_MEAN_nS_pF",
        "gKCC2_D_nS_pF",
        "CL_in_D",
    ]
)

figure_name = "matching_vivo_neuron_distributions"
cols = vivo_df.columns
run_cmd(f"mkdir {figure_direc}/{figure_name}")
for col in cols:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
    sns.histplot(
        data=vivo_df,
        x=col,
        stat="probability",
        edgecolor="w",
        color="gray",
        kde=True,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5, labelsize=6)
    plt.savefig(
        f"{figure_direc}/{figure_name}/{col}_histogram.pdf", bbox_inches="tight"
    )
    plt.close()

cols_together = [
    "gTON_CL_S_MEAN_nS_pF",
    "gKCC2_S_nS_pF",
    "gTRPC3_nS_pF",
    "gCA_nS_pF",
    "gL_nS_pF",
    "gSK_nS_pF",
    "gNAP_nS_pF",
    "gNAF_nS_pF",
    "gKDR_nS_pF",
    "soma_noise_intensity",
    "Eleak_mV",
    "CL_in_S",
    "connections",
    "total_weight",
    "gTON_STN_MEAN_nS_pF",
]

cols_together = [
    "gTON_CL_S_MEAN_nS_pF",
    "gKCC2_S_nS_pF",
    "gTRPC3_nS_pF",
    "gCA_nS_pF",
    "gL_nS_pF",
    "gSK_nS_pF",
    "gNAP_nS_pF",
    "gNAF_nS_pF",
    "gKDR_nS_pF",
    "Eleak_mV",
    "E_GABA",
    "gTON_STN_MEAN_nS_pF",
]

cols_labels = [
    "$g^{tonic,S}_{GABA}$ (nS/pF)",
    "$g_{KCC2}^S$ (nS/pF)",
    "$g_{TRPC3}$ (nS/pF)",
    "$g_{CA}$ (nS/pF)",
    "$g_{L}$ (nS/pF)",
    "$g_{SK}$ (nS/pF)",
    "$g_{NaP}$ (nS/pF)",
    "$g_{NaF}$ (nS/pF)",
    "$g_{KDR}$ (nS/pF)",
    "$E_{leak}$ (mV)",
    "$E_{GABA}$ (mV)",
    "$g_{STN}^{tonic}$ (nS/pF)",
]

vivo_df["E_GABA"] = chloride_to_egaba(vivo_df["CL_in_S"])

print(len(vivo_df))

fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(4) for j in range(3)]
for i, col in enumerate(cols_together):
    if col == "connections":
        sns.histplot(
            data=vivo_df,
            x=col,
            stat="probability",
            edgecolor="w",
            color="gray",
            kde=True,
            ax=axes[i],
            log_scale=col == "rel_error",
            binwidth=1,
        )
    else:
        sns.histplot(
            data=vivo_df,
            x=col,
            stat="probability",
            edgecolor="w",
            color="gray",
            kde=True,
            ax=axes[i],
            log_scale=col == "rel_error",
        )
    axes[i].set_xlabel(cols_labels[i])
for plot in axes:
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)
add_fig_labels(axes)
plt.savefig(f"{figure_direc}/{figure_name}/all_histogram.pdf", bbox_inches="tight")
plt.close()
run_cmd(f"open {figure_direc}/{figure_name}/all_histogram.pdf")
