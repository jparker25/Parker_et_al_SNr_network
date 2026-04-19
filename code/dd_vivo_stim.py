# import python modules
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os, sys, time
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import ks_2samp as ks2
from matplotlib import colors
import networkx as nx
from scipy import stats
from scipy import optimize

# import user modules
import network_analysis as na
import run_model
import plot_model_results as pmr

# sys.path.append("../")
from helpers import *
import experimental_analysis as expan

T2 = 35
T1 = 5
T0 = 3
run = True
vivo = True
correlate_egaba = True
correlate_wstrs = True
correlate_wgpes = False
size = 100
show = False
dopamine_depletion = True

N_vivo_sims = 3  # samples to pull from in vivo models
N_stim_seeds = 3  # stim seeds

exp_data = (
    expan.get_slice_baseline_data()
    if not vivo
    else expan.get_in_vivo_baseline_data_short(segment=2, DD=True)
)

exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]

exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(exp_data)
exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

exp_fr_outliers_removed, exp_cv_outliers_removed = (
    exp_data["pre_exp_freq"],
    exp_data["pre_exp_cv"],
)

_, gpe_pulse_exp, _, d1_pulse_exp = expan.get_dd_classification_results()
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
d1_seeds = []
gpe_seeds = []

# open figure to store stats


for slice_seed in slice_seeds:
    seeds_completed = open(
        f"data/dd_vivo_stim_seeds_completed_{int(slice_seed)}.txt", "w"
    )
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
        # sim_to_pull_from = f"{data_dir}_dd_search/sim_{int(row["Sim"]):04d}"
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

            sim_to_pull_from = f"{data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/sim_{int(dd_row["Sim"]):06d}"

            for stim_seeds in np.arange(1, N_stim_seeds + 1):
                stim_data_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/dd_sim_{int(dd_row["Sim"]):06d}/sim_{stim_seeds:04d}"
                gpestim = False  # True to run GPe stim experiment
                d1stim = True
                dopamine_depletion = True
                status = True
                dynamics = False

                params = run_model.get_default_params()
                params["T"] = T2
                params["save_dir"] = stim_data_dir
                params["size"] = size
                params["status"] = 1 if status else 0
                params["dynamics"] = 1 if dynamics else 0

                xgpe, xgpe_dd, xd1, xd1_dd = expan.ephys_exp_data()

                np.random.seed(
                    int(slice_seed * row["Sim"] + int(dd_row["Sim"]) + stim_seeds)
                )
                d1_seeds.append(
                    int(slice_seed * row["Sim"] + int(dd_row["Sim"]) + stim_seeds)
                )
                xs, ys, lbound, ubound, zs = expan.ephys_sample(
                    xd1_dd["Conductance"], xd1_dd["Time constant"]
                )

                wstrs = xs.copy() / 40
                tausyn_dend_dist = zs.copy() * 1000

                wgpes = np.zeros(size)
                tausyn_dist = np.zeros(size)

                params["W_gpe"] = wgpes  # xs / 100  # wgpes
                params["gpe_stim"] = np.zeros(size)
                params["gpe_poisson"] = np.zeros(
                    size
                )  # 1 means continuous stim, 0 means pulsed
                params["gpe_stim_percent"] = np.zeros(size)
                params["gpe_freq"] = np.zeros(size)
                params["gpe_start_time"] = np.zeros(size)
                params["gpe_base_length"] = np.zeros(size)
                params["gpe_stim_length"] = np.zeros(size)
                params["gpe_post_length"] = np.zeros(size)

                params["W_str"] = wstrs  # wstrs
                params["str_stim"] = np.ones(size)
                params["str_poisson"] = np.zeros(
                    size
                )  # 1 means continuous stim, 0 means pulsed
                params["str_stim_percent"] = np.ones(size)
                params["str_freq"] = 20 * np.ones(size)
                params["str_start_time"] = T1 * np.ones(size)
                params["str_base_length"] = np.ones(size)
                params["str_stim_length"] = np.ones(size)
                params["str_post_length"] = np.ones(size)

                tauexc_dist = run_model.positive_normal_distribution(5, 0, size)

                params["tausyn"] = tausyn_dist
                params["tauexc"] = tauexc_dist
                params["tausyn_dend"] = tausyn_dend_dist

                csv = f"{sim_to_pull_from}/heterogeneity.csv"
                df = pd.read_csv(csv)
                run_model.heterogeneity_from_simulation(sim_to_pull_from, **params)

                _, connections, total_weight = pmr.find_connections(
                    size, sim_to_pull_from
                )
                kcc2_D_vals = df["gKCC2_D_nS_pF"].values
                tonic_CL_D_vals = df["gTON_CL_D_MEAN_nS_pF"].values
                egaba_vals = np.zeros((size, 3))
                for i in range(size):
                    kcc2_val = kcc2_D_vals[i]
                    tonic_val = tonic_CL_D_vals[i]
                    egaba_vals[i] = [
                        kcc2_val,
                        tonic_val,
                        run_model.chloride_to_egaba(
                            optimize.newton(
                                run_model.chloride_dynamics_ss,
                                10,
                                args=(
                                    kcc2_val,
                                    tonic_val,
                                    -55,
                                ),
                            )
                        ),
                    ]
                sorted_egaba_vals = egaba_vals[egaba_vals[:, 2].argsort()]
                sorted_connections_ind = total_weight.argsort()
                sorted_wstrs = wstrs[wstrs.argsort()]
                sorted_tausyn_dend_dist = tausyn_dend_dist[wstrs.argsort()]
                sorted_wstrs_ind = wstrs.argsort()
                sorted_wgpes = wgpes[wgpes.argsort()]
                sorted_wgpes_ind = wgpes.argsort()
                for i in range(size):
                    egaba_vals[sorted_connections_ind[i], :] = sorted_egaba_vals[i, :]
                    if correlate_wstrs:
                        wstrs[sorted_connections_ind[i]] = sorted_wstrs[size - 1 - i]
                        tausyn_dend_dist[sorted_connections_ind[i]] = (
                            sorted_tausyn_dend_dist[size - 1 - i]
                        )
                    if correlate_wgpes:
                        wgpes[sorted_connections_ind[i]] = sorted_wgpes[size - 1 - i]

                df["tausyn"] = tausyn_dist
                df["tauexc"] = tauexc_dist
                df["tausyn_dend"] = tausyn_dend_dist
                update_dict = {
                    "W_str": wstrs,
                    "W_gpe": wgpes,
                    "tausyn": tausyn_dist,  # tausyn_dist if gpestim else np.zeros(size),
                    "tauexc": tauexc_dist,
                    "tausyn_dend_dist": tausyn_dend_dist,  # tausyn_dend_dist if d1stim else np.zeros(size),
                }
                run_model.update_by_dict(df, update_dict, **params)

                run_model.set_clin(**params)
                run_model.weights_from_simulation(sim_to_pull_from, **params)

                if gpestim or d1stim:
                    run_model.add_stim(**params)

                if stim_seeds == 0:
                    updated_df = pd.read_csv(f"{stim_data_dir}/heterogeneity.csv")
                    vivo_changes = [
                        "gTON_STN_MEAN_nS_pF",
                        "gTON_CL_S_MEAN_nS_pF",
                        "gTON_CL_D_MEAN_nS_pF",
                        "soma_noise_intensity",
                        "dend_noise_intensity",
                    ]
                    vivo_models[f"{slice_seed}_{int(row["Sim"]):04d}"] = [
                        updated_df[x].values for x in vivo_changes
                    ]

                run_model.run(compile=True, **params)

                bw_fr = 10
                bw_cv = 0.2
                pmr.plot_comparison_results(
                    stim_data_dir,
                    T1,
                    T0,
                    vivo,
                    DD=dopamine_depletion,
                    size=size,
                    bw_fr=bw_fr,
                    bw_cv=bw_cv,
                    show=show,
                )
                pmr.plot_network_statistics(
                    stim_data_dir, T1, T0, size, vivo=vivo, show=show
                )
                pmr.plot_egaba(stim_data_dir, T1, T0, size=size, show=show)
                pmr.plot_heterogeneity_with_mf(
                    stim_data_dir,
                    T2,
                    T1,
                    params,
                    d1stim,
                    gpestim,
                    size,
                    show=show,
                    vivo=vivo,
                )
                kspval, ttestpval, area, _, _, _ = pmr.plot_modulation_factors(
                    stim_data_dir,
                    gpe_pulse_exp if gpestim else d1_pulse_exp,
                    params,
                    T1,
                    T2,
                    d1stim,
                    gpestim,
                    size=size,
                    show=show,
                )
                d1_mfs[f"{slice_seed}_{int(row["Sim"]):04d}_{stim_seeds}"] = [
                    kspval,
                    ttestpval,
                    area,
                ]

    stim_dir = f"{data_dir}_dd_gpe_stim"
    for index, row in sorted_results.iterrows():
        # sim_to_pull_from = f"{data_dir}_dd_search/sim_{int(row["Sim"]):04d}"
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

            sim_to_pull_from = f"{data_dir}_dd_search/in_vivo_seed_{int(row["Sim"]):04d}/sim_{int(dd_row["Sim"]):06d}"

            for stim_seeds in np.arange(1, N_stim_seeds + 1):
                stim_data_dir = f"{stim_dir}/in_vivo_seed_{int(row["Sim"]):04d}/dd_sim_{int(dd_row["Sim"]):06d}/sim_{stim_seeds:04d}"
                gpestim = True  # True to run GPe stim experiment
                d1stim = False
                dopamine_depletion = True
                status = True
                dynamics = False

                params = run_model.get_default_params()
                params["T"] = T2
                params["save_dir"] = stim_data_dir
                params["size"] = size
                params["status"] = 1 if status else 0
                params["dynamics"] = 1 if dynamics else 0

                xgpe, xgpe_dd, xd1, xd1_dd = expan.ephys_exp_data()

                np.random.seed(
                    int(slice_seed * row["Sim"] + int(dd_row["Sim"]) * 2 + stim_seeds)
                )

                wstrs = np.zeros(size)
                tausyn_dend_dist = np.zeros(size)

                xs, ys, lbound, ubound, zs = expan.ephys_sample(
                    xgpe_dd["Conductance"], xgpe_dd["Time constant"]
                )
                wgpes = xs.copy() / 100 / 10
                tausyn_dist = zs.copy() * 1000

                params["W_gpe"] = wgpes  # xs / 100  # wgpes
                params["gpe_stim"] = np.ones(size)
                params["gpe_poisson"] = np.zeros(
                    size
                )  # 1 means continuous stim, 0 means pulsed
                params["gpe_stim_percent"] = np.ones(size)
                params["gpe_freq"] = 20 * np.ones(size)
                params["gpe_start_time"] = T1 * np.ones(size)
                params["gpe_base_length"] = np.ones(size)
                params["gpe_stim_length"] = np.ones(size)
                params["gpe_post_length"] = np.ones(size)

                params["W_str"] = wstrs  # wstrs
                params["str_stim"] = np.zeros(size)
                params["str_poisson"] = np.zeros(
                    size
                )  # 1 means continuous stim, 0 means pulsed
                params["str_stim_percent"] = np.zeros(size)
                params["str_freq"] = np.zeros(size)
                params["str_start_time"] = np.zeros(size)
                params["str_base_length"] = np.zeros(size)
                params["str_stim_length"] = np.zeros(size)
                params["str_post_length"] = np.zeros(size)

                tauexc_dist = run_model.positive_normal_distribution(5, 0, size)

                params["tausyn"] = tausyn_dist
                params["tauexc"] = tauexc_dist
                params["tausyn_dend"] = tausyn_dend_dist

                csv = f"{sim_to_pull_from}/heterogeneity.csv"
                df = pd.read_csv(csv)
                run_model.heterogeneity_from_simulation(sim_to_pull_from, **params)

                _, connections, total_weight = pmr.find_connections(
                    size, sim_to_pull_from
                )
                kcc2_D_vals = df["gKCC2_D_nS_pF"].values
                tonic_CL_D_vals = df["gTON_CL_D_MEAN_nS_pF"].values
                egaba_vals = np.zeros((size, 3))
                for i in range(size):
                    kcc2_val = kcc2_D_vals[i]
                    tonic_val = tonic_CL_D_vals[i]
                    egaba_vals[i] = [
                        kcc2_val,
                        tonic_val,
                        run_model.chloride_to_egaba(
                            optimize.newton(
                                run_model.chloride_dynamics_ss,
                                10,
                                args=(
                                    kcc2_val,
                                    tonic_val,
                                    -55,
                                ),
                            )
                        ),
                    ]
                sorted_egaba_vals = egaba_vals[egaba_vals[:, 2].argsort()]
                sorted_connections_ind = total_weight.argsort()
                sorted_wstrs = wstrs[wstrs.argsort()]
                sorted_tausyn_dend_dist = tausyn_dend_dist[wstrs.argsort()]
                sorted_wstrs_ind = wstrs.argsort()
                sorted_wgpes = wgpes[wgpes.argsort()]
                sorted_wgpes_ind = wgpes.argsort()
                for i in range(size):
                    egaba_vals[sorted_connections_ind[i], :] = sorted_egaba_vals[i, :]
                    if correlate_wstrs:
                        wstrs[sorted_connections_ind[i]] = sorted_wstrs[size - 1 - i]
                        tausyn_dend_dist[sorted_connections_ind[i]] = (
                            sorted_tausyn_dend_dist[size - 1 - i]
                        )
                    if correlate_wgpes:
                        wgpes[sorted_connections_ind[i]] = sorted_wgpes[size - 1 - i]

                df["tausyn"] = tausyn_dist
                df["tauexc"] = tauexc_dist
                df["tausyn_dend"] = tausyn_dend_dist
                update_dict = {
                    "W_str": wstrs,
                    "W_gpe": wgpes,
                    "tausyn": tausyn_dist,  # tausyn_dist if gpestim else np.zeros(size),
                    "tauexc": tauexc_dist,
                    "tausyn_dend_dist": tausyn_dend_dist,  # tausyn_dend_dist if d1stim else np.zeros(size),
                }
                run_model.update_by_dict(df, update_dict, **params)

                run_model.set_clin(**params)
                run_model.weights_from_simulation(sim_to_pull_from, **params)

                if gpestim or d1stim:
                    run_model.add_stim(**params)

                run_model.run(compile=True, **params)

                bw_fr = 10
                bw_cv = 0.2
                pmr.plot_comparison_results(
                    stim_data_dir,
                    T1,
                    T0,
                    vivo,
                    DD=dopamine_depletion,
                    size=size,
                    bw_fr=bw_fr,
                    bw_cv=bw_cv,
                    show=show,
                )
                pmr.plot_network_statistics(
                    stim_data_dir, T1, T0, size, vivo=vivo, show=show
                )
                pmr.plot_egaba(stim_data_dir, T1, T0, size=size, show=show)
                pmr.plot_heterogeneity_with_mf(
                    stim_data_dir,
                    T2,
                    T1,
                    params,
                    d1stim,
                    gpestim,
                    size,
                    show=show,
                    vivo=vivo,
                )
                kspval, ttestpval, area, _, _, _ = pmr.plot_modulation_factors(
                    stim_data_dir,
                    gpe_pulse_exp if gpestim else d1_pulse_exp,
                    params,
                    T1,
                    T2,
                    d1stim,
                    gpestim,
                    size=size,
                    show=show,
                )
                gpe_mfs[f"{slice_seed}_{int(row["Sim"]):04d}_{stim_seeds}"] = [
                    kspval,
                    ttestpval,
                    area,
                ]
    seeds_completed.write(f"Completed Seed: {slice_seed}\n")
    seeds_completed.close()

d1_seeds = np.asarray(d1_seeds)
gpe_seeds = np.asarray(gpe_seeds)
print(len(d1_seeds), len(np.unique(d1_seeds)))
print(len(gpe_seeds), len(np.unique(gpe_seeds)))
print(len(np.unique(np.concatenate([d1_seeds, gpe_seeds]))))

sys.exit()
fig, ax = plt.subplots(2, 3, figsize=(16, 6), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(3)]
count = 0
for key in vivo_models:
    axes[0].scatter(vivo_models[key][0], np.ones(size) * count)
    axes[1].scatter(vivo_models[key][1], np.ones(size) * count)
    axes[2].scatter(vivo_models[key][2], np.ones(size) * count)
    axes[3].scatter(vivo_models[key][3], np.ones(size) * count)
    axes[4].scatter(vivo_models[key][4], np.ones(size) * count)
    count += 1
plt.show()

sys.exit()

fig, ax = plt.subplots(2, 3, figsize=(16, 6), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(2) for j in range(3)]

count = 0
for key in d1_mfs:
    axes[0].bar(
        count, height=d1_mfs[key][0], width=1, edgecolor="w", color="blue", align="edge"
    )
    axes[1].bar(
        count, height=d1_mfs[key][1], width=1, edgecolor="w", color="blue", align="edge"
    )
    axes[2].bar(
        count, height=d1_mfs[key][2], width=1, edgecolor="w", color="blue", align="edge"
    )
    axes[3].bar(
        count,
        height=gpe_mfs[key][0],
        width=1,
        edgecolor="w",
        color="blue",
        align="edge",
    )
    axes[4].bar(
        count,
        height=gpe_mfs[key][1],
        width=1,
        edgecolor="w",
        color="blue",
        align="edge",
    )
    axes[5].bar(
        count,
        height=gpe_mfs[key][2],
        width=1,
        edgecolor="w",
        color="blue",
        align="edge",
    )
    count += 1
labels = [key for key in d1_mfs]
for i in range(len(axes)):
    axes[i].set_xticks(range(count))
    axes[i].set_xticklabels(labels, rotation=90, fontsize=2)
    axes[i].set_xlim([0, count])

axes[0].set_ylabel("D1 KS")
axes[1].set_ylabel("D1 Ttest")
axes[2].set_ylabel("D1 Area")
axes[3].set_ylabel("GPe KS")
axes[4].set_ylabel("GPe Ttest")
axes[5].set_ylabel("GPe Area")

makeNice(axes)
add_fig_labels(axes)
fig.savefig("/Users/johnparker/Desktop/d1_gpe_stim_test.pdf", bbox_inches="tight")
plt.close()

run_cmd("open /Users/johnparker/Desktop/d1_gpe_stim_test.pdf")
