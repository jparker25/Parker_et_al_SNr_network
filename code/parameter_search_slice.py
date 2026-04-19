import numpy as np
import multiprocessing, sys
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy.stats import ks_2samp
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats
from scipy import optimize

# import user modules
sys.path.append("../")
sys.path.append("../../")
from helpers import *
import network_analysis as na
import run_model
import plot_model_results as pmr
import experimental_analysis as expan


def score_sim(df):
    score = np.zeros(len(df))
    count = 0
    for index, row in df.iterrows():
        score[count] = np.sum(
            np.array(
                [
                    row["KS_FR_pval"] > 0.05,
                    row["KS_CV_pval"] > 0.05,
                    row["Ttest_FR_pval"] < 0.05,
                    row["Ttest_CV_pval"] < 0.05,
                    row["Rel. Error FR"] < 0.4,
                    row["Rel. Error CV"] < 0.4,
                    row["Percent Firing"] > 0.75,
                ]
            )
        )
        count += 1
    df["score"] = score
    return df


def label_point(x, y, val, ax):
    for s in range(val.shape[0]):
        ax.text(x[s], y[s] + 0.01, str(int(val[s])))


def run_mult_commands(save_dir, sim_params):
    size = 100
    sim_params["save_dir"] = save_dir
    run_cmd(f"mkdir -p {save_dir}", print_out=False)
    sim_params["gSTN_ton_std_pct"] = 0

    run_model.generate_heterogeneity_matrix(**sim_params)
    run_model.scale_dend_current_noise(**sim_params)

    run_model.match_soma_dendrite_conductances(**sim_params)
    run_model.generate_random_weights(**sim_params)

    df = pd.read_csv(f"{save_dir}/heterogeneity.csv")

    _, _, total_weight = pmr.find_connections(
        size,
        save_dir,
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
    for i in range(size):
        egaba_vals[sorted_connections_ind[i], :] = sorted_egaba_vals[i, :]

    update_dict = {
        "gKCC2_S_nS_pF": egaba_vals[:, 0],
        "gTON_CL_S_MEAN_nS_pF": egaba_vals[:, 1],
        "gKCC2_D_nS_pF": egaba_vals[:, 0],
        "gTON_CL_D_MEAN_nS_pF": egaba_vals[:, 1],
    }
    run_model.update_by_dict(df, update_dict, **sim_params)
    run_model.set_clin(**sim_params)
    run_model.run(**sim_params)

    pmr.plot_comparison_results(
        save_dir,
        5,  # T, default 5
        3,  # T0, default 3
        False,  # vivo, true
        size=100,  # default 100
        bw_fr=10,
        bw_cv=0.2,
        show=False,
        all_features=False,
    )
    pmr.plot_egaba(save_dir, 5, 3, size=100, show=False)


if __name__ == "__main__":
    size = 100
    T = 5
    T0 = 3
    vivo = False
    run = False
    plots = False
    analyze = True
    gen_data_file = False
    size = 100

    exp_data = (
        expan.get_slice_baseline_data()
        if not vivo
        else expan.get_in_vivo_baseline_data_short(segment=2, DD=False)
    )

    exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]

    exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(
        exp_data
    )
    exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
    exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

    exp_fr_outliers_removed, exp_cv_outliers_removed = (
        exp_data["pre_exp_freq"],
        exp_data["pre_exp_cv"],
    )

    data_dir = "data/param_search_slice_correlated"

    df = pd.read_csv(f"{data_dir}/results.csv")
    df = df[
        (df["KS_FR_pval"] >= 0.05)
        & (df["Ttest_FR_pval"] >= 0.05)
        & (df["Ttest_CV_pval"] >= 0.05)
        # & (df["KS_CV_pval"] >= 0.05)
    ]

    if run:
        param_set = []
        sim = 1
        for seed in [5]:
            for w in np.arange(0.1, 0.5, 0.1):
                for heterogeneity in np.arange(0.1, 0.5, 0.1):
                    for gkcc2 in np.arange(0.001, 0.008, 0.001):
                        for gtonic in np.arange(0.001, 0.007, 0.001):
                            for noise in [0.001, 0.01]:
                                save_dir = f"{data_dir}/test/sim_{sim:04d}"

                                params = run_model.get_default_params()
                                params["print_out"] = False

                                params["T"] = T
                                params["prob"] = (
                                    1.16 / 81
                                )  # from Higgs & Wilson, #0.015
                                params["dynamics"] = 0
                                params["status"] = 0
                                params["save_dir"] = save_dir

                                params["gKCC2_S_mean"] = gkcc2
                                params["gKCC2_D_mean"] = gkcc2
                                params["gGABA_S_ton_mean"] = gtonic
                                params["gGABA_D_ton_mean"] = gtonic

                                params["gGABA_S_ton_noise_sigma"] = 0.001
                                params["gGABA_S_ton_noise_theta"] = 0.001
                                params["gGABA_D_ton_noise_sigma"] = 0.001
                                params["gGABA_D_ton_noise_theta"] = 0.001

                                params["current_noise_intensity_S"] = noise
                                params["current_noise_intensity_D"] = noise
                                params["current_noise_intensity_theta_S"] = 0.1
                                params["current_noise_intensity_theta_D"] = 0.1
                                params["current_noise_intensity_S_pct"] = heterogeneity
                                params["current_noise_intensity_D_pct"] = heterogeneity

                                params["Eleak_mean"] = -58
                                params["Eleak_std_pct"] = 0.025
                                params["W_SNr_mean"] = w
                                params["W_SNr_std_pct"] = heterogeneity
                                params = run_model.uniform_conductance_heterogeneity(
                                    heterogeneity, **params
                                )
                                params["gSTN_ton_std_pct"] = 0
                                param_set.append([save_dir, params])
                                sim += 1

        # run_cmd(f"g++ -std=c++17 -O2 SNr_adapted.cc -o SNr")
        pool = multiprocessing.Pool(
            multiprocessing.cpu_count() - 2
        )  # Create a worker pool based on CPUs on machine
        pool.starmap(
            run_mult_commands, param_set
        )  # Analyze the trials of each cell, in parallel
        pool.close()  # Close the worker pool
        pool.join()  # Bring the workers back together
        # sys.exit()
    if plots:
        sim = 1
        for seed in [5]:
            for w in np.arange(0.1, 0.5, 0.1):
                for heterogeneity in np.arange(0.1, 0.5, 0.1):
                    for gkcc2 in np.arange(0.001, 0.008, 0.001):
                        for gtonic in np.arange(0.001, 0.007, 0.001):
                            for noise in [0.001, 0.01]:
                                save_dir = f"{data_dir}/test/sim_{sim:04d}"
                                bw_fr = 10
                                bw_cv = 0.2
                                pmr.plot_comparison_results(
                                    save_dir,
                                    T,
                                    T0,
                                    vivo,
                                    size=size,
                                    bw_fr=bw_fr,
                                    bw_cv=bw_cv,
                                    show=False,
                                )
                                """
                                pmr.plot_network_statistics(
                                    save_dir, T, T0, size, vivo=vivo, show=False
                                )
                                """
                                sim += 1
                                print(sim - 1)

    if analyze:
        if gen_data_file:
            data = []
            neurons = pd.DataFrame()

            bw_fr = 10
            bw_cv = 0.2
            sim = 1
            for seed in [5]:
                for w in np.arange(0.1, 0.5, 0.1):
                    for heterogeneity in np.arange(0.1, 0.5, 0.1):
                        for gkcc2 in np.arange(0.001, 0.008, 0.001):
                            for gtonic in np.arange(0.001, 0.007, 0.001):
                                for noise in [0.001, 0.01]:
                                    save_dir = f"{data_dir}/test/sim_{sim:04d}"
                                    sim_spikes, sim_df = pmr.gather_sim_data(
                                        save_dir, T, T0, size=size
                                    )
                                    sim_fr = sim_df["pre_exp_freq"]
                                    sim_cv = sim_df["pre_exp_cv"]

                                    percent_firing = len(sim_fr[sim_fr > 0]) / len(
                                        sim_fr
                                    )
                                    meta_data = pmr.get_neuron_meta_data(
                                        save_dir, size=size
                                    )

                                    meta_data["pre_exp_fr"] = sim_fr
                                    meta_data["pre_exp_cv"] = sim_cv

                                    meta_data["pre_exp_fr_zscore"] = stats.zscore(
                                        meta_data["pre_exp_fr"]
                                    )
                                    meta_data["pre_exp_cv_zscore"] = stats.zscore(
                                        meta_data["pre_exp_cv"]
                                    )

                                    _, connections, total_weight = pmr.find_connections(
                                        size, save_dir
                                    )

                                    meta_data["connections"] = connections
                                    meta_data["total_weight"] = total_weight

                                    meta_data = pd.DataFrame(meta_data)
                                    meta_data["sim"] = sim

                                    neurons = pd.concat([neurons, meta_data])

                                    meta_data = meta_data[
                                        (np.abs(meta_data["pre_exp_fr_zscore"]) <= 3)
                                        & (np.abs(meta_data["pre_exp_cv_zscore"]) <= 3)
                                        & (np.abs(meta_data["pre_exp_fr"]) > 0)
                                    ]

                                    sim_fr = meta_data["pre_exp_fr"]
                                    sim_cv = meta_data["pre_exp_cv"]

                                    rel_error_fr = na.percent_error_fr(
                                        sim_fr, exp_fr_outliers_removed, bin_width=bw_fr
                                    )
                                    rel_error_cv = na.percent_error_cv(
                                        sim_cv, exp_cv_outliers_removed, bin_width=bw_cv
                                    )

                                    avg_fr = np.mean(sim_fr)
                                    avg_cv = np.mean(sim_cv)

                                    data.append(
                                        [
                                            sim,
                                            w,
                                            heterogeneity,
                                            gkcc2,
                                            gtonic,
                                            noise,
                                            ks_2samp(
                                                sim_fr, exp_fr_outliers_removed
                                            ).pvalue,
                                            ks_2samp(
                                                sim_cv, exp_cv_outliers_removed
                                            ).pvalue,
                                            stats.ttest_ind(
                                                sim_fr,
                                                exp_fr_outliers_removed,
                                            ).pvalue,
                                            stats.ttest_ind(
                                                sim_cv,
                                                exp_cv_outliers_removed,
                                            ).pvalue,
                                            rel_error_fr,
                                            rel_error_cv,
                                            (rel_error_fr + rel_error_cv) / 2,
                                            percent_firing,
                                            avg_fr,
                                            avg_cv,
                                            np.std(sim_fr),
                                            np.std(sim_cv),
                                        ]
                                    )
                                    sim += 1
                                    print(sim)

            df = pd.DataFrame(
                data,
                columns=[
                    "Sim",
                    "Wsnr",
                    "Heterogeneity",
                    "gKCC2",
                    "gGABA_tonic",
                    "current_noise",
                    "KS_FR_pval",
                    "KS_CV_pval",
                    "Ttest_FR_pval",
                    "Ttest_CV_pval",
                    "Rel. Error FR",
                    "Rel. Error CV",
                    "Avg. Error",
                    "Percent Firing",
                    "Avg. FR",
                    "Avg. CV",
                    "STD FR",
                    "STD CV",
                ],
            )
            neurons.to_csv(f"{data_dir}/neurons.csv")
            df.to_csv(f"{data_dir}/results.csv")
            data = np.asarray(data)
            np.savetxt(
                f"{data_dir}/results.txt", data, fmt="%f", newline="\n", delimiter="\t"
            )
        else:
            neurons = pd.read_csv(f"{data_dir}/neurons.csv")
            match_neurons = pd.DataFrame()
            for i in range(len(exp_fr_outliers_removed)):
                if (
                    exp_fr_outliers_removed.iloc[i] > 0
                    and exp_cv_outliers_removed.iloc[i] > 0
                ):
                    neurons["rel_error"] = (
                        np.abs(neurons["pre_exp_fr"] - exp_fr_outliers_removed.iloc[i])
                        * 100
                        / exp_fr_outliers_removed.iloc[i]
                        + np.abs(
                            neurons["pre_exp_cv"] - exp_cv_outliers_removed.iloc[i]
                        )
                        * 100
                        / exp_cv_outliers_removed.iloc[i]
                    ) / 2
                    match_neurons = pd.concat(
                        [
                            match_neurons,
                            neurons[neurons.rel_error == neurons.rel_error.min()],
                        ]
                    )
                match_neurons.to_csv(f"{data_dir}/match_neurons.csv")
