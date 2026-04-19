import numpy as np
import multiprocessing, sys
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy.stats import ks_2samp
import pandas as pd
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


def run_mult_commands(save_dir, df, exp_fr_outliers_removed, sim_params):
    size = 100
    run_cmd(f"mkdir -p {save_dir}", print_out=False)
    sim_params["gSTN_ton_std_pct"] = 0
    run_model.heterogeneity_from_csv(df, exp_fr_outliers_removed, **sim_params)

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
    run_model.run(compile=False, **sim_params)

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
    analyze = True
    gen_data_file = True
    Nseeds = 300

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

    data_dir = "data/param_seed_finder_slice_correlated"

    results = pd.read_csv(f"{data_dir}/results.csv")
    results = results[
        (results["KS_FR_pval"] >= 0.05)
        & (results["Ttest_FR_pval"] >= 0.05)
        & (results["Ttest_CV_pval"] >= 0.05)
    ]

    sorted_results = results.sort_values(by="Avg. Error")

    param_set = []
    if run:
        for seed in np.arange(1, Nseeds + 1):
            np.random.seed(seed)
            csv = "data/param_search_slice_correlated/match_neurons.csv"
            df = pd.read_csv(csv)
            save_dir = f"{data_dir}/sim_{seed:04d}"

            params = run_model.get_default_params()
            params["T"] = T
            params["save_dir"] = save_dir
            params["size"] = size
            params["print_out"] = False
            params["status"] = 0
            params["dynamics"] = 0

            params["W_SNr_mean"] = np.mean(df["total_weight"]) / np.mean(
                df["connections"]
            )
            params["prob"] = np.mean(df["connections"]) / 100  # 0.015
            params["W_SNr_std_pct"] = np.std(df["total_weight"]) / np.mean(
                df["total_weight"]
            )
            param_set.append([save_dir, df, exp_fr_outliers_removed, params])

        pool = multiprocessing.Pool(
            multiprocessing.cpu_count() - 2
        )  # Create a worker pool based on CPUs on machine
        pool.starmap(
            run_mult_commands, param_set
        )  # Analyze the trials of each cell, in parallel
        pool.close()  # Close the worker pool
        pool.join()  # Bring the workers back together
    elif analyze and gen_data_file:
        data = []
        bw_fr = 10
        bw_cv = 0.2
        for seed in np.arange(1, Nseeds + 1):
            save_dir = f"{data_dir}/sim_{seed:04d}"
            sim_spikes, sim_df = pmr.gather_sim_data(save_dir, T, T0, size=size)
            sim_fr = sim_df["pre_exp_freq"]
            sim_cv = sim_df["pre_exp_cv"]
            percent_firing = len(sim_fr[sim_fr > 0]) / len(sim_fr)

            meta_data = dict()
            meta_data["pre_exp_fr"] = sim_fr
            meta_data["pre_exp_cv"] = sim_cv

            meta_data["pre_exp_fr_zscore"] = stats.zscore(meta_data["pre_exp_fr"])
            meta_data["pre_exp_cv_zscore"] = stats.zscore(meta_data["pre_exp_cv"])

            meta_data = pd.DataFrame(meta_data)

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
            avg_fr_error = (avg_fr - np.mean(exp_fr_outliers_removed)) / np.mean(
                exp_fr_outliers_removed
            )
            avg_cv_error = (avg_cv - np.mean(exp_cv_outliers_removed)) / np.mean(
                exp_cv_outliers_removed
            )

            data.append(
                [
                    seed,
                    ks_2samp(sim_fr, exp_fr_outliers_removed).pvalue,
                    ks_2samp(sim_cv, exp_cv_outliers_removed).pvalue,
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
                    avg_fr_error,
                    avg_cv_error,
                    np.std(sim_fr),
                    np.std(sim_cv),
                ]
            )
        df = pd.DataFrame(
            data,
            columns=[
                "Seed",
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
                "Avg. FR Rel. Error",
                "Avg. CV Rel. Error",
                "STD FR",
                "STD CV",
            ],
        )

        df.to_csv(f"{data_dir}/results.csv")
        data = np.asarray(data)
        np.savetxt(
            f"{data_dir}/results.txt", data, fmt="%f", newline="\n", delimiter="\t"
        )
