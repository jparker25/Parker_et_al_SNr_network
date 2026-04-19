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


def run_mult_commands(
    save_dir, df, update_dict, sim_params, slice_seed, correlate_egaba, sim, size
):
    run_cmd(f"mkdir -p {save_dir}", print_out=False)
    run_model.heterogeneity_from_simulation(
        f"data/param_seed_finder_slice_correlated/sim_{slice_seed:04d}",
        **sim_params,
    )

    sim_params["W_SNr_mean"] = np.mean(df["total_weight"]) / np.mean(df["connections"])
    sim_params["prob"] = 2 * np.mean(df["connections"]) / 100  # 0.015
    sim_params["W_SNr_std_pct"] = np.std(df["total_weight"]) / np.mean(
        df["total_weight"]
    )
    run_model.generate_random_weights(**sim_params)
    run_model.update_by_dict(df, update_dict, **sim_params)

    if correlate_egaba:
        csv = f"{save_dir}/heterogeneity.csv"
        df = pd.read_csv(csv)
        _, connections, total_weight = pmr.find_connections(
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

        update_dict["gKCC2_S_nS_pF"] = egaba_vals[:, 0]

        update_dict["gTON_CL_S_MEAN_nS_pF"] = egaba_vals[:, 1]

        update_dict["gKCC2_D_nS_pF"] = egaba_vals[:, 0]

        update_dict["gTON_CL_D_MEAN_nS_pF"] = egaba_vals[:, 1]

        run_model.update_by_dict(df, update_dict, **sim_params)
    run_model.set_clin(**sim_params)

    run_model.run(**sim_params)

    pmr.plot_comparison_results(
        save_dir,
        5,  # T, default 5
        3,  # T0, default 3
        True,  # vivo, true
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
    run = False
    analyze = True
    gen_data_file = True
    correlate_egaba = True
    vivo = True

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

    slice_results = pd.read_csv(f"data/param_seed_finder_slice_correlated/results.csv")

    slice_results = slice_results[
        (slice_results["KS_FR_pval"] >= 0.05)
        & (slice_results["Ttest_FR_pval"] >= 0.05)
        & (slice_results["Ttest_CV_pval"] >= 0.05)
    ]
    avg_err_sorted = slice_results.sort_values(by="Avg. Error")

    slice_seeds = [int(row["Seed"]) for index, row in avg_err_sorted.iterrows()]

    vivo_sims = 0
    for slice_seed in slice_seeds:
        data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
        results = pd.read_csv(f"{data_dir}/results.csv")
        results = results[
            (results["KS_FR_pval"] >= 0.05)
            & (results["Ttest_FR_pval"] >= 0.05)
            & (results["Ttest_CV_pval"] >= 0.05)
            & (results["KS_CV_pval"] >= 0.05)
        ]
        vivo_sims += len(results)
    extra_noise_factor = 4

    if run:
        for slice_seed in slice_seeds:
            data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
            param_set = []
            sim = 1
            for seed in [5]:
                for vivo_tonic_scale in np.arange(1.1, 3.1, 0.3):
                    for vivo_noise_scale in [2, 5, 10, 20]:
                        for current_noise_scale in [2, 5, 10, 20]:
                            for gstn_mu in np.arange(0.01, 0.06, 0.02):
                                for gstn_pct in np.arange(0.1, 0.6, 0.1):

                                    save_dir = f"{data_dir}/sim_{sim:04d}"
                                    sim_params = run_model.get_default_params()

                                    sim_params["save_dir"] = save_dir

                                    sim_params["T"] = T
                                    sim_params["prob"] = 0.015 * 2
                                    sim_params["dynamics"] = 0
                                    sim_params["status"] = 0
                                    sim_params["print_out"] = False
                                    save_dir = f"{data_dir}/sim_{sim:04d}"

                                    sim_params["gSTN_ton_mean"] = gstn_mu
                                    sim_params["gSTN_ton_std_pct"] = gstn_pct
                                    sim_params["gSTN_ton_noise_sigma"] = (
                                        0.001 * vivo_noise_scale
                                    )
                                    sim_params["gSTN_ton_noise_theta"] = 0.001

                                    csv = f"data/param_seed_finder_slice_correlated/sim_{slice_seed:04d}/heterogeneity.csv"
                                    df = pd.read_csv(csv)
                                    ones = np.ones(size)
                                    update_dict = {
                                        "gTON_STN_MEAN_nS_pF": run_model.positive_normal_distribution(
                                            sim_params["gSTN_ton_mean"],
                                            sim_params["gSTN_ton_mean"]
                                            * sim_params["gSTN_ton_std_pct"],
                                            sim_params["size"],
                                        ),
                                        "gTON_STN_SIGMA": ones
                                        * sim_params["gSTN_ton_noise_sigma"],
                                        "gTON_STN_THETA": ones
                                        * sim_params["gSTN_ton_noise_theta"],
                                        "gTON_CL_S_MEAN_nS_pF": vivo_tonic_scale
                                        * df["gTON_CL_S_MEAN_nS_pF"].values,
                                        "gTON_CL_S_SIGMA": vivo_noise_scale
                                        * df["gTON_CL_S_SIGMA"].values,
                                        "gTON_CL_D_MEAN_nS_pF": vivo_tonic_scale
                                        * df["gTON_CL_D_MEAN_nS_pF"].values,
                                        "gTON_CL_D_SIGMA": vivo_noise_scale
                                        * df["gTON_CL_D_SIGMA"].values,
                                        "soma_noise_intensity": current_noise_scale
                                        * extra_noise_factor
                                        * df["soma_noise_intensity"].values,
                                        "dend_noise_intensity": current_noise_scale
                                        * extra_noise_factor
                                        * df["dend_noise_intensity"].values,
                                    }

                                    param_set.append(
                                        [
                                            save_dir,
                                            df,
                                            update_dict,
                                            sim_params,
                                            slice_seed,
                                            correlate_egaba,
                                            sim,
                                            size,
                                        ]
                                    )
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

    if analyze:
        if gen_data_file:
            for slice_seed in slice_seeds:
                data_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
                print(data_dir)
                data = []
                neurons = pd.DataFrame()

                bw_fr = 10
                bw_cv = 0.2
                sim = 1
                for seed in [5]:
                    for vivo_tonic_scale in np.arange(1.1, 3.1, 0.3):
                        for vivo_noise_scale in [2, 5, 10, 20]:
                            for current_noise_scale in [2, 5, 10, 20]:
                                for gstn_mu in np.arange(0.01, 0.06, 0.02):
                                    for gstn_pct in np.arange(0.1, 0.6, 0.1):

                                        save_dir = f"{data_dir}/sim_{sim:04d}"
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

                                        _, connections, total_weight = (
                                            pmr.find_connections(size, save_dir)
                                        )

                                        meta_data["connections"] = connections
                                        meta_data["total_weight"] = total_weight

                                        meta_data = pd.DataFrame(meta_data)
                                        meta_data["sim"] = sim

                                        neurons = pd.concat([neurons, meta_data])

                                        meta_data = meta_data[
                                            (
                                                np.abs(meta_data["pre_exp_fr_zscore"])
                                                <= 3
                                            )
                                            & (
                                                np.abs(meta_data["pre_exp_cv_zscore"])
                                                <= 3
                                            )
                                            & (np.abs(meta_data["pre_exp_fr"]) > 0)
                                        ]

                                        sim_fr = meta_data["pre_exp_fr"]
                                        sim_cv = meta_data["pre_exp_cv"]

                                        rel_error_fr = na.percent_error_fr(
                                            sim_fr,
                                            exp_fr_outliers_removed,
                                            bin_width=bw_fr,
                                        )
                                        rel_error_cv = na.percent_error_cv(
                                            sim_cv,
                                            exp_cv_outliers_removed,
                                            bin_width=bw_cv,
                                        )

                                        avg_fr = np.mean(sim_fr)
                                        avg_cv = np.mean(sim_cv)

                                        avg_fr_error = (
                                            avg_fr - np.mean(exp_fr_outliers_removed)
                                        ) / np.mean(exp_fr_outliers_removed)
                                        avg_cv_error = (
                                            avg_cv - np.mean(exp_cv_outliers_removed)
                                        ) / np.mean(exp_cv_outliers_removed)

                                        data.append(
                                            [
                                                sim,
                                                vivo_tonic_scale,
                                                vivo_noise_scale,
                                                current_noise_scale
                                                * extra_noise_factor,
                                                gstn_mu,
                                                gstn_pct,
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
                                                avg_fr_error,
                                                avg_cv_error,
                                                np.std(sim_fr),
                                                np.std(sim_cv),
                                            ]
                                        )
                                        print(sim)
                                        sim += 1

                df = pd.DataFrame(
                    data,
                    columns=[
                        "Sim",
                        "vivo_tonic_scale",
                        "vivo_noise_scale",
                        "current_noise_scale",
                        "gstn_mu",
                        "gstn_pct",
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
                neurons.to_csv(f"{data_dir}/neurons.csv")
                df.to_csv(f"{data_dir}/results.csv")
                data = np.asarray(data)
                np.savetxt(
                    f"{data_dir}/results.txt",
                    data,
                    fmt="%f",
                    newline="\n",
                    delimiter="\t",
                )
        else:
            for slice_seed in slice_seeds:
                data_dir = f"data/param_search_vivo_from_slice_{slice_seed}"
                print(data_dir)
                neurons = pd.read_csv(f"{data_dir}/neurons.csv")
                match_neurons = pd.DataFrame()
                for i in range(len(exp_fr_outliers_removed)):
                    if (
                        exp_fr_outliers_removed.iloc[i] > 0
                        and exp_cv_outliers_removed.iloc[i] > 0
                    ):
                        neurons["rel_error"] = (
                            np.abs(
                                neurons["pre_exp_fr"] - exp_fr_outliers_removed.iloc[i]
                            )
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
