import numpy as np
import multiprocessing, sys
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy.stats import ks_2samp
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import optimize


# import user modules
sys.path.append("../")
sys.path.append("../../")
from helpers import *
import network_analysis as na
import run_model
import plot_model_results as pmr
import experimental_analysis as expan


# def run_mult_commands(save_dir, df, update_dict, sim_params):
def run_mult_commands(
    save_dir, source_dir, kcc2_S, kcc2_D, trpc3, tonic_CL_S, tonic_CL_D, stn, noise
):
    sim_params = run_model.get_default_params()
    sim_params["save_dir"] = save_dir

    sim_params["T"] = 5  # T
    sim_params["dynamics"] = 0
    sim_params["status"] = 0
    sim_params["print_out"] = False

    csv = f"{source_dir}/heterogeneity.csv"
    df = pd.read_csv(csv)
    run_model.heterogeneity_from_simulation(
        f"{source_dir}",
        **sim_params,
    )

    run_model.weights_from_simulation(f"{source_dir}", **sim_params)

    update_dict = {
        "soma_noise_intensity": (noise * df["soma_noise_intensity"].values),
        "dend_noise_intensity": (noise * df["dend_noise_intensity"].values),
        "gTRPC3_nS_pF": (trpc3 * df["gTRPC3_nS_pF"].values),
        "gKCC2_S_nS_pF": (kcc2_S * df["gKCC2_S_nS_pF"].values),
        "gTON_CL_S_MEAN_nS_pF": (tonic_CL_S * df["gTON_CL_S_MEAN_nS_pF"].values),
        "gKCC2_D_nS_pF": (kcc2_D * df["gKCC2_D_nS_pF"].values),
        "gTON_CL_D_MEAN_nS_pF": (tonic_CL_D * df["gTON_CL_D_MEAN_nS_pF"].values),
        "gTON_STN_MEAN_nS_pF": (stn * df["gTON_STN_MEAN_nS_pF"].values),
    }
    run_cmd(f"mkdir -p {save_dir}", print_out=False)
    run_model.update_by_dict(df, update_dict, **sim_params)
    run_model.set_clin(**sim_params)
    run_model.run(**sim_params)

    bw_fr = 10
    bw_cv = 0.2
    pmr.plot_comparison_results(
        save_dir,
        5,
        3,
        True,
        DD=True,
        size=100,
        bw_fr=bw_fr,
        bw_cv=bw_cv,
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
    vivo = True
    dopamine_depletion = True
    correlate_egaba = True
    correlate_wstrs = True
    correlate_wgpes = False

    N_vivo_sims = 3

    exp_data = (
        expan.get_slice_baseline_data()
        if not vivo
        else expan.get_in_vivo_baseline_data_short(segment=2, DD=dopamine_depletion)
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
        sorted_results = results.sort_values(by="Avg. Error")
        sorted_results = sorted_results.iloc[:N_vivo_sims]

    # decrease_scale = np.arange(0.25, 1.25, 0.25)
    decrease_scale = [0.33, 0.66, 1]
    increase_scale = [1, 1.25, 1.5]  # np.arange(1, 1.75, 0.25)
    noise_increase_scale = [1, 1.5, 2]

    for slice_seed in slice_seeds:
        slice_dir = (
            f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}_dd_search"
        )
        results = pd.read_csv(f"{slice_dir}/results.csv")
        results = results[
            (results["KS_FR_pval"] >= 0.05)
            & (results["Ttest_FR_pval"] >= 0.05)
            & (results["Ttest_CV_pval"] >= 0.05)
            & (results["KS_CV_pval"] >= 0.05)
        ]
        sorted_results = results.sort_values(by="Avg. Error")
        sorted_results = sorted_results.iloc[:N_vivo_sims]["Sim"].values
        for vivo_sim in sorted_results:
            dd_results = pd.read_csv(
                f"{slice_dir}_dd_search/in_vivo_seed_{int(vivo_sim):04d}/results.csv"
            )
            dd_results = dd_results[
                (dd_results["KS_FR_pval"] >= 0.05)
                & (dd_results["Ttest_FR_pval"] >= 0.05)
                & (dd_results["Ttest_CV_pval"] >= 0.05)
                & (dd_results["KS_CV_pval"] >= 0.05)
            ]
            sorted_dd_results = dd_results.sort_values(by="Avg. Error")

            print(slice_seed, vivo_sim, len(sorted_dd_results))

    if run:
        for slice_seed in slice_seeds:
            slice_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
            results = pd.read_csv(f"{slice_dir}/results.csv")
            results = results[
                (results["KS_FR_pval"] >= 0.05)
                & (results["Ttest_FR_pval"] >= 0.05)
                & (results["Ttest_CV_pval"] >= 0.05)
                & (results["KS_CV_pval"] >= 0.05)
            ]
            sorted_results = results.sort_values(by="Avg. Error")
            sorted_results = sorted_results.iloc[:N_vivo_sims]["Sim"].values
            for vivo_sim in sorted_results:
                source_dir = f"{slice_dir}/sim_{int(vivo_sim):04d}"

                param_set = []
                sim = 1
                for kcc2_S in decrease_scale:
                    for kcc2_D in decrease_scale:
                        for trpc3 in decrease_scale:
                            for tonic_CL_S in decrease_scale:
                                for tonic_CL_D in decrease_scale:
                                    for stn in increase_scale:
                                        for noise in noise_increase_scale:

                                            save_dir = f"{slice_dir}_dd_search/in_vivo_seed_{int(vivo_sim):04d}/sim_{sim:06d}"
                                            param_set.append(
                                                [
                                                    save_dir,
                                                    source_dir,
                                                    kcc2_S,
                                                    kcc2_D,
                                                    trpc3,
                                                    tonic_CL_S,
                                                    tonic_CL_D,
                                                    stn,
                                                    noise,
                                                ]
                                            )
                                            sim += 1
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
                slice_dir = f"data/param_search_vivo_from_slice/slice_seed_{slice_seed}"
                results = pd.read_csv(f"{slice_dir}/results.csv")
                results = results[
                    (results["KS_FR_pval"] >= 0.05)
                    & (results["Ttest_FR_pval"] >= 0.05)
                    & (results["Ttest_CV_pval"] >= 0.05)
                    & (results["KS_CV_pval"] >= 0.05)
                ]
                sorted_results = results.sort_values(by="Avg. Error")
                sorted_results = sorted_results.iloc[:N_vivo_sims]["Sim"].values
                for vivo_sim in sorted_results:
                    source_dir = f"{slice_dir}/sim_{int(vivo_sim):04d}"
                    data = []
                    neurons = pd.DataFrame()

                    bw_fr = 10
                    bw_cv = 0.2
                    sim = 1
                    for seed in [5]:
                        for kcc2_S in decrease_scale:
                            for kcc2_D in decrease_scale:
                                for trpc3 in decrease_scale:
                                    for tonic_CL_S in decrease_scale:
                                        for tonic_CL_D in decrease_scale:
                                            for stn in increase_scale:
                                                for noise in noise_increase_scale:

                                                    save_dir = f"{slice_dir}_dd_search/in_vivo_seed_{int(vivo_sim):04d}/sim_{sim:06d}"

                                                    sim_spikes, sim_df = (
                                                        pmr.gather_sim_data(
                                                            save_dir, T, T0, size=size
                                                        )
                                                    )
                                                    sim_fr = sim_df["pre_exp_freq"]
                                                    sim_cv = sim_df["pre_exp_cv"]

                                                    percent_firing = len(
                                                        sim_fr[sim_fr > 0]
                                                    ) / len(sim_fr)
                                                    meta_data = (
                                                        pmr.get_neuron_meta_data(
                                                            save_dir, size=size
                                                        )
                                                    )

                                                    meta_data["pre_exp_fr"] = sim_fr
                                                    meta_data["pre_exp_cv"] = sim_cv

                                                    meta_data["pre_exp_fr_zscore"] = (
                                                        stats.zscore(
                                                            meta_data["pre_exp_fr"]
                                                        )
                                                    )
                                                    meta_data["pre_exp_cv_zscore"] = (
                                                        stats.zscore(
                                                            meta_data["pre_exp_cv"]
                                                        )
                                                    )

                                                    _, connections, total_weight = (
                                                        pmr.find_connections(
                                                            size, save_dir
                                                        )
                                                    )

                                                    meta_data["connections"] = (
                                                        connections
                                                    )
                                                    meta_data["total_weight"] = (
                                                        total_weight
                                                    )

                                                    meta_data = pd.DataFrame(meta_data)
                                                    meta_data["sim"] = sim

                                                    neurons = pd.concat(
                                                        [neurons, meta_data]
                                                    )

                                                    meta_data = meta_data[
                                                        (
                                                            np.abs(
                                                                meta_data[
                                                                    "pre_exp_fr_zscore"
                                                                ]
                                                            )
                                                            <= 3
                                                        )
                                                        & (
                                                            np.abs(
                                                                meta_data[
                                                                    "pre_exp_cv_zscore"
                                                                ]
                                                            )
                                                            <= 3
                                                        )
                                                        & (
                                                            np.abs(
                                                                meta_data["pre_exp_fr"]
                                                            )
                                                            > 0
                                                        )
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
                                                        avg_fr
                                                        - np.mean(
                                                            exp_fr_outliers_removed
                                                        )
                                                    ) / np.mean(exp_fr_outliers_removed)
                                                    avg_cv_error = (
                                                        avg_cv
                                                        - np.mean(
                                                            exp_cv_outliers_removed
                                                        )
                                                    ) / np.mean(exp_cv_outliers_removed)

                                                    data.append(
                                                        [
                                                            sim,
                                                            kcc2_S,
                                                            kcc2_D,
                                                            trpc3,
                                                            tonic_CL_S,
                                                            tonic_CL_D,
                                                            stn,
                                                            noise,
                                                            ks_2samp(
                                                                sim_fr,
                                                                exp_fr_outliers_removed,
                                                            ).pvalue,
                                                            ks_2samp(
                                                                sim_fr,
                                                                exp_fr_outliers_removed,
                                                            ).statistic,
                                                            ks_2samp(
                                                                sim_cv,
                                                                exp_cv_outliers_removed,
                                                            ).pvalue,
                                                            ks_2samp(
                                                                sim_cv,
                                                                exp_cv_outliers_removed,
                                                            ).statistic,
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
                                                            (
                                                                rel_error_fr
                                                                + rel_error_cv
                                                            )
                                                            / 2,
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
                            "kcc2_S_change",
                            "kcc2_D_change",
                            "trpc3_change",
                            "tonic_CL_S_change",
                            "tonic_CL_D_change",
                            "stn_change",
                            "noise_change",
                            "KS_FR_pval",
                            "KS_FR_statistic",
                            "KS_CV_pval",
                            "KS_CV_statistic",
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
                    df.to_csv(
                        f"{slice_dir}_dd_search/in_vivo_seed_{int(vivo_sim):04d}/results.csv"
                    )
                    data = np.asarray(data)
                    np.savetxt(
                        f"{slice_dir}_dd_search/in_vivo_seed_{int(vivo_sim):04d}/results.txt",
                        data,
                        fmt="%f",
                        newline="\n",
                        delimiter="\t",
                    )
        else:
            df = pd.read_csv(f"{data_dir}/results.csv")
            print(df.shape)
            df = df[
                (df["KS_FR_pval"] > 0.05)
                & (df["KS_CV_pval"] > 0.05)
                & (df["Ttest_FR_pval"] > 0.05)
                & (df["Ttest_CV_pval"] > 0.05)
            ]

            print(df.shape)
            dfsort = df.sort_values(by="Avg. Error")
            print(dfsort["Sim"])

            """df2 = pd.read_csv(f"{data_dir2}/results.csv")
            df2 = df2[
                (df2["KS_FR_pval"] > 0.05)
                & (df2["KS_CV_pval"] > 0.05)
                & (df2["Ttest_FR_pval"] > 0.05)
                & (df2["Ttest_CV_pval"] > 0.05)
            ]"""

            sys.exit()

            df["Avg_KS_statistic"] = (df["KS_CV_statistic"] + df["KS_FR_statistic"]) / 2

            prediction_stat = ["Avg_KS_statistic", "KS_FR_statistic", "KS_CV_statistic"]
            keeps = [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [0, 1, 2, 4],
                [0, 1, 2, 3],
            ]
            removed = ["None", "KCC2", "TRPC3", "Tonic GABA", "STN", "Noise"]

            for predict in prediction_stat:

                feature_df = df[
                    [
                        "kcc2_change",
                        "trpc3_change",
                        "tonic_CL_change",
                        "stn_change",
                        "noise_change",
                        predict,
                    ]
                ]
                zscore = np.abs(stats.zscore(feature_df))
                feature_df = feature_df[
                    (np.abs(stats.zscore(feature_df)) < 3).all(axis=1)
                ]

                norm_df = StandardScaler().fit_transform(feature_df)
                count = 0
                for keep in keeps:
                    X_train = norm_df[:, keep]
                    y_train = norm_df[:, 5]

                    reg = LinearRegression().fit(X_train, y_train)
                    forest = RandomForestRegressor(
                        n_estimators=100, random_state=0, oob_score=True
                    ).fit(X_train, y_train)

                    # print(reg.score(X_train, y_train))
                    # print(reg.coef_, reg.intercept_)

                    # print(forest.oob_score_)
                    # print(forest.feature_importances_)

                    print("\n", predict, " ", removed[count])
                    predictions = forest.predict(X_train)
                    linear_predict = reg.predict(X_train)

                    # Evaluating the model
                    # print("Linear Regression")
                    mse = mean_squared_error(y_train, linear_predict)
                    print(f"Linear Regression: Mean Squared Error: {mse}")

                    r2 = r2_score(y_train, linear_predict)
                    # print(f"R-squared: {r2}\n")

                    # print("Random Forest")
                    mse = mean_squared_error(y_train, predictions)
                    print(f"Random Forest: Mean Squared Error: {mse}")

                    r2 = r2_score(y_train, predictions)
                    # print(f"R-squared: {r2}")
                    count += 1

            """
            fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300)
            axes = [ax[i, j] for i in range(3) for j in range(2)]
            sns.histplot(df["kcc2_change"], ax=axes[0], stat="probability")
            sns.histplot(df["trpc3_change"], ax=axes[1], stat="probability")
            sns.histplot(df["tonic_CL_change"], ax=axes[2], stat="probability")
            sns.histplot(df["stn_change"], ax=axes[3], stat="probability")
            sns.histplot(df["noise_change"], ax=axes[4], stat="probability")
            plt.show()
            """
