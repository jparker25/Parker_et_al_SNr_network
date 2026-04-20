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
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# import user modules
sys.path.append("../")
sys.path.append("../../")
from helpers import *
import network_analysis as na
import run_model
import plot_model_results as pmr
import experimental_analysis as expan

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8


if __name__ == "__main__":
    size = 100
    T = 5
    T0 = 3
    vivo = False
    run = False
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

    data_dir = "data/example_networks_heterogeneity_noise_gkcc2_2"

    N = 20

    if run:
        param_set = []
        sim = 1
        for seed in [5]:
            for w in [0.1]:
                for heterogeneity in np.linspace(0.0, 0.5, N):
                    for gkcc2 in [0.005]:
                        for gtonic in [0.005]:
                            for noise in np.linspace(0.0, 0.025, N):
                                save_dir = f"{data_dir}/sim_{sim:04d}"

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

                                params["gGABA_S_ton_noise_sigma"] = 0.00
                                params["gGABA_S_ton_noise_theta"] = 0.00
                                params["gGABA_D_ton_noise_sigma"] = 0.00
                                params["gGABA_D_ton_noise_theta"] = 0.00

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
                                # param_set.append([save_dir, params])
                                params["save_dir"] = save_dir
                                run_cmd(f"mkdir -p {save_dir}", print_out=False)
                                params["gSTN_ton_std_pct"] = 0

                                run_model.generate_heterogeneity_matrix(**params)
                                run_model.scale_dend_current_noise(**params)

                                run_model.match_soma_dendrite_conductances(**params)
                                run_model.generate_random_weights(**params)

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
                                sorted_egaba_vals = egaba_vals[
                                    egaba_vals[:, 2].argsort()
                                ]
                                sorted_connections_ind = total_weight.argsort()
                                for i in range(size):
                                    egaba_vals[sorted_connections_ind[i], :] = (
                                        sorted_egaba_vals[i, :]
                                    )

                                update_dict = {
                                    "gKCC2_S_nS_pF": egaba_vals[:, 0],
                                    "gTON_CL_S_MEAN_nS_pF": egaba_vals[:, 1],
                                    "gKCC2_D_nS_pF": egaba_vals[:, 0],
                                    "gTON_CL_D_MEAN_nS_pF": egaba_vals[:, 1],
                                }
                                run_model.update_by_dict(df, update_dict, **params)
                                run_model.set_clin(**params)
                                run_model.run(**params)

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
                                sim += 1

    if analyze:
        if gen_data_file:
            data = []
            neurons = pd.DataFrame()

            bw_fr = 10
            bw_cv = 0.2
            sim = 1
            for seed in [5]:
                for w in [0.1]:
                    for heterogeneity in np.linspace(0.0, 0.5, N):
                        for gkcc2 in [0.005]:
                            for gtonic in [0.005]:
                                for noise in np.linspace(0.0, 0.025, N):
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

                                    w_avg_err = na.weighted_avg_error(
                                        sim_fr,
                                        exp_fr_outliers_removed,
                                        sim_cv,
                                        exp_cv_outliers_removed,
                                        bin_width_fr=bw_fr,
                                        bin_width_cv=bw_cv,
                                    )

                                    avg_fr = np.mean(sim_fr)
                                    avg_cv = np.mean(sim_cv)

                                    abs_diff_fr = np.abs(
                                        avg_fr - np.mean(exp_fr_outliers_removed)
                                    )

                                    avg_fr_rel_error = abs_diff_fr / np.mean(
                                        exp_fr_outliers_removed
                                    )

                                    abs_diff_cv = np.abs(
                                        avg_cv - np.mean(exp_cv_outliers_removed)
                                    )

                                    avg_cv_rel_error = abs_diff_cv / np.mean(
                                        exp_cv_outliers_removed
                                    )

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
                                            w_avg_err,
                                            percent_firing,
                                            avg_fr,
                                            avg_cv,
                                            avg_fr_rel_error,
                                            avg_cv_rel_error,
                                            (avg_fr_rel_error + avg_cv_rel_error) / 2,
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
                    "Weighted Avg. Error",
                    "Percent Firing",
                    "Avg. FR",
                    "Avg. CV",
                    "Avg. FR Rel. Error",
                    "Avg. CV Rel. Error",
                    "Avg. FR CV Avg. Error",
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
            df = pd.read_csv(f"{data_dir}/results.csv")
            df["Rel. Error FR-zscore"] = (
                df["Rel. Error FR"] - df["Rel. Error FR"].mean()
            ) / df["Rel. Error FR"].std()
            df["Rel. Error CV-zscore"] = (
                df["Rel. Error CV"] - df["Rel. Error CV"].mean()
            ) / df["Rel. Error CV"].std()
            df["Avg. Error Z-score"] = (
                0.5 * df["Rel. Error FR-zscore"] + 0.5 * df["Rel. Error CV-zscore"]
            )

            df["Rel. Error FR-MinMax"] = (
                df["Rel. Error FR"] - df["Rel. Error FR"].min()
            ) / (df["Rel. Error FR"].max() - df["Rel. Error FR"].min())
            df["Rel. Error CV-MinMax"] = (
                df["Rel. Error CV"] - df["Rel. Error CV"].min()
            ) / (df["Rel. Error CV"].max() - df["Rel. Error CV"].min())
            df["Avg. Error MinMax"] = (
                0.5 * df["Rel. Error FR-MinMax"] + 0.5 * df["Rel. Error CV-MinMax"]
            )

            # Plot heatmap
            indicators = [
                "Rel. Error FR",
                "Rel. Error CV",
                "Avg. Error MinMax",
                "Avg. FR Rel. Error",
                "Avg. CV Rel. Error",
                "Avg. FR CV Avg. Error",
            ]
            fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300, tight_layout=True)
            axes = [ax[i, j] for i in range(2) for j in range(3)]

            for i in range(len(axes)):
                heatmap_data = df.pivot(
                    index="current_noise", columns="Heterogeneity", values=indicators[i]
                )
                heatmap_data = heatmap_data.sort_index(ascending=False)

                heatmap = sns.heatmap(heatmap_data, annot=False, cmap="jet", ax=axes[i])
                plt.gca().invert_yaxis()

                axes[i].set_xlabel("Heterogeneity")
                axes[i].set_ylabel("Current Noise")
                axes[i].set_title(indicators[i])
                heatmap.set_yticklabels(
                    [
                        "{:.3}".format(float(label.get_text()))
                        for label in heatmap.get_yticklabels()
                    ]
                )
                heatmap.set_xticklabels(
                    [
                        "{:.3}".format(float(label.get_text()))
                        for label in heatmap.get_xticklabels()
                    ]
                )

            add_fig_labels(axes)
            fig.savefig(f"{data_dir}/heatmap_discrete.pdf")
            plt.close()

            # Original data
            x = df["Heterogeneity"].values
            y = df["current_noise"].values

            # Create grid to interpolate onto
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)

            z0 = df[indicators[2]].values
            z1 = df[indicators[2]].values
            z2 = df[indicators[2]].values

            shared_min = min(z0.min(), z1.min(), z2.min())
            shared_max = max(z0.max(), z1.max(), z2.max())

            # Plotting
            fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300, tight_layout=True)
            axes = [ax[i, j] for i in range(2) for j in range(3)]

            for i in range(len(axes)):

                # Interpolate data onto grid
                z = df[indicators[i]].values
                zi = griddata((x, y), z, (xi, yi), method="cubic")

                # Apply Gaussian smoothing
                zi_smoothed = gaussian_filter(zi, sigma=3, mode="nearest")

                if i in [2]:
                    vmin, vmax = shared_min, shared_max
                else:
                    vmin = zi_smoothed.min()
                    vmax = zi_smoothed.max()
                heatmap = axes[i].imshow(
                    zi_smoothed,
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    origin="lower",
                    cmap="coolwarm",
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(heatmap)
                axes[i].set_xlabel("Hetereogeneity")
                axes[i].set_ylabel("Current Noise")
                axes[i].set_title(indicators[i])

            add_fig_labels(axes)
            fig.savefig(f"{data_dir}/heatmap_smoothed.pdf")
            plt.close()

            run_cmd(f"open {data_dir}/heatmap_discrete.pdf")
            run_cmd(f"open {data_dir}/heatmap_smoothed.pdf")
