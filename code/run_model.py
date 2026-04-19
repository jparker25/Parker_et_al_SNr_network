import sys, os
import numpy as np
import pandas as pd
from scipy import optimize

# import user modules
sys.path.append("../")
from helpers import *


def chloride_dynamics_ss(
    x,
    gkcc2,
    gtonic,
    v,
    R=8.314,
    T=308,
    F=96.485,
    clout=120,
    hco3in=11.8,
    hco3out=25,
    EK=-90,
):
    rtf = R * T / F
    ehco3 = rtf * np.log(hco3in / hco3out)
    egaba = rtf * np.log((4 * x + hco3in) / (4 * clout + hco3out))
    ecl = rtf * np.log(x / clout)
    chi = (ehco3 - egaba) / (ehco3 - ecl)
    return gkcc2 * (ecl - EK) - chi * gtonic * (v - ecl)


def chloride_to_egaba(
    clin, R=8.314, T=308, F=96.485, clout=120, hco3in=11.8, hco3out=25
):
    return (R * T / F) * np.log((4 * clin + hco3in) / (4 * clout + hco3out))


def positive_normal_distribution(mean, std, size):
    dist = np.random.normal(loc=mean, scale=std, size=size)
    for i in range(size):
        while dist[i] < 0:
            dist[i] = np.random.normal(loc=mean, scale=std, size=1)
    return dist


def heterogeneity_from_simulation(source, **kwargs):
    if kwargs["print_out"]:
        print("Pulling heterogeneity matrix from: ", source)
    run_cmd(f"mkdir -p {kwargs['save_dir']}/", print_out=kwargs["print_out"])
    run_cmd(
        f"cp {source}/heterogeneity.csv {kwargs['save_dir']}/heterogeneity.csv",
        print_out=kwargs["print_out"],
    )
    new_heterogeneity_matrix = pd.read_csv(f"{kwargs['save_dir']}/heterogeneity.csv")
    save_heterogeneity_matrix(new_heterogeneity_matrix, **kwargs)
    for col in new_heterogeneity_matrix.columns:
        if col in kwargs.keys():
            kwargs[col] = new_heterogeneity_matrix[col]
    return kwargs


def weights_from_simulation(source, **kwargs):
    if kwargs["print_out"]:
        print("Pulling adjacency matrix from: ", source)
    run_cmd(
        f"cp {source}/weights.txt {kwargs['save_dir']}/weights.txt",
        print_out=False,
    )


def update_by_dict(param_values, update_dict, **kwargs):
    param_values.update(update_dict)
    save_heterogeneity_matrix(param_values, **kwargs)
    return param_values


def add_noise(**kwargs):
    if kwargs["print_out"]:
        print("Generating noise parameters distributions...")
    param_values = pd.read_csv(
        f"{kwargs['save_dir']}/heterogeneity.csv", dtype=np.float64
    )
    param_values["gTON_STN_MEAN_nS_pF"] = positive_normal_distribution(
        kwargs["gSTN_ton_mean"],
        kwargs["gSTN_ton_mean"] * kwargs["gSTN_ton_std_pct"],
        kwargs["size"],
    )
    param_values["gTON_STN_SIGMA"] = kwargs["gSTN_ton_noise_sigma"]
    param_values["gTON_STN_THETA"] = kwargs["gSTN_ton_noise_theta"]
    param_values["gTON_CL_S_SIGMA"] = kwargs["gGABA_S_ton_noise_sigma"]
    param_values["gTON_CL_S_THETA"] = kwargs["gGABA_S_ton_noise_theta"]
    param_values["gTON_CL_D_SIGMA"] = kwargs["gGABA_D_ton_noise_sigma"]
    param_values["gTON_CL_D_THETA"] = kwargs["gGABA_D_ton_noise_theta"]
    param_values["soma_noise_intensity"] = positive_normal_distribution(
        kwargs["current_noise_intensity_S"],
        kwargs["current_noise_intensity_S"] * kwargs["current_noise_intensity_S_pct"],
        kwargs["size"],
    )
    param_values["soma_noise_intensity_theta"] = kwargs[
        "current_noise_intensity_theta_S"
    ]
    param_values["dend_noise_intensity"] = positive_normal_distribution(
        kwargs["current_noise_intensity_D"],
        kwargs["current_noise_intensity_D"] * kwargs["current_noise_intensity_D_pct"],
        kwargs["size"],
    )
    param_values["dend_noise_intensity_theta"] = kwargs[
        "current_noise_intensity_theta_D"
    ]
    save_heterogeneity_matrix(param_values, **kwargs)


def scale_dend_current_noise(scale=0.5, **kwargs):
    if kwargs["print_out"]:
        print("Scaling dendrite to soma current noise...")
    param_values = pd.read_csv(
        f"{kwargs['save_dir']}/heterogeneity.csv", dtype=np.float64
    )
    param_values["dend_noise_intensity"] = 0.4 * param_values["soma_noise_intensity"]
    param_values["dend_noise_intensity_theta"] = (
        0.4 * param_values["soma_noise_intensity_theta"]
    )
    save_heterogeneity_matrix(param_values, **kwargs)


def add_stim(**kwargs):
    if kwargs["print_out"]:
        print("Generating stimulation parameters...")
    ones = np.ones(kwargs["size"])
    param_values = pd.read_csv(
        f"{kwargs['save_dir']}/heterogeneity.csv", dtype=np.float64
    )
    param_values.update(
        {
            "gpe_stim": ones * kwargs["gpe_stim"],
            "gpe_stim_freqs": ones * kwargs["gpe_freq"],
            "gpe_poisson": ones * kwargs["gpe_poisson"],
            "W_gpe": ones * kwargs["W_gpe"],
            "gpe_start_time": ones * kwargs["gpe_start_time"],
            "gpe_base_length": ones * kwargs["gpe_base_length"],
            "gpe_stim_length": ones * kwargs["gpe_stim_length"],
            "gpe_post_length": ones * kwargs["gpe_post_length"],
            "str_stim": ones * kwargs["str_stim"],
            "str_stim_freqs": ones * kwargs["str_freq"],
            "str_poisson": ones * kwargs["str_poisson"],
            "W_str": ones * kwargs["W_str"],
            "str_start_time": ones * kwargs["str_start_time"],
            "str_base_length": ones * kwargs["str_base_length"],
            "str_stim_length": ones * kwargs["str_stim_length"],
            "str_post_length": ones * kwargs["str_post_length"],
        }
    )
    save_heterogeneity_matrix(param_values, **kwargs)


def match_soma_dendrite_conductances(**kwargs):
    if kwargs["print_out"]:
        print("Matching soma and dendrite parameters...")
    param_values = pd.read_csv(
        f"{kwargs['save_dir']}/heterogeneity.csv", dtype=np.float64
    )
    param_values["gTON_CL_D_MEAN_nS_pF"] = param_values["gTON_CL_S_MEAN_nS_pF"]
    param_values["gTON_CL_D_THETA"] = 0.4 * param_values["gTON_CL_S_THETA"]
    param_values["gTON_CL_D_SIGMA"] = 0.4 * param_values["gTON_CL_S_SIGMA"]
    param_values["gKCC2_D_nS_pF"] = param_values["gKCC2_S_nS_pF"]
    save_heterogeneity_matrix(param_values, **kwargs)


def set_clin(**kwargs):
    if kwargs["print_out"]:
        print("Generating CLin values from gKCC2 and gTON_CL...")
    param_values = pd.read_csv(
        f"{kwargs['save_dir']}/heterogeneity.csv", dtype=np.float64
    )
    param_values["CL_in_S"] = np.asarray(
        [
            optimize.newton(
                chloride_dynamics_ss,
                10,
                args=(row["gKCC2_S_nS_pF"], row["gTON_CL_S_MEAN_nS_pF"], -55),
                maxiter=200,
                tol=1.4e-12,
            )
            for index, row in param_values.iterrows()
        ]
    )
    param_values["CL_in_D"] = np.asarray(
        [
            optimize.newton(
                chloride_dynamics_ss,
                10,
                args=(row["gKCC2_D_nS_pF"], row["gTON_CL_D_MEAN_nS_pF"], -55),
                maxiter=200,
                tol=1.4e-12,
            )
            for index, row in param_values.iterrows()
        ]
    )
    save_heterogeneity_matrix(param_values, **kwargs)


def generate_random_weights(**kwargs):
    if kwargs["print_out"]:
        print("Generating adjacency matrix...")
    weights = np.zeros((kwargs["size"], kwargs["size"]))
    for i in range(kwargs["size"]):  # source
        for j in range(kwargs["size"]):  # target
            if np.random.rand() < kwargs["prob"] and i != j:
                weights[i][j] = positive_normal_distribution(
                    kwargs["W_SNr_mean"],
                    kwargs["W_SNr_mean"] * kwargs["W_SNr_std_pct"],
                    1,
                )

    np.savetxt(
        f"{kwargs['save_dir']}/weights.txt",
        weights,
        fmt="%f",
        newline="\n",
        delimiter=" ",
    )


def heterogeneity_from_csv(df, exp_fr_outliers_removed, size=100, **kwargs):
    if kwargs["print_out"]:
        print("Generating heterogeneity matrix from csv...")
    bins = np.arange(0, np.max(exp_fr_outliers_removed) + 5, 5)
    heights, _ = np.histogram(
        exp_fr_outliers_removed,
        bins=bins,
    )
    heights = heights / len(df)
    heights = np.rint(heights * size).astype(int)
    if np.sum(heights) > size:
        while np.sum(heights) > size:
            height_index = np.random.randint(low=0, high=heights.shape[0] - 1, size=1)
            if heights[height_index] > 0:
                heights[height_index] -= 1
    data_frames = []
    for i in range(len(bins) - 1):
        sample = df[(df["pre_exp_fr"] > bins[i]) & (df["pre_exp_fr"] <= bins[i + 1])]
        if len(sample) > 0:
            data_frames.append(sample.sample(n=heights[i]))
    sim_df = pd.concat(data_frames)
    if len(sim_df) < size:
        sim_df = pd.concat([sim_df, df.sample(n=size - len(sim_df))])

    df = sim_df
    df.to_csv(f"{kwargs['save_dir']}/parameter_df.csv")
    save_heterogeneity_matrix(df, **kwargs)


def save_heterogeneity_matrix(df, **kwargs):
    run_cmd(f"mkdir -p {kwargs['save_dir']}/model_setup", print_out=False)
    for col in df.columns:
        if col[0:2] != "Un":
            np.savetxt(
                f"{kwargs['save_dir']}/model_setup/{col}.txt",
                df[col],
                fmt="%f",
                delimiter="\t",
                newline="\n",
            )
    df.to_csv(f"{kwargs['save_dir']}/heterogeneity.csv")


def generate_heterogeneity_matrix(**kwargs):
    if kwargs["print_out"]:
        print("Generating heterogeneity matrix from distributions...")
    param_dict = {
        "gTON_STN_MEAN_nS_pF": positive_normal_distribution(
            kwargs["gSTN_ton_mean"],
            kwargs["gSTN_ton_mean"] * kwargs["gSTN_ton_std_pct"],
            kwargs["size"],
        ),
        "gTON_STN_SIGMA": kwargs["gSTN_ton_noise_sigma"],
        "gTON_STN_THETA": kwargs["gSTN_ton_noise_theta"],
        "gTON_CL_S_MEAN_nS_pF": positive_normal_distribution(
            kwargs["gGABA_S_ton_mean"],
            kwargs["gGABA_S_ton_mean"] * kwargs["gGABA_S_ton_std_pct"],
            kwargs["size"],
        ),
        "gTON_CL_S_SIGMA": kwargs["gGABA_S_ton_noise_sigma"],
        "gTON_CL_S_THETA": kwargs["gGABA_S_ton_noise_theta"],
        "gTON_CL_D_MEAN_nS_pF": positive_normal_distribution(
            kwargs["gGABA_D_ton_mean"],
            kwargs["gGABA_D_ton_mean"] * kwargs["gGABA_D_ton_std_pct"],
            kwargs["size"],
        ),
        "gTON_CL_D_SIGMA": kwargs["gGABA_D_ton_noise_sigma"],
        "gTON_CL_D_THETA": kwargs["gGABA_D_ton_noise_theta"],
        "gKCC2_S_nS_pF": positive_normal_distribution(
            kwargs["gKCC2_S_mean"],
            kwargs["gKCC2_S_mean"] * kwargs["gKCC2_S_std_pct"],
            kwargs["size"],
        ),
        "gKCC2_D_nS_pF": positive_normal_distribution(
            kwargs["gKCC2_D_mean"],
            kwargs["gKCC2_D_mean"] * kwargs["gKCC2_D_std_pct"],
            kwargs["size"],
        ),
        "gTRPC3_nS_pF": positive_normal_distribution(
            kwargs["gTRPC3_mean"],
            kwargs["gTRPC3_mean"] * kwargs["gTRPC3_std_pct"],
            kwargs["size"],
        ),
        "gHCN_nS_pF": positive_normal_distribution(
            kwargs["gHCN_mean"],
            kwargs["gHCN_mean"] * kwargs["gHCN_std_pct"],
            kwargs["size"],
        ),
        "gCA_nS_pF": positive_normal_distribution(
            kwargs["gCa_mean"],
            kwargs["gCa_mean"] * kwargs["gCa_std_pct"],
            kwargs["size"],
        ),
        "gL_nS_pF": positive_normal_distribution(
            kwargs["gL_mean"], kwargs["gL_mean"] * kwargs["gL_std_pct"], kwargs["size"]
        ),
        "gSK_nS_pF": positive_normal_distribution(
            kwargs["gSK_mean"],
            kwargs["gSK_mean"] * kwargs["gSK_std_pct"],
            kwargs["size"],
        ),
        "gNAP_nS_pF": positive_normal_distribution(
            kwargs["gNaP_mean"],
            kwargs["gNaP_mean"] * kwargs["gNaP_std_pct"],
            kwargs["size"],
        ),
        "gNAF_nS_pF": positive_normal_distribution(
            kwargs["gNaF_mean"],
            kwargs["gNaF_mean"] * kwargs["gNaF_std_pct"],
            kwargs["size"],
        ),
        "gKDR_nS_pF": positive_normal_distribution(
            kwargs["gKdr_mean"],
            kwargs["gKdr_mean"] * kwargs["gKdr_std_pct"],
            kwargs["size"],
        ),
        "gSD_nS": positive_normal_distribution(
            kwargs["gSD_mean"],
            kwargs["gSD_mean"] * kwargs["gSD_std_pct"],
            kwargs["size"],
        ),
        "soma_noise_intensity": positive_normal_distribution(
            kwargs["current_noise_intensity_S"],
            kwargs["current_noise_intensity_S"]
            * kwargs["current_noise_intensity_S_pct"],
            kwargs["size"],
        ),
        "soma_noise_intensity_theta": kwargs["current_noise_intensity_theta_S"],
        "dend_noise_intensity": positive_normal_distribution(
            kwargs["current_noise_intensity_D"],
            kwargs["current_noise_intensity_D"]
            * kwargs["current_noise_intensity_D_pct"],
            kwargs["size"],
        ),
        "dend_noise_intensity_theta": kwargs["current_noise_intensity_theta_D"],
        "Eleak_mV": np.random.normal(
            loc=kwargs["Eleak_mean"],
            scale=np.abs(kwargs["Eleak_mean"] * kwargs["Eleak_std_pct"]),
            size=kwargs["size"],
        ),
        "CL_in_S": positive_normal_distribution(
            kwargs["clin_S_mean"],
            kwargs["clin_S_mean"] * kwargs["clin_S_std_pct"],
            kwargs["size"],
        ),
        "CL_in_D": positive_normal_distribution(
            kwargs["clin_D_mean"],
            kwargs["clin_D_mean"] * kwargs["clin_D_std_pct"],
            kwargs["size"],
        ),
        "Iapp": kwargs["iapp"],
        "Iapp_dend": kwargs["iapp_dend"],
        "tausyn": positive_normal_distribution(
            kwargs["tausyn_mean"], kwargs["tausyn_std"], kwargs["size"]
        ),
        "tauexc": positive_normal_distribution(
            kwargs["tauexc_mean"], kwargs["tauexc_std"], kwargs["size"]
        ),
        "tausyn_dend_dist": positive_normal_distribution(
            kwargs["tausyn_dend_mean"], kwargs["tausyn_dend_std"], kwargs["size"]
        ),
        "gpe_stim": kwargs["gpe_stim"],
        "gpe_stim_freqs": np.asarray(
            [
                (
                    kwargs["gpe_freq"]
                    if np.random.rand() < kwargs["gpe_stim_percent"]
                    else 0
                )
                for i in range(kwargs["size"])
            ]
        ),
        "gpe_poisson": kwargs["gpe_poisson"],
        "W_gpe": kwargs["W_gpe"],
        "gpe_start_time": kwargs["gpe_start_time"],
        "gpe_base_length": kwargs["gpe_base_length"],
        "gpe_stim_length": kwargs["gpe_stim_length"],
        "gpe_post_length": kwargs["gpe_post_length"],
        "str_stim": kwargs["str_stim"],
        "str_stim_freqs": np.asarray(
            [
                (
                    kwargs["str_freq"]
                    if np.random.rand() < kwargs["str_stim_percent"]
                    else 0
                )
                for i in range(kwargs["size"])
            ]
        ),
        "str_poisson": kwargs["str_poisson"],
        "W_str": kwargs["W_str"],
        "str_start_time": kwargs["str_start_time"],
        "str_base_length": kwargs["str_base_length"],
        "str_stim_length": kwargs["str_stim_length"],
        "str_post_length": kwargs["str_post_length"],
    }
    df = pd.DataFrame.from_dict(param_dict)
    # df_matrix = df.to_numpy()
    save_heterogeneity_matrix(df, **kwargs)


def get_default_params():
    return {
        "print_out": True,
        "save_dir": "",
        "size": 100,
        "T": 10,
        "dt": 0.025,
        "DT": 0.05,
        "prob": 0.015,
        "seed": 5,
        "gpe_stim": 0,
        "gpe_stim_percent": 0,
        "gpe_freq": 0,
        "gpe_poisson": 0,
        "W_gpe": 0.2,
        "gpe_start_time": 0,
        "gpe_base_length": 0,
        "gpe_stim_length": 0,
        "gpe_post_length": 0,
        "str_stim": 0,
        "str_stim_percent": 0,
        "str_freq": 0,
        "str_poisson": 0,
        "W_str": 2,
        "str_start_time": 0,
        "str_stim_length": 0,
        "str_base_length": 0,
        "str_post_length": 0,
        "status": 1,
        "dynamics": 1,
        "W_SNr_mean": 0.1,
        "W_SNr_std_pct": 0,
        "iapp": 0,
        "iapp_dend": 0,
        # noise
        "current_noise_intensity_S": 0,
        "current_noise_intensity_S_pct": 0,
        "current_noise_intensity_D": 0,
        "current_noise_intensity_D_pct": 0,
        "current_noise_intensity_theta_S": 0,
        "current_noise_intensity_theta_D": 0,
        "gSTN_ton_noise_sigma": 0,
        "gSTN_ton_noise_theta": 0,
        "gGABA_S_ton_noise_sigma": 0,
        "gGABA_S_ton_noise_theta": 0,
        "gGABA_D_ton_noise_sigma": 0,
        "gGABA_D_ton_noise_theta": 0,
        # conductances
        "gNaP_mean": 0.175,
        "gNaP_std_pct": 0,
        "gCa_mean": 0.7,
        "gCa_std_pct": 0,
        "gSK_mean": 0.009,
        "gSK_std_pct": 0,
        "gL_mean": 0.04,
        "gL_std_pct": 0,
        "gTRPC3_mean": 0.1,
        "gTRPC3_std_pct": 0,
        "gHCN_mean": 0.0,
        "gHCN_std_pct": 0,
        "gKdr_mean": 50,
        "gKdr_std_pct": 0,
        "gNaF_mean": 35,
        "gNaF_std_pct": 0,
        "gKCC2_S_mean": 0.15,
        "gKCC2_S_std_pct": 0,
        "gKCC2_D_mean": 0.15,
        "gKCC2_D_std_pct": 0,
        "gSD_mean": 26.5,
        "gSD_std_pct": 0,
        # Tonic Conductances
        "gSTN_ton_mean": 0,
        "gSTN_ton_std_pct": 0,
        "gGABA_S_ton_mean": 0,
        "gGABA_S_ton_std_pct": 0,
        "gGABA_D_ton_mean": 0,
        "gGABA_D_ton_std_pct": 0,
        # Reversals
        "Eleak_mean": -60,
        "Eleak_std_pct": 0,
        # Chloride
        "clin_S_mean": 4,
        "clin_S_std_pct": 0,
        "clin_D_mean": 4,
        "clin_D_std_pct": 0,
        # time constants
        "tausyn_mean": 3,
        "tausyn_std": 0,
        "tauexc_mean": 5,
        "tauexc_std": 0,
        "tausyn_dend_mean": 7.2,
        "tausyn_dend_std": 0,
    }


def uniform_conductance_heterogeneity(heterogeneity, **kwargs):
    kwargs["gNaP_std_pct"] = heterogeneity
    kwargs["gCa_std_pct"] = heterogeneity
    kwargs["gSK_std_pct"] = heterogeneity
    kwargs["gL_std_pct"] = heterogeneity
    kwargs["gTRPC3_std_pct"] = heterogeneity
    kwargs["gHCN_std_pct"] = heterogeneity
    kwargs["gKdr_std_pct"] = heterogeneity
    kwargs["gNaF_std_pct"] = heterogeneity
    kwargs["gKCC2_S_std_pct"] = heterogeneity
    kwargs["gKCC2_D_std_pct"] = heterogeneity
    kwargs["gGABA_S_ton_std_pct"] = heterogeneity
    kwargs["gGABA_D_ton_std_pct"] = heterogeneity
    kwargs["gSTN_ton_std_pct"] = heterogeneity
    return kwargs


def gen_cpp_flags(**kwargs):
    cpp_flags = f'-save_dir {kwargs["save_dir"]}\
    -s {kwargs["size"]}\
    -T {kwargs["T"]}\
    -dt {kwargs["dt"]}\
    -DT {kwargs["DT"]}\
    -status {kwargs["status"]}\
    -dynamics {kwargs["dynamics"]}\
    -sd {kwargs["seed"]}'
    return cpp_flags


def save_params(**kwargs):
    with open(f"{kwargs['save_dir']}/param_dict.txt", "w") as file:
        for key in kwargs:
            file.write(f"{key}:{kwargs[key]}\n")


def run(compile=False, **kwargs):
    if kwargs["print_out"]:
        print("Running simulation...")
    run_cmd(f"mkdir -p {kwargs["save_dir"]}", print_out=False)
    save_params(**kwargs)
    if compile:
        run_cmd(f"g++ -std=c++17 -O2 SNr_adapted.cc -o SNr -Wno-vla-cxx-extension")
    for i in range(kwargs["size"]):
        run_cmd(f"mkdir -p {kwargs["save_dir"]}/Neuron_{i}", print_out=False)
    run_cmd(f"./SNr {gen_cpp_flags(**kwargs)}")
    if kwargs["print_out"]:
        print("Simulation completed")
