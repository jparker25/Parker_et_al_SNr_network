import numpy as np
import multiprocessing, sys
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
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
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

import matplotlib as mpl

DPI_SIZE = 600
mpl.rcParams["savefig.dpi"] = DPI_SIZE  # or whatever you need
mpl.rcParams["figure.dpi"] = DPI_SIZE
# mpl.use("pdf")

# Adjust figure settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8


def slice_seed_data(sim_path):
    """
    Reads in simulation data from the specified path and calculates firing rates and coefficients of variation (CV) for each neuron.

    Parameters:
    sim_path (str): Path to the simulation data directory.

    Returns:
    tuple: Two numpy arrays containing firing rates and CVs for each neuron.
    """
    # Create empty array to hold data
    rates = np.zeros(100)
    cvs = np.zeros(100)
    # Loop through all neurons in sim_path
    for i in range(100):
        if os.path.getsize(f"{sim_path}/Neuron_{i}/spike_times.txt") > 0:
            spikes = np.loadtxt(f"{sim_path}/Neuron_{i}/spike_times.txt")
            spikes = spikes[(spikes >= 3) & (spikes <= 5)]  # only keep final 2 seconds
            rates[i] = len(spikes) / 2
            cvs[i] = (
                np.std(np.diff(spikes)) / np.mean(np.diff(spikes))
                if len(spikes) > 1
                else 0
            )
        else:
            rates[i] = 0
            cvs[i] = 0
    return rates, cvs


# where to save figure
FIG_DIR = "/Users/johnparker/paper_repos/Parker_et_al_SNr_network/figures"

# open figure to store stats
stats_file = open(f"{FIG_DIR}/network_generation_figure_stats.txt", "w")

model_color = "#b00096"

model_color = "#165b05"

# fig, ax = plt.subplots(3, 3, figsize=(8, 6), dpi=300, tight_layout=True)
# axes = [ax[i, j] for i in range(3) for j in range(3)]

fig = plt.figure(figsize=(8, 6), dpi=DPI_SIZE, tight_layout=True)
gs = GridSpec(3, 12, figure=fig)
axes = [
    fig.add_subplot(gs[0, :6]),
    fig.add_subplot(gs[0, 6:9]),
    fig.add_subplot(gs[0, 9:]),
]
remaining_sub_plots = [
    fig.add_subplot(gs[i, j : j + 4]) for i in range(1, 3) for j in range(0, 12, 4)
]
axes.extend(remaining_sub_plots)

img_path = f"{FIG_DIR}/snr_cell_schematic.png"
img = mpimg.imread(img_path)
axes[0].imshow(img)
axes[0].axis("off")


# Grab experimental data
exp_data = expan.get_slice_baseline_data()

# redefine variables
exp_fr, exp_cv = exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]

# find experimental outliers
exp_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(exp_data)
exp_fr_low_outliers, exp_cv_low_outliers = low_outliers[0], low_outliers[1]
exp_fr_high_outliers, exp_cv_high_outliers = high_outliers[0], high_outliers[1]

sim_rates, sim_cvs = slice_seed_data(
    f"data/example_networks_heterogeneity_noise_gkcc2_2/sim_0001"
)
sim_data = pd.DataFrame({"pre_exp_freq": sim_rates, "pre_exp_cv": sim_cvs})
sim_data, high_outliers, low_outliers = expan.find_and_remove_outliers_as_df(sim_data)

# perform and plot linear regression on experimental and example model
pmr.plot_linear_fit(
    sim_data["pre_exp_freq"], sim_data["pre_exp_cv"], axes[1], color=model_color
)
pmr.plot_linear_fit(
    exp_data["pre_exp_freq"], exp_data["pre_exp_cv"], axes[1], color="black"
)
axes[1].set_xlim([0, 20])
axes[1].set_xlabel("Firing Rate (Hz)")
axes[1].set_ylabel("CV")

slope, intercept, r, p, std_err = stats.linregress(
    sim_data["pre_exp_freq"], sim_data["pre_exp_cv"]
)

stats_file.write("### PANEL B ###\n")
ks_pval = ks_2samp(
    sim_data["pre_exp_freq"].values, exp_data["pre_exp_freq"].values
).pvalue
ttest_pval = ttest_ind(
    sim_data["pre_exp_freq"].values, exp_data["pre_exp_freq"].values
).pvalue
stats_file.write("FR Comparison, Slice Model vs Experimental:\n")
stats_file.write(
    f"\tFR, Model: {np.mean(sim_data["pre_exp_freq"].values):.3f} +/- {np.std(sim_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(
    f"\tFR, Exp: {np.mean(exp_data["pre_exp_freq"].values):.3f} +/- {np.std(exp_data['pre_exp_freq'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")

stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")

ks_pval = ks_2samp(sim_data["pre_exp_cv"].values, exp_data["pre_exp_cv"].values).pvalue
ttest_pval = ttest_ind(
    sim_data["pre_exp_cv"].values, exp_data["pre_exp_cv"].values
).pvalue

stats_file.write("CV Comparison, Slice Model vs Experimental:\n")
stats_file.write(
    f"\tCV, Model: {np.mean(sim_data["pre_exp_cv"].values):.3f} +/- {np.std(sim_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(
    f"\tCV, Exp: {np.mean(exp_data["pre_exp_cv"].values):.3f} +/- {np.std(exp_data['pre_exp_cv'].values):.3f}\n",
)
stats_file.write(f"\tKS p-value: {ks_pval:.3f}\n")

stats_file.write(f"\tT-test p-value: {ttest_pval:.3f}\n\n")


stats_file.write("Model Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")

slope, intercept, r, p, std_err = stats.linregress(
    exp_data["pre_exp_freq"], exp_data["pre_exp_cv"]
)
stats_file.write("Experimental Linear Regression:\n")
stats_file.write(f"\tSlope: {slope:.3f}\n")
stats_file.write(f"\tIntercept: {intercept:.3f}\n")
stats_file.write(f"\tR: {r:.3f}\n")
stats_file.write(f"\tR^2: {r**2:.3f}\n")
stats_file.write(f"\tP: {p:.3f}\n")
stats_file.write(f"\tStd Err: {std_err:.3f}\n\n")


### heatmaps
csv_path = "/Users/johnparker/snr_dynamics/python_code/network/data/example_networks_heterogeneity_noise_gkcc2_2"
df = pd.read_csv(f"{csv_path}/results.csv")
df["Rel. Error FR-zscore"] = (df["Rel. Error FR"] - df["Rel. Error FR"].mean()) / df[
    "Rel. Error FR"
].std()
df["Rel. Error CV-zscore"] = (df["Rel. Error CV"] - df["Rel. Error CV"].mean()) / df[
    "Rel. Error CV"
].std()
df["Avg. Error Z-score"] = (
    0.5 * df["Rel. Error FR-zscore"] + 0.5 * df["Rel. Error CV-zscore"]
)

df["Rel. Error FR-MinMax"] = (df["Rel. Error FR"] - df["Rel. Error FR"].min()) / (
    df["Rel. Error FR"].max() - df["Rel. Error FR"].min()
)
df["Rel. Error CV-MinMax"] = (df["Rel. Error CV"] - df["Rel. Error CV"].min()) / (
    df["Rel. Error CV"].max() - df["Rel. Error CV"].min()
)
df["Avg. Error MinMax"] = (
    0.5 * df["Rel. Error FR-MinMax"] + 0.5 * df["Rel. Error CV-MinMax"]
)

indicators = [
    "Rel. Error FR-MinMax",
    "Rel. Error CV-MinMax",
    "Avg. Error MinMax",
    "Avg. FR Rel. Error",
    "Avg. CV Rel. Error",
    "Avg. FR CV Avg. Error",
]

### connections
example_sim = 200
total_connections = []
for i in range(400):
    adjacency, connections, total_weight = pmr.find_connections(
        100, f"{csv_path}/sim_{int(i+1):04d}"
    )
    total_connections.extend(connections)

bins = np.arange(0, np.max(total_connections) + 2)
sns.histplot(
    total_connections,
    kde=False,
    stat="proportion",
    bins=bins,
    color=model_color,
    edgecolor="w",
    ax=axes[2],
)
ylims = axes[2].get_ylim()
axes[2].vlines(
    np.mean(total_connections),
    ylims[0],
    ylims[1],
    linestyle="dashed",
    color=model_color,
)
axes[2].vlines(
    np.median(total_connections),
    ylims[0],
    ylims[1],
    linestyle="dotted",
    color=model_color,
)


# Print statistical tests
stats_file.write("### PANEL C ###\n")
stats_file.write("Connections from 400 sims:\n")
stats_file.write(
    f"\tMean total connections: {np.mean(total_connections)}\n",
)
stats_file.write(
    f"\tMedian total connections: {np.median(total_connections)}\n",
)
stats_file.write(
    f"\tSTD total connections: {np.std(total_connections)}\n",
)
stats_file.write(
    f"\tSEM total connections: {stats.sem(total_connections)}\n",
)
stats_file.close()

axes[2].set_xticks(bins + 0.5)
axes[2].set_xticklabels(bins.astype(int))
axes[2].set_xlabel("# SNr Connections", fontsize=8)
axes[2].set_ylabel("Proportion", fontsize=8)
axes[2].set_xlim([bins[0], bins[-1]])
axes[2].set_xlim([0, 7])

x = df["Heterogeneity"].values
y = df["current_noise"].values

# Create grid to interpolate onto
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

z0 = df[indicators[3]].values
z1 = df[indicators[4]].values
z2 = df[indicators[4]].values

shared_min = min(z0.min(), z1.min(), z2.min())
shared_max = max(z0.max(), z1.max(), z2.max())

# Plotting
for i in range(3, len(axes)):

    # Interpolate data onto grid
    z = df[indicators[i - 3]].values
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    # Apply Gaussian smoothing
    zi_smoothed = gaussian_filter(zi, sigma=3, mode="nearest")

    vmin = zi_smoothed.min()
    vmax = zi_smoothed.max()
    heatmap = axes[i].imshow(
        zi_smoothed,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="seismic",
        aspect="auto",
        vmin=0 if i < 5 else vmin,
        vmax=1 if i < 5 else vmax,
    )
    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=6)
    axes[i].set_xlabel("Hetereogeneity ($\\kappa_h$)")
    axes[i].set_ylabel("Noise Volatility ($\\sigma_{\\eta_i}$)")
    # axes[i].set_title(indicators[i - 2])

for i, plot in enumerate(axes[1:]):
    plot.spines["top"].set_visible(False if i == 0 or i == 1 else True)
    plot.spines["right"].set_visible(False if i == 0 or i == 1 else True)
    if i > 2:
        plot.spines["top"].set_linewidth(0.5)
        plot.spines["right"].set_linewidth(0.5)
    plot.spines["left"].set_linewidth(0.5)
    plot.spines["bottom"].set_linewidth(0.5)
    plot.tick_params(width=0.5, labelsize=6)

# makeNice(axes[1])
add_fig_labels(axes)
fig.savefig(f"{FIG_DIR}/heatmap_smoothed.pdf", bbox_inches="tight", dpi=DPI_SIZE)
plt.close()

run_cmd(f"open {FIG_DIR}/heatmap_smoothed.pdf")
