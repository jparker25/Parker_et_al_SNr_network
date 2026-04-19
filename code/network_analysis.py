# import python modules
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
import scipy.stats
import sys

# import user modules
sys.path.append("../")
from helpers import *


################################################################################
#########   ANALYSIS    ########################################################
################################################################################


def find_and_remove_outliers(dist, zval=3):
    zscores = stats.zscore(dist)
    low_outliers = dist[zscores < -zval]
    up_outliers = dist[zscores > zval]
    outliers = low_outliers.shape[0] + up_outliers.shape[0]
    dist = dist[np.abs(zscores) <= zval]
    return dist, outliers, low_outliers, up_outliers


def find_and_remove_outliers_fr_cv(data, fr_threshold=0, zval=3):
    zscores = np.abs(stats.zscore(data))
    zscores = stats.zscore(data)
    fr_low_outliers = data[zscores[:, 0] < -zval, 0]
    cv_low_outliers = data[zscores[:, 1] < -zval, 1]
    fr_high_outliers = data[zscores[:, 0] >= zval, 0]
    cv_high_outliers = data[zscores[:, 1] >= zval, 1]
    data = data[
        (np.abs(zscores[:, 0]) < zval)
        & (np.abs(zscores[:, 1]) < zval)
        & (data[:, 0] > fr_threshold)
    ]
    return (
        data[:, 0],
        data[:, 1],
        fr_low_outliers,
        cv_low_outliers,
        fr_high_outliers,
        cv_high_outliers,
    )


def find_and_remove_outliers_columns(
    df, columns=["pre_exp_freq", "pre_exp_cv"], threshold=[0, -1], zvals=[3, 3]
):
    count = 0
    low_outliers = []
    high_outliers = []
    df_copy = df.copy()
    for col in columns:
        zscores = np.abs(stats.zscore(df[col]))
        df_copy = df_copy.loc[(zscores < zvals[count]) & (df[col] > threshold[count])]
        low_outliers.append(df[stats.zscore(df[col]) <= -zvals[count]][col].values)
        high_outliers.append(df[stats.zscore(df[col]) >= zvals[count]][col].values)
        count += 1
    return df_copy, high_outliers, low_outliers


def distributions_match(dist1, dist2, alpha=0.05):
    return ks_2samp(dist1, dist2).pvalue > alpha, ks_2samp(dist1, dist2).pvalue


def percent_error_fr(sim_data, exp_data, bin_width=10):
    max_fr = np.max([np.max(sim_data), np.max(exp_data)])
    max_fr = np.ceil(max_fr / bin_width) * bin_width
    bins = np.arange(0, max_fr + bin_width, bin_width)
    sim_hist, be = np.histogram(sim_data, bins=bins, density=True)
    exp_hist, _ = np.histogram(exp_data, bins=bins, density=True)
    return np.sum(np.abs(sim_hist - exp_hist)) / np.sum(exp_hist)


def weighted_avg_error(
    sim_fr, exp_fr, sim_cv, exp_cv, bin_width_fr=10, bin_width_cv=0.2
):
    max_fr = np.max([np.max(sim_fr), np.max(exp_fr)])
    max_fr = np.ceil(max_fr / bin_width_fr) * bin_width_fr
    bins = np.arange(0, max_fr + bin_width_fr, bin_width_fr)
    sim_hist_fr, _ = np.histogram(sim_fr, bins=bins, density=True)
    exp_hist_fr, _ = np.histogram(exp_fr, bins=bins, density=True)
    fr_err = np.sum(np.abs(sim_hist_fr - exp_hist_fr)) / np.sum(exp_hist_fr)
    alpha = 1 / np.var(np.abs(sim_hist_fr - exp_hist_fr))

    max_cv = np.max([np.max(sim_cv), np.max(exp_cv)])
    bins = np.arange(0, max_cv + bin_width_cv, bin_width_cv)
    sim_hist_cv, be = np.histogram(sim_cv, bins=bins, density=True)
    exp_hist_cv, _ = np.histogram(exp_cv, bins=bins, density=True)
    cv_err = np.sum(np.abs(sim_hist_cv - exp_hist_cv)) / np.sum(exp_hist_cv)
    beta = 1 / np.var(np.abs(sim_hist_cv - exp_hist_cv))

    weight_fr = alpha / (alpha + beta)
    weight_cv = beta / (alpha + beta)

    return weight_fr * fr_err + weight_cv * cv_err


def percent_error_cv(sim_data, exp_data, bin_width=0.2):
    max_cv = np.max([np.max(sim_data), np.max(exp_data)])
    bins = np.arange(0, max_cv + bin_width, bin_width)
    sim_hist, be = np.histogram(sim_data, bins=bins, density=True)
    exp_hist, _ = np.histogram(exp_data, bins=bins, density=True)
    return np.sum(np.abs(sim_hist - exp_hist)) / np.sum(exp_hist)


def sse(sim_data, exp_data, bin_width=10):
    max_fr = np.max([np.max(sim_data), np.max(exp_data)])
    max_fr = np.ceil(max_fr / 10) * 10
    bins = np.arange(0, max_fr + bin_width, bin_width)
    sim_hist, _ = np.histogram(sim_data[sim_data > 0], bins=bins, density=True)
    exp_hist, _ = np.histogram(exp_data[exp_data > 0], bins=bins, density=True)
    return np.sum((sim_hist - exp_hist) ** 2)


def metrics_fr(sim_data, exp_data, bin_width=10):
    max_fr = np.max([np.max(sim_data), np.max(exp_data)])
    max_fr = np.ceil(max_fr / bin_width) * bin_width
    bins = np.arange(0, max_fr + bin_width, bin_width)
    sim_hist, _ = np.histogram(sim_data, bins=bins, density=True)
    exp_hist, _ = np.histogram(exp_data, bins=bins, density=True)
    chi, p = scipy.stats.chisquare(sim_hist, exp_hist)
    kld = scipy.stats.entropy(sim_hist, exp_hist)
    emd = scipy.stats.wasserstein_distance(sim_hist, exp_hist)
    return chi, kld, emd


def metrics_cv(sim_data, exp_data):
    max_cv = np.max([np.max(sim_data), np.max(exp_data)])
    bins = np.arange(0, max_cv + 0.2, 0.2)
    sim_hist, _ = np.histogram(sim_data, bins=bins, density=True)
    exp_hist, _ = np.histogram(exp_data, bins=bins, density=True)
    chi, p = scipy.stats.chisquare(sim_hist, exp_hist)
    kld = scipy.stats.entropy(sim_hist, exp_hist)
    emd = scipy.stats.wasserstein_distance(sim_hist, exp_hist)
    return chi, kld, emd


def ks2test(sim_data, exp_data, alpha=0.05):
    pval = ks_2samp(sim_data[sim_data > 0], exp_data).pvalue
    return (
        f"Distributions different, {pval:0.02f}"
        if pval < alpha
        else f"Distributions the same, {pval:0.02f}"
    )
