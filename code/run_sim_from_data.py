# import python modules
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import random
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import networkx as nx
from scipy import optimize
from scipy import stats
import os

# import user modules
import experimental_analysis as expan
from helpers import *
import plot_model_results as pmr
import network_analysis as na
import run_model

# path to data_processed to pull
sim_to_pull_from = "data_processed/data_processed/param_search_vivo_from_slice/slice_seed_9/sim_0005"

# where to save reproduced simulation
save_dir = "reproduced_sim/"

# only true if running DD sim
dopamine_depletion = False

# only true if running stimulations
d1 = False 
gpe = False

# true to see simulation progress
status = True

# true to record neural dynamics
dynamics = False

# true to display figures (will still save)
show = False

####### DO NOT EDIT BELOW ###################
T0 = 3  # End of transient period
T1 = 5  # End of baseline period
T2 = 35 if d1 or gpe else 5  # End of stim period
size = 100

csv = f"{sim_to_pull_from}/heterogeneity.csv"
df = pd.read_csv(csv)
params = run_model.get_default_params()
params["T"] = T2
params["save_dir"] = f"{save_dir}/{sim_to_pull_from}"
params["size"] = size
params["status"] = 1 if status else 0
params["dynamics"] = 1 if dynamics else 0
params = run_model.heterogeneity_from_simulation(sim_to_pull_from, **params)
run_model.weights_from_simulation(sim_to_pull_from, **params)

if gpe or d1:
    if gpe:
        params["gpe_stim"] = 1
        params["gpe_poisson"] = 0  # 1 means continuous stim, 0 means pulsed
        params["gpe_stim_percent"] = 1
        params["gpe_freq"] = 20
        params["gpe_start_time"] = T1
        params["gpe_base_length"] = 1
        params["gpe_stim_length"] = 1
        params["gpe_post_length"] = 1

    if d1:
        params["str_stim"] = 1
        params["str_poisson"] = 0  # 1 means continuous stim, 0 means pulsed
        params["str_stim_percent"] = 1
        params["str_freq"] = 20
        params["str_start_time"] = T1
        params["str_base_length"] = 1
        params["str_stim_length"] = 1
        params["str_post_length"] = 1
run_model.add_stim(**params)

run_model.run(compile=True, **params)

bw_fr = 10
bw_cv = 0.2
pmr.plot_comparison_results(
    f"{save_dir}/{sim_to_pull_from}",
    T1,
    T0,
    True,
    DD=dopamine_depletion,
    size=size,
    bw_fr=bw_fr,
    bw_cv=bw_cv,
    show=show,
)
pmr.plot_network_statistics(
    f"{save_dir}/{sim_to_pull_from}", T1, T0, size, vivo=True, show=show
)
