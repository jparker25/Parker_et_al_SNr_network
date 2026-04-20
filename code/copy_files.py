import os, sys

snr_path = "/Users/johnparker/snr_dynamics/python_code/network"


def cp_file(file):
    os.system(f"cp {snr_path}/{file} .")


file_list = [
    "parameter_search_slice.py",
    "seed_finder_slice.py",
    "parameter_search_vivo.py",
    "naive_vivo_stim.py",
    "parameter_search_dopamine_depletion.py",
    "dd_vivo_stim.py",
    "plot_model_results.py",
    "network_analysis.py",
    "run_model.py",
    "experimental_analysis.py",
    "helpers.py",
    "network_generation_figure.py",
    "example_networks_heterogeneity_noise.py",
    "slice_figure.py",
    "in_vivo_figure_resubmission.py",
    "fit_rates_figure.py",
    "ml_decision_tree.py",
    "naive_stim_figure.py",
    "dd_stim_figure.py",
    "response_dist_measures_figure.py",
    "parameter_distributions_figures.py",
    "experimental_stats_figure.py",
    "choice_decision_tree.py",
    "trpc3_change_figure.py",
    "stim_gradients.py",
    "compare_wstr_correlation_results.py",
    "response_violin_figure.py",
]

for f in file_list:
    cp_file(f)
