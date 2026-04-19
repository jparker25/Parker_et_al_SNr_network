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
]

for f in file_list:
    cp_file(f)
