This repository is for the paper:

**Reduced TRPC3 conductance underlies altered SNr activity under dopamine depletion: predictions from data-driven network models** John E. Parker, Asier Aristieta, Ya Emma Gao, Aryn H. Gittis, Jonathan E. Rubin bioRxiv 2025.08.15.670540; doi: [https://doi.org/10.1101/2025.08.15.670540](https://doi.org/10.1101/2025.08.15.670540)

Please use the above citation if using the SNr Model in a publication. Upon publication, the above citation will be updated. 

A Python virtual environment is advised and is assumed in all instructions in this README. Details of required Python and C++ versions can be found at the end of this README.

For any questions or issues please contact the owner of this repository.

# Running Code
In this section, details on how to reproduce the data is described in detail. Generally, very little needs to be altered. If there are any bugs, please notify the owner of this repository. Please see Figure 9 of the mansucript for a overview of the computational pipeline. For dependencies see the sections at the end of this README.

## SNr Model

## Slice Scripts
1. `parameter_search_slice.py`<br>
    * **Description**: The file `parameter_search_slice.py` sweeps through potential slice condition parameters (1344 combinations for $N=1344$ possible networks) and stores the results in `data/param_search_slice_correlated`. For the initial slice models before tuning, run: 
    ```
    $ python parameter_search_slice.py
    ``` 
    * For initial sweep of parameters (1344 combinations), run with:
        * `vivo = False`
        * `run = True`
        * `plots = True`
        * `analyze = True`
        * `gen_data_file = True`
    * To generate the database of neurons that best match the experimental data ($n=252$), stored in `data/param_search_slice_correlated/match_neurons.csv`, change the parameters to:
        * `vivo = False`
        * `run = False`
        * `plots = False`
        * `analyze = True`
        * `gen_data_file = False`
2. `seed_finder.py`<br>
    * **Description**: The file `seed_finder.py` generates $N=300$ potential tuned networks utilizing the database of best fit neurons stored in `match_neurons.csv` and determines best networks compared to experimental data. 
    ```
    $ python seed_finder.py
    ``` 
    * For initial generation of tuned networks that will be stored in `data/param_seed_finder_slice_correlated/sim_[SEED]`:
        * `vivo = False`
        * `run = True`
        * `analyze = False`
        * `gen_data_file = False`
    * For analysis of tuned network performance will be stored in `data/param_seed_finder_slice_correlated`, as `results.csv` and `results.txt`.
        * `vivo = False`
        * `run = False`
        * `analyze = True`
        * `gen_data_file = True`

## _in vivo_ Networks
1. `parameter_search_vivo.py` <br>
    * **Description**: Sweeps through parameter combinations in the _in vivo_ setting ($N=1680$) for each of the tuned slice networks. Network simulations are stored in `data/param_search_vivo_from_slice/slice_seed_[SEED]/sim_[SIM]` and analysis is stored in `results.txt` and `results.csv` at `data/param_search_vivo_from_slice/slice_seed_[SEED]/`.
    ```
    $ python parameter_search_vivo.py
    ```
    * To run the all networks and perform analysis:
        * `vivo = True`
        * `run = True`
        * `analyze = True`
        * `gen_data_file = True`
        * `correlate_egaba = True`
2. `naive_vivo_stim.py`<br>
    * **Description**: Uses the best three _in vivo_ networks produced from each slice seed and applies three instances of GPe and D1 stimulation. Network simulations are stored in `data/slice_seed_[SEED]_naive_[STIM]_stim/in_vivo_seed_[SEED]/sim_[SIM]`.
    ```
    $ python naive_vivo_stim.py
    ```
    * To run the all networks and perform analysis:
        * `vivo = True`
        * `run = True`
        * `correlate_egaba = True`
        * `correlate_wstrs = True`
        * `correlate_wgpe = False`

3. `parameter_search_dopamine_depletion.py`<br>
    * **Description**: Applies 2,187 dopamine depletion parameter combinations to the top three _in vivo_ simulations. Network simulations are stored in `data/slice_seed_[SEED]_dd_search/in_vivo_seed_[SEED]/sim_[SIM]`. Fit analysis is stored in `results.txt` and `results.csv` at `data/slice_seed_[SEED]_dd_search/in_vivo_seed_[SEED]/`.
    ```
    $ python parameter_search_dopamine_depletion.py
    ```
    * To run all simulations and perform analysis:
        * `vivo = True`
        * `run = True`
        * `analyze = True`
        * `gen_data_file = True`
        * `dopamine_depletion = True`
        * `correlate_egaba = True`
        * `correlate_wstrs = True`
        * `correlate_wgpe = False`

4. `dd_vivo_stim.py`<br>
    * **Description**: Uses the best three dopamine depleted _in vivo_ networks produced from each _in vivo_ seed and applies three instances of GPe and D1 stimulation. Network simulations are stored in `data/slice_seed_[SEED]_dd_[STIM]_stim/in_vivo_seed_[SEED]/dd_sim_[SEED]/sim_[SIM]`.
    ```
    $ python dd_vivo_stim.py
    ```
    * To run the all networks and perform analysis:
        * `vivo = True`
        * `run = True`
        * `correlate_egaba = True`
        * `correlate_wstrs = True`
        * `correlate_wgpe = False`
        * `dopamine depletion = True`

## STReaC 
**To run these files, please use the [STReaC toolbox](https://github.com/jparker25/streac). Files need to be moved to the local respository containing the STReaC toolbox before being executed.**
1. `run_naive_in_vivo_stim.py`<br>
    * **Description**: Runs STReaC toolbox on naive stimulation simulation. Only resulting CSV is kept due to storage limitations. 
    ```
    $ python run_naive_in_vivo_stim.py
    ```
    * To run STReaC, set the following:
        * `direc = [PATH TO SNr CODE]`
        * `remove_data = True` _Turn to false if all STReaC output is desired_
        * `run_d1 = True`
        * `run_gpe = True`
        * `dopamine_depletion = False`
        * `vivo = True`
2. `run_dd_in_vivo_stim.py`<br>
    * **Description**: Runs STReaC toolbox on DD stimulation simulation. Only resulting CSV is kept due to storage limitations. 
    ```
    $ python run_dd_in_vivo_stim.py
    ```
    * To run STReaC, set the following:
        * `direc = [PATH TO SNr CODE]`
        * `remove_data = True` _Turn to false if all STReaC output is desired_
        * `run_d1 = True`
        * `run_gpe = True`
        * `dopamine_depletion = True`
        * `vivo = True`

## Figure Generation



## Helper Files
1. `helpers.py`
2. `network_analysis.py`
3. `run_model.py`
4. `plot_model_results.py`
5. `experimental_analysis.py`


# Data
The corresponding data to the model is quite large. The `data_processed` contains only the necessary files to recreate each simulation. Contact the owner of this repository to for all raw data or visit LINK_TO_DATA???.


# Python Virtual Environment
Python version 3.12.13 was used to implement this repository.

Please see https://docs.python.org/3/library/venv.html for instructions on how to create a virtual environment on your machine.

Then, read in required Python modules via `$ pip install -r requirements.txt`

# C++
