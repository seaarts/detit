# detit
Determinantal choice modeling for data-driven subset selection.

This readme describes how to install the `detit` package and how to replicate the findings in our paper *A Determinantal discrete choice model for subset selection*.

<p align="left">
<a href="https://github.com/seaarts/detit/actions?query=workflow%3ATests"><img alt="Tests" src=https://github.com/seaarts/detit/actions/workflows/tests.yml/badge.svg></a>
<a href="https://github.com/seaarts/detit/actions?query=workflow%3ALinting"><img alt="Linting" src=https://github.com/seaarts/detit/actions/workflows/linting.yml/badge.svg></a>
<a href="https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/seaarts/detit/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

## Installation
To install the package, first copy the directory `detit` to some location of your preference.

### Install the requirements

Make sure you are inside the `detit` directory, and that you have `pip` installed.

It is recommended to create a fresh virtual environment, e.g. using `venv`.
```bash
python -m venv env
```
This should create an `env` directory inside `detit`.

After creating the environment, be sure to activate it with:
```bash
source env/bin/activate
```

Then run the following command to install the requirements with `pip`.
```bash
pip install requirements.txt
```
This should install a list of required packages.

### Install detit
The implementation is packaged as python package and must built before use.

Building the environment requires `twine` and `bash`. This is best installed outside of your virtual enviroment.
```bash
deactivate
pip install twine
pip install build
```
Next the package is built from the `detit` directory:
```bash
python3 -m build
```
Your package should now be installed. 

## Reproducing the simulation study
This secton explains how to reproduce the simulation study. Doing to assumes the package is installed, anything necessary beyond this is contained in the `study_simulations` subdirectory. To use its contents, first install any additioanl requirements.
```bash
cd simulations
pip install requirements_simu.txt
```
The simulations can be run in two ways (i) from the command line using the `.py`-files, or (ii) via the jupyter notebooks.

### Running experiments from the command line
Experiments are to be run from the `detit` directory. In `detit` the following commands run the logit and MNL experiments:
```bash
python study_simulations/simu_logisitc.py
python study_simulation/simu_mnlogit.py
```
This generates training and testing data, and fits logistic and MNL models the the training data, respectively.

The determinantal choice model takes considerably longer to train, so the command for each radius $r > 0$ is given separately. E.g.
```bash
python study_simulations/simu_detit.py 0.5
```
Runs the simulation for $r = 0.5$. For all simulations, the data is stored in `study_simulations/data`. One should run the logistic regression `simu_logit.py` first, as this generates the testing data used by the other two programs.

### Running experiments from the notebooks
The notebooks are more self-explanatory - they guide you through with text as you go.

### Visualizing results
The notebook `simulation_results.ipynb` visualizes the simulation results. It requires the simulations to have been run, so that the `data` subdirectory is populated for `r = 0.1, 0.3, ..., 2.7`.


## Reproducing the wireless interference study
The wireless interference application is contained in the `study_interference` directory. This is still under development, and does not yet make full use of the `detit` package. Nevertheless, the directory has notebooks that can replicate the experiments:
1. `make_figure_3.ipynb` is a simple notebook used to generate Figure 3.
2. `run_mcmc.ipynb` uses `TensorFlow` to generate posterior samples - this can take a while (it took us 8h on an M1 mac). This outputs `5k_halft.npy`.
3. `make_results.ipynb` uses the output `5k_halft.npy` to explore results and generate graphs.