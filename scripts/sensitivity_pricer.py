import sys
import yaml
import torch
import shutil
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

#################### some I/O ####################
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ANN_pricer import ANN

# load the config 
with open(PROJECT_ROOT / "configs" / "sensitivity_config.yaml", "r") as f:
    config = yaml.safe_load(f)


RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
if config["model_dir"] == "lastest":
    LAST_RUN_DIR = sorted(RUNS_DIR.iterdir())[-1] 
else:
    LAST_RUN_DIR = PROJECT_ROOT / "outputs" / "runs" / config["model_dir"]

# to load the model
model_dir = LAST_RUN_DIR / "model_architecture.yaml"

# to load the best checkpoint
ckpt_dir = LAST_RUN_DIR / "checkpoints" / "model_best.pt"

# to save the generated figures
fig_dir = LAST_RUN_DIR / "figures"


#################### grid for interpolation and extrapolation ####################
# load the parameters on which the NN was trained
with open(LAST_RUN_DIR / "synth.yaml", "r") as f:
    data_config = yaml.safe_load(f)


# get the free parameters and their ranges
free_params = config["parameters"]["variables"]
param_ranges = {}
for param in free_params:
    param_ranges[param] = data_config["data"]["presets"]["main"]["heston_params"][param]

# take fixed parameters (middle of the range)


# expand the grid so we can see interpolation and extrapolation (only 2 params) 
factor = config["grid"]["expansion_factor"]

# create the grid of parameters 



#################### results from the NN ####################

# load the model architecture

# load best checkpoint

# predict IVs on the created grid


#################### results from Heston model + root finder ####################
# load the solver

# compute the IVs on the same grid (same parameters)


#################### compute the surface plots ####################
# create the error
if config["error_metric"] == "rmse":

elif config["error_metric"] == "mse":

# plot the surfaces

