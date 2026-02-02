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
if config["model_dir"] == "latest":
    LAST_RUN_DIR = sorted(RUNS_DIR.iterdir())[-1] 
else:
    LAST_RUN_DIR = PROJECT_ROOT / "outputs" / "runs" / config["model_dir"]

# to load the model
model_dir = LAST_RUN_DIR / "model_architecture_copy.yaml"
with open(model_dir, "r") as f:
    model_cfg = yaml.safe_load(f)

# to load the best checkpoint
ckpt_dir = LAST_RUN_DIR / "checkpoints" / "model_best.pt"

# to save the generated figures
fig_dir = LAST_RUN_DIR / "figures"


#################### grid for interpolation and extrapolation ####################
# load the parameters on which the NN was trained
with open(LAST_RUN_DIR / "synth_copy.yaml", "r") as f:
    data_config = yaml.safe_load(f)

# get the parameters and their ranges (all parameters from synth_copy.yaml)
heston_params = data_config["data"]["heston_params"]
grid_params = data_config["data"]["grid"]

param_ranges = {}
for source in (heston_params, grid_params):
    for name, value in source.items():
        if isinstance(value, dict):
            continue
        param_ranges[name] = value

# take values for the fixed parameters 
fixed_params = config["parameters"]["fixed_values"]

# expand the grid so we can see interpolation and extrapolation (only 2 params) 
factor = config["grid"]["expansion_factor"]
n_points = config["grid"]["n_points"]

# create the grid of parameters 
for pair_params in config["parameters"]["pairs"]:
    param1, param2 = pair_params
    p1_value = param_ranges[param1]
    p1_min, p1_max = (p1_value if isinstance(p1_value, (list, tuple)) else (p1_value, p1_value)) #NOTE: necesario ahora porque hay fixed params

    p2_value = param_ranges[param2]
    p2_min, p2_max = (p2_value if isinstance(p2_value, (list, tuple)) else (p2_value, p2_value)) #NOTE: necesario ahora porque hay fixed params
    
    p1_range = p1_max - p1_min
    p2_range = p2_max - p2_min
    
    p1_expanded_max = p1_max * factor
    p1_expanded_min = p1_min / factor
    p2_expanded_max = p2_max * factor
    p2_expanded_min = p2_min / factor
    
    p1_values = torch.linspace(p1_expanded_min, p1_expanded_max, n_points)
    p2_values = torch.linspace(p2_expanded_min, p2_expanded_max, n_points)
    
    P1, P2 = torch.meshgrid(p1_values, p2_values, indexing='ij')


#################### results from the NN ####################
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load the model architecture
model = ANN(
    input_dim=model_cfg["input"]["dim"],
    hidden_dims=model_cfg["hidden"]["dims"],
    output_dim=model_cfg["output"]["dim"],
    activation=model_cfg["hidden"]["activation"],
    dropout_rate=model_cfg["hidden"]["dropout_rate"],
    initialization=model_cfg["hidden"]["initialization"],
).to(device)

# load best checkpoint
ckpt = torch.load(ckpt_dir, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# predict IVs on the created grid


raise RuntimeError("STOP")
#################### results from Heston model + root finder ####################
# load the solver

# compute the IVs on the same grid (same parameters)


#################### compute the surface plots ####################
# create the error
if config["error_metric"] == "rmse":
    pass
elif config["error_metric"] == "mse":
    pass

# plot the surfaces
