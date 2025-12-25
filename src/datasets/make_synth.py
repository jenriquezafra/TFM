# Generate synthetic dataset
#   - samplear parámetros de Heston + (K,tau,r)
#   - compute IV surface 
#   - saves in standard format (parquet?) (save parameters + IVs)


import yaml
import numpy as np
from pathlib import Path
from scipy.stats import qmc



PROJECT_PATH = Path("__file__").resolve().parent
config_path = PROJECT_PATH / "configs" / "synth.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def generate_params_Heston(n_samples, param_ranges, seed=None):
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    