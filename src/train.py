# Training loops / used in pricer_nn and calibration_nn
#   - load data
#   - loss, 
#   - optimizer, 
#   - early stopping, 
#   - checkpoints, 
#   - basic logging

import yaml
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.stats import norm
from pathlib import Path

######################################## CONFIG LOAD ########################################
PROJECT_PATH = Path("__file__").resolve().parent
config_path = PROJECT_PATH / "configs" / "train.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


CONFIG_COMPILER = config["compile"]

optimizer = CONFIG_COMPILER["name"]
optimizer_lr = CONFIG_COMPILER["learning_rate"]
loss = config["loss"]["name"]
metrics = config["metrics"]


######################################## IMPORT THE MODEL ########################################


######################################## MODEL TRAINING ########################################

