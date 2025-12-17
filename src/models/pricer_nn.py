# Define the pricer NN (architecture only)
#   - map (params, K, T, r) -> price/IV
#   - only architecture + forward
#   - DONT INCLUDE: loss function, optimizer, training, etc

import yaml
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.stats import norm
from pathlib import Path

######################################## CONFIG LOAD ########################################
PROJECT_PATH = Path("__file__").resolve().parent
config_path = PROJECT_PATH / "configs" / "model.yaml"


with open(config_path, "r") as f:
    config = yaml.safe_load(f)

CONFIG_INPUT = config["input"]
CONFIG_MODEL = config["model"]
CONFIG_OUTPUT = config["output"]

N_input = CONFIG_INPUT["dim"]
layers_neurons = CONFIG_MODEL["trunk"]["dense_layers"]
layers_activation = CONFIG_MODEL["trunk"]["activation"]
layers_kernel_regularizer = CONFIG_MODEL["trunk"]["kernel_regularizer"]

N_ouput = CONFIG_OUTPUT["dim"]
output_activation = CONFIG_OUTPUT["activation"]
output_name = CONFIG_OUTPUT["name"]




######################################## MODEL BUILDING ########################################

def build_heston_pricer_nn():
    # input layer
    inputs = keras.Input(shape=(N_input,), name="heston_params_m_tau")

    # hidden layers
    x = layers.Dense(
        layers_neurons[0], 
        activation=layers_activation, 
        kernel_regularizer=regularizers.l2(layers_kernel_regularizer[1])
        )(inputs)
    
    x = layers.Dense(
        layers_neurons[1], 
        activation=layers_activation, 
        kernel_regularizer=regularizers.l2(layers_kernel_regularizer[1])
        )(x)
    
    x = layers.Dense(
        layers_neurons[2], 
        activation=layers_activation, 
        kernel_regularizer=regularizers.l2(layers_kernel_regularizer[1])
        )(x)    
    
    x = layers.Dense(
        layers_neurons[3], 
        activation=layers_activation, 
        kernel_regularizer=regularizers.l2(layers_kernel_regularizer[1])
        )(x)    
    
    # output layer
    outputs = layers.Dense(
        N_ouput,
        activation=output_activation,
        name=output_name
    )(x)

    # build the model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=config["name"]
    )

    return model

