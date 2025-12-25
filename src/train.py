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
from scipy.stats import norm
from pathlib import Path
from src.models.pricer_nn import build_heston_pricer_nn

PROJECT_PATH = Path("__file__").resolve().parent
config_path = PROJECT_PATH / "configs" / "train.yaml"
output_path = #TODO:

######################################## TRAIN MODEL ########################################

def main():
    # load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    optimizer_cfg = config["compile"]["optimizer"]
    fit_cfg = config["fit"]
    callbacks_cfg = config["callbacks"]
    output_path = PROJECT_PATH / config["output_path"]
    # load data
    # TODO: implement data loading
    X_train, y_train = None, None
    X_val, y_val = None, None

    # build model
    model = build_heston_pricer_nn()

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=optimizer_cfg["learning_rate"]), 
              loss=config["loss"]["name"],
              metrics=config["metrics"]) 
    # TODO: improve optimizer and loss function

    # set the callbacks
    callbacks = []
    if callbacks_cfg["early_stopping"]["enabled"]:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=callbacks_cfg["early_stopping"]["monitor"],
                patience=callbacks_cfg["early_stopping"]["patience"],
                restore_best_weights=callbacks_cfg["early_stopping"]["restore_best_weights"],
            )
        )

    if callbacks_cfg["checkpoints"]["enabled"]:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_path / "checkpoints" / "model_epoch_{epoch:02d}.keras",
                monitor=callbacks_cfg["checkpoints"]["monitor"],
                save_best_only=callbacks_cfg["checkpoint"]["save_best_only"],
            )
        )

    # fit model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=fit_cfg["epochs"],
        batch_size=fit_cfg["batch_size"],
        callbacks=callbacks, 
        )
    # save model
    model.save(output_path / "final")


if __name__ == "__main__":
    main()