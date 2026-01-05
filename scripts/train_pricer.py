import sys
import yaml
import torch
import shutil
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

#################### some I/O ####################
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ANN_pricer import ANN
from src.utils.callbacks import save_checkpoints, EarlyStopping

config_path = PROJECT_ROOT / "configs" / "model_training.yaml"
model_config_path = PROJECT_ROOT / "configs" / "model_architecture.yaml"

# to save the run outputs
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
run_dir = RUNS_DIR / run_id

# create files for the runs
(run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
(run_dir / "metrics").mkdir()
(run_dir / "figures").mkdir()


# to save the config used
shutil.copy(
    PROJECT_ROOT / "configs" / "model_training.yaml",
    run_dir / "model_training.yaml"
)
shutil.copy(
    PROJECT_ROOT / "configs" / "model_architecture.yaml",
    run_dir / "model_architecture.yaml"
)

#################### read the config and data ####################
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cfg_data = config["data"]
shuffle = cfg_data["shuffle"]
seed = config["meta"]["seed"]

# to save the checkpoints
cfg_ckpt = config["callbacks"]["checkpoint"]
ckpt_enabled = bool(cfg_ckpt.get("enabled", True))



ckpt_dir = run_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

ckpt_best_path = ckpt_dir / cfg_ckpt["filename_best"] 
ckpt_last_path = ckpt_dir / cfg_ckpt["filename_last"]

best_val = float("inf")

# Early stopping
cfg_es = config["callbacks"]["early_stopping"]
es_enable = bool(cfg_es.get("enabled", True))

if cfg_es["monitor"] != "val_loss":
    raise ValueError("Right now only 'val_loss' is supported")

early_stopper = None
if es_enable:
    early_stopper = EarlyStopping(
        patience=cfg_es["patience"],
        min_delta=cfg_es["min_delta"],
        warmup_epochs=cfg_es["warmup_epochs"],
        mode=cfg_es["mode"]
    )

# load the data
data_path = PROJECT_ROOT / cfg_data["dir"] / "train.parquet"
train_path = PROJECT_ROOT / cfg_data["dir"] / "train.parquet"
val_path = PROJECT_ROOT / cfg_data["dir"] / "val.parquet"

data_df = pd.read_parquet(data_path)
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)


#################### train the model ####################
### read the data already splitted

train_df_X = train_df.iloc[:, :-1]
train_df_y = train_df.iloc[:, -1]

val_df_X = val_df.iloc[:, :-1]
val_df_y = val_df.iloc[:, -1]

X_train = torch.from_numpy(train_df_X.values).float()
y_train = torch.from_numpy(train_df_y.values).float().view(-1, 1)

X_val = torch.from_numpy(val_df_X.values).float()
y_val = torch.from_numpy(val_df_y.values).float().view(-1, 1)

device = "mps" if torch.backends.mps.is_available() else "cpu"

### change the datasets to tensors
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

n_train = len(train_ds)
n_val = len(val_ds)

batch_size = config["loop"]["batch_size"]
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)

### load the model
with open(model_config_path, "r") as f:
    model_cfg = yaml.safe_load(f)


model = ANN(
    input_dim=model_cfg["input"]["dim"],
    hidden_dims=model_cfg["hidden"]["dims"],
    output_dim=model_cfg["output"]["dim"],
    activation=model_cfg["hidden"]["activation"],
    dropout_rate=model_cfg["hidden"]["dropout_rate"],
    initialization=model_cfg["hidden"]["initialization"],
).to(device)

### loss 
loss_name = (config["loss"]["name"]).lower()
if loss_name == "mse":
    loss_fn = nn.MSELoss()
elif loss_name == "mae":
    loss_fn = nn.L1Loss()
else:
    raise ValueError(f"Loss function: '{loss_name}' not implemented")

### optimizer
#NOTE: implementar mejor si quiero usar otro optmizer
opt_name = (config["optimizer"]["name"]).lower()
if opt_name == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["learn_rate"],
        weight_decay=config["optimizer"]["weight_decay"]
        )
else: 
    raise ValueError(f"Optimizer: '{opt_name}' not implemented")

### training with validation
epochs = config["loop"]["epochs"]

# to save some metrics 
metrics_dir = run_dir / "metrics"
history = []

for epoch in range(1, epochs+1): # each epoch
    # train
    model.train()
    train_sum = 0.0

    for xb, yb in train_loader: # each batch
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad() # para no acumular los gradientes antiguos
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        train_sum += loss.item() * xb.size(0) # loss.item() es una media del escalar de loss para una epoch

    train_loss = train_sum / n_train 

    # validation
    model.eval()
    val_sum = 0.0
    with torch.no_grad(): 
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_sum += loss_fn(pred, yb).item() * xb.size(0)

    val_loss = val_sum / n_val

    # save metrics
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    })

    # save checkpoints
    if ckpt_enabled:
        # always save the last
        save_checkpoints(
            path=ckpt_last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            loss_name=loss_name,
            )

        # save best if improved
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoints(
                path=ckpt_best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                loss_name=loss_name,
                )
            
    # Early stopping
    if es_enable and early_stopper is not None:
        if early_stopper.step(epoch=epoch,value=val_loss):
            print(
                f"Early stopping at epoch {epoch} | best val_loss={early_stopper.best:.6f}"
                f"| patience={cfg_es['patience']} | min_delta={cfg_es['min_delta']}"
            )
            break


    # some logging
    if epoch==1 or epoch%5==0:
        print(
            f"Epoch {epoch:3d}/{epochs} | train {loss_name}: {train_loss:.6f} | val {loss_name}: {val_loss:.6f}"
        )
    

#################### some outputs ####################
# save the metrics
hist_df = pd.DataFrame(history)
hist_df.to_parquet(metrics_dir / "metrics.parquet", index=False)
print(f"Saved metrics to: {metrics_dir}")

# plot some figures
fig_dir = run_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

epochs_arr = hist_df["epoch"].to_numpy()
train_arr = hist_df["train_loss"].to_numpy()
val_arr = hist_df["val_loss"].to_numpy()


# loss curve (semilog)
plt.figure()
plt.plot(epochs_arr, train_arr, label="train")
plt.plot(epochs_arr, val_arr, label="val")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel(f"{loss_name} (log)")
plt.grid(True, which="major")
plt.grid(True, which="minor", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "loss_curve.png", dpi=300)
plt.close()

# gap (semilog)
plt.figure()
plt.plot(epochs_arr, val_arr - train_arr)
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("val - train")
plt.grid(True, which="major")
plt.grid(True, which="minor", alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "generalization_gap.png", dpi=300)
plt.close()

print(f"Saved all figures on: {fig_dir}")