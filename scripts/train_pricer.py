import sys
import time
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
    run_dir / "model_training_copy.yaml"
)
shutil.copy(
    PROJECT_ROOT / "configs" / "model_architecture.yaml",
    run_dir / "model_architecture_copy.yaml"
)

shutil.copy(
    PROJECT_ROOT / "configs" / "synth.yaml",
    run_dir / "synth_copy.yaml"
)

#################### auxiliary functions ####################

def _format_seconds(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

#################### read the config and data ####################
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cfg_data = config["data"]
shuffle = cfg_data["shuffle"]

# set the random seed
seed = config["meta"]["seed"]
g = torch.Generator()
g.manual_seed(seed)

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

batch_size_train = config["loop"]["batch_size_train"]
batch_size_val = config["loop"]["batch_size_val"]
if batch_size_val == "all":
    batch_size_val = n_val

meta_opt_name = (config["meta"]["optimizer"]).lower()
if batch_size_train == "all":
    batch_size_train = n_train
elif meta_opt_name in ("mix", "lbfgs", "l-bfgs"):
    batch_size_train = min(int(batch_size_train) * 10, n_train)

train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size_train, 
    shuffle=shuffle,
    generator=g)

val_loader = DataLoader(
    val_ds, 
    batch_size=batch_size_val, 
    shuffle=False)      #NOTE: i think shuffling is not needed for validation

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
elif loss_name == "rmse":
    loss_fn = lambda pred, target: torch.sqrt(nn.MSELoss()(pred, target))
else:
    raise ValueError(f"Loss function: '{loss_name}' not implemented")

### optimizer
def _normalize_optimizer_name(name: str) -> str:
    name = name.lower()
    if name == "adam":
        return "adam"
    if name in ("l-bfgs", "lbfgs"):
        return "lbfgs"
    raise ValueError(f"Optimizer '{name}' not supported. Use 'adam' or 'L-BFGS'")


def _get_optimizer_cfg(cfg, normalized_name: str):
    for item in cfg["optimizers"]:
        raw_name = item.get("name", "").lower()
        if raw_name == "mix":
            continue
        try:
            item_name = _normalize_optimizer_name(raw_name)
        except ValueError:
            continue
        if item_name == normalized_name:
            return item
    raise ValueError(f"Optimizer: '{normalized_name}' not found in config optimizers list")


def _build_optimizer(normalized_name: str):
    opt_cfg = _get_optimizer_cfg(config, normalized_name)
    if normalized_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg["learn_rate"],
            weight_decay=opt_cfg["weight_decay"],
        )
    return torch.optim.LBFGS(
        model.parameters(),
        lr=opt_cfg["learn_rate"],
        max_iter=opt_cfg["max_iter"],
        line_search_fn=opt_cfg["line_search_fn"],
        history_size=opt_cfg["historic_size"],
    )


# mix setup
meta_opt_name = (config["meta"]["optimizer"]).lower()
mix_enabled = meta_opt_name == "mix"
mix_step = None
mix_first_opt_name = None
mix_second_opt_name = None

if mix_enabled:
    mix_cfg = None
    for item in config["optimizers"]:
        if item.get("name", "").lower() == "mix":
            mix_cfg = item
            break
    if mix_cfg is None:
        raise ValueError("Mix optimizer configuration not found in config optimizers list")

    mix_step = int(mix_cfg["step_size"])
    if mix_step <= 0:
        raise ValueError("Mix step_size must be > 0")

    mix_first_opt_name = _normalize_optimizer_name(mix_cfg["first_optimizer"])
    mix_second_opt_name = "lbfgs" if mix_first_opt_name == "adam" else "adam"
    active_opt_name = mix_first_opt_name
else:
    active_opt_name = _normalize_optimizer_name(meta_opt_name)

optimizer = _build_optimizer(active_opt_name)

# callback of StepLR
from src.utils.callbacks import build_step_lr
lr_scheduler = None
cfg_lr = config["callbacks"]["lr_scheduler"]


def _build_lr_scheduler_for(optimizer_obj):
    if cfg_lr["enabled"]:
        if cfg_lr["name"].lower() == "step":
            return build_step_lr(
                optimizer=optimizer_obj,
                step_size=cfg_lr["step_size"],
                gamma=cfg_lr["gamma"]
            )
        raise ValueError(f"LR Scheduler: '{cfg_lr['name']}' not implemented")
    return None


lr_scheduler = _build_lr_scheduler_for(optimizer)

### training with validation
epochs = config["loop"]["epochs"]

# to save some metrics 
metrics_dir = run_dir / "metrics"
epoch_times = []
history = []

for epoch in range(1, epochs+1): # each epoch
    epoch_start = time.time()

    # switch optimizers every mix_step epochs if mix mode is enabled
    if mix_enabled:
        block_idx = (epoch - 1) // mix_step
        desired_opt_name = mix_first_opt_name if (block_idx % 2 == 0) else mix_second_opt_name
        if desired_opt_name != active_opt_name:
            active_opt_name = desired_opt_name
            optimizer = _build_optimizer(active_opt_name)
            lr_scheduler = _build_lr_scheduler_for(optimizer)
            print(f"[mix] epoch {epoch}: switched optimizer to {active_opt_name}")


    # train
    model.train()
    train_sum = 0.0

    for xb, yb in train_loader: # each batch
        xb, yb = xb.to(device), yb.to(device)

        if active_opt_name == "lbfgs":
            def closure():
                optimizer.zero_grad()
                pred_local = model(xb)
                loss_local = loss_fn(pred_local, yb)
                loss_local.backward()
                return loss_local

            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
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

    # step the lr scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # save metrics
    history.append({
        "epoch": epoch,
        "optimizer": active_opt_name,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": current_lr, 
    })

    # log epoch time
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    N = min(5, len(epoch_times))
    avg_epoch_time = sum(epoch_times[-N:]) / N
    epochs_left = epochs - epoch
    eta_sec = avg_epoch_time * epochs_left

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
            f"Epoch {epoch:3d}/{epochs} | "
            f"opt: {active_opt_name} | "
            f"train {loss_name}: {train_loss:.6f} | "
            f"val {loss_name}: {val_loss:.6f} | "
            f"ETA {_format_seconds(eta_sec)}"
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
lr_arr = hist_df["lr"].to_numpy()

plots_cfg = config["outputs"]

# loss curve (semilog)
if plots_cfg["loss_curve"]:
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
    print(f"Saved loss curve figure on {fig_dir}")

# gap (semilog)
if plots_cfg["gap_curve"]:
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
    print(f"Saved generalization gap figure on {fig_dir}")

# learning rate curve
if plots_cfg["lr_curve"]:
    plt.figure()
    plt.plot(epochs_arr, lr_arr)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_rate_curve.png", dpi=300)
    plt.close()
    print(f"Saved learning rate curve figure on {fig_dir}")
