import sys
import time
import re
import yaml
import torch
import shutil
import subprocess
import pandas as pd
import numpy as np
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
calibration_config_path = PROJECT_ROOT / "configs" / "calibration.yaml"
experiment_logs_dir = PROJECT_ROOT / "outputs" / "experiment_logs"
calibration_folder_pattern = re.compile(r"^Calibration_(\d+)$")

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


def _count_feature_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, round_decimals: int | None = 12) -> int:
    if round_decimals is not None:
        df_a = df_a.round(round_decimals)
        df_b = df_b.round(round_decimals)

    hash_a = pd.util.hash_pandas_object(df_a, index=False).to_numpy()
    hash_b = pd.util.hash_pandas_object(df_b, index=False).to_numpy()
    return int(np.intersect1d(hash_a, hash_b).size)


def _batch_losses(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return (pred - target).pow(2).view(-1)
    if loss_name == "mae":
        return (pred - target).abs().view(-1)
    if loss_name == "rmse":
        # Keep squared errors; sqrt is applied after trimmed aggregation.
        return (pred - target).pow(2).view(-1)
    raise ValueError(f"Loss function: '{loss_name}' not implemented for per-sample losses")


def _trimmed_mean(values: np.ndarray, trim_top_fraction: float) -> float:
    if values.size == 0:
        return float("nan")
    if trim_top_fraction <= 0:
        return float(np.mean(values))

    n_trim = int(np.floor(values.size * trim_top_fraction))
    if n_trim <= 0:
        return float(np.mean(values))
    if n_trim >= values.size:
        raise ValueError(
            f"trim_top_fraction={trim_top_fraction} removes all validation samples "
            f"(n={values.size})."
        )

    # Keep the lowest values and trim only the top-tail hardest samples.
    values_sorted = np.sort(values)
    return float(np.mean(values_sorted[: values.size - n_trim]))


def _load_yaml_dict(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_calibration_output_root(calibration_cfg_path: Path) -> Path:
    cfg = _load_yaml_dict(calibration_cfg_path)
    out_raw = cfg.get("outputs", {}).get("dir", "outputs/calibration")
    return _resolve_path(out_raw, base_dir=PROJECT_ROOT)


def _resolve_calibration_quotes_override(calibration_cfg_path: Path) -> tuple[Path | None, bool]:
    cfg = _load_yaml_dict(calibration_cfg_path)
    quotes_raw = cfg.get("data", {}).get("market_quotes_path", None)

    if quotes_raw is not None:
        configured_quotes_path = _resolve_path(quotes_raw, base_dir=PROJECT_ROOT)
        if configured_quotes_path.exists():
            return None, True
        print(
            f"Warning: calibration quotes file not found at {configured_quotes_path}. "
            "Trying fallback quotes file."
        )

    fallback_candidates = [
        PROJECT_ROOT / "data" / "market" / "market_quotes_liu_35.csv",
        PROJECT_ROOT / "data" / "market" / "smoke_quotes.csv",
    ]
    for fallback in fallback_candidates:
        if fallback.exists():
            print(f"Using fallback calibration quotes: {fallback}")
            return fallback, True

    print("Warning: no calibration quotes file found. Skipping post-training calibration.")
    return None, False


def _latest_calibration_summary_path(*, calibration_root: Path, run_name: str) -> Path | None:
    run_cal_dir = calibration_root / run_name
    if not run_cal_dir.exists():
        return None

    latest_id = None
    latest_summary_path = None
    for child in run_cal_dir.iterdir():
        if not child.is_dir():
            continue
        match = calibration_folder_pattern.match(child.name)
        if match is None:
            continue
        calibration_id = int(match.group(1))
        summary_path = child / "summary.yaml"
        if not summary_path.exists():
            continue
        if latest_id is None or calibration_id > latest_id:
            latest_id = calibration_id
            latest_summary_path = summary_path

    return latest_summary_path


def _as_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _fmt_scientific(value) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.8e}"


def _fmt_pct(value) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:+.2f}%"


def _run_post_training_calibration(run_name: str) -> None:
    if not calibration_config_path.exists():
        print(f"Warning: calibration config not found at {calibration_config_path}")
        return

    quotes_override, can_run = _resolve_calibration_quotes_override(calibration_config_path)
    if not can_run:
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "calibrate_cann.py"),
        "--config",
        str(calibration_config_path),
        "--model-dir",
        run_name,
    ]
    if quotes_override is not None:
        cmd.extend(["--quotes", str(quotes_override)])

    print("Running calibrate_cann.py for this trained run...")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Warning: calibrate_cann.py failed")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        return

    calibration_root = _resolve_calibration_output_root(calibration_config_path)
    summary_path = _latest_calibration_summary_path(
        calibration_root=calibration_root,
        run_name=run_name,
    )
    if summary_path is None:
        print(
            f"Warning: calibration finished but summary.yaml was not found "
            f"under {calibration_root / run_name}"
        )
        return

    summary = _load_yaml_dict(summary_path)
    print(
        "Calibration metrics | "
        f"weighted_mse: {_fmt_scientific(summary.get('weighted_mse'))} | "
        f"residual_rmse: {_fmt_scientific(summary.get('residual_rmse'))} | "
        f"objective_fun: {_fmt_scientific(summary.get('objective_fun'))}"
    )


def _refresh_optimizer_logs() -> bool:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_optimizer_experiment_logs.py"),
    ]
    print("Refreshing optimizer experiment logs...")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Warning: build_optimizer_experiment_logs.py failed")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        return False
    return True


def _print_run_vs_best_history(*, run_name: str, optimizer_mode: str) -> None:
    if optimizer_mode not in {"mix", "adam"}:
        return

    log_path = experiment_logs_dir / f"{optimizer_mode}_experiments.csv"
    if not log_path.exists():
        print(f"Warning: optimizer log file not found: {log_path}")
        return

    log_df = pd.read_csv(log_path)
    if log_df.empty:
        print(f"Warning: optimizer log file is empty: {log_path}")
        return

    row_df = log_df[log_df["run_id"] == run_name]
    if row_df.empty:
        print(f"Warning: run '{run_name}' not found in {log_path.name}")
        return

    row = row_df.iloc[0]
    print(f"Run comparison vs historical best [{optimizer_mode.upper()}]")
    print(
        f"- train best_val_loss: {_fmt_scientific(row.get('best_val_loss'))} "
        f"| best historical: {_fmt_scientific(row.get('best_hist_train_val_loss'))} "
        f"({str(row.get('best_hist_train_run_id', 'n/a'))}) "
        f"| delta: {_fmt_pct(row.get('train_vs_best_hist_pct'))}"
    )
    print(
        f"- calib weighted_mse: {_fmt_scientific(row.get('calib_weighted_mse'))} "
        f"| best historical: {_fmt_scientific(row.get('best_hist_calib_weighted_mse'))} "
        f"({str(row.get('best_hist_calib_run_id', 'n/a'))}) "
        f"| delta: {_fmt_pct(row.get('calib_vs_best_hist_pct'))}"
    )
    print(
        f"- calib residual_rmse: {_fmt_scientific(row.get('calib_residual_rmse'))}"
    )

#################### read the config and data ####################
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cfg_data = config["data"]
shuffle = cfg_data["shuffle"]
integrity_cfg = cfg_data.get("integrity_checks", {})
integrity_enabled = bool(integrity_cfg.get("enabled", False))
integrity_round_decimals = integrity_cfg.get("round_decimals", 12)
if integrity_round_decimals is not None:
    integrity_round_decimals = int(integrity_round_decimals)

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

# Early stopping
cfg_es = config["callbacks"]["early_stopping"]
es_enable = bool(cfg_es.get("enabled", True))
es_monitor = cfg_es.get("monitor", "val_loss")
supported_es_monitors = {"val_loss", "val_loss_trimmed"}
if es_monitor not in supported_es_monitors:
    raise ValueError(f"Unsupported early-stopping monitor '{es_monitor}'. Use one of {supported_es_monitors}")

trim_top_fraction = float(cfg_es.get("trim_top_fraction", 0.0))
if trim_top_fraction < 0 or trim_top_fraction >= 1:
    raise ValueError("early_stopping.trim_top_fraction must be in [0, 1)")

requires_trimmed_monitor = es_monitor == "val_loss_trimmed"
if requires_trimmed_monitor and trim_top_fraction <= 0:
    raise ValueError(
        "Using monitor='val_loss_trimmed' requires early_stopping.trim_top_fraction > 0"
    )

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

if integrity_enabled:
    overlap_train_val = _count_feature_overlap(
        train_df_X,
        val_df_X,
        round_decimals=integrity_round_decimals
    )
    if overlap_train_val > 0:
        print(
            f"[warning] data integrity check: detected {overlap_train_val} overlapping "
            "feature rows between train and val (possible leakage)."
        )
    else:
        print("[info] data integrity check: no overlapping feature rows between train and val.")

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
optimizers_by_name = {}

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
    optimizers_by_name[mix_first_opt_name] = _build_optimizer(mix_first_opt_name)
    optimizers_by_name[mix_second_opt_name] = _build_optimizer(mix_second_opt_name)
else:
    active_opt_name = _normalize_optimizer_name(meta_opt_name)
    optimizers_by_name[active_opt_name] = _build_optimizer(active_opt_name)

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


if mix_enabled:
    # In MIX mode we only schedule ADAM's learning rate.
    lr_schedulers_by_name = {
        opt_name: (_build_lr_scheduler_for(opt_obj) if opt_name == "adam" else None)
        for opt_name, opt_obj in optimizers_by_name.items()
    }
else:
    lr_schedulers_by_name = {
        opt_name: _build_lr_scheduler_for(opt_obj)
        for opt_name, opt_obj in optimizers_by_name.items()
    }

optimizer = optimizers_by_name[active_opt_name]
lr_scheduler = lr_schedulers_by_name[active_opt_name]

### training with validation
epochs = config["loop"]["epochs"]

# to save some metrics 
metrics_dir = run_dir / "metrics"
epoch_times = []
history = []
best_monitor = float("inf")
best_monitor_name = es_monitor

for epoch in range(1, epochs+1): # each epoch
    epoch_start = time.time()

    # switch optimizers every mix_step epochs if mix mode is enabled
    if mix_enabled:
        block_idx = (epoch - 1) // mix_step
        desired_opt_name = mix_first_opt_name if (block_idx % 2 == 0) else mix_second_opt_name
        if desired_opt_name != active_opt_name:
            active_opt_name = desired_opt_name
            optimizer = optimizers_by_name[active_opt_name]
            lr_scheduler = lr_schedulers_by_name[active_opt_name]
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
    val_losses_for_trim = [] if requires_trimmed_monitor else None
    with torch.no_grad(): 
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            batch_loss = loss_fn(pred, yb)
            val_sum += batch_loss.item() * xb.size(0)

            if requires_trimmed_monitor:
                per_sample = _batch_losses(pred=pred, target=yb, loss_name=loss_name)
                val_losses_for_trim.append(per_sample.detach().cpu().numpy())

    val_loss = val_sum / n_val
    val_loss_trimmed = None
    if requires_trimmed_monitor:
        val_losses_np = np.concatenate(val_losses_for_trim, axis=0)
        trimmed_base = _trimmed_mean(val_losses_np, trim_top_fraction=trim_top_fraction)
        val_loss_trimmed = float(np.sqrt(trimmed_base)) if loss_name == "rmse" else trimmed_base

    monitor_values = {
        "val_loss": val_loss,
        "val_loss_trimmed": val_loss_trimmed if val_loss_trimmed is not None else val_loss,
    }
    monitor_value = monitor_values[best_monitor_name]

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
        "val_loss_trimmed": val_loss_trimmed,
        "monitor_name": best_monitor_name,
        "monitor_value": monitor_value,
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
        if monitor_value < best_monitor:
            best_monitor = monitor_value
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
        if early_stopper.step(epoch=epoch, value=monitor_value):
            print(
                f"Early stopping at epoch {epoch} | best {best_monitor_name}={early_stopper.best:.6f} "
                f"| patience={cfg_es['patience']} | min_delta={cfg_es['min_delta']}"
            )
            break


    # some logging
    if epoch==1 or epoch%5==0:
        trimmed_str = ""
        if val_loss_trimmed is not None:
            trimmed_str = f" | val_trim {loss_name}: {val_loss_trimmed:.6f}"
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"opt: {active_opt_name} | "
            f"train {loss_name}: {train_loss:.6f} | "
            f"val {loss_name}: {val_loss:.6f} | "
            f"monitor ({best_monitor_name}): {monitor_value:.6f}"
            f"{trimmed_str} | "
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

# run sensitivity plots automatically for this run
sensitivity_cfg_path = PROJECT_ROOT / "configs" / "sensitivity_config.yaml"
if sensitivity_cfg_path.exists():
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sensitivity_pricer.py"),
        "--config",
        str(sensitivity_cfg_path),
        "--model-dir",
        run_dir.name,
    ]
    print("Running sensitivity_pricer.py to generate 3x2 grid...")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"Sensitivity plots generated in {fig_dir}")
    else:
        print("Warning: sensitivity_pricer.py failed")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
else:
    print(f"Warning: sensitivity config not found at {sensitivity_cfg_path}")

# run calibration automatically for this run
_run_post_training_calibration(run_name=run_dir.name)

# refresh logs and show comparison against historical best
if _refresh_optimizer_logs():
    _print_run_vs_best_history(run_name=run_dir.name, optimizer_mode=meta_opt_name)
