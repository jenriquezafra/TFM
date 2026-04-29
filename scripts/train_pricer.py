import sys
import time
import re
import argparse
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
from src.models.normalization import (
    build_normalization_stats,
    denormalize_target,
    normalize_features,
    normalize_target,
    save_normalization_stats,
)
from src.utils.callbacks import save_checkpoints, EarlyStopping

config_path = PROJECT_ROOT / "configs" / "model_training.yaml"
model_config_path = PROJECT_ROOT / "configs" / "model_architecture.yaml"
calibration_config_path = PROJECT_ROOT / "configs" / "calibration.yaml"
experiment_logs_dir = PROJECT_ROOT / "outputs" / "experiment_logs"
calibration_folder_pattern = re.compile(r"^Calibration_(\d+)$")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANN pricer.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(config_path),
        help="Path to training YAML config (absolute or relative to project root).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(model_config_path),
        help="Path to model architecture YAML config (absolute or relative to project root).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output folder name under outputs/runs. Defaults to a timestamp.",
    )
    return parser.parse_args()


def _resolve_config_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


cli_args = _parse_cli_args()
config_path = _resolve_config_path(cli_args.config)
model_config_path = _resolve_config_path(cli_args.model_config)

if not config_path.exists():
    raise FileNotFoundError(f"Training config not found: {config_path}")
if not model_config_path.exists():
    raise FileNotFoundError(f"Model architecture config not found: {model_config_path}")

DEFAULT_FEATURE_COLUMNS = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]
TARGET_FALLBACK_CANDIDATES = ("iv_brent", "IV")

# to save the run outputs
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
run_id = cli_args.run_name.strip() if cli_args.run_name else datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
    raise ValueError(f"Invalid --run-name: {run_id!r}")
run_dir = RUNS_DIR / run_id

# create files for the runs
(run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
(run_dir / "metrics").mkdir()
(run_dir / "figures").mkdir()


# to save the config used
shutil.copy(
    config_path,
    run_dir / "model_training_copy.yaml"
)
shutil.copy(
    model_config_path,
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


def _validate_bin_edges(edges, *, name: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} bin edges must have at least 2 values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} bin edges contain non-finite values")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} bin edges must be strictly increasing")
    return arr


def _resolve_target_column(df: pd.DataFrame, preferred: str | None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(TARGET_FALLBACK_CANDIDATES)
    candidates.append(str(df.columns[-1]))

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not resolve target column from candidates={candidates}")


def _resolve_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    preferred: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if preferred is not None:
        if not isinstance(preferred, (list, tuple)):
            raise TypeError("data.feature_columns must be a list/tuple when provided")
        cols = [str(col) for col in preferred]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise KeyError(f"Configured feature columns are missing in dataset: {missing}")
        return cols

    if all(col in df.columns for col in DEFAULT_FEATURE_COLUMNS):
        return list(DEFAULT_FEATURE_COLUMNS)

    cols = [str(col) for col in df.columns if str(col) != target_col]
    if not cols:
        raise ValueError("No feature columns found after removing target column")
    return cols


def _apply_row_filters(df: pd.DataFrame, *, filters_cfg: dict, split_name: str) -> pd.DataFrame:
    if not bool(filters_cfg.get("enabled", False)):
        return df

    out = df.copy()
    n_before = len(out)

    finite_columns = [str(col) for col in filters_cfg.get("finite_columns", [])]
    for col in finite_columns:
        if col not in out.columns:
            raise KeyError(f"row_filters.finite_columns includes missing column '{col}' on split '{split_name}'")
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
        out = out.loc[np.isfinite(values)].copy()

    equals = filters_cfg.get("equals", {}) or {}
    for col, value in equals.items():
        if col not in out.columns:
            raise KeyError(f"row_filters.equals includes missing column '{col}' on split '{split_name}'")
        out = out.loc[out[col] == value].copy()

    greater_equal = filters_cfg.get("greater_equal", {}) or {}
    for col, value in greater_equal.items():
        if col not in out.columns:
            raise KeyError(f"row_filters.greater_equal includes missing column '{col}' on split '{split_name}'")
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
        out = out.loc[values >= float(value)].copy()

    less_equal = filters_cfg.get("less_equal", {}) or {}
    for col, value in less_equal.items():
        if col not in out.columns:
            raise KeyError(f"row_filters.less_equal includes missing column '{col}' on split '{split_name}'")
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
        out = out.loc[values <= float(value)].copy()

    out = out.reset_index(drop=True)
    removed = n_before - len(out)
    print(f"[data] row_filters on {split_name}: kept={len(out)} removed={removed}")
    if out.empty:
        raise ValueError(f"row_filters removed all rows on split '{split_name}'")
    return out


def _drop_non_finite_rows(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    split_name: str,
) -> pd.DataFrame:
    required_cols = feature_cols + [target_col]
    values = df.loc[:, required_cols].to_numpy(dtype=np.float64)
    finite_mask = np.all(np.isfinite(values), axis=1)
    dropped = int((~finite_mask).sum())
    if dropped > 0:
        print(f"[data] dropped non-finite rows on {split_name}: {dropped}")
    out = df.loc[finite_mask].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"All rows are non-finite on split '{split_name}' for selected features/target")
    return out


def _predict_model(
    *,
    model: torch.nn.Module,
    x_features: np.ndarray,
    batch_size: int,
    device: str,
    norm_stats: dict,
) -> np.ndarray:
    x_norm = normalize_features(x_features, norm_stats)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, x_norm.shape[0], batch_size):
            stop = min(start + batch_size, x_norm.shape[0])
            x_chunk = torch.from_numpy(x_norm[start:stop]).to(device)
            y_chunk = model(x_chunk).detach().cpu().numpy().reshape(-1)
            outputs.append(y_chunk.astype(np.float64, copy=False))
    pred = np.concatenate(outputs, axis=0)
    return denormalize_target(pred, norm_stats)


def _build_global_metrics(split: str, residual: np.ndarray) -> dict:
    abs_err = np.abs(residual)
    mse = float(np.mean(residual**2))
    return {
        "split": split,
        "n_samples": int(residual.size),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_err)),
        "abs_err_p50": float(np.quantile(abs_err, 0.50)),
        "abs_err_p90": float(np.quantile(abs_err, 0.90)),
        "abs_err_p99": float(np.quantile(abs_err, 0.99)),
    }


def _build_bin_metrics(
    *,
    split: str,
    feature_name: str,
    feature_values: np.ndarray,
    residual: np.ndarray,
    bin_edges: np.ndarray,
) -> list[dict]:
    idx = np.digitize(feature_values, bin_edges[1:-1], right=False)
    rows = []
    for i in range(bin_edges.size - 1):
        mask = idx == i
        if not np.any(mask):
            continue
        err = residual[mask]
        mse = float(np.mean(err**2))
        rows.append(
            {
                "split": split,
                "feature": feature_name,
                "bin_left": float(bin_edges[i]),
                "bin_right": float(bin_edges[i + 1]),
                "n_samples": int(mask.sum()),
                "share_pct": float(100.0 * mask.mean()),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(err))),
            }
        )
    return rows


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
    optimizer_mode = str(optimizer_mode).strip().lower()
    if optimizer_mode not in {"mix", "mix_half", "adam"}:
        return

    log_family = "mix" if optimizer_mode in {"mix", "mix_half"} else "adam"
    log_path = experiment_logs_dir / f"{log_family}_experiments.csv"
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
train_path = PROJECT_ROOT / cfg_data["dir"] / "train.parquet"
val_path = PROJECT_ROOT / cfg_data["dir"] / "val.parquet"
test_path = PROJECT_ROOT / cfg_data["dir"] / "test.parquet"

train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

row_filters_cfg = cfg_data.get("row_filters", {}) or {}
train_df = _apply_row_filters(train_df, filters_cfg=row_filters_cfg, split_name="train")
val_df = _apply_row_filters(val_df, filters_cfg=row_filters_cfg, split_name="val")
test_df = _apply_row_filters(test_df, filters_cfg=row_filters_cfg, split_name="test")


#################### train the model ####################
### read the data already splitted

target_col = _resolve_target_column(train_df, cfg_data.get("target_column"))
feature_cols = _resolve_feature_columns(
    train_df,
    target_col=target_col,
    preferred=cfg_data.get("feature_columns"),
)
print(f"[data] target column: {target_col}")
print(f"[data] feature columns ({len(feature_cols)}): {feature_cols}")

missing_in_val = [col for col in feature_cols + [target_col] if col not in val_df.columns]
missing_in_test = [col for col in feature_cols + [target_col] if col not in test_df.columns]
if missing_in_val:
    raise KeyError(f"Validation split is missing required columns: {missing_in_val}")
if missing_in_test:
    raise KeyError(f"Test split is missing required columns: {missing_in_test}")

train_df = _drop_non_finite_rows(
    train_df,
    feature_cols=feature_cols,
    target_col=target_col,
    split_name="train",
)
val_df = _drop_non_finite_rows(
    val_df,
    feature_cols=feature_cols,
    target_col=target_col,
    split_name="val",
)
test_df = _drop_non_finite_rows(
    test_df,
    feature_cols=feature_cols,
    target_col=target_col,
    split_name="test",
)

train_df_X = train_df.loc[:, feature_cols]
train_df_y = train_df.loc[:, target_col]

val_df_X = val_df.loc[:, feature_cols]
val_df_y = val_df.loc[:, target_col]

test_df_X = test_df.loc[:, feature_cols]
test_df_y = test_df.loc[:, target_col]

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

pre_cfg = config.get("preprocessing", {})
norm_cfg = pre_cfg.get("normalization", {})
norm_enabled = bool(norm_cfg.get("enabled", False))
norm_eps = float(norm_cfg.get("eps", 1.0e-12))
norm_target_enabled = bool(norm_cfg.get("normalize_target", norm_enabled))

normalization_stats = build_normalization_stats(
    train_features=train_df_X.to_numpy(dtype=np.float64),
    train_target=train_df_y.to_numpy(dtype=np.float64),
    feature_names=feature_cols,
    target_name=target_col,
    enabled=norm_enabled,
    eps=norm_eps,
    normalize_target=norm_target_enabled,
)
norm_stats_path = save_normalization_stats(run_dir=run_dir, stats=normalization_stats)
print(
    f"[preprocessing] normalization enabled={norm_enabled} "
    f"(target={norm_target_enabled}) | stats={norm_stats_path}"
)

eval_cfg = config.get("evaluation", {})
eval_enabled = bool(eval_cfg.get("enabled", True))
eval_batch_size = int(eval_cfg.get("batch_size", 8192))
if eval_batch_size <= 0:
    raise ValueError("evaluation.batch_size must be > 0")

eval_bins_cfg = eval_cfg.get("bins", {})
tau_bins = _validate_bin_edges(
    eval_bins_cfg.get("tau", [0.05, 0.25, 0.5, 1.0, 2.0, 3.0]),
    name="tau",
)
moneyness_bins = _validate_bin_edges(
    eval_bins_cfg.get("moneyness", [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]),
    name="moneyness",
)

X_train_np = normalize_features(train_df_X.to_numpy(dtype=np.float64), normalization_stats)
y_train_np = normalize_target(train_df_y.to_numpy(dtype=np.float64).reshape(-1, 1), normalization_stats)
X_val_np = normalize_features(val_df_X.to_numpy(dtype=np.float64), normalization_stats)
y_val_np = normalize_target(val_df_y.to_numpy(dtype=np.float64).reshape(-1, 1), normalization_stats)

X_train = torch.from_numpy(X_train_np).float()
y_train = torch.from_numpy(y_train_np).float()

X_val = torch.from_numpy(X_val_np).float()
y_val = torch.from_numpy(y_val_np).float()

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

def _normalize_training_mode(name: str) -> str:
    raw = str(name).strip().lower()
    if raw in ("l-bfgs", "lbfgs"):
        return "lbfgs"
    if raw in ("mix_half", "mix-half", "mix_once"):
        return "mix_half"
    return raw


meta_opt_name = _normalize_training_mode(config["meta"]["optimizer"])
supported_training_modes = {"adam", "lbfgs", "mix", "mix_half"}
if meta_opt_name not in supported_training_modes:
    raise ValueError(
        "Unsupported training mode on meta.optimizer. "
        "Use one of: 'adam', 'L-BFGS', 'mix', 'mix_half'"
    )

if batch_size_train == "all":
    batch_size_train_adam = n_train
else:
    batch_size_train_adam = int(batch_size_train)

lbfgs_cfg = None
for item in config["optimizers"]:
    raw_name = str(item.get("name", "")).lower()
    if raw_name in ("l-bfgs", "lbfgs"):
        lbfgs_cfg = item
        break
lbfgs_full_batch = bool(lbfgs_cfg.get("full_batch", False)) if lbfgs_cfg is not None else False

if lbfgs_full_batch:
    batch_size_train_lbfgs = n_train
else:
    batch_size_train_lbfgs = batch_size_train_adam
    if meta_opt_name in ("mix", "mix_half", "lbfgs"):
        batch_size_train_lbfgs = min(batch_size_train_lbfgs * 10, n_train)

train_loaders_by_name = {
    "adam": DataLoader(
        train_ds,
        batch_size=batch_size_train_adam,
        shuffle=shuffle,
        generator=g,
    ),
    "lbfgs": DataLoader(
        train_ds,
        batch_size=batch_size_train_lbfgs,
        # L-BFGS full-batch is deterministic and safer without shuffling.
        shuffle=(shuffle and not lbfgs_full_batch),
        generator=g,
    ),
}

print(
    "[data] train batch size | "
    f"adam: {batch_size_train_adam} | "
    f"lbfgs: {batch_size_train_lbfgs} (full_batch={lbfgs_full_batch})"
)

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


# mixed optimizer setup
mix_enabled = meta_opt_name == "mix"
mix_half_enabled = meta_opt_name == "mix_half"
mix_step = None
mix_half_switch_epoch = None
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
elif mix_half_enabled:
    mix_first_opt_name = "adam"
    mix_second_opt_name = "lbfgs"
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


if mix_enabled or mix_half_enabled:
    # In mixed modes we only schedule ADAM's learning rate.
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
epochs = int(config["loop"]["epochs"])
if mix_half_enabled:
    # Adam for first half, L-BFGS for second half, with a single switch.
    mix_half_switch_epoch = max(2, (epochs // 2) + 1)
    if mix_half_switch_epoch > epochs:
        mix_half_switch_epoch = epochs + 1
    adam_end_epoch = min(epochs, mix_half_switch_epoch - 1)
    if mix_half_switch_epoch <= epochs:
        print(
            f"[mix_half] schedule: adam epochs 1-{adam_end_epoch}, "
            f"lbfgs epochs {mix_half_switch_epoch}-{epochs}"
        )
    else:
        print(f"[mix_half] schedule: adam epochs 1-{epochs} (lbfgs not reached)")

# to save some metrics 
metrics_dir = run_dir / "metrics"
epoch_times = []
history = []
best_monitor = float("inf")
best_monitor_name = es_monitor

for epoch in range(1, epochs+1): # each epoch
    epoch_start = time.time()

    # optimizer switching policy for mixed modes
    if mix_enabled:
        block_idx = (epoch - 1) // mix_step
        desired_opt_name = mix_first_opt_name if (block_idx % 2 == 0) else mix_second_opt_name
        if desired_opt_name != active_opt_name:
            active_opt_name = desired_opt_name
            optimizer = optimizers_by_name[active_opt_name]
            lr_scheduler = lr_schedulers_by_name[active_opt_name]
            print(f"[mix] epoch {epoch}: switched optimizer to {active_opt_name}")
    elif mix_half_enabled:
        desired_opt_name = mix_first_opt_name if epoch < mix_half_switch_epoch else mix_second_opt_name
        if desired_opt_name != active_opt_name:
            active_opt_name = desired_opt_name
            optimizer = optimizers_by_name[active_opt_name]
            lr_scheduler = lr_schedulers_by_name[active_opt_name]
            print(f"[mix_half] epoch {epoch}: switched optimizer to {active_opt_name}")


    # train
    model.train()
    train_sum = 0.0
    train_loader = train_loaders_by_name[active_opt_name]

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

if eval_enabled:
    eval_model = ANN(
        input_dim=model_cfg["input"]["dim"],
        hidden_dims=model_cfg["hidden"]["dims"],
        output_dim=model_cfg["output"]["dim"],
        activation=model_cfg["hidden"]["activation"],
        dropout_rate=model_cfg["hidden"]["dropout_rate"],
        initialization=model_cfg["hidden"]["initialization"],
    ).to(device)
    best_ckpt = torch.load(ckpt_best_path, map_location=torch.device(device))
    eval_model.load_state_dict(best_ckpt["model_state"])
    eval_model.eval()

    split_payload = {
        "train": (train_df_X, train_df_y),
        "val": (val_df_X, val_df_y),
        "test": (test_df_X, test_df_y),
    }
    global_rows = []
    bin_rows = []
    for split_name, (x_df, y_series) in split_payload.items():
        x_np = x_df.to_numpy(dtype=np.float64)
        y_true = y_series.to_numpy(dtype=np.float64).reshape(-1)
        y_pred = _predict_model(
            model=eval_model,
            x_features=x_np,
            batch_size=eval_batch_size,
            device=device,
            norm_stats=normalization_stats,
        )
        residual = y_pred - y_true

        global_rows.append(_build_global_metrics(split=split_name, residual=residual))

        if "moneyness" in x_df.columns:
            m_vals = x_df["moneyness"].to_numpy(dtype=np.float64)
            bin_rows.extend(
                _build_bin_metrics(
                    split=split_name,
                    feature_name="moneyness",
                    feature_values=m_vals,
                    residual=residual,
                    bin_edges=moneyness_bins,
                )
            )
        if "tau" in x_df.columns:
            tau_vals = x_df["tau"].to_numpy(dtype=np.float64)
            bin_rows.extend(
                _build_bin_metrics(
                    split=split_name,
                    feature_name="tau",
                    feature_values=tau_vals,
                    residual=residual,
                    bin_edges=tau_bins,
                )
            )

    eval_global_df = pd.DataFrame(global_rows)
    eval_global_df.to_parquet(metrics_dir / "eval_global.parquet", index=False)
    eval_global_df.to_csv(metrics_dir / "eval_global.csv", index=False)

    eval_bins_df = pd.DataFrame(bin_rows)
    if not eval_bins_df.empty:
        eval_bins_df.to_parquet(metrics_dir / "eval_by_region.parquet", index=False)
        eval_bins_df.to_csv(metrics_dir / "eval_by_region.csv", index=False)

    eval_summary = {
        "normalization": {
            "enabled": norm_enabled,
            "normalize_target": norm_target_enabled,
            "stats_file": str(norm_stats_path),
        },
        "global": {
            row["split"]: {
                "n_samples": int(row["n_samples"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "abs_err_p50": float(row["abs_err_p50"]),
                "abs_err_p90": float(row["abs_err_p90"]),
                "abs_err_p99": float(row["abs_err_p99"]),
            }
            for row in global_rows
        },
        "bins": {
            "moneyness": moneyness_bins.tolist(),
            "tau": tau_bins.tolist(),
        },
        "artifacts": {
            "global_parquet": str(metrics_dir / "eval_global.parquet"),
            "global_csv": str(metrics_dir / "eval_global.csv"),
            "region_parquet": str(metrics_dir / "eval_by_region.parquet"),
            "region_csv": str(metrics_dir / "eval_by_region.csv"),
        },
    }
    with open(metrics_dir / "eval_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_summary, f, sort_keys=False)

    print("Saved evaluation metrics:")
    for row in global_rows:
        print(
            f"  [{row['split']}] mse={row['mse']:.8e} "
            f"rmse={row['rmse']:.8e} mae={row['mae']:.8e}"
        )
else:
    print("Evaluation skipped (evaluation.enabled=False)")

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
