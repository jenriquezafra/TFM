import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "outputs" / "calibration"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "experiment_logs"
CALIBRATION_DIR_PATTERN = re.compile(r"^Calibration_(\d+)$")


def _normalize_optimizer_name(name: Any) -> str:
    raw = str(name or "").strip().lower()
    if raw in {"l-bfgs", "lbfgs"}:
        return "lbfgs"
    return raw


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _get_optimizer_cfg(training_cfg: dict[str, Any], opt_name: str) -> dict[str, Any]:
    target = _normalize_optimizer_name(opt_name)
    for item in training_cfg.get("optimizers", []):
        if _normalize_optimizer_name(item.get("name")) == target:
            return item
    return {}


def _read_metrics_summary(metrics_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "epochs_ran": None,
        "best_epoch": None,
        "best_optimizer": None,
        "best_train_loss": None,
        "best_val_loss": None,
        "best_gap": None,
        "last_epoch": None,
        "last_optimizer": None,
        "last_train_loss": None,
        "last_val_loss": None,
        "last_gap": None,
        "val_last_vs_best_pct": None,
    }

    if not metrics_path.exists():
        return out

    df = pd.read_parquet(metrics_path)
    if df.empty:
        return out

    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)

    out["epochs_ran"] = int(len(df))

    last_row = df.iloc[-1]
    out["last_epoch"] = int(last_row["epoch"]) if "epoch" in df.columns else int(len(df))
    out["last_optimizer"] = (
        str(last_row["optimizer"]) if "optimizer" in df.columns and pd.notna(last_row["optimizer"]) else None
    )
    out["last_train_loss"] = float(last_row["train_loss"]) if "train_loss" in df.columns else None
    out["last_val_loss"] = float(last_row["val_loss"]) if "val_loss" in df.columns else None
    if out["last_train_loss"] is not None and out["last_val_loss"] is not None:
        out["last_gap"] = out["last_val_loss"] - out["last_train_loss"]

    if "val_loss" not in df.columns:
        return out

    valid_val = df["val_loss"].dropna()
    if valid_val.empty:
        return out

    best_idx = int(df["val_loss"].idxmin())
    best_row = df.loc[best_idx]
    out["best_epoch"] = int(best_row["epoch"]) if "epoch" in df.columns else (best_idx + 1)
    out["best_optimizer"] = (
        str(best_row["optimizer"]) if "optimizer" in df.columns and pd.notna(best_row["optimizer"]) else None
    )
    out["best_train_loss"] = float(best_row["train_loss"]) if "train_loss" in df.columns else None
    out["best_val_loss"] = float(best_row["val_loss"])
    if out["best_train_loss"] is not None:
        out["best_gap"] = out["best_val_loss"] - out["best_train_loss"]

    if out["last_val_loss"] is not None and out["best_val_loss"] not in (None, 0.0):
        out["val_last_vs_best_pct"] = ((out["last_val_loss"] / out["best_val_loss"]) - 1.0) * 100.0

    return out


def _read_latest_calibration_summary(*, calibration_dir: Path, run_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "calib_available": False,
        "calib_latest_id": None,
        "calib_output_dir": None,
        "calib_timestamp": None,
        "calib_success": None,
        "calib_objective_fun": None,
        "calib_weighted_mse": None,
        "calib_residual_rmse": None,
        "calib_n_quotes": None,
    }

    run_calibration_dir = calibration_dir / run_id
    if not run_calibration_dir.exists():
        return out

    latest_id = None
    latest_dir: Path | None = None
    for child in run_calibration_dir.iterdir():
        if not child.is_dir():
            continue
        match = CALIBRATION_DIR_PATTERN.match(child.name)
        if match is None:
            continue
        calib_id = int(match.group(1))
        if latest_id is None or calib_id > latest_id:
            latest_id = calib_id
            latest_dir = child

    if latest_dir is None:
        return out

    summary_yaml = latest_dir / "summary.yaml"
    if not summary_yaml.exists():
        return out

    summary = _load_yaml(summary_yaml)
    out["calib_available"] = True
    out["calib_latest_id"] = latest_id
    out["calib_output_dir"] = str(latest_dir)
    out["calib_timestamp"] = summary.get("timestamp")
    out["calib_success"] = summary.get("success")
    out["calib_objective_fun"] = summary.get("objective_fun")
    out["calib_weighted_mse"] = summary.get("weighted_mse")
    out["calib_residual_rmse"] = summary.get("residual_rmse")
    out["calib_n_quotes"] = summary.get("n_quotes")
    return out


def _build_run_row(run_dir: Path, *, calibration_dir: Path) -> dict[str, Any] | None:
    training_cfg_path = run_dir / "model_training_copy.yaml"
    metrics_path = run_dir / "metrics" / "metrics.parquet"
    if not training_cfg_path.exists():
        return None

    training_cfg = _load_yaml(training_cfg_path)
    if not training_cfg:
        return None

    optimizer_mode = _normalize_optimizer_name(training_cfg.get("meta", {}).get("optimizer"))
    if optimizer_mode not in {"mix", "mix_half", "adam"}:
        return None

    adam_cfg = _get_optimizer_cfg(training_cfg, "adam")
    lbfgs_cfg = _get_optimizer_cfg(training_cfg, "lbfgs")
    mix_cfg = _get_optimizer_cfg(training_cfg, "mix")
    mix_half_cfg = _get_optimizer_cfg(training_cfg, "mix_half")

    callbacks_cfg = training_cfg.get("callbacks", {})
    es_cfg = callbacks_cfg.get("early_stopping", {})
    lr_cfg = callbacks_cfg.get("lr_scheduler", {})
    loop_cfg = training_cfg.get("loop", {})
    data_cfg = training_cfg.get("data", {})

    metrics_summary = _read_metrics_summary(metrics_path)
    calibration_summary = _read_latest_calibration_summary(
        calibration_dir=calibration_dir,
        run_id=run_dir.name,
    )

    row: dict[str, Any] = {
        "run_id": run_dir.name,
        "run_modified_at": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "optimizer_mode": optimizer_mode,
        "seed": training_cfg.get("meta", {}).get("seed"),
        "data_dir": data_cfg.get("dir"),
        "loss_name": training_cfg.get("loss", {}).get("name"),
        "epochs_config": loop_cfg.get("epochs"),
        "batch_size_train": loop_cfg.get("batch_size_train"),
        "batch_size_val": loop_cfg.get("batch_size_val"),
        "es_enabled": es_cfg.get("enabled"),
        "es_monitor": es_cfg.get("monitor"),
        "es_patience": es_cfg.get("patience"),
        "es_min_delta": es_cfg.get("min_delta"),
        "scheduler_enabled": lr_cfg.get("enabled"),
        "scheduler_name": lr_cfg.get("name"),
        "scheduler_step_size": lr_cfg.get("step_size"),
        "scheduler_gamma": lr_cfg.get("gamma"),
        "adam_lr": adam_cfg.get("learn_rate"),
        "adam_weight_decay": adam_cfg.get("weight_decay"),
        "lbfgs_lr": lbfgs_cfg.get("learn_rate"),
        "lbfgs_max_iter": lbfgs_cfg.get("max_iter"),
        "lbfgs_history_size": lbfgs_cfg.get("historic_size"),
        "lbfgs_full_batch": lbfgs_cfg.get("full_batch"),
        "mix_step_size": mix_cfg.get("step_size") if optimizer_mode == "mix" else None,
        "mix_first_optimizer": mix_cfg.get("first_optimizer") if optimizer_mode == "mix" else None,
        "mix_half_first_optimizer": (
            mix_half_cfg.get("first_optimizer") if optimizer_mode == "mix_half" else None
        ),
    }
    row.update(metrics_summary)
    row.update(calibration_summary)
    return row


def _add_historical_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    out["best_hist_train_run_id"] = None
    out["best_hist_train_val_loss"] = None
    out["train_vs_best_hist_pct"] = None

    out["best_hist_calib_run_id"] = None
    out["best_hist_calib_weighted_mse"] = None
    out["calib_vs_best_hist_pct"] = None

    for idx, row in out.iterrows():
        others = out.drop(index=idx)

        train_candidates = others[pd.to_numeric(others["best_val_loss"], errors="coerce").notna()]
        if not train_candidates.empty:
            best_train_idx = pd.to_numeric(train_candidates["best_val_loss"], errors="coerce").idxmin()
            best_train_row = train_candidates.loc[best_train_idx]
            best_train_val = float(best_train_row["best_val_loss"])
            out.at[idx, "best_hist_train_run_id"] = str(best_train_row["run_id"])
            out.at[idx, "best_hist_train_val_loss"] = best_train_val
            this_train_val = pd.to_numeric(pd.Series([row["best_val_loss"]]), errors="coerce").iloc[0]
            if pd.notna(this_train_val) and best_train_val != 0:
                out.at[idx, "train_vs_best_hist_pct"] = ((float(this_train_val) / best_train_val) - 1.0) * 100.0

        calib_candidates = others[pd.to_numeric(others["calib_weighted_mse"], errors="coerce").notna()]
        if not calib_candidates.empty:
            best_calib_idx = pd.to_numeric(calib_candidates["calib_weighted_mse"], errors="coerce").idxmin()
            best_calib_row = calib_candidates.loc[best_calib_idx]
            best_calib_mse = float(best_calib_row["calib_weighted_mse"])
            out.at[idx, "best_hist_calib_run_id"] = str(best_calib_row["run_id"])
            out.at[idx, "best_hist_calib_weighted_mse"] = best_calib_mse
            this_calib_mse = pd.to_numeric(pd.Series([row["calib_weighted_mse"]]), errors="coerce").iloc[0]
            if pd.notna(this_calib_mse) and best_calib_mse != 0:
                out.at[idx, "calib_vs_best_hist_pct"] = ((float(this_calib_mse) / best_calib_mse) - 1.0) * 100.0

    return out


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    df = _add_historical_comparison(df)
    if not df.empty and "run_modified_at" in df.columns:
        df = df.sort_values("run_modified_at", ascending=False)
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build experiment logs for mixed-family (mix + mix_half) and ADAM runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing model run folders.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing calibration outputs grouped by run id.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for optimizer experiment CSV logs.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    calibration_dir = args.calibration_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_rows: list[dict[str, Any]] = []
    adam_rows: list[dict[str, Any]] = []

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        row = _build_run_row(run_dir, calibration_dir=calibration_dir)
        if row is None:
            continue
        if row["optimizer_mode"] in {"mix", "mix_half"}:
            mix_rows.append(row)
        elif row["optimizer_mode"] == "adam":
            adam_rows.append(row)

    mix_out = out_dir / "mix_experiments.csv"
    adam_out = out_dir / "adam_experiments.csv"
    _write_csv(mix_rows, mix_out)
    _write_csv(adam_rows, adam_out)

    print(f"Wrote MIX-family log (mix + mix_half): {mix_out} ({len(mix_rows)} runs)")
    print(f"Wrote ADAM log: {adam_out} ({len(adam_rows)} runs)")


if __name__ == "__main__":
    main()
