from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.pinn.model import build_pinn_model


def _resolve_device(device_pref: str) -> torch.device:
    pref = str(device_pref).lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref in {"cpu", "cuda", "mps"}:
        return torch.device(pref)
    raise ValueError(f"Unsupported device '{device_pref}'")


def _predict_in_batches(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    input_affine: dict | None = None,
) -> np.ndarray:
    x_eval = x
    if input_affine is not None:
        a = np.asarray(input_affine.get("a", []), dtype=np.float32).reshape(1, -1)
        b = np.asarray(input_affine.get("b", []), dtype=np.float32).reshape(1, -1)
        if a.shape[1] != x.shape[1] or b.shape[1] != x.shape[1]:
            raise ValueError(
                "input_scaling dimension mismatch in evaluator: "
                f"x has {x.shape[1]}, a has {a.shape[1]}, b has {b.shape[1]}."
            )
        x_eval = a + b * x

    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, x_eval.shape[0], batch_size):
            stop = min(start + batch_size, x_eval.shape[0])
            xb = torch.from_numpy(x_eval[start:stop]).to(device)
            yb = model(xb).detach().cpu().numpy()
            preds.append(yb.astype(np.float64, copy=False))
    return np.concatenate(preds, axis=0).reshape(-1)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    return {
        "n_samples": int(y_true.size),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
    }


def _split_indices_from_training_cfg(
    *,
    n_samples: int,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = int(round(n_samples * val_fraction))
    n_val = max(1, min(n_val, n_samples - 1))
    return idx[n_val:], idx[:n_val]


def _load_input_affine_from_train_summary(*, run_dir: Path, x_dim: int) -> dict | None:
    summary_path = run_dir / "train" / "metrics" / "train_summary.yaml"
    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return None

    scaling = payload.get("input_scaling", {})
    if not isinstance(scaling, dict):
        return None
    if not bool(scaling.get("enabled", False)):
        return None

    a = np.asarray(scaling.get("a", []), dtype=np.float32).reshape(-1)
    b = np.asarray(scaling.get("b", []), dtype=np.float32).reshape(-1)
    if a.size != x_dim or b.size != x_dim:
        return None

    return {
        "a": a.tolist(),
        "b": b.tolist(),
        "method": str(scaling.get("method", "unknown")),
    }


def evaluate_pinn_run(
    *,
    run_dir: Path,
    model_config: dict,
    training_config: dict,
    evaluation_config: dict,
    dataset_file: Path | str | None = None,
    checkpoint_file: Path | str | None = None,
    split_indices_file: Path | str | None = None,
) -> dict:
    """
    Evaluate supervised PINN run (all/train/val metrics).
    """
    run_dir = Path(run_dir)
    dataset_path = (
        Path(dataset_file)
        if dataset_file is not None
        else run_dir / "data" / "supervised_dataset.npz"
    )
    checkpoint_path = (
        Path(checkpoint_file)
        if checkpoint_file is not None
        else run_dir / "train" / "checkpoints" / "model_best.pt"
    )
    split_idx_path = (
        Path(split_indices_file)
        if split_indices_file is not None
        else run_dir / "train" / "metrics" / "split_indices.npz"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    arrays = np.load(dataset_path)
    x = arrays["X"].astype(np.float32)
    y = arrays["y"].astype(np.float32).reshape(-1)

    meta_cfg = training_config.get("meta", {})
    data_cfg = training_config.get("data", {})
    seed = int(meta_cfg.get("seed", 42))
    val_fraction = float(data_cfg.get("val_fraction", 0.2))

    if split_idx_path.exists():
        split = np.load(split_idx_path)
        train_idx = split["train_idx"].astype(np.int64)
        val_idx = split["val_idx"].astype(np.int64)
    else:
        train_idx, val_idx = _split_indices_from_training_cfg(
            n_samples=x.shape[0],
            seed=seed,
            val_fraction=val_fraction,
        )

    device = _resolve_device(meta_cfg.get("device", "auto"))
    model = build_pinn_model(model_config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    input_affine = _load_input_affine_from_train_summary(run_dir=run_dir, x_dim=x.shape[1])

    batch_size = int(evaluation_config.get("batch_size", 8192))
    pred_all = _predict_in_batches(
        model=model,
        x=x,
        batch_size=batch_size,
        device=device,
        input_affine=input_affine,
    )

    metrics_all = _compute_metrics(y_true=y, y_pred=pred_all)
    metrics_train = _compute_metrics(y_true=y[train_idx], y_pred=pred_all[train_idx])
    metrics_val = _compute_metrics(y_true=y[val_idx], y_pred=pred_all[val_idx])

    metrics_payload = {
        "dataset_file": str(dataset_path),
        "checkpoint_file": str(checkpoint_path),
        "split_indices_file": str(split_idx_path),
        "device": str(device),
        "input_scaling_applied": bool(input_affine is not None),
        "input_scaling_method": (None if input_affine is None else input_affine.get("method")),
        "metrics_all": metrics_all,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
    }

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_yaml = eval_dir / "eval_metrics.yaml"
    with open(metrics_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_payload, f, sort_keys=False)

    metrics_table = pd.DataFrame(
        [
            {"split": "all", **metrics_all},
            {"split": "train", **metrics_train},
            {"split": "val", **metrics_val},
        ]
    )
    metrics_csv = eval_dir / "eval_metrics.csv"
    metrics_table.to_csv(metrics_csv, index=False)

    return {
        "metrics_yaml": str(metrics_yaml),
        "metrics_csv": str(metrics_csv),
        "metrics_all": metrics_all,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
    }
