from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


NORMALIZATION_STATS_REL_PATH = Path("metrics") / "normalization_stats.yaml"


def is_normalization_enabled(stats: dict[str, Any] | None) -> bool:
    return bool(stats is not None and stats.get("enabled", False))


def build_normalization_stats(
    *,
    train_features: np.ndarray,
    train_target: np.ndarray,
    feature_names: list[str],
    target_name: str,
    enabled: bool,
    eps: float = 1.0e-12,
    normalize_target: bool = True,
) -> dict[str, Any]:
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("normalization eps must be > 0")

    if not enabled:
        return {
            "enabled": False,
            "normalize_target": False,
            "eps": eps,
            "feature_names": list(feature_names),
            "target_name": target_name,
            "x_mean": [],
            "x_std": [],
            "y_mean": 0.0,
            "y_std": 1.0,
        }

    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"train_features must be 2D; got ndim={x.ndim}")
    if x.shape[1] != len(feature_names):
        raise ValueError(
            "feature_names size mismatch. "
            f"Expected {x.shape[1]}, got {len(feature_names)}"
        )
    if y.size != x.shape[0]:
        raise ValueError(
            "train_target length mismatch. "
            f"Expected {x.shape[0]}, got {y.size}"
        )

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0, ddof=0)
    x_std = np.where(np.abs(x_std) < eps, 1.0, x_std)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y, ddof=0))
    if abs(y_std) < eps:
        y_std = 1.0

    return {
        "enabled": True,
        "normalize_target": bool(normalize_target),
        "eps": eps,
        "feature_names": list(feature_names),
        "target_name": target_name,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize_features(features: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if not is_normalization_enabled(stats):
        return x.astype(np.float32, copy=False)

    x_mean = np.asarray(stats["x_mean"], dtype=np.float64)
    x_std = np.asarray(stats["x_std"], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D; got ndim={x.ndim}")
    if x.shape[1] != x_mean.size:
        raise ValueError(
            "features column mismatch for normalization. "
            f"Expected {x_mean.size}, got {x.shape[1]}"
        )

    out = (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    return out.astype(np.float32, copy=False)


def normalize_target(target: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    if not is_normalization_enabled(stats):
        return y.astype(np.float32, copy=False)
    if not bool(stats.get("normalize_target", False)):
        return y.astype(np.float32, copy=False)

    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])
    out = (y - y_mean) / y_std
    return out.astype(np.float32, copy=False)


def denormalize_target(target: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    if not is_normalization_enabled(stats):
        return y.astype(np.float64, copy=False)
    if not bool(stats.get("normalize_target", False)):
        return y.astype(np.float64, copy=False)

    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])
    out = y * y_std + y_mean
    return out.astype(np.float64, copy=False)


def normalization_stats_path(run_dir: Path) -> Path:
    return run_dir / NORMALIZATION_STATS_REL_PATH


def save_normalization_stats(*, run_dir: Path, stats: dict[str, Any]) -> Path:
    out_path = normalization_stats_path(run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    return out_path


def load_normalization_stats_from_run(run_dir: Path) -> dict[str, Any] | None:
    path = normalization_stats_path(run_dir)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return None
    return data
