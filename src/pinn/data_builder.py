from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset suffix '{path.suffix}'")
    if df.empty:
        raise ValueError(f"Dataset file is empty: {path}")
    return df


def _resolve_target_column(
    quotes_df: pd.DataFrame,
    *,
    target_column: str,
    fallback_target_columns: Sequence[str],
) -> str:
    if target_column in quotes_df.columns:
        return target_column
    for candidate in fallback_target_columns:
        if candidate in quotes_df.columns:
            return candidate
    raise KeyError(
        f"Target column '{target_column}' not found and no fallback matched. "
        f"Available columns: {list(quotes_df.columns)}"
    )


def build_supervised_xy(
    *,
    theta_star: Sequence[float],
    quotes_df: pd.DataFrame,
    feature_columns: Sequence[str] = ("moneyness", "tau", "r"),
    target_column: str = "price_market",
    fallback_target_columns: Sequence[str] = ("iv_market",),
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Build X,y from CaNN quotes and calibrated theta.
    """
    missing = [col for col in feature_columns if col not in quotes_df.columns]
    if missing:
        raise KeyError(
            f"Missing feature columns: {missing}. "
            f"Available columns: {list(quotes_df.columns)}"
        )

    target_col_used = _resolve_target_column(
        quotes_df,
        target_column=target_column,
        fallback_target_columns=fallback_target_columns,
    )

    theta = np.asarray(theta_star, dtype=np.float32).reshape(1, -1)
    if theta.size == 0:
        raise ValueError("theta_star cannot be empty")

    n_samples = int(len(quotes_df))
    theta_block = np.repeat(theta, repeats=n_samples, axis=0)
    market_block = quotes_df.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
    x = np.concatenate([theta_block, market_block], axis=1)
    y = quotes_df.loc[:, target_col_used].to_numpy(dtype=np.float32).reshape(-1, 1)

    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values")
    if not np.isfinite(y).all():
        raise ValueError("Target vector contains non-finite values")

    return x, y, target_col_used


def build_supervised_dataset(
    *,
    cann_quotes_path: Path,
    theta_star: Sequence[float],
    output_dir: Path,
    feature_columns: Sequence[str] = ("moneyness", "tau", "r"),
    target_column: str = "price_market",
    fallback_target_columns: Sequence[str] = ("iv_market",),
) -> Path:
    """
    Build and persist supervised dataset artifacts from CaNN outputs.
    """
    quotes_df = _read_table(cann_quotes_path)
    x, y, target_col_used = build_supervised_xy(
        theta_star=theta_star,
        quotes_df=quotes_df,
        feature_columns=feature_columns,
        target_column=target_column,
        fallback_target_columns=fallback_target_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "supervised_dataset.npz"
    np.savez(
        dataset_path,
        X=x,
        y=y,
        theta_star=np.asarray(theta_star, dtype=np.float32),
    )

    manifest = {
        "source_quotes_file": str(cann_quotes_path),
        "dataset_file": str(dataset_path),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "feature_columns": [f"theta_{i}" for i in range(len(theta_star))]
        + list(feature_columns),
        "target_column_requested": target_column,
        "target_column_used": target_col_used,
        "fallback_target_columns": list(fallback_target_columns),
    }
    manifest_path = output_dir / "supervised_dataset_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return dataset_path


def build_collocation_dataset(*, sampling_config: dict, output_dir: Path) -> Path:
    """
    Build collocation points for PDE residual training.
    """
    raise NotImplementedError(
        "PINN scaffold only: collocation dataset builder not implemented yet."
    )


def build_boundary_dataset(*, boundary_config: dict, output_dir: Path) -> Path:
    """
    Build boundary/initial-condition samples used by PINN constraints.
    """
    raise NotImplementedError(
        "PINN scaffold only: boundary dataset builder not implemented yet."
    )
