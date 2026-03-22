from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from src.pinn.contracts import CaNNArtifactsSpec


def validate_cann_artifacts(spec: CaNNArtifactsSpec) -> None:
    """
    Minimal validation for scaffold mode:
    confirm that CaNN artifacts exist before wiring PINN stages.
    """
    missing = []
    if not spec.calibration_dir.exists():
        missing.append(spec.calibration_dir)
    if not spec.summary_file.exists():
        missing.append(spec.summary_file)
    if not spec.quotes_file.exists():
        missing.append(spec.quotes_file)

    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing CaNN artifacts required by PINN scaffold: "
            f"{missing_str}"
        )


def load_quotes_table(quotes_file: Path, *, file_format: str = "auto") -> pd.DataFrame:
    """
    Read CaNN quotes file (CSV/Parquet).
    """
    if not quotes_file.exists():
        raise FileNotFoundError(f"Quotes file not found: {quotes_file}")

    fmt = str(file_format).lower()
    if fmt == "auto":
        suffix = quotes_file.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            fmt = "parquet"
        elif suffix in {".csv", ".txt"}:
            fmt = "csv"
        else:
            raise ValueError(
                f"Unsupported quotes suffix '{quotes_file.suffix}'. "
                "Use csv/parquet or pass file_format explicitly."
            )

    if fmt == "csv":
        df = pd.read_csv(quotes_file)
    elif fmt == "parquet":
        df = pd.read_parquet(quotes_file)
    else:
        raise ValueError(f"Unsupported quotes format '{file_format}'")

    if df.empty:
        raise ValueError(f"Quotes table is empty: {quotes_file}")
    return df


def validate_required_quote_columns(
    quotes_df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    context: str = "CaNN quotes",
) -> None:
    missing = [col for col in required_columns if col not in quotes_df.columns]
    if missing:
        raise KeyError(
            f"{context} missing columns: {missing}. "
            f"Available: {list(quotes_df.columns)}"
        )


def load_calibration_summary(summary_file: Path) -> dict:
    """
    Read calibration summary metadata from CaNN.
    """
    with open(summary_file, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {summary_file}")
    return payload


def extract_theta_star(summary_payload: dict, *, parameter_key: str = "theta_star") -> list[float]:
    """
    Extract calibrated parameter vector from CaNN summary.
    """
    theta_raw = summary_payload.get(parameter_key)
    if theta_raw is None:
        raise KeyError(f"'{parameter_key}' not present in calibration summary.")
    if not isinstance(theta_raw, (list, tuple)):
        raise TypeError(
            f"'{parameter_key}' must be list/tuple, got {type(theta_raw)!r}."
        )
    return [float(value) for value in theta_raw]


def load_cann_inputs(
    spec: CaNNArtifactsSpec,
    *,
    required_quote_columns: Sequence[str] = ("moneyness", "tau", "r"),
    quotes_format: str = "auto",
) -> tuple[list[float], list[str], pd.DataFrame]:
    """
    Convenience loader for the first PINN step.
    Returns calibrated params, parameter order and quotes table.
    """
    validate_cann_artifacts(spec)

    summary = load_calibration_summary(spec.summary_file)
    theta_star = extract_theta_star(summary, parameter_key=spec.parameter_key)

    param_order_raw = summary.get("parameter_order")
    if isinstance(param_order_raw, list) and len(param_order_raw) == len(theta_star):
        parameter_order = [str(name) for name in param_order_raw]
    else:
        parameter_order = [f"theta_{i}" for i in range(len(theta_star))]

    quotes_df = load_quotes_table(spec.quotes_file, file_format=quotes_format)
    validate_required_quote_columns(
        quotes_df,
        required_columns=required_quote_columns,
    )
    return theta_star, parameter_order, quotes_df
