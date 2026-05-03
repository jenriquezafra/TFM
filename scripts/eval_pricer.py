from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.model_inference import FEATURE_ORDER, load_model_from_run, predict_iv


DEFAULT_TAU_BINS = [0.05, 0.25, 0.5, 1.0, 2.0, 3.0]
DEFAULT_MONEYNESS_BINS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
DEFAULT_SPLITS = ("train", "val", "test")
FEATURE_LABELS = {
    "moneyness": r"$m$",
    "tau": r"$\tau$",
}
METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mape_pct": "MAPE (%)",
}
PLOT_TITLE_SIZE = 15
PLOT_LABEL_SIZE = 13
PLOT_TICK_SIZE = 11


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _validate_bin_edges(edges: list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} bin edges must contain at least two values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} bin edges contain non-finite values")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} bin edges must be strictly increasing")
    return arr


def _parse_bin_edges(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    chunks = [token.strip() for token in raw.split(",") if token.strip()]
    if len(chunks) < 2:
        raise ValueError("Bin edges must have at least two comma-separated values")
    return [float(token) for token in chunks]


def _resolve_data_dir(*, run_dir: Path, cli_data_dir: str | None) -> Path:
    if cli_data_dir:
        data_path = Path(cli_data_dir)
        return data_path if data_path.is_absolute() else PROJECT_ROOT / data_path

    run_cfg = _load_yaml_dict(run_dir / "model_training_copy.yaml")
    cfg_data_dir = run_cfg.get("data", {}).get("dir")
    if cfg_data_dir:
        return PROJECT_ROOT / cfg_data_dir

    default_cfg = _load_yaml_dict(PROJECT_ROOT / "configs" / "model_training.yaml")
    default_data_dir = default_cfg.get("data", {}).get("dir")
    if default_data_dir:
        return PROJECT_ROOT / default_data_dir

    raise FileNotFoundError(
        "Could not resolve dataset directory. Pass --data-dir explicitly."
    )


def _resolve_bins_from_config(
    *,
    run_dir: Path,
    tau_cli: list[float] | None,
    moneyness_cli: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if tau_cli is not None and moneyness_cli is not None:
        return (
            _validate_bin_edges(tau_cli, name="tau"),
            _validate_bin_edges(moneyness_cli, name="moneyness"),
        )

    run_cfg = _load_yaml_dict(run_dir / "model_training_copy.yaml")
    fallback_cfg = _load_yaml_dict(PROJECT_ROOT / "configs" / "model_training.yaml")
    bins_cfg = run_cfg.get("evaluation", {}).get("bins", {}) or fallback_cfg.get(
        "evaluation", {}
    ).get("bins", {})

    tau = tau_cli if tau_cli is not None else bins_cfg.get("tau", DEFAULT_TAU_BINS)
    moneyness = (
        moneyness_cli
        if moneyness_cli is not None
        else bins_cfg.get("moneyness", DEFAULT_MONEYNESS_BINS)
    )
    return (
        _validate_bin_edges(list(tau), name="tau"),
        _validate_bin_edges(list(moneyness), name="moneyness"),
    )


def _resolve_feature_columns(
    *, df: pd.DataFrame, target_col: str, normalization_stats: dict[str, Any] | None
) -> list[str]:
    if normalization_stats is not None:
        feature_names = normalization_stats.get("feature_names", [])
        if isinstance(feature_names, list) and feature_names:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise KeyError(f"Missing normalized feature columns in dataset: {missing}")
            return [str(col) for col in feature_names]

    if all(col in df.columns for col in FEATURE_ORDER):
        return list(FEATURE_ORDER)

    features = [c for c in df.columns if c != target_col]
    if not features:
        raise ValueError("No feature columns detected in dataset")
    return features


def _build_global_metrics(
    *,
    split: str,
    residual: np.ndarray,
    y_true: np.ndarray,
    mape_floor: float,
) -> dict[str, Any]:
    abs_err = np.abs(residual)
    mse = float(np.mean(residual**2))
    denom = np.maximum(np.abs(y_true), float(mape_floor))
    return {
        "split": split,
        "n_samples": int(residual.size),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape_pct": float(100.0 * np.mean(abs_err / denom)),
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
    y_true: np.ndarray,
    bin_edges: np.ndarray,
    mape_floor: float,
) -> list[dict[str, Any]]:
    idx = np.digitize(feature_values, bin_edges[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for i in range(bin_edges.size - 1):
        mask = idx == i
        if not np.any(mask):
            continue
        err = residual[mask]
        abs_err = np.abs(err)
        denom = np.maximum(np.abs(y_true[mask]), float(mape_floor))
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
                "mape_pct": float(100.0 * np.mean(abs_err / denom)),
            }
        )
    return rows


def _plot_global_metric(*, global_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if global_df.empty or metric not in global_df.columns:
        return
    splits = list(global_df["split"])
    values = global_df[metric].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(splits, values, color="#2f6db5")
    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("split", fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel(metric_label, fontsize=PLOT_LABEL_SIZE)
    if np.all(values > 0.0):
        ax.set_yscale("log")
    ax.set_title(f"Global {metric_label} by split", fontsize=PLOT_TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
    ax.grid(True, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_region_metric(
    *,
    bins_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    feature_order: tuple[str, ...] = ("moneyness", "tau"),
    split_order: tuple[str, ...] = DEFAULT_SPLITS,
) -> None:
    if bins_df.empty or metric not in bins_df.columns:
        return

    fig, axes = plt.subplots(1, len(feature_order), figsize=(7 * len(feature_order), 4.5))
    if len(feature_order) == 1:
        axes = [axes]

    for ax, feature in zip(axes, feature_order):
        feature_df = bins_df[bins_df["feature"] == feature]
        feature_label = FEATURE_LABELS.get(feature, feature)
        metric_label = METRIC_LABELS.get(metric, metric)
        if feature_df.empty:
            ax.text(0.5, 0.5, f"No data for {feature_label}", ha="center", va="center")
            ax.set_axis_off()
            continue

        for split in split_order:
            split_df = feature_df[feature_df["split"] == split].copy()
            if split_df.empty:
                continue
            split_df = split_df.sort_values("bin_left")
            x_center = 0.5 * (
                split_df["bin_left"].to_numpy(dtype=np.float64)
                + split_df["bin_right"].to_numpy(dtype=np.float64)
            )
            y_val = split_df[metric].to_numpy(dtype=np.float64)
            ax.plot(x_center, y_val, marker="o", linewidth=1.3, label=split)

        ax.set_xlabel(feature_label, fontsize=PLOT_LABEL_SIZE)
        ax.set_ylabel(metric_label, fontsize=PLOT_LABEL_SIZE)
        if np.all(feature_df[metric].to_numpy(dtype=np.float64) > 0.0):
            ax.set_yscale("log")
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.3)
        ax.set_title(f"{metric_label} by {feature_label} bins", fontsize=PLOT_TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
        ax.legend(fontsize=PLOT_TICK_SIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_split_list(raw_splits: str) -> tuple[str, ...]:
    splits = tuple(chunk.strip() for chunk in raw_splits.split(",") if chunk.strip())
    if not splits:
        raise ValueError("--splits must include at least one split")
    for split in splits:
        if split not in {"train", "val", "test"}:
            raise ValueError("Each split must be one of {train,val,test}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing ANN run on train/val/test with regional metrics."
    )
    parser.add_argument("--model-dir", default="latest", help="Run dir name under outputs/runs")
    parser.add_argument("--checkpoint-name", default="model_best.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory containing train.parquet/val.parquet/test.parquet",
    )
    parser.add_argument("--target-col", default="iv_brent")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--mape-floor", type=float, default=1.0e-4)
    parser.add_argument(
        "--tau-bins",
        default=None,
        help="Comma-separated tau bin edges, e.g. 0.05,0.25,0.5,1.0,2.0,3.0",
    )
    parser.add_argument(
        "--moneyness-bins",
        default=None,
        help="Comma-separated moneyness bin edges, e.g. 0.6,0.8,0.9,1.0,1.1,1.2,1.4",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    model, model_device, run_dir, _, normalization_stats = load_model_from_run(
        project_root=PROJECT_ROOT,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )

    data_dir = _resolve_data_dir(run_dir=run_dir, cli_data_dir=args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    split_names = _parse_split_list(args.splits)
    tau_bins, moneyness_bins = _resolve_bins_from_config(
        run_dir=run_dir,
        tau_cli=_parse_bin_edges(args.tau_bins),
        moneyness_cli=_parse_bin_edges(args.moneyness_bins),
    )

    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    global_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []

    for split in split_names:
        split_path = data_dir / f"{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_df = pd.read_parquet(split_path)
        target_col = args.target_col
        if target_col not in split_df.columns and target_col == "iv_brent" and "IV" in split_df.columns:
            target_col = "IV"
        if target_col not in split_df.columns:
            raise KeyError(f"Target column '{args.target_col}' not found in {split_path}")

        feature_cols = _resolve_feature_columns(
            df=split_df,
            target_col=target_col,
            normalization_stats=normalization_stats,
        )
        x_np = split_df[feature_cols].to_numpy(dtype=np.float64)
        y_true = split_df[target_col].to_numpy(dtype=np.float64).reshape(-1)

        y_pred = predict_iv(
            model=model,
            features=x_np,
            device=model_device,
            batch_size=args.batch_size,
            normalization_stats=normalization_stats,
        )
        residual = y_pred - y_true
        global_rows.append(
            _build_global_metrics(
                split=split,
                residual=residual,
                y_true=y_true,
                mape_floor=float(args.mape_floor),
            )
        )

        if "moneyness" in split_df.columns:
            bin_rows.extend(
                _build_bin_metrics(
                    split=split,
                    feature_name="moneyness",
                    feature_values=split_df["moneyness"].to_numpy(dtype=np.float64),
                    residual=residual,
                    y_true=y_true,
                    bin_edges=moneyness_bins,
                    mape_floor=float(args.mape_floor),
                )
            )
        if "tau" in split_df.columns:
            bin_rows.extend(
                _build_bin_metrics(
                    split=split,
                    feature_name="tau",
                    feature_values=split_df["tau"].to_numpy(dtype=np.float64),
                    residual=residual,
                    y_true=y_true,
                    bin_edges=tau_bins,
                    mape_floor=float(args.mape_floor),
                )
            )

    global_df = pd.DataFrame(global_rows)
    bins_df = pd.DataFrame(bin_rows)

    global_parquet = metrics_dir / "eval_global.parquet"
    global_csv = metrics_dir / "eval_global.csv"
    region_parquet = metrics_dir / "eval_by_region.parquet"
    region_csv = metrics_dir / "eval_by_region.csv"
    summary_yaml = metrics_dir / "eval_summary.yaml"

    global_df.to_parquet(global_parquet, index=False)
    global_df.to_csv(global_csv, index=False)
    if not bins_df.empty:
        bins_df.to_parquet(region_parquet, index=False)
        bins_df.to_csv(region_csv, index=False)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_name": args.checkpoint_name,
        "device": str(model_device),
        "data_dir": str(data_dir),
        "splits": list(split_names),
        "target_col": args.target_col,
        "batch_size": int(args.batch_size),
        "normalization": {
            "enabled": bool(normalization_stats is not None and normalization_stats.get("enabled", False)),
            "normalize_target": bool(normalization_stats is not None and normalization_stats.get("normalize_target", False)),
            "stats_file": str(run_dir / "metrics" / "normalization_stats.yaml"),
        },
        "bins": {
            "moneyness": moneyness_bins.tolist(),
            "tau": tau_bins.tolist(),
        },
        "global": {
            row["split"]: {
                "n_samples": int(row["n_samples"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mape_pct": float(row["mape_pct"]),
                "abs_err_p50": float(row["abs_err_p50"]),
                "abs_err_p90": float(row["abs_err_p90"]),
                "abs_err_p99": float(row["abs_err_p99"]),
            }
            for row in global_rows
        },
        "artifacts": {
            "eval_global_parquet": str(global_parquet),
            "eval_global_csv": str(global_csv),
            "eval_region_parquet": str(region_parquet),
            "eval_region_csv": str(region_csv),
            "eval_summary_yaml": str(summary_yaml),
        },
    }

    with open(summary_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"Run: {run_dir.name}")
    print(f"Data dir: {data_dir}")
    for row in global_rows:
        print(
            f"[{row['split']}] "
            f"mse={row['mse']:.8e} rmse={row['rmse']:.8e} mape={row['mape_pct']:.4f}%"
        )
    print(f"Saved: {global_parquet}")
    if not bins_df.empty:
        print(f"Saved: {region_parquet}")
    print(f"Saved: {summary_yaml}")

    if not args.no_plots:
        _plot_global_metric(
            global_df=global_df,
            metric="mse",
            out_path=figures_dir / "eval_global_mse.png",
        )
        _plot_global_metric(
            global_df=global_df,
            metric="mape_pct",
            out_path=figures_dir / "eval_global_mape.png",
        )
        _plot_region_metric(
            bins_df=bins_df,
            metric="mse",
            out_path=figures_dir / "eval_region_mse.png",
            split_order=split_names,
        )
        _plot_region_metric(
            bins_df=bins_df,
            metric="mape_pct",
            out_path=figures_dir / "eval_region_mape.png",
            split_order=split_names,
        )
        print(f"Saved plots to: {figures_dir}")


if __name__ == "__main__":
    main()
