from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


RUNS = [
    {
        "key": "baseline",
        "label": "Baseline PINN",
        "run_dir": "outputs/pinn/PINN_mix_scaled_param",
        "diagnostics": "greeks/baseline_diagnostics",
    },
    {
        "key": "payoff_aware",
        "label": "Payoff-aware PINN",
        "run_dir": "outputs/pinn/payoff_aware_pinn",
        "diagnostics": "greeks/payoff_aware_diagnostics",
    },
    {
        "key": "payoff_boundary_layer",
        "label": "Payoff-aware + boundary-layer",
        "run_dir": "outputs/pinn/payoff_aware_boundary_layer_pinn",
        "diagnostics": "greeks/payoff_boundary_layer_diagnostics",
    },
    {
        "key": "boundary_no_payoff",
        "label": "Boundary-layer no-payoff",
        "run_dir": "outputs/pinn/boundary_layer_pinn_no_payoff",
        "diagnostics": "greeks/boundary_layer_no_payoff_diagnostics",
    },
    {
        "key": "adaptive_collocation",
        "label": "Adaptive collocation",
        "run_dir": "outputs/pinn/adaptive_collocation",
        "diagnostics": "greeks/adaptive_collocation_diagnostics",
    },
    {
        "key": "derivative_consistency",
        "label": "Derivative-consistency",
        "run_dir": "outputs/pinn/derivative_consistency",
        "diagnostics": "greeks/derivative_consistency_diagnostics",
    },
    {
        "key": "acv_control_variate_best_gate",
        "label": "ACV control variate",
        "run_dir": "outputs/pinn/acv_hard_patch_control_variate_best_gate_tau_floor_5e4",
        "diagnostics": "diagnostics",
        "metrics_format": "acv_yaml",
        "points_format": "acv_surface",
        "points_file": "surface_diagnostics.csv",
        "train_summary": "metrics/train_summary.yaml",
    },
]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _run_paths(run: dict) -> dict[str, Path]:
    run_dir = PROJECT_ROOT / run["run_dir"]
    diag_dir = run_dir / run["diagnostics"]
    metrics_file = run.get("metrics_file", "metrics_by_region.csv")
    if run.get("metrics_format") == "acv_yaml":
        metrics_file = run.get("metrics_file", "metrics.yaml")
    points_file = run.get("points_file", "points_baseline_diagnostics.csv")
    train_summary = run.get("train_summary", "train/metrics/train_summary.yaml")
    return {
        "run_dir": run_dir,
        "diagnostics_dir": diag_dir,
        "metrics": diag_dir / metrics_file,
        "points": diag_dir / points_file,
        "train_summary": run_dir / train_summary,
    }


def _metric(df: pd.DataFrame, *, region: str, variable: str, column: str) -> float:
    row = df[(df["region"] == region) & (df["variable"] == variable)]
    if row.empty or column not in row.columns:
        return float("nan")
    return float(row.iloc[0][column])


def _acv_metric(metrics: dict, *, region: str, variable: str, column: str) -> float:
    value = metrics.get("metrics", {}).get(region, {}).get(variable, {}).get(column)
    if value is None:
        return float("nan")
    return float(value)


def _build_comparison_table(runs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        paths = _run_paths(run)
        if not paths["metrics"].exists():
            raise FileNotFoundError(f"Missing metrics file for {run['label']}: {paths['metrics']}")
        if run.get("metrics_format") == "acv_yaml":
            metrics = _load_yaml(paths["metrics"])

            def metric(region: str, variable: str, column: str) -> float:
                return _acv_metric(metrics, region=region, variable=variable, column=column)

        else:
            metrics = pd.read_csv(paths["metrics"])

            def metric(region: str, variable: str, column: str) -> float:
                return _metric(metrics, region=region, variable=variable, column=column)

        train_summary = _load_yaml(paths["train_summary"])
        rows.append(
            {
                "model_key": run["key"],
                "model": run["label"],
                "price_rmse": metric("full", "price", "rmse"),
                "delta_rmse": metric("full", "delta", "rmse"),
                "gamma_rmse": metric("full", "gamma", "rmse"),
                "vega_rmse": metric("full", "vega", "rmse"),
                "theta_rmse": metric("full", "theta", "rmse"),
                "hard_price_rmse": metric("hard", "price", "rmse"),
                "hard_delta_rmse": metric("hard", "delta", "rmse"),
                "hard_gamma_rmse": metric("hard", "gamma", "rmse"),
                "hard_vega_rmse": metric("hard", "vega", "rmse"),
                "hard_theta_rmse": metric("hard", "theta", "rmse"),
                "p99_gamma_error": metric("full", "gamma", "p99_abs_error"),
                "hard_p99_gamma_error": metric("hard", "gamma", "p99_abs_error"),
                "pde_residual_rmse": metric("full", "pde_residual", "rmse"),
                "hard_pde_residual_rmse": metric("hard", "pde_residual", "rmse"),
                "training_time_seconds": float(
                    train_summary.get("total_training_seconds", train_summary.get("n_steps", float("nan")))
                ),
                "best_val_total": float(train_summary.get("best_val_total", train_summary.get("best_loss", float("nan")))),
            }
        )
    return pd.DataFrame(rows)


def _format_markdown_table(df: pd.DataFrame) -> str:
    columns = [
        ("model", "Model"),
        ("price_rmse", "Price RMSE"),
        ("delta_rmse", "Delta RMSE"),
        ("gamma_rmse", "Gamma RMSE"),
        ("hard_delta_rmse", "Hard Delta RMSE"),
        ("hard_gamma_rmse", "Hard Gamma RMSE"),
        ("hard_p99_gamma_error", "Hard p99 Gamma Error"),
        ("training_time_seconds", "Training Time (s)"),
    ]
    out = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for key, _label in columns:
            value = row[key]
            if key == "model":
                cells.append(str(value))
            elif key == "training_time_seconds":
                cells.append("" if pd.isna(value) else f"{value:.1f}")
            else:
                cells.append("" if pd.isna(value) else f"{value:.6g}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _save_bar_chart(*, df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["price_rmse", "delta_rmse", "gamma_rmse", "hard_delta_rmse", "hard_gamma_rmse"]
    labels = ["Price", "Delta", "Gamma", "Hard Delta", "Hard Gamma"]
    x = np.arange(len(metrics))
    width = 0.12

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    for i, (_, row) in enumerate(df.iterrows()):
        y = [float(row[m]) for m in metrics]
        ax.bar(x + (i - (len(df) - 1) / 2) * width, y, width=width, label=row["model"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("PINN Ablation Study: Key RMSE Metrics")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _read_points(run: dict) -> pd.DataFrame:
    paths = _run_paths(run)
    if not paths["points"].exists():
        raise FileNotFoundError(f"Missing points diagnostics for {run['label']}: {paths['points']}")
    df = pd.read_csv(paths["points"])
    if run.get("points_format") == "acv_surface":
        for variable in ["price", "delta", "gamma", "vega", "theta", "rho"]:
            error_col = f"{variable}_error"
            abs_error_col = f"abs_error_{variable}"
            if error_col in df.columns and abs_error_col not in df.columns:
                df[abs_error_col] = df[error_col].abs()
    df["model_key"] = run["key"]
    df["model"] = run["label"]
    return df


def _region_masks(points: pd.DataFrame, *, epsilon_m: float = 0.03, epsilon_tau: float = 0.05) -> dict[str, np.ndarray]:
    m = points["moneyness"].to_numpy(dtype=np.float64)
    tau = points["tau"].to_numpy(dtype=np.float64)
    atm = np.abs(np.log(np.maximum(m, np.finfo(np.float64).tiny))) < epsilon_m
    short = tau < epsilon_tau
    return {
        "full": np.ones(len(points), dtype=bool),
        "hard": atm & short,
        "short_maturity": short,
        "atm": atm,
    }


def _save_distribution_plots(*, runs: list[dict], output_dir: Path) -> None:
    points_by_model = {run["key"]: _read_points(run) for run in runs}
    variables = ["price", "delta", "gamma", "vega"]
    regions = ["full", "hard", "short_maturity", "atm"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for variable in variables:
        col = f"abs_error_{variable}"
        for region in regions:
            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            plotted = False
            for run in runs:
                points = points_by_model[run["key"]]
                if col not in points.columns:
                    continue
                mask = _region_masks(points)[region]
                values = points.loc[mask, col].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                ax.ecdf(np.maximum(values, 1.0e-16), label=run["label"])
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xscale("log")
            ax.set_xlabel(f"|{variable} error| (log scale)")
            ax.set_ylabel("empirical CDF")
            ax.set_title(f"{variable} absolute-error distribution: {region}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"dist_{region}_{variable}.png", dpi=240)
            plt.close(fig)


def _heatmap_matrix(points: pd.DataFrame, *, value_col: str, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.sort(points[x_col].unique())
    y_values = np.sort(points[y_col].unique())
    pivot = points.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=y_values, columns=x_values)
    return pivot.to_numpy(dtype=np.float64), x_values, y_values


def _save_heatmap_grid(*, runs: list[dict], output_dir: Path) -> None:
    variables = [
        ("price", "abs_error_price", "Price absolute error", False),
        ("delta", "abs_error_delta", "Delta absolute error", True),
        ("gamma", "abs_error_gamma", "Gamma absolute error", True),
        ("vega", "abs_error_vega", "Vega absolute error", True),
        ("pde_residual", "pde_residual_abs", "PDE residual absolute value", True),
    ]
    points_by_model = {run["key"]: _read_points(run) for run in runs}
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, col, title, log_scale in variables:
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), sharex=True, sharey=True)
        axes_flat = axes.reshape(-1)
        all_values: list[np.ndarray] = []
        matrices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for run in runs:
            points = points_by_model[run["key"]].copy()
            if key == "pde_residual":
                if "pde_residual" not in points.columns:
                    matrices.append((np.full((1, 1), np.nan), np.array([0.0]), np.array([0.0])))
                    continue
                points[col] = np.abs(points["pde_residual"].to_numpy(dtype=np.float64))
            matrix, x_values, y_values = _heatmap_matrix(points, value_col=col, x_col="tau", y_col="moneyness")
            matrices.append((matrix, x_values, y_values))
            valid = matrix[np.isfinite(matrix)]
            if valid.size:
                all_values.append(valid)

        norm = None
        vmin = None
        vmax = None
        if all_values:
            flat = np.concatenate(all_values)
            if log_scale:
                flat = flat[flat > 0.0]
                if flat.size:
                    vmin = max(float(np.percentile(flat, 5.0)), float(np.finfo(np.float64).tiny))
                    vmax = float(np.percentile(flat, 95.0))
                    if vmax <= vmin:
                        vmax = vmin * 10.0
                    norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                vmin = float(np.percentile(flat, 5.0))
                vmax = float(np.percentile(flat, 95.0))
                if vmax <= vmin:
                    vmax = None

        last_im = None
        for ax, run, (matrix, x_values, y_values) in zip(axes_flat, runs, matrices):
            plot_matrix = matrix
            if log_scale and vmin is not None:
                plot_matrix = np.where(np.isfinite(matrix), np.maximum(matrix, vmin), np.nan)
            last_im = ax.imshow(
                plot_matrix,
                origin="lower",
                aspect="auto",
                extent=[float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max())],
                cmap="magma",
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
            )
            ax.set_title(run["label"], fontsize=9)
            ax.set_xlabel("tau")
            ax.set_ylabel("moneyness")
        for ax in axes_flat[len(runs):]:
            ax.axis("off")
        fig.suptitle(title)
        if last_im is not None:
            fig.colorbar(last_im, ax=axes_flat[: len(runs)].tolist(), shrink=0.82)
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_comparison_{key}.png", dpi=240)
        plt.close(fig)


def _save_hard_zoom_heatmaps(*, runs: list[dict], output_dir: Path) -> None:
    points_by_model = {run["key"]: _read_points(run) for run in runs}
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable in ("gamma", "delta"):
        col = f"abs_error_{variable}"
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), sharex=True, sharey=True)
        axes_flat = axes.reshape(-1)
        last_im = None
        for ax, run in zip(axes_flat, runs):
            points = points_by_model[run["key"]]
            mask = (points["tau"] <= 0.12) & (points["moneyness"].between(0.94, 1.06))
            zoom = points.loc[mask].copy()
            matrix, x_values, y_values = _heatmap_matrix(zoom, value_col=col, x_col="tau", y_col="moneyness")
            valid = matrix[np.isfinite(matrix) & (matrix > 0)]
            norm = None
            plot_matrix = matrix
            if valid.size:
                vmin = max(float(np.percentile(valid, 5.0)), float(np.finfo(np.float64).tiny))
                vmax = float(np.percentile(valid, 95.0))
                if vmax <= vmin:
                    vmax = vmin * 10.0
                norm = LogNorm(vmin=vmin, vmax=vmax)
                plot_matrix = np.where(np.isfinite(matrix), np.maximum(matrix, vmin), np.nan)
            last_im = ax.imshow(
                plot_matrix,
                origin="lower",
                aspect="auto",
                extent=[float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max())],
                cmap="magma",
                norm=norm,
            )
            ax.set_title(run["label"], fontsize=9)
            ax.set_xlabel("tau")
            ax.set_ylabel("moneyness")
        for ax in axes_flat[len(runs):]:
            ax.axis("off")
        fig.suptitle(f"Hard-region zoom: {variable} absolute error")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes_flat[: len(runs)].tolist(), shrink=0.82)
        fig.tight_layout()
        fig.savefig(output_dir / f"hard_zoom_{variable}.png", dpi=240)
        plt.close(fig)


def build_ablation_study(*, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = _build_comparison_table(RUNS)
    table_path = tables_dir / "ablation_comparison.csv"
    table.to_csv(table_path, index=False)
    markdown_path = tables_dir / "ablation_comparison.md"
    markdown_path.write_text(_format_markdown_table(table), encoding="utf-8")

    _save_bar_chart(df=table, output_path=figures_dir / "ablation_key_metrics.png")
    _save_heatmap_grid(runs=RUNS, output_dir=figures_dir)
    _save_hard_zoom_heatmaps(runs=RUNS, output_dir=figures_dir)
    _save_distribution_plots(runs=RUNS, output_dir=figures_dir)

    summary = {
        "runs": RUNS,
        "outputs": {
            "comparison_csv": str(table_path),
            "comparison_markdown": str(markdown_path),
            "figures_dir": str(figures_dir),
        },
        "best_by_metric": {
            metric: str(table.loc[table[metric].idxmin(), "model"])
            for metric in [
                "price_rmse",
                "delta_rmse",
                "gamma_rmse",
                "hard_delta_rmse",
                "hard_gamma_rmse",
                "hard_p99_gamma_error",
            ]
            if table[metric].notna().any()
        },
    }
    summary_path = output_dir / "ablation_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return {
        "comparison_csv": table_path,
        "comparison_markdown": markdown_path,
        "summary": summary_path,
        "figures_dir": figures_dir,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PINN Greek-improvement ablation study.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pinn" / "ablation_study",
        help="Output directory for ablation tables and figures.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outputs = build_ablation_study(output_dir=output_dir)
    print("PINN ablation study completed")
    print(f"Comparison CSV: {outputs['comparison_csv']}")
    print(f"Comparison Markdown: {outputs['comparison_markdown']}")
    print(f"Summary: {outputs['summary']}")
    print(f"Figures dir: {outputs['figures_dir']}")


if __name__ == "__main__":
    main()
