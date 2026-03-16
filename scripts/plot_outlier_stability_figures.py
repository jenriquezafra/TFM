from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("val", "test")
INTERVAL_RE = re.compile(r"[\[\(]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate concise figures for outlier/surface stability study."
    )
    parser.add_argument("--model-dir", default="Liu_like_v01")
    parser.add_argument("--analysis-dir", default=None)
    return parser.parse_args()


def _resolve_analysis_dir(args: argparse.Namespace) -> Path:
    if args.analysis_dir:
        path = Path(args.analysis_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "outputs" / "runs" / args.model_dir / "outliers_analysis"


def _interval_left(label: str) -> float:
    match = INTERVAL_RE.search(str(label))
    if not match:
        return np.inf
    return float(match.group(1))


def _sorted_interval_labels(labels: list[str]) -> list[str]:
    clean = [str(x) for x in labels if str(x).lower() != "nan"]
    return sorted(clean, key=_interval_left)


def _plot_mse_breakdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    splits = summary_df["split"].tolist()
    total = summary_df["global_mse"].to_numpy(dtype=np.float64)
    hard = summary_df["mse_contrib_outliers_hard"].to_numpy(dtype=np.float64)
    region_non_out = summary_df["mse_contrib_region_excluding_hard_outliers"].to_numpy(dtype=np.float64)
    rest = np.maximum(total - hard - region_non_out, 0.0)
    share = np.divide(hard, np.maximum(total, 1e-18))

    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x, hard, label="Hard outliers", color="#d7263d")
    ax.bar(x, region_non_out, bottom=hard, label="Low m, low tau (non-outliers)", color="#f4a259")
    ax.bar(x, rest, bottom=hard + region_non_out, label="Rest", color="#2a9d8f")
    ax.set_xticks(x, splits)
    ax.set_ylabel("MSE contribution")
    ax.set_yscale("log")
    ax.set_title("MSE decomposition by split")
    ax.grid(True, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    for i, s in enumerate(share):
        ax.text(i, total[i] * 1.15, f"outliers {100.0*s:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_outlier_map(analysis_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12.0, 4.5), sharex=True, sharey=True)
    if len(SPLITS) == 1:
        axes = [axes]

    for ax, split in zip(axes, SPLITS):
        out_path_split = analysis_dir / f"{split}_outliers_detailed.parquet"
        ref_path_split = analysis_dir / f"{split}_reference_non_outliers.parquet"
        if not out_path_split.exists():
            ax.text(0.5, 0.5, f"No outliers file for {split}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df_out = pd.read_parquet(out_path_split)
        if ref_path_split.exists():
            df_ref = pd.read_parquet(ref_path_split)
            ax.scatter(
                df_ref["tau"],
                df_ref["moneyness"],
                s=10,
                alpha=0.22,
                color="#9aa0a6",
                label="Reference non-outliers",
            )

        sc = ax.scatter(
            df_out["tau"],
            df_out["moneyness"],
            c=df_out["abs_error"],
            cmap="magma",
            s=44,
            edgecolors="black",
            linewidths=0.35,
            label="Outliers",
        )
        ax.axvline(0.25, color="#1d3557", linestyle="--", linewidth=1.0)
        ax.axhline(0.8, color="#1d3557", linestyle="--", linewidth=1.0)
        ax.set_title(split)
        ax.set_xlabel("tau")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("moneyness")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("abs_error")
    fig.suptitle("Outlier location on surface", y=1.07)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps_by_metric(
    *,
    analysis_dir: Path,
    metric_col: str,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12.0, 4.8), sharex=True, sharey=True)
    if len(SPLITS) == 1:
        axes = [axes]
    image = None

    for ax, split in zip(axes, SPLITS):
        csv_path = analysis_dir / f"{split}_surface_grad_by_region.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, f"No region file for {split}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df = pd.read_csv(csv_path)
        df = df[(df["m_bin"].astype(str).str.lower() != "nan") & (df["tau_bin"].astype(str).str.lower() != "nan")]
        if df.empty or metric_col not in df.columns:
            ax.text(0.5, 0.5, f"No metric {metric_col} for {split}", ha="center", va="center")
            ax.set_axis_off()
            continue

        x_labels = _sorted_interval_labels(df["m_bin"].unique().tolist())
        y_labels = _sorted_interval_labels(df["tau_bin"].unique().tolist())
        x_map = {k: i for i, k in enumerate(x_labels)}
        y_map = {k: i for i, k in enumerate(y_labels)}

        mat = np.full((len(y_labels), len(x_labels)), np.nan, dtype=np.float64)
        for _, row in df.iterrows():
            y = y_map[str(row["tau_bin"])]
            x = x_map[str(row["m_bin"])]
            mat[y, x] = float(row[metric_col])

        image = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("moneyness bin")
        ax.set_title(split)

    axes[0].set_ylabel("tau bin")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.04, pad=0.02)
        cbar.set_label(metric_col)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_greek_lifts(analysis_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12.0, 4.5), sharey=True)
    if len(SPLITS) == 1:
        axes = [axes]

    for ax, split in zip(axes, SPLITS):
        csv_path = analysis_dir / f"{split}_outliers_vs_ref_greek_metrics.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, f"No greek file for {split}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df = pd.read_csv(csv_path)
        if "median_lift_out_vs_ref" not in df.columns:
            ax.text(0.5, 0.5, f"No lift metric for {split}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df = df[df["metric"] != "abs_gamma"].copy()
        df["greek"] = df["metric"].str.replace("abs_", "", regex=False)
        df = df.sort_values("median_lift_out_vs_ref", ascending=False)

        ax.bar(df["greek"], df["median_lift_out_vs_ref"], color="#264653")
        ax.set_yscale("log")
        ax.set_title(split)
        ax.set_xlabel("greek")
        ax.grid(True, axis="y", which="major")
        ax.grid(True, axis="y", which="minor", alpha=0.3)

    axes[0].set_ylabel("median lift (outliers / reference)")
    fig.suptitle("Greek lift in outliers", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = _resolve_analysis_dir(args)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis dir not found: {analysis_dir}")

    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / "all_splits_outlier_stability_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    _plot_mse_breakdown(summary_df, figures_dir / "01_mse_decomposition.png")
    _plot_outlier_map(analysis_dir, figures_dir / "02_outlier_surface_map.png")
    _plot_heatmaps_by_metric(
        analysis_dir=analysis_dir,
        metric_col="abs_error_p90",
        title="Surface instability map: abs_error_p90",
        out_path=figures_dir / "03_surface_abs_error_p90_heatmap.png",
    )
    _plot_heatmaps_by_metric(
        analysis_dir=analysis_dir,
        metric_col="grad_norm_p90",
        title="Surface sensitivity map: grad_norm_p90",
        out_path=figures_dir / "04_surface_grad_norm_p90_heatmap.png",
    )
    _plot_greek_lifts(analysis_dir, figures_dir / "05_greek_lift_outliers_vs_ref.png")

    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
