from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
TABLE_SIZE = 11
ANNOTATION_SIZE = 10


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def _build_heatmap(
    *,
    df: pd.DataFrame,
    value_col: str,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = df["tau"].to_numpy(dtype=np.float64)
    moneyness = df["moneyness"].to_numpy(dtype=np.float64)
    values = df[value_col].to_numpy(dtype=np.float64)

    tau_edges = np.linspace(float(np.min(tau)), float(np.max(tau)), int(n_bins) + 1)
    m_edges = np.linspace(float(np.min(moneyness)), float(np.max(moneyness)), int(n_bins) + 1)
    tau_idx = np.clip(np.digitize(tau, bins=tau_edges, right=False) - 1, 0, n_bins - 1)
    m_idx = np.clip(np.digitize(moneyness, bins=m_edges, right=False) - 1, 0, n_bins - 1)

    sums = np.zeros((n_bins, n_bins), dtype=np.float64)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(sums, (tau_idx, m_idx), values)
    np.add.at(counts, (tau_idx, m_idx), 1)
    mean_values = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return mean_values, tau_edges, m_edges


def _plot_heatmap(
    *,
    ax,
    matrix: np.ndarray,
    tau_edges: np.ndarray,
    m_edges: np.ndarray,
    title: str,
    cbar_label: str,
) -> None:
    valid = np.isfinite(matrix) & (matrix > 0.0)
    norm = None
    matrix_plot = matrix
    if np.any(valid):
        vals = matrix[valid]
        vmin = max(float(np.nanpercentile(vals, 5.0)), float(np.finfo(np.float64).tiny))
        vmax = float(np.nanpercentile(vals, 95.0))
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
        matrix_plot = np.where(np.isfinite(matrix), np.maximum(matrix, vmin), np.nan)

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(
        matrix_plot,
        origin="lower",
        aspect="auto",
        extent=[m_edges[0], m_edges[-1], tau_edges[0], tau_edges[-1]],
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(r"$m$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\tau$", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)


def build_figure(*, points_csv: Path, metrics_yaml: Path, out_path: Path, n_bins: int) -> None:
    df = pd.read_csv(points_csv)
    with open(metrics_yaml, "r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f) or {}

    rel_heat, tau_edges, m_edges = _build_heatmap(df=df, value_col="rel_abs_error", n_bins=n_bins)

    fig = plt.figure(figsize=(12.0, 8.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[1.0, 1.0])

    ax_scatter = fig.add_subplot(outer[0, 0])
    y_ref = df["price_cos"].to_numpy(dtype=np.float64)
    y_pred = df["price_pinn"].to_numpy(dtype=np.float64)
    lo = float(np.nanmin([np.min(y_ref), np.min(y_pred)]))
    hi = float(np.nanmax([np.max(y_ref), np.max(y_pred)]))
    ax_scatter.scatter(y_ref, y_pred, s=8, alpha=0.35, edgecolors="none", color="#2f5d8a")
    ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black")
    ax_scatter.set_xlabel(r"$V_{\mathrm{COS}}$", fontsize=LABEL_SIZE)
    ax_scatter.set_ylabel(r"$V_{\mathrm{PINN}}$", fontsize=LABEL_SIZE)
    ax_scatter.set_title("A. PINN-COS price parity", fontsize=TITLE_SIZE)
    ax_scatter.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_scatter.grid(True, alpha=0.25)

    ax_table = fig.add_subplot(outer[0, 1])
    ax_table.axis("off")
    rel_abs = df["rel_abs_error"].to_numpy(dtype=np.float64)
    table_rows = [
        ["Points", "10,000"],
        ["MSE", _fmt_sci(float(metrics["mse"]))],
        ["RMSE", _fmt_sci(float(metrics["rmse"]))],
        ["MAPE", f"{float(metrics['mape_pct']):.2f}%"],
        [r"Median rel. $|e|$", f"{100.0 * float(np.median(rel_abs)):.2f}%"],
        [r"p95 rel. $|e|$", f"{100.0 * float(np.percentile(rel_abs, 95.0)):.2f}%"],
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_SIZE)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e6f2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f7f9fb" if row % 2 == 0 else "white")
        cell.set_edgecolor("#8a9aa8")
    ax_table.set_title("B. Global pricing metrics", pad=14, fontsize=TITLE_SIZE)

    ax_rel = fig.add_subplot(outer[1, 0])
    _plot_heatmap(
        ax=ax_rel,
        matrix=rel_heat,
        tau_edges=tau_edges,
        m_edges=m_edges,
        title="C. Relative absolute-error map",
        cbar_label=r"mean $|e|_{\mathrm{rel}}$",
    )

    ax_cdf = fig.add_subplot(outer[1, 1])
    rel_err = np.sort(df["rel_abs_error"].to_numpy(dtype=np.float64))
    cdf = np.linspace(0.0, 1.0, rel_err.size, endpoint=False)
    ax_cdf.plot(rel_err, cdf, linewidth=1.5, color="#2f5d8a")
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel(r"$|e|_{\mathrm{rel}}$", fontsize=LABEL_SIZE)
    ax_cdf.set_ylabel("Empirical CDF", fontsize=LABEL_SIZE)
    ax_cdf.set_title("D. Relative-error distribution", fontsize=TITLE_SIZE)
    ax_cdf.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_cdf.grid(True, which="major", alpha=0.3)
    ax_cdf.grid(True, which="minor", alpha=0.18)
    stat_text = "\n".join(
        [
            f"median = {100.0 * float(np.median(rel_err)):.2f}%",
            f"p95 = {100.0 * float(np.percentile(rel_err, 95.0)):.2f}%",
            f"P(rel. |e| > 20%) = {100.0 * float(np.mean(rel_err > 0.20)):.2f}%",
        ]
    )
    ax_cdf.text(
        0.04,
        0.96,
        stat_text,
        transform=ax_cdf.transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_SIZE,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#8a9aa8", "alpha": 0.95},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the composite PINN pricing benchmark figure.")
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=Path("outputs/pinn/PINN_mix_scaled/cos_benchmark/points_pinn_vs_cos.csv"),
    )
    parser.add_argument(
        "--metrics-yaml",
        type=Path,
        default=Path("outputs/pinn/PINN_mix_scaled/cos_benchmark/metrics.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("thesis/figures/pinn/pricing/pricing_benchmark_composite.png"),
    )
    parser.add_argument("--n-bins", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(
        points_csv=_resolve(args.points_csv),
        metrics_yaml=_resolve(args.metrics_yaml),
        out_path=_resolve(args.out),
        n_bins=int(args.n_bins),
    )
    print(f"Saved: {_resolve(args.out)}")


if __name__ == "__main__":
    main()
