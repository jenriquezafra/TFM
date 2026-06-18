from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
TABLE_SIZE = 13
ANNOTATION_SIZE = 10


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns are available: {candidates}")


def _prepare_pricing_frame(df: pd.DataFrame, *, mape_floor: float) -> pd.DataFrame:
    out = df.copy()
    ref_col = _first_existing(out, ("price_cos", "ref_price", "price_ref"))
    pred_col = _first_existing(out, ("price_pinn", "pinn_price", "pred_price", "price_pred"))
    out["price_ref_plot"] = out[ref_col].to_numpy(dtype=np.float64)
    out["price_pinn_plot"] = out[pred_col].to_numpy(dtype=np.float64)

    if "abs_error" in out.columns:
        abs_error = out["abs_error"].to_numpy(dtype=np.float64)
    elif "abs_error_price" in out.columns:
        abs_error = out["abs_error_price"].to_numpy(dtype=np.float64)
    elif "error" in out.columns:
        abs_error = np.abs(out["error"].to_numpy(dtype=np.float64))
    elif "error_price" in out.columns:
        abs_error = np.abs(out["error_price"].to_numpy(dtype=np.float64))
    else:
        abs_error = np.abs(out["price_pinn_plot"].to_numpy() - out["price_ref_plot"].to_numpy())
    out["abs_error_plot"] = abs_error

    if "rel_abs_error" in out.columns:
        rel_abs = out["rel_abs_error"].to_numpy(dtype=np.float64)
    elif "rel_abs_error_price" in out.columns:
        rel_abs = out["rel_abs_error_price"].to_numpy(dtype=np.float64)
    else:
        denom = np.maximum(np.abs(out["price_ref_plot"].to_numpy(dtype=np.float64)), float(mape_floor))
        rel_abs = abs_error / denom
    out["rel_abs_error_plot"] = rel_abs
    return out


def _load_or_compute_metrics(df: pd.DataFrame, metrics_yaml: Path | None) -> dict[str, float]:
    if metrics_yaml is not None and metrics_yaml.exists():
        with open(metrics_yaml, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if {"mse", "rmse", "mape_pct"}.issubset(loaded):
            return {
                "mse": float(loaded["mse"]),
                "rmse": float(loaded["rmse"]),
                "mape_pct": float(loaded["mape_pct"]),
            }

    abs_error = df["abs_error_plot"].to_numpy(dtype=np.float64)
    rel_abs = df["rel_abs_error_plot"].to_numpy(dtype=np.float64)
    mse = float(np.nanmean(abs_error**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape_pct": 100.0 * float(np.nanmean(rel_abs)),
    }


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


def _heatmap_norm(matrices: list[np.ndarray]) -> LogNorm | None:
    vals = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0.0)] for matrix in matrices])
    if vals.size == 0:
        return None
    vmin = max(float(np.nanpercentile(vals, 5.0)), float(np.finfo(np.float64).tiny))
    vmax = float(np.nanpercentile(vals, 95.0))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _format_log_tick(value: float, _pos: int) -> str:
    if value <= 0.0 or not np.isfinite(value):
        return ""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0**exponent)
    mantissa = int(round(mantissa))
    if mantissa == 1:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa}\times10^{{{exponent}}}$"


def _plot_heatmap(
    *,
    ax,
    matrix: np.ndarray,
    tau_edges: np.ndarray,
    m_edges: np.ndarray,
    title: str,
    norm: LogNorm | None,
    interpolation: str,
    colorbar_ticks: list[float] | None,
) -> None:
    matrix_plot = matrix.T
    if norm is not None:
        matrix_plot = np.where(np.isfinite(matrix_plot), np.maximum(matrix_plot, float(norm.vmin)), np.nan)

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(
        matrix_plot,
        origin="lower",
        aspect="auto",
        extent=[tau_edges[0], tau_edges[-1], m_edges[0], m_edges[-1]],
        cmap=cmap,
        norm=norm,
        interpolation=interpolation,
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.ax.yaxis.set_ticks_position("right")
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    cbar.ax.tick_params(labelsize=TICK_SIZE)


def _plot_histogram(ax, rel_err: np.ndarray) -> None:
    positive = rel_err[np.isfinite(rel_err) & (rel_err > 0.0)]
    if positive.size == 0:
        return
    lo = max(float(np.nanpercentile(positive, 0.5)), float(np.finfo(np.float64).tiny))
    hi = float(np.nanpercentile(positive, 99.5))
    if hi <= lo:
        hi = lo * 10.0
    bins = np.logspace(np.log10(lo), np.log10(hi), 38)
    ax.hist(positive, bins=bins, density=True, color="#2f5d8a", alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|e|_{\mathrm{rel}}$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
    ax.set_title("D. Relative-error histogram", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.18)
    stat_text = "\n".join(
        [
            f"median = {100.0 * float(np.median(rel_err)):.2f}%",
            f"p95 = {100.0 * float(np.percentile(rel_err, 95.0)):.2f}%",
            f"P(rel. |e| > 20%) = {100.0 * float(np.mean(rel_err > 0.20)):.2f}%",
        ]
    )
    ax.text(
        0.04,
        0.96,
        stat_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_SIZE,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#8a9aa8", "alpha": 0.95},
    )


def build_figure(
    *,
    points_csv: Path,
    metrics_yaml: Path | None,
    out_path: Path,
    n_bins: int,
    mape_floor: float,
    tail_panel: str,
    reference_symbol: str,
    model_name: str,
    scale_points_csv: list[Path],
    heatmap_interpolation: str,
    colorbar_ticks: list[float] | None,
) -> None:
    df = _prepare_pricing_frame(pd.read_csv(points_csv), mape_floor=mape_floor)
    metrics = _load_or_compute_metrics(df, metrics_yaml)

    rel_heat, tau_edges, m_edges = _build_heatmap(df=df, value_col="rel_abs_error_plot", n_bins=n_bins)
    norm_matrices = [rel_heat]
    for scale_csv in scale_points_csv:
        scale_df = _prepare_pricing_frame(pd.read_csv(scale_csv), mape_floor=mape_floor)
        scale_heat, _, _ = _build_heatmap(df=scale_df, value_col="rel_abs_error_plot", n_bins=n_bins)
        norm_matrices.append(scale_heat)
    rel_norm = _heatmap_norm(norm_matrices)

    fig = plt.figure(figsize=(12.0, 8.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[1.0, 1.0])

    ax_scatter = fig.add_subplot(outer[0, 0])
    y_ref = df["price_ref_plot"].to_numpy(dtype=np.float64)
    y_pred = df["price_pinn_plot"].to_numpy(dtype=np.float64)
    lo = float(np.nanmin([np.min(y_ref), np.min(y_pred)]))
    hi = float(np.nanmax([np.max(y_ref), np.max(y_pred)]))
    ax_scatter.scatter(y_ref, y_pred, s=8, alpha=0.35, edgecolors="none", color="#2f5d8a")
    ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black")
    ax_scatter.set_xlabel(rf"$V_{{\mathrm{{{reference_symbol}}}}}$", fontsize=LABEL_SIZE)
    ax_scatter.set_ylabel(r"$V_{\mathrm{PINN}}$", fontsize=LABEL_SIZE)
    ax_scatter.set_title(f"A. {model_name} price parity", fontsize=TITLE_SIZE)
    ax_scatter.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_scatter.grid(True, alpha=0.25)

    ax_table = fig.add_subplot(outer[0, 1])
    ax_table.axis("off")
    rel_abs = df["rel_abs_error_plot"].to_numpy(dtype=np.float64)
    table_rows = [
        ["Points", f"{len(df):,}"],
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
    table.scale(1.18, 1.82)
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
        norm=rel_norm,
        interpolation=heatmap_interpolation,
        colorbar_ticks=colorbar_ticks,
    )

    ax_cdf = fig.add_subplot(outer[1, 1])
    rel_err = np.sort(df["rel_abs_error_plot"].to_numpy(dtype=np.float64))
    if tail_panel == "hist":
        _plot_histogram(ax_cdf, rel_err)
    else:
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
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("thesis/figures/pinn/pricing/pricing_benchmark_composite.png"),
    )
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--mape-floor", type=float, default=1.0e-4)
    parser.add_argument("--tail-panel", choices=("cdf", "hist"), default="hist")
    parser.add_argument("--reference-symbol", default="COS")
    parser.add_argument("--model-name", default="PINN-COS")
    parser.add_argument("--scale-points-csv", type=Path, action="append", default=[])
    parser.add_argument("--heatmap-interpolation", default="bilinear")
    parser.add_argument("--colorbar-tick", type=float, action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(
        points_csv=_resolve(args.points_csv),
        metrics_yaml=_resolve(args.metrics_yaml) if args.metrics_yaml else None,
        out_path=_resolve(args.out),
        n_bins=int(args.n_bins),
        mape_floor=float(args.mape_floor),
        tail_panel=str(args.tail_panel),
        reference_symbol=str(args.reference_symbol),
        model_name=str(args.model_name),
        scale_points_csv=[_resolve(path) for path in args.scale_points_csv],
        heatmap_interpolation=str(args.heatmap_interpolation),
        colorbar_ticks=args.colorbar_tick,
    )
    print(f"Saved: {_resolve(args.out)}")


if __name__ == "__main__":
    main()
