from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GREEKS = ("delta", "gamma", "vega", "theta", "rho")
GREEK_LABELS = {
    "delta": "DELTA",
    "gamma": "GAMMA",
    "vega": "VEGA",
    "theta": "THETA",
    "rho": "RHO",
}
MAPE_FLOOR = 1.0e-4
TITLE_SIZE = 20
LABEL_SIZE = 17
TICK_SIZE = 15
HIST_TICK_SIZE = 18


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(index="moneyness", columns="tau", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return (
        pivot.to_numpy(dtype=np.float64),
        pivot.columns.to_numpy(dtype=np.float64),
        pivot.index.to_numpy(dtype=np.float64),
    )


def _log_norm(matrices: list[np.ndarray]) -> LogNorm | None:
    vals = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0.0)] for matrix in matrices])
    if vals.size == 0:
        return None
    vmin = max(float(np.nanpercentile(vals, 2.0)), float(np.finfo(np.float64).tiny))
    vmax = float(np.nanpercentile(vals, 98.0))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _format_log_tick(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return ""
    exponent = np.log10(value)
    rounded = int(np.round(exponent))
    if np.isclose(exponent, rounded, atol=1.0e-8):
        return rf"$10^{{{rounded}}}$"
    floor_exp = int(np.floor(exponent))
    coeff = value / (10.0**floor_exp)
    return rf"${coeff:.1f}\times10^{{{floor_exp}}}$"


def _apply_log_ticks(cbar, norm: LogNorm | None) -> None:
    if norm is None:
        return
    lo = float(norm.vmin)
    hi = float(norm.vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        return
    exp_min = int(np.floor(np.log10(lo)))
    exp_max = int(np.ceil(np.log10(hi)))
    powers = [10.0**exp for exp in range(exp_min, exp_max + 1)]
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    min_edge_spacing = 0.35
    interior = [
        tick
        for tick in powers
        if lo < tick < hi
        and np.log10(tick) - log_lo >= min_edge_spacing
        and log_hi - np.log10(tick) >= min_edge_spacing
    ]
    ticks = [lo] + interior + [hi]
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))


def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.22)


def _style_hist_axis(ax) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=HIST_TICK_SIZE)
    ax.xaxis.get_offset_text().set_size(HIST_TICK_SIZE)
    ax.yaxis.get_offset_text().set_size(HIST_TICK_SIZE)
    ax.grid(True, alpha=0.22)


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _normalize_greek_columns(df: pd.DataFrame, *, mape_floor: float = MAPE_FLOOR) -> pd.DataFrame:
    out = df.copy()
    for greek in GREEKS:
        error_col = f"error_{greek}"
        abs_error_col = f"abs_error_{greek}"
        rel_abs_error_col = f"rel_abs_error_{greek}"
        ref_col = _first_existing(out, (f"ref_{greek}", f"{greek}_ref"))
        pred_col = _first_existing(out, (f"pinn_{greek}", f"{greek}_pred"))
        alt_error_col = _first_existing(out, (f"{greek}_error",))

        if error_col not in out.columns:
            if alt_error_col is not None:
                out[error_col] = out[alt_error_col]
            elif pred_col is not None and ref_col is not None:
                out[error_col] = out[pred_col] - out[ref_col]
            else:
                raise KeyError(f"Cannot infer {error_col}; missing prediction/reference columns for {greek}.")

        if abs_error_col not in out.columns:
            out[abs_error_col] = out[error_col].abs()

        if rel_abs_error_col not in out.columns:
            if ref_col is None:
                raise KeyError(f"Cannot infer {rel_abs_error_col}; missing reference column for {greek}.")
            denominator = np.maximum(out[ref_col].abs().to_numpy(dtype=np.float64), float(mape_floor))
            out[rel_abs_error_col] = out[abs_error_col].to_numpy(dtype=np.float64) / denominator

    return out


def _five_panel_axes(fig: plt.Figure, *, right: float = 0.98) -> list[plt.Axes]:
    grid = fig.add_gridspec(
        2,
        6,
        left=0.06,
        right=right,
        bottom=0.08,
        top=0.92,
        wspace=0.56,
        hspace=0.52,
    )
    positions = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (1, slice(1, 3)),
        (1, slice(3, 5)),
    ]
    return [fig.add_subplot(grid[row, cols]) for row, cols in positions]


def _save_map_composite(*, df: pd.DataFrame, value_prefix: str, out_path: Path) -> None:
    grids = [_grid(df, f"{value_prefix}_{greek}") for greek in GREEKS]
    matrices = [matrix for matrix, _, _ in grids]
    norm = _log_norm(matrices)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f2f2f2")

    fig = plt.figure(figsize=(15.8, 9.2))
    axes = _five_panel_axes(fig, right=0.88)
    image = None
    for ax, greek, (matrix, tau, m) in zip(axes, GREEKS, grids):
        plot_matrix = matrix
        if norm is not None:
            plot_matrix = np.where(np.isfinite(matrix), np.maximum(matrix, norm.vmin), np.nan)
        image = ax.imshow(
            plot_matrix,
            origin="lower",
            aspect="auto",
            extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(GREEK_LABELS[greek], fontsize=TITLE_SIZE)
        ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE)
        _style_axis(ax)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _save_hist_composite(*, df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(15.8, 9.2))
    axes = _five_panel_axes(fig)
    for ax, greek in zip(axes, GREEKS):
        values = df[f"error_{greek}"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        lo, hi = np.nanpercentile(values, [0.2, 99.8])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        ax.hist(values, bins=90, range=(lo, hi), density=True, alpha=0.86, color="#2F5D8A")
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_title(GREEK_LABELS[greek], fontsize=TITLE_SIZE)
        ax.set_xlabel("error (PINN - ref.)", fontsize=LABEL_SIZE)
        ax.set_ylabel("density", fontsize=LABEL_SIZE)
        _style_hist_axis(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _plot_matrix(matrix: np.ndarray, norm: LogNorm | None) -> np.ndarray:
    if norm is None:
        return matrix
    return np.where(np.isfinite(matrix), np.maximum(matrix, norm.vmin), np.nan)


def _save_pair_map_composite(
    *,
    baseline: pd.DataFrame,
    improved: pd.DataFrame,
    improved_label: str,
    value_prefix: str,
    out_path: Path,
) -> None:
    baseline_grids = [_grid(baseline, f"{value_prefix}_{greek}") for greek in GREEKS]
    improved_grids = [_grid(improved, f"{value_prefix}_{greek}") for greek in GREEKS]
    matrices = [matrix for matrix, _, _ in baseline_grids + improved_grids]
    norm = _log_norm(matrices)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, axes = plt.subplots(len(GREEKS), 2, figsize=(10.8, 19.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.06, top=0.88, wspace=0.16, hspace=0.28)

    image = None
    columns = [("Baseline PINN", baseline_grids), (improved_label, improved_grids)]
    for col_idx, (col_label, grids) in enumerate(columns):
        axes[0, col_idx].set_title(col_label, fontsize=TITLE_SIZE)
        for row_idx, (greek, (matrix, tau, m)) in enumerate(zip(GREEKS, grids)):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                _plot_matrix(matrix, norm),
                origin="lower",
                aspect="auto",
                extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            if row_idx == len(GREEKS) - 1:
                ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
            if col_idx == 0:
                ax.set_ylabel(f"{GREEK_LABELS[greek]}\n$m$", fontsize=LABEL_SIZE)
            _style_axis(ax)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            location="top",
            fraction=0.025,
            pad=0.025,
        )
        cbar.set_label("Relative Absolute Error", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _save_pair_hist_composite(
    *,
    baseline: pd.DataFrame,
    improved: pd.DataFrame,
    improved_label: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    fig = plt.figure(figsize=(18.5, 10.2))
    axes = _five_panel_axes(fig, right=0.98)
    baseline_color = "#8F969E"
    improved_color = "#D95F02"

    for ax, greek in zip(axes, GREEKS):
        baseline_values = baseline[f"error_{greek}"].to_numpy(dtype=np.float64)
        improved_values = improved[f"error_{greek}"].to_numpy(dtype=np.float64)
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        improved_values = improved_values[np.isfinite(improved_values)]
        combined = np.concatenate([baseline_values, improved_values])
        lo, hi = np.nanpercentile(combined, [0.2, 99.8])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
        limit = max(abs(float(lo)), abs(float(hi)))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        lo, hi = -limit, limit

        ax.hist(
            baseline_values,
            bins=95,
            range=(lo, hi),
            density=True,
            alpha=0.50,
            color=baseline_color,
            label="Baseline PINN",
        )
        ax.hist(
            improved_values,
            bins=95,
            range=(lo, hi),
            density=True,
            alpha=0.78,
            color=improved_color,
            label=improved_label,
        )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks(np.linspace(lo, hi, 5))
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_title(GREEK_LABELS[greek], fontsize=TITLE_SIZE)
        ax.set_xlabel("error (PINN - ref.)", fontsize=LABEL_SIZE)
        ax.set_ylabel("density", fontsize=LABEL_SIZE)
        _style_hist_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=LABEL_SIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def build_all(
    *,
    baseline_csv: Path,
    sobolev_csv: Path,
    out_dir: Path,
    acv_csv: Path | None = None,
    include_individual: bool = True,
    include_sobolev_comparison: bool = False,
) -> None:
    baseline = _normalize_greek_columns(pd.read_csv(baseline_csv))
    sobolev = _normalize_greek_columns(pd.read_csv(sobolev_csv))
    if include_individual:
        _save_map_composite(
            df=baseline,
            value_prefix="rel_abs_error",
            out_path=out_dir / "no_sobolev_rel_abs_error_maps_5greeks.png",
        )
        _save_hist_composite(
            df=baseline,
            out_path=out_dir / "no_sobolev_error_hist_5greeks.png",
        )
        _save_map_composite(
            df=sobolev,
            value_prefix="rel_abs_error",
            out_path=out_dir / "sobolev_rel_abs_error_maps_5greeks.png",
        )
        _save_hist_composite(
            df=sobolev,
            out_path=out_dir / "sobolev_error_hist_5greeks.png",
        )

    if include_sobolev_comparison:
        _save_pair_map_composite(
            baseline=baseline,
            improved=sobolev,
            improved_label="Sobolev PINN",
            value_prefix="rel_abs_error",
            out_path=out_dir / "baseline_vs_sobolev_rel_abs_error_maps_10panel.png",
        )
        _save_pair_hist_composite(
            baseline=baseline,
            improved=sobolev,
            improved_label="Sobolev PINN",
            out_path=out_dir / "baseline_vs_sobolev_error_hist_overlay_5greeks.png",
        )
        _save_pair_hist_composite(
            baseline=baseline,
            improved=sobolev,
            improved_label="Sobolev PINN",
            out_path=out_dir / "baseline_vs_sobolev_error_hist_overlay_logy_5greeks.png",
            log_y=True,
        )

    if include_individual and acv_csv is not None:
        acv = _normalize_greek_columns(pd.read_csv(acv_csv))
        _save_map_composite(
            df=acv,
            value_prefix="rel_abs_error",
            out_path=out_dir / "acv_rel_abs_error_maps_5greeks.png",
        )
        _save_hist_composite(
            df=acv,
            out_path=out_dir / "acv_error_hist_5greeks.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build composite PINN Greek diagnostic figures.")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs/pinn/PINN_mix_scaled_param/greeks/benchmark_cf/points_pinn_vs_heston_cf_greeks.csv"),
    )
    parser.add_argument(
        "--sobolev-csv",
        type=Path,
        default=Path(
            "outputs/pinn/PINN_mix_scaled_param/greeks/"
            "benchmark_cf_sobolev_best_PINN_mix_scaled_param_20260402_224714/"
            "points_pinn_vs_heston_cf_greeks.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("thesis/figures/pinn/greeks"))
    parser.add_argument(
        "--acv-csv",
        type=Path,
        default=None,
        help="Optional ACV surface diagnostics CSV with columns like delta_pred, delta_ref, delta_error.",
    )
    parser.add_argument(
        "--sobolev-comparison",
        action="store_true",
        help="Also build two baseline-vs-Sobolev comparison figures.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only build the two baseline-vs-Sobolev comparison figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(
        baseline_csv=_resolve(args.baseline_csv),
        sobolev_csv=_resolve(args.sobolev_csv),
        out_dir=_resolve(args.out_dir),
        acv_csv=_resolve(args.acv_csv) if args.acv_csv is not None else None,
        include_individual=not bool(args.comparison_only),
        include_sobolev_comparison=bool(args.sobolev_comparison or args.comparison_only),
    )
    print(f"Saved Greek composites to: {_resolve(args.out_dir)}")


if __name__ == "__main__":
    main()
