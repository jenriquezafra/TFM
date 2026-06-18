from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GREEKS = ("delta", "gamma", "vega", "theta", "rho")
GREEK_LABELS = {
    "delta": "DELTA",
    "gamma": "GAMMA",
    "vega": "VEGA",
    "theta": "THETA",
    "rho": "RHO",
}
PINN_VARIANT_LABELS = ("Baseline PINN", "Sobolev PINN", "Sobolev + ACV")
MAPE_FLOOR = 1.0e-4
TITLE_SIZE = 20
LABEL_SIZE = 17
TICK_SIZE = 15


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


def _binned_grid(
    df: pd.DataFrame,
    value_col: str,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = df["tau"].to_numpy(dtype=np.float64)
    m = df["moneyness"].to_numpy(dtype=np.float64)
    values = df[value_col].to_numpy(dtype=np.float64)
    tau_edges = np.linspace(float(np.nanmin(tau)), float(np.nanmax(tau)), int(n_bins) + 1)
    m_edges = np.linspace(float(np.nanmin(m)), float(np.nanmax(m)), int(n_bins) + 1)
    tau_idx = np.clip(np.digitize(tau, bins=tau_edges, right=False) - 1, 0, int(n_bins) - 1)
    m_idx = np.clip(np.digitize(m, bins=m_edges, right=False) - 1, 0, int(n_bins) - 1)

    sums = np.zeros((int(n_bins), int(n_bins)), dtype=np.float64)
    counts = np.zeros((int(n_bins), int(n_bins)), dtype=np.int64)
    finite = np.isfinite(values)
    np.add.at(sums, (m_idx[finite], tau_idx[finite]), values[finite])
    np.add.at(counts, (m_idx[finite], tau_idx[finite]), 1)
    matrix = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    return matrix, tau_centers, m_centers


def _regular_or_binned_grid(
    df: pd.DataFrame,
    value_col: str,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tau = df["tau"].nunique()
    n_m = df["moneyness"].nunique()
    if n_tau * n_m == len(df):
        return _grid(df, value_col)
    return _binned_grid(df, value_col, n_bins=n_bins)


def _log_norm(matrices: list[np.ndarray], *, vmin: float | None = None, vmax: float | None = None) -> LogNorm:
    vals = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0.0)] for matrix in matrices])
    if vals.size == 0:
        return LogNorm(vmin=MAPE_FLOOR, vmax=1.0)
    lo = max(float(np.nanpercentile(vals, 2.0)), float(np.finfo(np.float64).tiny))
    hi = float(np.nanpercentile(vals, 98.0))
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    if hi <= lo:
        hi = lo * 10.0
    return LogNorm(vmin=lo, vmax=hi)


def _format_log_tick(value: float, _pos: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return ""
    exponent = np.log10(value)
    rounded = int(np.round(exponent))
    if np.isclose(exponent, rounded, atol=1.0e-8):
        return rf"$10^{{{rounded}}}$"
    return f"{value:.1e}"


def _apply_log_ticks(cbar, norm: LogNorm) -> None:
    lo = float(norm.vmin)
    hi = float(norm.vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        return
    exp_min = int(np.floor(np.log10(lo)))
    exp_max = int(np.ceil(np.log10(hi)))
    powers = [10.0**exp for exp in range(exp_min, exp_max + 1)]
    ticks = [lo] + [tick for tick in powers if lo < tick < hi] + [hi]
    unique_ticks: list[float] = []
    for tick in ticks:
        if not any(np.isclose(tick, existing, rtol=1.0e-6, atol=0.0) for existing in unique_ticks):
            unique_ticks.append(float(tick))
    cbar.set_ticks(unique_ticks)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))


def _abs_error_col(df: pd.DataFrame, quantity: str) -> pd.Series:
    candidates = (
        f"abs_error_{quantity}",
        f"{quantity}_abs_error",
        f"abs_{quantity}_error",
    )
    for col in candidates:
        if col in df.columns:
            return df[col].abs()
    error_candidates = (
        f"error_{quantity}",
        f"{quantity}_error",
        "error" if quantity == "price" else "",
    )
    for col in error_candidates:
        if col and col in df.columns:
            return df[col].abs()
    raise KeyError(f"Cannot infer absolute error column for {quantity}.")


def _rel_error_col(df: pd.DataFrame, quantity: str) -> pd.Series:
    candidates = (
        f"rel_abs_error_{quantity}",
        f"relative_abs_error_{quantity}",
        f"{quantity}_rel_abs_error",
    )
    for col in candidates:
        if col in df.columns:
            return df[col].abs()

    abs_error = _abs_error_col(df, quantity)
    ref_candidates = (
        f"ref_{quantity}",
        f"{quantity}_ref",
        f"{quantity}_heston_cf",
    )
    for col in ref_candidates:
        if col in df.columns:
            denom = np.maximum(df[col].abs().to_numpy(dtype=np.float64), MAPE_FLOOR)
            return pd.Series(abs_error.to_numpy(dtype=np.float64) / denom, index=df.index)
    raise KeyError(f"Cannot infer relative absolute error column for {quantity}.")


def _prepare_pinn_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_error_price"] = _abs_error_col(out, "price")
    for greek in GREEKS:
        out[f"abs_error_{greek}"] = _abs_error_col(out, greek)
        out[f"rel_abs_error_{greek}"] = _rel_error_col(out, greek)
    return out


def _plot_matrix(matrix: np.ndarray, norm: LogNorm) -> np.ndarray:
    return np.where(np.isfinite(matrix), np.maximum(matrix, norm.vmin), np.nan)


def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.22)


def _normalize_ann_iv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for greek in GREEKS:
        ref_col = f"{greek}_heston_cf"
        error_col = f"{greek}_error"
        if ref_col not in out.columns or error_col not in out.columns:
            raise KeyError(f"Missing ANN-IV columns for {greek}: {ref_col}, {error_col}")
        out[f"error_{greek}"] = out[error_col]
        out[f"abs_error_{greek}"] = out[error_col].abs()
        denom = np.maximum(out[ref_col].abs().to_numpy(dtype=np.float64), MAPE_FLOOR)
        out[f"rel_abs_error_{greek}"] = out[f"abs_error_{greek}"].to_numpy(dtype=np.float64) / denom
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


def save_ann_pair_map(*, baseline: pd.DataFrame, sobolev: pd.DataFrame, out_path: Path) -> None:
    baseline_grids = [_grid(baseline, f"rel_abs_error_{greek}") for greek in GREEKS]
    sobolev_grids = [_grid(sobolev, f"rel_abs_error_{greek}") for greek in GREEKS]
    norm = _log_norm([matrix for matrix, _, _ in baseline_grids + sobolev_grids])
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, axes = plt.subplots(len(GREEKS), 2, figsize=(10.8, 19.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.06, top=0.88, wspace=0.16, hspace=0.28)

    image = None
    columns = [("Baseline ANN", baseline_grids), ("Sobolev ANN", sobolev_grids)]
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
                interpolation="bicubic",
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
        cbar.set_label("relative absolute error", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_ann_pair_hist(
    *,
    baseline: pd.DataFrame,
    sobolev: pd.DataFrame,
    out_path: Path,
    log_y: bool,
) -> None:
    fig = plt.figure(figsize=(18.5, 10.2))
    axes = _five_panel_axes(fig)
    baseline_color = "#8F969E"
    sobolev_color = "#D95F02"

    for ax, greek in zip(axes, GREEKS):
        baseline_values = baseline[f"error_{greek}"].to_numpy(dtype=np.float64)
        sobolev_values = sobolev[f"error_{greek}"].to_numpy(dtype=np.float64)
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        sobolev_values = sobolev_values[np.isfinite(sobolev_values)]
        combined = np.concatenate([baseline_values, sobolev_values])
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
            label="Baseline ANN",
        )
        ax.hist(
            sobolev_values,
            bins=95,
            range=(lo, hi),
            density=True,
            alpha=0.78,
            color=sobolev_color,
            label="Sobolev ANN",
        )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks(np.linspace(lo, hi, 5))
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_title(GREEK_LABELS[greek], fontsize=TITLE_SIZE)
        ax.set_xlabel(r"$e$", fontsize=LABEL_SIZE)
        ax.set_ylabel("density", fontsize=LABEL_SIZE)
        _style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=LABEL_SIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_pricing_map(*, points: pd.DataFrame, out_path: Path, n_bins: int = 24) -> None:
    df = points.copy()
    if "abs_error_price" not in df.columns:
        if "abs_error" in df.columns:
            df["abs_error_price"] = df["abs_error"]
        elif "error_price" in df.columns:
            df["abs_error_price"] = df["error_price"].abs()
        elif "error" in df.columns:
            df["abs_error_price"] = df["error"].abs()
        else:
            raise KeyError("Cannot infer price absolute error column.")
    matrix, tau, m = _regular_or_binned_grid(df, "abs_error_price", n_bins=n_bins)
    norm = _log_norm([matrix])
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    image = ax.imshow(
        _plot_matrix(matrix, norm),
        origin="lower",
        aspect="auto",
        extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title("Baseline mix PINN pricing", fontsize=TITLE_SIZE)
    ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE)
    _style_axis(ax)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", location="top", fraction=0.08, pad=0.08)
    cbar.set_label("absolute error", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    _apply_log_ticks(cbar, norm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_pinn_pricing_triptych(*, variants: list[pd.DataFrame], out_path: Path) -> None:
    grids = [_regular_or_binned_grid(_prepare_pinn_errors(df), "abs_error_price", n_bins=181) for df in variants]
    norm = _log_norm([matrix for matrix, _, _ in grids])
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 12.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.07, top=0.87, hspace=0.22)

    image = None
    for ax, label, (matrix, tau, m) in zip(axes, PINN_VARIANT_LABELS, grids):
        image = ax.imshow(
            _plot_matrix(matrix, norm),
            origin="lower",
            aspect="auto",
            extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
        )
        ax.set_title(label, fontsize=TITLE_SIZE)
        ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE)
        _style_axis(ax)
    axes[-1].set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            location="top",
            fraction=0.04,
            pad=0.03,
        )
        cbar.set_label("price absolute error", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_pinn_pricing_pair(*, variants: list[pd.DataFrame], out_path: Path) -> None:
    labels = PINN_VARIANT_LABELS[:2]
    grids = [
        _regular_or_binned_grid(_prepare_pinn_errors(df), "abs_error_price", n_bins=181)
        for df in variants[:2]
    ]
    norm = _log_norm([matrix for matrix, _, _ in grids])
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 9.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.08, top=0.84, hspace=0.24)

    image = None
    for ax, label, (matrix, tau, m) in zip(axes, labels, grids):
        image = ax.imshow(
            _plot_matrix(matrix, norm),
            origin="lower",
            aspect="auto",
            extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
        )
        ax.set_title(label, fontsize=TITLE_SIZE)
        ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE)
        _style_axis(ax)
    axes[-1].set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            location="top",
            fraction=0.05,
            pad=0.035,
        )
        cbar.set_label("price absolute error", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_pinn_greeks_pair(*, variants: list[pd.DataFrame], out_path: Path) -> None:
    labels = PINN_VARIANT_LABELS[:2]
    prepared = [_prepare_pinn_errors(df) for df in variants[:2]]
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")
    all_grids = [
        [
            _regular_or_binned_grid(df, f"rel_abs_error_{greek}", n_bins=181)
            for df in prepared
        ]
        for greek in GREEKS
    ]
    norm = _log_norm([matrix for row in all_grids for matrix, _, _ in row])

    fig, axes = plt.subplots(len(GREEKS), 2, figsize=(10.8, 18.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.06, top=0.88, wspace=0.14, hspace=0.28)

    for col_idx, label in enumerate(labels):
        axes[0, col_idx].set_title(label, fontsize=TITLE_SIZE)

    image = None
    for row_idx, greek in enumerate(GREEKS):
        for col_idx, (matrix, tau, m) in enumerate(all_grids[row_idx]):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                _plot_matrix(matrix, norm),
                origin="lower",
                aspect="auto",
                extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
                cmap=cmap,
                norm=norm,
                interpolation="bicubic",
            )
            if col_idx == 0:
                ax.set_ylabel(f"{GREEK_LABELS[greek]}\n$m$", fontsize=LABEL_SIZE)
            if row_idx == len(GREEKS) - 1:
                ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
            _style_axis(ax)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            location="top",
            fraction=0.026,
            pad=0.025,
        )
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        cbar.set_label("relative absolute error", fontsize=LABEL_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def build_all(
    *,
    ann_baseline_csv: Path,
    ann_sobolev_csv: Path,
    pricing_csv: Path,
    pinn_baseline_csv: Path,
    pinn_sobolev_csv: Path,
    pinn_acv_csv: Path,
    ann_out_dir: Path,
    pricing_out_dir: Path,
    greeks_out_dir: Path,
) -> None:
    baseline = _normalize_ann_iv(pd.read_csv(ann_baseline_csv))
    sobolev = _normalize_ann_iv(pd.read_csv(ann_sobolev_csv))
    pricing = pd.read_csv(pricing_csv)
    pinn_variants = [
        pd.read_csv(pinn_baseline_csv),
        pd.read_csv(pinn_sobolev_csv),
        pd.read_csv(pinn_acv_csv),
    ]

    save_ann_pair_map(
        baseline=baseline,
        sobolev=sobolev,
        out_path=ann_out_dir / "ann_iv_baseline_vs_sobolev_rel_error_maps_10panel.png",
    )
    save_ann_pair_hist(
        baseline=baseline,
        sobolev=sobolev,
        out_path=ann_out_dir / "ann_iv_baseline_vs_sobolev_error_hist_overlay_5greeks.png",
        log_y=False,
    )
    save_ann_pair_hist(
        baseline=baseline,
        sobolev=sobolev,
        out_path=ann_out_dir / "ann_iv_baseline_vs_sobolev_error_hist_overlay_logy_5greeks.png",
        log_y=True,
    )
    save_pricing_map(
        points=pricing,
        out_path=pricing_out_dir / "baseline_mix_fixed_params_price_abs_error_map.png",
    )
    save_pinn_pricing_triptych(
        variants=pinn_variants,
        out_path=pricing_out_dir / "pinn_baseline_sobolev_acv_price_abs_error_maps_3row.png",
    )
    save_pinn_pricing_pair(
        variants=pinn_variants,
        out_path=pricing_out_dir / "pinn_baseline_sobolev_price_abs_error_maps_2row.png",
    )
    save_pinn_greeks_pair(
        variants=pinn_variants,
        out_path=greeks_out_dir / "pinn_baseline_sobolev_greek_rel_error_maps_10panel.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build requested thesis figures for ANN Sobolev and PINN pricing.")
    parser.add_argument(
        "--ann-baseline-csv",
        type=Path,
        default=Path("outputs/ann_iv_greeks/Liu_like_tanh_1M_v01_pinn_params_grid161/points_ann_iv_greeks.csv"),
    )
    parser.add_argument(
        "--ann-sobolev-csv",
        type=Path,
        default=Path("outputs/ann_iv_greeks/ANN_IV_Sobolev_v01_pinn_params_grid161/points_ann_iv_greeks.csv"),
    )
    parser.add_argument(
        "--pricing-csv",
        type=Path,
        default=Path("outputs/pinn/PINN_mix_scaled_fixed_theta/greeks/pricing_baseline_fixed_theta_final/points_baseline_diagnostics.csv"),
    )
    parser.add_argument(
        "--pinn-baseline-csv",
        type=Path,
        default=Path("outputs/pinn/PINN_mix_scaled_fixed_theta/greeks/pricing_baseline_fixed_theta_final/points_baseline_diagnostics.csv"),
    )
    parser.add_argument(
        "--pinn-sobolev-csv",
        type=Path,
        default=Path("outputs/pinn/PINN_mix_scaled_fixed_theta/greeks/pricing_sobolev_fixed_theta_final/points_baseline_diagnostics.csv"),
    )
    parser.add_argument(
        "--pinn-acv-csv",
        type=Path,
        default=Path(
            "outputs/pinn/acv_hard_patch_sobolev_control_variate_best_gate_tau_floor_5e4/"
            "diagnostics_fixed_theta_final/surface_diagnostics.csv"
        ),
    )
    parser.add_argument("--ann-out-dir", type=Path, default=Path("thesis/figures/nn"))
    parser.add_argument("--pricing-out-dir", type=Path, default=Path("thesis/figures/pinn/pricing"))
    parser.add_argument("--greeks-out-dir", type=Path, default=Path("thesis/figures/pinn/greeks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(
        ann_baseline_csv=_resolve(args.ann_baseline_csv),
        ann_sobolev_csv=_resolve(args.ann_sobolev_csv),
        pricing_csv=_resolve(args.pricing_csv),
        pinn_baseline_csv=_resolve(args.pinn_baseline_csv),
        pinn_sobolev_csv=_resolve(args.pinn_sobolev_csv),
        pinn_acv_csv=_resolve(args.pinn_acv_csv),
        ann_out_dir=_resolve(args.ann_out_dir),
        pricing_out_dir=_resolve(args.pricing_out_dir),
        greeks_out_dir=_resolve(args.greeks_out_dir),
    )
    print(f"Saved ANN figures to: {_resolve(args.ann_out_dir)}")
    print(f"Saved pricing figures to: {_resolve(args.pricing_out_dir)}")
    print(f"Saved PINN Greek figures to: {_resolve(args.greeks_out_dir)}")


if __name__ == "__main__":
    main()
