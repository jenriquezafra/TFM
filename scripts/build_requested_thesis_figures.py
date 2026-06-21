from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GREEKS = ("delta", "gamma", "vega", "theta", "rho")
PINN_VALUE_ROWS = ("price", "delta", "gamma", "vega", "theta", "rho")
GREEK_LABELS = {
    "price": "PRICE",
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
    floor_exp = int(np.floor(exponent))
    coeff = value / (10.0**floor_exp)
    return rf"${coeff:.1f}\times10^{{{floor_exp}}}$"


def _apply_log_ticks(cbar, norm: LogNorm) -> None:
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


def _value_col(df: pd.DataFrame, quantity: str) -> pd.Series:
    candidates = (
        f"pinn_{quantity}",
        f"{quantity}_pred",
        f"pred_{quantity}",
        quantity,
    )
    for col in candidates:
        if col in df.columns:
            return df[col]
    raise KeyError(f"Cannot infer value column for {quantity}.")


def _prepare_pinn_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for quantity in PINN_VALUE_ROWS:
        out[f"value_{quantity}"] = _value_col(out, quantity)
    return out


def _ann_value_col(df: pd.DataFrame, quantity: str) -> pd.Series:
    candidates = (
        f"{quantity}_ann_iv",
        f"ann_{quantity}",
        quantity,
    )
    for col in candidates:
        if col in df.columns:
            return df[col]
    raise KeyError(f"Cannot infer ANN-IV value column for {quantity}.")


def _prepare_ann_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for quantity in PINN_VALUE_ROWS:
        out[f"value_{quantity}"] = _ann_value_col(out, quantity)
    return out


def _plot_matrix(matrix: np.ndarray, norm: LogNorm) -> np.ndarray:
    return np.where(np.isfinite(matrix), np.maximum(matrix, norm.vmin), np.nan)


def _linear_limits(matrices: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(vals, 1.0))
    hi = float(np.nanpercentile(vals, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0e-6)
        return lo - pad, hi + pad
    return lo, hi


def _value_cmap_norm(quantity: str, matrices: list[np.ndarray]):
    vals = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    if vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(np.abs(vals), 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f2f2f2")
    return cmap, TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), None, None


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
        cbar.set_label("Relative Absolute Error", fontsize=LABEL_SIZE)
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
        ax.set_xlabel("error (ANN - ref.)", fontsize=LABEL_SIZE)
        ax.set_ylabel("density", fontsize=LABEL_SIZE)
        _style_hist_axis(ax)

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
        cbar.set_label("Relative Absolute Error", fontsize=LABEL_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_pinn_price_greek_value_pair(*, variants: list[pd.DataFrame], out_path: Path) -> None:
    labels = PINN_VARIANT_LABELS[:2]
    prepared = [_prepare_pinn_values(df) for df in variants[:2]]
    all_grids = [
        [
            _regular_or_binned_grid(df, f"value_{quantity}", n_bins=181)
            for df in prepared
        ]
        for quantity in PINN_VALUE_ROWS
    ]

    fig, axes = plt.subplots(
        len(PINN_VALUE_ROWS),
        2,
        figsize=(11.2, 21.5),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.055, top=0.94, wspace=0.14, hspace=0.30)

    for col_idx, label in enumerate(labels):
        axes[0, col_idx].set_title(label, fontsize=TITLE_SIZE)

    for row_idx, quantity in enumerate(PINN_VALUE_ROWS):
        row_grids = all_grids[row_idx]
        row_matrices = [matrix for matrix, _, _ in row_grids]
        cmap, norm, vmin, vmax = _value_cmap_norm(quantity, row_matrices)

        image = None
        for col_idx, (matrix, tau, m) in enumerate(row_grids):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
            )
            if col_idx == 0:
                ax.set_ylabel(f"{GREEK_LABELS[quantity]}\n$m$", fontsize=LABEL_SIZE)
            if row_idx == len(PINN_VALUE_ROWS) - 1:
                ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
            _style_axis(ax)

        if image is not None:
            cbar = fig.colorbar(
                image,
                ax=axes[row_idx, :].tolist(),
                orientation="vertical",
                fraction=0.045,
                pad=0.025,
            )
            cbar.ax.tick_params(labelsize=max(TICK_SIZE - 2, 10))
            cbar.set_label("value", fontsize=max(LABEL_SIZE - 2, 10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_pinn_price_greek_value_pair_landscape(*, variants: list[pd.DataFrame], out_path: Path) -> None:
    labels = PINN_VARIANT_LABELS[:2]
    prepared = [_prepare_pinn_values(df) for df in variants[:2]]
    all_grids = [
        [
            _regular_or_binned_grid(df, f"value_{quantity}", n_bins=181)
            for quantity in PINN_VALUE_ROWS
        ]
        for df in prepared
    ]
    fig, axes = plt.subplots(
        2,
        len(PINN_VALUE_ROWS),
        figsize=(22.0, 7.6),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.78, wspace=0.18, hspace=0.18)

    for col_idx, quantity in enumerate(PINN_VALUE_ROWS):
        axes[0, col_idx].set_title(GREEK_LABELS[quantity], fontsize=TITLE_SIZE)

        matrices = [all_grids[row_idx][col_idx][0] for row_idx in range(2)]
        cmap, norm, vmin, vmax = _value_cmap_norm(quantity, matrices)

        image = None
        for row_idx, label in enumerate(labels):
            matrix, tau, m = all_grids[row_idx][col_idx]
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=[float(tau.min()), float(tau.max()), float(m.min()), float(m.max())],
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
            )
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n$m$", fontsize=LABEL_SIZE)
            if row_idx == 1:
                ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE)
            _style_axis(ax)

        if image is not None:
            cbar = fig.colorbar(
                image,
                ax=axes[:, col_idx].tolist(),
                orientation="horizontal",
                location="top",
                fraction=0.085,
                pad=0.12,
            )
            cbar.ax.tick_params(labelsize=max(TICK_SIZE - 3, 10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_ann_sobolev_value_surfaces_3d(*, points: pd.DataFrame, out_path: Path) -> None:
    prepared = _prepare_ann_values(points)
    grids = [
        (quantity, _regular_or_binned_grid(prepared, f"value_{quantity}", n_bins=161))
        for quantity in PINN_VALUE_ROWS
    ]

    fig = plt.figure(figsize=(18.5, 10.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.93, wspace=0.05, hspace=0.16)
    cmap = plt.get_cmap("RdBu_r")

    for idx, (quantity, (matrix, tau, m)) in enumerate(grids, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        tau_grid, m_grid = np.meshgrid(tau, m)

        vals = matrix[np.isfinite(matrix)]
        if vals.size:
            z_abs = float(np.nanpercentile(np.abs(vals), 99.0))
            if not np.isfinite(z_abs) or z_abs <= 0.0:
                z_abs = max(float(np.nanmax(np.abs(vals))), 1.0)
        else:
            z_abs = 1.0
        norm = TwoSlopeNorm(vmin=-z_abs, vcenter=0.0, vmax=z_abs)

        ax.plot_surface(
            tau_grid,
            m_grid,
            matrix,
            cmap=cmap,
            norm=norm,
            linewidth=0.0,
            antialiased=True,
            rcount=90,
            ccount=90,
            shade=True,
        )
        ax.set_title(GREEK_LABELS[quantity], fontsize=TITLE_SIZE, pad=10)
        ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_zlabel("value", fontsize=LABEL_SIZE, labelpad=8)
        ax.tick_params(axis="both", labelsize=max(TICK_SIZE - 3, 9), pad=2)
        ax.zaxis.set_tick_params(labelsize=max(TICK_SIZE - 4, 8), pad=2)
        ax.view_init(elev=26, azim=-132)
        ax.set_box_aspect((1.55, 1.0, 0.62))
        ax.grid(True, alpha=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pinn_sobolev_acv_kink_value_surfaces_3d(*, points: pd.DataFrame, out_path: Path) -> None:
    prepared = _prepare_pinn_values(points)
    grids = [
        (quantity, _regular_or_binned_grid(prepared, f"value_{quantity}", n_bins=121))
        for quantity in PINN_VALUE_ROWS
    ]

    fig = plt.figure(figsize=(18.5, 10.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.93, wspace=0.05, hspace=0.16)
    cmap = plt.get_cmap("RdBu_r")

    for idx, (quantity, (matrix, tau, m)) in enumerate(grids, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        tau_grid, m_grid = np.meshgrid(tau, m)

        vals = matrix[np.isfinite(matrix)]
        if vals.size:
            z_abs = float(np.nanpercentile(np.abs(vals), 99.0))
            if not np.isfinite(z_abs) or z_abs <= 0.0:
                z_abs = max(float(np.nanmax(np.abs(vals))), 1.0)
        else:
            z_abs = 1.0
        norm = TwoSlopeNorm(vmin=-z_abs, vcenter=0.0, vmax=z_abs)

        ax.plot_surface(
            tau_grid,
            m_grid,
            matrix,
            cmap=cmap,
            norm=norm,
            linewidth=0.0,
            antialiased=True,
            rcount=90,
            ccount=90,
            shade=True,
        )
        ax.set_title(GREEK_LABELS[quantity], fontsize=TITLE_SIZE, pad=10)
        ax.set_xlabel(r"$\tau$", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_ylabel(r"$m$", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_zlabel("value", fontsize=LABEL_SIZE, labelpad=8)
        ax.tick_params(axis="both", labelsize=max(TICK_SIZE - 3, 9), pad=2)
        ax.zaxis.set_tick_params(labelsize=max(TICK_SIZE - 4, 8), pad=2)
        ax.view_init(elev=28, azim=-136)
        ax.set_box_aspect((1.35, 1.0, 0.70))
        ax.grid(True, alpha=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    save_pinn_price_greek_value_pair(
        variants=pinn_variants,
        out_path=greeks_out_dir / "pinn_baseline_sobolev_price_greek_value_maps_12panel.png",
    )
    save_pinn_price_greek_value_pair_landscape(
        variants=pinn_variants,
        out_path=greeks_out_dir / "pinn_baseline_sobolev_price_greek_value_maps_presentation.png",
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
