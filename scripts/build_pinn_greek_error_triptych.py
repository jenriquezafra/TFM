from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter


GREEKS = ("delta", "gamma", "vega", "theta", "rho")
GREEK_LABELS = {
    "delta": "DELTA",
    "gamma": "GAMMA",
    "vega": "VEGA",
    "theta": "THETA",
    "rho": "RHO",
}
MODEL_LABELS = ("Baseline PINN", "Sobolev PINN", "Sobolev + ACV")
MAPE_FLOOR = 1.0e-4
TITLE_SIZE = 20
LABEL_SIZE = 17
TICK_SIZE = 15


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _normalize_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for greek in GREEKS:
        abs_col = f"abs_error_{greek}"
        err_col = f"error_{greek}"
        ref_col = _first_existing(out, (f"ref_{greek}", f"{greek}_ref"))
        pred_col = _first_existing(out, (f"pinn_{greek}", f"{greek}_pred"))
        alt_err_col = _first_existing(out, (f"{greek}_error",))

        if abs_col not in out.columns:
            if err_col in out.columns:
                out[abs_col] = out[err_col].abs()
            elif alt_err_col is not None:
                out[err_col] = out[alt_err_col]
                out[abs_col] = out[err_col].abs()
            elif pred_col is not None and ref_col is not None:
                out[err_col] = out[pred_col] - out[ref_col]
                out[abs_col] = out[err_col].abs()
            else:
                raise KeyError(f"Cannot infer absolute error columns for {greek}.")

        if ref_col is None:
            raise KeyError(f"Cannot infer relative error for {greek}; missing reference column.")
        denom = np.maximum(out[ref_col].abs().to_numpy(dtype=np.float64), MAPE_FLOOR)
        out[f"rel_abs_error_{greek}"] = out[abs_col].to_numpy(dtype=np.float64) / denom

    return out


def _grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(index="moneyness", columns="tau", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return (
        pivot.to_numpy(dtype=np.float64),
        pivot.columns.to_numpy(dtype=np.float64),
        pivot.index.to_numpy(dtype=np.float64),
    )


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
        log_tick = float(np.log10(tick))
        too_close = any(abs(log_tick - float(np.log10(existing))) < 0.14 for existing in unique_ticks)
        if not too_close and not any(np.isclose(tick, existing, rtol=1.0e-6, atol=0.0) for existing in unique_ticks):
            unique_ticks.append(float(tick))
    cbar.set_ticks(unique_ticks)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))


def _plot_matrix(matrix: np.ndarray, norm: LogNorm) -> np.ndarray:
    return np.where(np.isfinite(matrix), np.maximum(matrix, norm.vmin), np.nan)


def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.22)


def save_triptych(*, frames: list[pd.DataFrame], value_prefix: str, cbar_label: str, out_path: Path) -> None:
    grids_by_greek = [
        [_grid(df, f"{value_prefix}_{greek}") for df in frames]
        for greek in GREEKS
    ]
    norm = _log_norm([matrix for row in grids_by_greek for matrix, _, _ in row])
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, axes = plt.subplots(len(GREEKS), len(frames), figsize=(15.8, 18.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.06, top=0.88, wspace=0.14, hspace=0.28)

    for col_idx, label in enumerate(MODEL_LABELS[: len(frames)]):
        axes[0, col_idx].set_title(label, fontsize=TITLE_SIZE)

    image = None
    for row_idx, greek in enumerate(GREEKS):
        for col_idx, (matrix, tau, m) in enumerate(grids_by_greek[row_idx]):
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
            fraction=0.025,
            pad=0.025,
        )
        cbar.set_label(cbar_label, fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        _apply_log_ticks(cbar, norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3-column PINN Greek error heatmaps.")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "outputs/pinn/PINN_mix_scaled_fixed_theta/greeks/"
            "pricing_baseline_fixed_theta_final/points_baseline_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--sobolev-csv",
        type=Path,
        default=Path(
            "outputs/pinn/PINN_mix_scaled_fixed_theta/greeks/"
            "pricing_sobolev_fixed_theta_final/points_baseline_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--sobolev-acv-csv",
        type=Path,
        default=Path(
            "outputs/pinn/acv_hard_patch_sobolev_control_variate_best_gate_tau_floor_5e4/"
            "diagnostics_fixed_theta_final/surface_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures/pinn_greeks/acv_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [
        _normalize_errors(pd.read_csv(_resolve(args.baseline_csv))),
        _normalize_errors(pd.read_csv(_resolve(args.sobolev_csv))),
        _normalize_errors(pd.read_csv(_resolve(args.sobolev_acv_csv))),
    ]
    out_dir = _resolve(args.out_dir)
    save_triptych(
        frames=frames,
        value_prefix="abs_error",
        cbar_label="absolute error",
        out_path=out_dir / "pinn_baseline_sobolev_acv_greek_abs_error_maps_15panel.png",
    )
    save_triptych(
        frames=frames,
        value_prefix="rel_abs_error",
        cbar_label="relative absolute error",
        out_path=out_dir / "pinn_baseline_sobolev_acv_greek_rel_abs_error_maps_15panel.png",
    )
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
