from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

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


def _log_norm(matrices: list[np.ndarray]) -> LogNorm | None:
    vals = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0.0)] for matrix in matrices])
    if vals.size == 0:
        return None
    vmin = max(float(np.nanpercentile(vals, 2.0)), float(np.finfo(np.float64).tiny))
    vmax = float(np.nanpercentile(vals, 98.0))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.22)


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
        ax.set_xlabel(r"$e$", fontsize=LABEL_SIZE)
        ax.set_ylabel("density", fontsize=LABEL_SIZE)
        _style_axis(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def build_all(*, baseline_csv: Path, sobolev_csv: Path, out_dir: Path) -> None:
    baseline = pd.read_csv(baseline_csv)
    sobolev = pd.read_csv(sobolev_csv)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(
        baseline_csv=_resolve(args.baseline_csv),
        sobolev_csv=_resolve(args.sobolev_csv),
        out_dir=_resolve(args.out_dir),
    )
    print(f"Saved Greek composites to: {_resolve(args.out_dir)}")


if __name__ == "__main__":
    main()
