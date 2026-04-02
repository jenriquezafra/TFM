from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT))

from src.greeks.chain_rule import apply_moneyness_to_spot_chain_rule
from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian
from src.greeks.names import build_greek_index_spec, parse_feature_order
from src.greeks.pinn_adapter import DEFAULT_PINN_FEATURE_ORDER, load_pinn_price_adapter


DEFAULT_FIXED_VALUES = {
    "tau": 1.0,
    "moneyness": 1.0,
    "v": 0.04,
    "rho": -0.7,
    "kappa": 2.0,
    "gamma": 0.3,
    "bar_v": 0.04,
    "r": 0.01,
}


def _none_if_empty(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _parse_dtype(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("dtype must be one of {float64, float32}")


def _parse_feature_order(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parse_feature_order(parts, fallback=DEFAULT_PINN_FEATURE_ORDER)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _parse_fixed_values(raw: str | None) -> dict[str, float]:
    if raw is None or not raw.strip():
        return {}

    out: dict[str, float] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid fixed-values token '{token}'. Use format name=value,name=value"
            )
        name, value = token.split("=", maxsplit=1)
        out[name.strip()] = float(value.strip())
    return out


def _resolve_defaults(feature_order: list[str], fixed_overrides: dict[str, float]) -> np.ndarray:
    base = np.zeros(len(feature_order), dtype=np.float64)
    for i, name in enumerate(feature_order):
        if name in fixed_overrides:
            base[i] = fixed_overrides[name]
        elif name in DEFAULT_FIXED_VALUES:
            base[i] = DEFAULT_FIXED_VALUES[name]
        else:
            base[i] = 0.0
    return base


def _default_output_path(*, run_dir: Path, x_feature: str, y_feature: str, user_output: str | None) -> Path:
    if user_output:
        out = Path(user_output)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        return out

    out_dir = run_dir / "greeks"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"surface_{_safe_name(x_feature)}_{_safe_name(y_feature)}.csv"


def _default_plot_path(*, csv_path: Path, user_plot_path: str | None) -> Path:
    if user_plot_path:
        out = Path(user_plot_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        return out
    return csv_path.with_suffix(".png")


def _default_plots_dir(*, csv_path: Path, user_plots_dir: str | None) -> Path:
    if user_plots_dir:
        out = Path(user_plots_dir)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        return out
    return csv_path.with_suffix("")


def _plot_surface(
    df: pd.DataFrame,
    *,
    x_feature: str,
    y_feature: str,
    metric: str,
    out_path: Path,
) -> None:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in output columns: {list(df.columns)}")

    x_values = np.sort(df[x_feature].unique())
    y_values = np.sort(df[y_feature].unique())

    pivot = (
        df.pivot(index=x_feature, columns=y_feature, values=metric)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    z = pivot.to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[y_values.min(), y_values.max(), x_values.min(), x_values.max()],
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel(y_feature)
    ax.set_ylabel(x_feature)
    ax.set_title(metric)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _greek_column_name(out_df: pd.DataFrame, greek_name: str) -> str:
    if greek_name in out_df.columns:
        return f"greek_{greek_name}"
    return greek_name


def _resolve_metric_name(
    *,
    metric_requested: str,
    columns: list[str],
    greek_aliases: dict[str, str],
) -> str:
    if metric_requested in columns:
        return metric_requested
    if metric_requested in greek_aliases and greek_aliases[metric_requested] in columns:
        return greek_aliases[metric_requested]
    greek_prefixed = f"greek_{metric_requested}"
    if greek_prefixed in columns:
        return greek_prefixed
    raise KeyError(
        f"Metric '{metric_requested}' not available. "
        f"Available columns: {columns}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 2D surface of value/greeks from trained PINN.")
    parser.add_argument("--run-dir", default="latest")
    parser.add_argument("--checkpoint-name", default="model_best.pt")
    parser.add_argument("--architecture-config", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--feature-order", default=None)

    parser.add_argument("--x-feature", required=True)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--x-points", type=int, default=81)

    parser.add_argument("--y-feature", required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--y-points", type=int, default=81)

    parser.add_argument(
        "--fixed-values",
        default="",
        help="Comma-separated defaults for non-surface features, e.g. rho=-0.7,kappa=2.0",
    )

    parser.add_argument("--spot-feature", default="moneyness")
    parser.add_argument("--vol-feature", default="v")
    parser.add_argument("--tau-feature", default="tau")
    parser.add_argument("--rate-feature", default="r")
    parser.add_argument("--theta-sign", default="minus_dv_dtau", choices=["minus_dv_dtau", "dv_dtau"])
    parser.add_argument("--strike", type=float, default=None)

    parser.add_argument("--chunk-size-values", type=int, default=4096)
    parser.add_argument("--chunk-size-jac", type=int, default=512)
    parser.add_argument("--chunk-size-hess", type=int, default=64)

    parser.add_argument("--metric", default="value", help="Column to plot as heatmap")
    parser.add_argument(
        "--all-greeks",
        action="store_true",
        help="Generate one plot per available Greek column (delta/gamma/vega/theta/rho).",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory for --all-greeks plots. Default: sibling folder derived from CSV path.",
    )
    return parser


def run_surface(args: argparse.Namespace) -> dict:
    if args.x_points < 2 or args.y_points < 2:
        raise ValueError("x-points and y-points must be >= 2")
    if args.x_feature == args.y_feature:
        raise ValueError("x-feature and y-feature must be different")
    if args.all_greeks and args.plot_path:
        raise ValueError("--plot-path cannot be used with --all-greeks. Use --plots-dir instead.")

    dtype = _parse_dtype(args.dtype)
    feature_order = _parse_feature_order(args.feature_order)

    loaded = load_pinn_price_adapter(
        project_root=PROJECT_ROOT,
        run_dir=args.run_dir,
        checkpoint_name=args.checkpoint_name,
        architecture_config_path=args.architecture_config,
        device=args.device,
        dtype=dtype,
        feature_order=feature_order,
    )
    resolved_feature_order = loaded.feature_order

    if args.x_feature not in resolved_feature_order:
        raise KeyError(
            f"x-feature '{args.x_feature}' not found in feature-order={resolved_feature_order}"
        )
    if args.y_feature not in resolved_feature_order:
        raise KeyError(
            f"y-feature '{args.y_feature}' not found in feature-order={resolved_feature_order}"
        )

    idx_x = resolved_feature_order.index(args.x_feature)
    idx_y = resolved_feature_order.index(args.y_feature)

    fixed_overrides = _parse_fixed_values(args.fixed_values)
    base = _resolve_defaults(resolved_feature_order, fixed_overrides)

    x_values = np.linspace(float(args.x_min), float(args.x_max), int(args.x_points), dtype=np.float64)
    y_values = np.linspace(float(args.y_min), float(args.y_max), int(args.y_points), dtype=np.float64)

    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    n_rows = xx.size

    x_grid = np.repeat(base.reshape(1, -1), repeats=n_rows, axis=0)
    x_grid[:, idx_x] = xx.reshape(-1)
    x_grid[:, idx_y] = yy.reshape(-1)

    x_t = torch.from_numpy(x_grid).to(device=loaded.device, dtype=dtype)
    diff = derivatives_batch(
        loaded.price_fn,
        x_t,
        chunk_size_values=args.chunk_size_values,
        chunk_size_jac=args.chunk_size_jac,
        chunk_size_hess=args.chunk_size_hess,
        dtype=dtype,
        device=loaded.device,
    )

    values = diff.values.detach().cpu().numpy().reshape(-1)
    jacobian = diff.jacobian.detach().cpu()
    hessian = diff.hessian.detach().cpu()

    spec = build_greek_index_spec(
        resolved_feature_order,
        spot_feature=args.spot_feature,
        vol_feature=_none_if_empty(args.vol_feature),
        tau_feature=_none_if_empty(args.tau_feature),
        rate_feature=_none_if_empty(args.rate_feature),
    )

    jac_for_greeks = jacobian
    hess_for_greeks = hessian
    if args.strike is not None and args.spot_feature == "moneyness":
        jac_for_greeks, hess_for_greeks = apply_moneyness_to_spot_chain_rule(
            jacobian_wrt_m=jac_for_greeks,
            hessian_wrt_m=hess_for_greeks,
            idx_moneyness=spec.idx_spot,
            strike=float(args.strike),
        )

    greek_map = greeks_from_jacobian_hessian(
        jac_for_greeks,
        hess_for_greeks,
        idx_spot=spec.idx_spot,
        idx_vol=spec.idx_vol,
        idx_tau=spec.idx_tau,
        idx_rate=spec.idx_rate,
        theta_is_minus_dv_dtau=(args.theta_sign == "minus_dv_dtau"),
    )

    jac_np = jacobian.numpy()
    hess_np = hessian.numpy()

    out_df = pd.DataFrame(
        {
            args.x_feature: xx.reshape(-1),
            args.y_feature: yy.reshape(-1),
            "value": values,
            f"jac_{args.x_feature}": jac_np[:, idx_x],
            f"jac_{args.y_feature}": jac_np[:, idx_y],
            f"hess_{args.x_feature}__{args.x_feature}": hess_np[:, idx_x, idx_x],
            f"hess_{args.y_feature}__{args.y_feature}": hess_np[:, idx_y, idx_y],
            f"hess_{args.x_feature}__{args.y_feature}": hess_np[:, idx_x, idx_y],
        }
    )

    greek_aliases: dict[str, str] = {}
    for greek_name, greek_tensor in greek_map.items():
        out_col = _greek_column_name(out_df, greek_name)
        greek_aliases[greek_name] = out_col
        out_df[out_col] = greek_tensor.detach().cpu().numpy().reshape(-1)

    if args.strike is not None and args.spot_feature == "moneyness":
        out_df["spot_from_moneyness_strike"] = float(args.strike)

    out_csv = _default_output_path(
        run_dir=loaded.run_dir,
        x_feature=args.x_feature,
        y_feature=args.y_feature,
        user_output=args.output_csv,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    metric_resolved = _resolve_metric_name(
        metric_requested=args.metric,
        columns=list(out_df.columns),
        greek_aliases=greek_aliases,
    )
    greek_columns_ordered = list(greek_aliases.values())
    saved_plot_paths: list[Path] = []
    if not args.no_plot:
        if args.all_greeks:
            plots_dir = _default_plots_dir(csv_path=out_csv, user_plots_dir=args.plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            stem = out_csv.stem
            for greek_col in greek_columns_ordered:
                plot_path = plots_dir / f"{stem}_{_safe_name(greek_col)}.png"
                _plot_surface(
                    out_df,
                    x_feature=args.x_feature,
                    y_feature=args.y_feature,
                    metric=greek_col,
                    out_path=plot_path,
                )
                saved_plot_paths.append(plot_path)
                print(f"Saved plot: {plot_path}")
        else:
            plot_path = _default_plot_path(csv_path=out_csv, user_plot_path=args.plot_path)
            _plot_surface(
                out_df,
                x_feature=args.x_feature,
                y_feature=args.y_feature,
                metric=metric_resolved,
                out_path=plot_path,
            )
            saved_plot_paths.append(plot_path)
            print(f"Saved plot: {plot_path}")

    print(f"Run dir: {loaded.run_dir}")
    print(f"Checkpoint: {loaded.checkpoint_path}")
    print(f"Architecture config: {loaded.architecture_config_path}")
    print(f"Feature order: {resolved_feature_order}")
    print(f"Device: {loaded.device}")
    print(f"Rows: {len(out_df)}")
    print(f"Metric requested: {args.metric}")
    print(f"Metric plotted: {metric_resolved}")
    print(f"All greeks mode: {bool(args.all_greeks)}")
    if args.all_greeks:
        print(f"Greek columns plotted: {greek_columns_ordered}")
    print(f"Plots generated: {len(saved_plot_paths)}")
    print(f"Saved CSV: {out_csv}")

    return {
        "run_dir": str(loaded.run_dir),
        "checkpoint": str(loaded.checkpoint_path),
        "architecture_config": str(loaded.architecture_config_path),
        "feature_order": list(resolved_feature_order),
        "device": str(loaded.device),
        "rows": int(len(out_df)),
        "metric_requested": str(args.metric),
        "metric_plotted": str(metric_resolved),
        "all_greeks": bool(args.all_greeks),
        "greek_columns_plotted": list(greek_columns_ordered if args.all_greeks else []),
        "plots_generated": int(len(saved_plot_paths)),
        "saved_csv": str(out_csv),
        "saved_plots": [str(p) for p in saved_plot_paths],
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_surface(args)


if __name__ == "__main__":
    main()
