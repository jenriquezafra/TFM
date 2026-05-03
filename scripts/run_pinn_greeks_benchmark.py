from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.greeks.chain_rule import apply_moneyness_to_spot_chain_rule
from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian
from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.greeks.names import build_greek_index_spec, parse_feature_order
from src.greeks.pinn_adapter import DEFAULT_PINN_FEATURE_ORDER, load_pinn_price_adapter


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pinn_greeks_benchmark.yaml"
GREEK_LABELS = {
    "delta": "DELTA",
    "gamma": "GAMMA",
    "vega": "VEGA",
    "theta": "THETA",
    "rho": "RHO",
}
FEATURE_LABELS = {
    "moneyness": r"$m$",
    "tau": r"$\tau$",
}
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
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


def _greek_label(name: str) -> str:
    return GREEK_LABELS.get(name, name)


def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PINN Greeks against Heston CF Greeks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to benchmark YAML config.",
    )
    return parser


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _none_if_empty(raw: str | None) -> str | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt.lower() in {"", "none", "null"}:
        return None
    return txt


def _parse_dtype(raw: str) -> torch.dtype:
    key = str(raw).strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("dtype must be one of {float64, float32}")


def _parse_feature_order(raw: str | list[str] | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        return parse_feature_order(parts, fallback=DEFAULT_PINN_FEATURE_ORDER)
    if isinstance(raw, list):
        return parse_feature_order(raw, fallback=DEFAULT_PINN_FEATURE_ORDER)
    raise ValueError("feature_order must be null, comma-separated string, or list")


def _load_train_summary(run_dir: Path) -> dict:
    path = run_dir / "train" / "metrics" / "train_summary.yaml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def _resolve_collocation_manifest_path(
    *,
    run_dir: Path,
    explicit_path: str | None,
) -> Path | None:
    if explicit_path is not None:
        path = _resolve_path(explicit_path, base=PROJECT_ROOT)
        if not path.exists():
            raise FileNotFoundError(f"Collocation manifest not found: {path}")
        return path

    summary = _load_train_summary(run_dir)
    summary_manifest = summary.get("collocation_manifest_file")
    if summary_manifest:
        path = _resolve_path(summary_manifest, base=PROJECT_ROOT)
        if path.exists():
            return path

    execution_path = run_dir / "pipeline_execution.yaml"
    if execution_path.exists():
        execution = _load_yaml(execution_path)
        stages = execution.get("stages", {})
        if isinstance(stages, dict):
            train_stage = stages.get("train", {})
            if isinstance(train_stage, dict):
                collocation_raw = train_stage.get("collocation_manifest_file")
                if collocation_raw:
                    path = _resolve_path(collocation_raw, base=PROJECT_ROOT)
                    if path.exists():
                        return path
    return None


def _sample_rows(x: np.ndarray, *, n_points: int, seed: int) -> np.ndarray:
    if n_points <= 0 or n_points >= x.shape[0]:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_points, replace=False)
    return x[idx]


def _resolve_defaults(feature_order: list[str], fixed_values: dict[str, float] | None) -> np.ndarray:
    fixed = fixed_values or {}
    out = np.zeros(len(feature_order), dtype=np.float64)
    for i, name in enumerate(feature_order):
        if name in fixed:
            out[i] = float(fixed[name])
        elif name in DEFAULT_FIXED_VALUES:
            out[i] = float(DEFAULT_FIXED_VALUES[name])
        else:
            out[i] = 0.0
    return out


def _load_input_points(
    *,
    run_dir: Path,
    feature_order: list[str],
    benchmark_cfg: dict,
) -> np.ndarray:
    input_csv = benchmark_cfg.get("input_csv")
    if input_csv:
        csv_path = _resolve_path(input_csv, base=PROJECT_ROOT)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in feature_order if c not in df.columns]
        if missing:
            raise KeyError(f"Input CSV is missing required columns: {missing}")
        x = df.loc[:, feature_order].to_numpy(dtype=np.float64)
        return _sample_rows(
            x,
            n_points=int(benchmark_cfg.get("n_points", 0)),
            seed=int(benchmark_cfg.get("seed", 42)),
        )

    mode = str(benchmark_cfg.get("source_mode", "collocation")).strip().lower()
    if mode == "surface_grid":
        grid_cfg = benchmark_cfg.get("surface_grid", {})
        if not isinstance(grid_cfg, dict):
            raise ValueError("benchmark.surface_grid must be a dictionary when source_mode=surface_grid")
        x_feature = str(grid_cfg.get("x_feature", "moneyness"))
        y_feature = str(grid_cfg.get("y_feature", "tau"))
        if x_feature == y_feature:
            raise ValueError("surface_grid x_feature and y_feature must be different")
        if x_feature not in feature_order:
            raise KeyError(f"surface_grid x_feature '{x_feature}' not found in feature_order={feature_order}")
        if y_feature not in feature_order:
            raise KeyError(f"surface_grid y_feature '{y_feature}' not found in feature_order={feature_order}")

        x_min = float(grid_cfg.get("x_min"))
        x_max = float(grid_cfg.get("x_max"))
        y_min = float(grid_cfg.get("y_min"))
        y_max = float(grid_cfg.get("y_max"))
        x_points = int(grid_cfg.get("x_points", 161))
        y_points = int(grid_cfg.get("y_points", 161))
        if x_points < 2 or y_points < 2:
            raise ValueError("surface_grid x_points and y_points must be >= 2")
        if x_max <= x_min:
            raise ValueError("surface_grid x_max must be > x_min")
        if y_max <= y_min:
            raise ValueError("surface_grid y_max must be > y_min")

        base = _resolve_defaults(feature_order, benchmark_cfg.get("fixed_values"))
        idx_x = feature_order.index(x_feature)
        idx_y = feature_order.index(y_feature)
        x_axis = np.linspace(x_min, x_max, x_points, dtype=np.float64)
        y_axis = np.linspace(y_min, y_max, y_points, dtype=np.float64)
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
        n = xx.size
        out = np.repeat(base.reshape(1, -1), repeats=n, axis=0)
        out[:, idx_x] = xx.reshape(-1)
        out[:, idx_y] = yy.reshape(-1)
        return out

    if mode == "fixed_point":
        base = _resolve_defaults(feature_order, benchmark_cfg.get("fixed_values"))
        n_repeat = int(benchmark_cfg.get("n_points", 1))
        n_repeat = max(1, n_repeat)
        return np.repeat(base.reshape(1, -1), repeats=n_repeat, axis=0)
    if mode != "collocation":
        raise ValueError("benchmark.source_mode must be one of {'collocation', 'fixed_point', 'surface_grid'}")

    manifest_path = _resolve_collocation_manifest_path(
        run_dir=run_dir,
        explicit_path=benchmark_cfg.get("collocation_manifest"),
    )
    if manifest_path is None:
        raise FileNotFoundError(
            "Could not resolve collocation manifest. Set benchmark.input_csv or benchmark.collocation_manifest."
        )
    manifest = _load_yaml(manifest_path)
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid datasets section in collocation manifest: {manifest_path}")
    dataset_key = str(benchmark_cfg.get("collocation_dataset", "interior")).strip()
    if dataset_key not in datasets:
        raise KeyError(f"Dataset '{dataset_key}' not found in manifest datasets keys={list(datasets.keys())}")
    data_path = _resolve_path(datasets[dataset_key], base=PROJECT_ROOT)
    if not data_path.exists():
        raise FileNotFoundError(f"Collocation dataset file not found: {data_path}")
    df = pd.read_parquet(data_path)
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise KeyError(f"Collocation dataset missing required columns: {missing}")
    x = df.loc[:, feature_order].to_numpy(dtype=np.float64)
    return _sample_rows(
        x,
        n_points=int(benchmark_cfg.get("n_points", 0)),
        seed=int(benchmark_cfg.get("seed", 42)),
    )


def _compute_metrics(*, y_pred: np.ndarray, y_ref: np.ndarray, mape_floor: float) -> dict[str, float]:
    err = y_pred - y_ref
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    denom = np.maximum(np.abs(y_ref), float(mape_floor))
    mape_pct = float(100.0 * np.mean(np.abs(err) / denom))
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape_pct,
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def _build_error_heatmap(
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(np.min(x_axis))
    x_max = float(np.max(x_axis))
    y_min = float(np.min(y_axis))
    y_max = float(np.max(y_axis))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError("Non-finite axis values found for heatmap.")
    if abs(x_max - x_min) < 1.0e-12:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1.0e-12:
        y_min -= 0.5
        y_max += 0.5

    x_edges = np.linspace(x_min, x_max, int(n_bins) + 1)
    y_edges = np.linspace(y_min, y_max, int(n_bins) + 1)

    x_idx = np.clip(np.digitize(x_axis, bins=x_edges, right=False) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(y_axis, bins=y_edges, right=False) - 1, 0, n_bins - 1)

    sums = np.zeros((n_bins, n_bins), dtype=np.float64)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(sums, (y_idx, x_idx), values)
    np.add.at(counts, (y_idx, x_idx), 1)
    mean = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return mean, x_edges, y_edges


def _save_error_hist(*, error: np.ndarray, out_path: Path, greek: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(error, bins=80, alpha=0.85, color="#2F5D8A")
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel(r"$e$", fontsize=LABEL_SIZE)
    ax.set_ylabel("count", fontsize=LABEL_SIZE)
    ax.set_title(f"{_greek_label(greek)}: error histogram", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _save_heatmap(
    *,
    matrix: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_label: str,
    y_label: str,
    out_path: Path,
    title: str,
    cbar_label: str,
    log_scale: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f2f2f2")

    norm = None
    plot_matrix = matrix
    if log_scale:
        valid = np.isfinite(matrix) & (matrix > 0.0)
        if np.any(valid):
            vals = matrix[valid]
            vmin = float(np.nanpercentile(vals, 5.0))
            vmax = float(np.nanpercentile(vals, 95.0))
            vmin = max(vmin, float(np.finfo(np.float64).tiny))
            if vmax <= vmin:
                vmax = vmin * 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
            plot_matrix = np.where(np.isfinite(matrix), np.maximum(matrix, vmin), np.nan)

    im = ax.imshow(
        plot_matrix,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel(_feature_label(x_label), fontsize=LABEL_SIZE)
    ax.set_ylabel(_feature_label(y_label), fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global", {})
    benchmark_cfg = cfg.get("benchmark", {})
    if not isinstance(global_cfg, dict):
        raise ValueError("config.global must be a dictionary")
    if not isinstance(benchmark_cfg, dict):
        raise ValueError("config.benchmark must be a dictionary")

    dtype = _parse_dtype(global_cfg.get("dtype", "float64"))
    feature_order = _parse_feature_order(global_cfg.get("feature_order"))

    loaded = load_pinn_price_adapter(
        project_root=PROJECT_ROOT,
        run_dir=str(global_cfg.get("run_dir", "latest")),
        checkpoint_name=str(global_cfg.get("checkpoint_name", "model_best.pt")),
        architecture_config_path=global_cfg.get("architecture_config"),
        device=str(global_cfg.get("device", "auto")),
        dtype=dtype,
        feature_order=feature_order,
    )
    feature_order_resolved = list(loaded.feature_order)
    x_eval = _load_input_points(
        run_dir=loaded.run_dir,
        feature_order=feature_order_resolved,
        benchmark_cfg=benchmark_cfg,
    )

    if x_eval.shape[0] == 0:
        raise ValueError("No benchmark points available.")

    spot_feature = str(global_cfg.get("spot_feature", "moneyness"))
    vol_feature = _none_if_empty(global_cfg.get("vol_feature", "v"))
    tau_feature = _none_if_empty(global_cfg.get("tau_feature", "tau"))
    rate_feature = _none_if_empty(global_cfg.get("rate_feature", "r"))
    theta_sign = str(global_cfg.get("theta_sign", "minus_dv_dtau"))

    spec = build_greek_index_spec(
        feature_order_resolved,
        spot_feature=spot_feature,
        vol_feature=vol_feature,
        tau_feature=tau_feature,
        rate_feature=rate_feature,
    )
    if vol_feature is None:
        raise ValueError("vol_feature cannot be null for Heston benchmark (needs current variance v0).")
    if spec.idx_tau is None:
        raise ValueError("tau_feature must be present in feature_order for benchmark.")
    if spec.idx_rate is None:
        raise ValueError("rate_feature must be present in feature_order for benchmark.")

    idx = {name: i for i, name in enumerate(feature_order_resolved)}
    for req in ("rho", "kappa", "gamma", "bar_v"):
        if req not in idx:
            raise KeyError(f"Required Heston feature '{req}' missing in feature_order={feature_order_resolved}")

    strike = float(global_cfg.get("strike", 1.0))
    if strike <= 0.0:
        raise ValueError("strike must be > 0")

    x_t = torch.from_numpy(x_eval).to(device=loaded.device, dtype=dtype)
    t0 = time.perf_counter()
    diff = derivatives_batch(
        loaded.price_fn,
        x_t,
        chunk_size_values=int(global_cfg.get("chunk_size_values", 4096)),
        chunk_size_jac=int(global_cfg.get("chunk_size_jac", 512)),
        chunk_size_hess=int(global_cfg.get("chunk_size_hess", 64)),
        dtype=dtype,
        device=loaded.device,
    )
    pinn_seconds = float(time.perf_counter() - t0)

    jacobian = diff.jacobian.detach().cpu()
    hessian = diff.hessian.detach().cpu()

    jac_for_greeks = jacobian
    hess_for_greeks = hessian
    if spot_feature == "moneyness":
        jac_for_greeks, hess_for_greeks = apply_moneyness_to_spot_chain_rule(
            jacobian_wrt_m=jac_for_greeks,
            hessian_wrt_m=hess_for_greeks,
            idx_moneyness=spec.idx_spot,
            strike=strike,
        )

    greek_map = greeks_from_jacobian_hessian(
        jac_for_greeks,
        hess_for_greeks,
        idx_spot=spec.idx_spot,
        idx_vol=spec.idx_vol,
        idx_tau=spec.idx_tau,
        idx_rate=spec.idx_rate,
        theta_is_minus_dv_dtau=(theta_sign == "minus_dv_dtau"),
    )
    pinn_greeks = {
        name: tensor.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        for name, tensor in greek_map.items()
    }

    cf_cfg = benchmark_cfg.get("cf_integration", {})
    if not isinstance(cf_cfg, dict):
        raise ValueError("benchmark.cf_integration must be a dictionary")
    cf_settings = HestonCFGreeksSettings(
        u_min=float(cf_cfg.get("u_min", 1.0e-6)),
        u_max=float(cf_cfg.get("u_max", 200.0)),
        n_u=int(cf_cfg.get("n_u", 1200)),
    )
    option_type = str(benchmark_cfg.get("option_type", "call")).strip().lower()

    n_rows = x_eval.shape[0]
    ref_data: dict[str, np.ndarray] = {
        "delta": np.full(n_rows, np.nan, dtype=np.float64),
        "gamma": np.full(n_rows, np.nan, dtype=np.float64),
        "vega": np.full(n_rows, np.nan, dtype=np.float64),
        "theta": np.full(n_rows, np.nan, dtype=np.float64),
        "rho": np.full(n_rows, np.nan, dtype=np.float64),
    }
    failed_ref = 0
    t1 = time.perf_counter()
    for i in range(n_rows):
        row = x_eval[i, :]
        tau = float(row[spec.idx_tau])
        r = float(row[spec.idx_rate])
        v0 = float(row[spec.idx_vol])
        rho = float(row[idx["rho"]])
        kappa = float(row[idx["kappa"]])
        gamma_val = float(row[idx["gamma"]])
        bar_v = float(row[idx["bar_v"]])

        spot_raw = float(row[spec.idx_spot])
        S0 = spot_raw * strike if spot_feature == "moneyness" else spot_raw
        try:
            out = heston_cf_greeks_scalar(
                option_type=option_type,
                S0=S0,
                K=strike,
                tau=tau,
                r=r,
                rho=rho,
                kappa=kappa,
                gamma=gamma_val,
                bar_v=bar_v,
                v0=v0,
                settings=cf_settings,
            )
            for greek_name in ref_data.keys():
                ref_data[greek_name][i] = float(out[greek_name])
        except Exception:
            failed_ref += 1

    ref_seconds = float(time.perf_counter() - t1)

    compare_order = [g for g in ("delta", "gamma", "vega", "theta", "rho") if g in pinn_greeks]
    if not compare_order:
        raise RuntimeError("No common Greek columns found to benchmark.")

    valid_mask = np.ones(n_rows, dtype=bool)
    for greek_name in compare_order:
        valid_mask &= np.isfinite(pinn_greeks[greek_name]) & np.isfinite(ref_data[greek_name])
    if not np.any(valid_mask):
        raise RuntimeError("No valid points after filtering finite PINN/reference Greek values.")

    x_valid = x_eval[valid_mask]
    points_df = pd.DataFrame(x_valid, columns=feature_order_resolved)

    mape_floor = float(benchmark_cfg.get("mape_floor", 1.0e-6))
    if "n_bins" in benchmark_cfg:
        n_bins = int(benchmark_cfg.get("n_bins", 24))
    else:
        source_mode = str(benchmark_cfg.get("source_mode", "collocation")).strip().lower()
        if source_mode == "surface_grid":
            grid_cfg = benchmark_cfg.get("surface_grid", {})
            if isinstance(grid_cfg, dict):
                n_bins = int(max(int(grid_cfg.get("x_points", 161)), int(grid_cfg.get("y_points", 161))))
            else:
                n_bins = 161
        else:
            n_bins = 24
    # Keep tau on the horizontal axis to match Greek surface plots.
    x_axis_values = x_valid[:, spec.idx_tau]
    y_axis_values = x_valid[:, spec.idx_spot]
    x_axis_label = str(tau_feature)
    y_axis_label = spot_feature

    metrics_rows: list[dict] = []
    output_subdir = str(benchmark_cfg.get("output_subdir", "benchmark_cf")).strip() or "benchmark_cf"
    out_dir = loaded.run_dir / "greeks" / output_subdir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in fig_dir.glob("error_vs_ref_*.png"):
        stale_path.unlink(missing_ok=True)

    for greek_name in compare_order:
        y_pred = pinn_greeks[greek_name][valid_mask]
        y_ref = ref_data[greek_name][valid_mask]
        err = y_pred - y_ref
        abs_err = np.abs(err)
        rel_abs_err = abs_err / np.maximum(np.abs(y_ref), mape_floor)

        points_df[f"pinn_{greek_name}"] = y_pred
        points_df[f"ref_{greek_name}"] = y_ref
        points_df[f"error_{greek_name}"] = err
        points_df[f"abs_error_{greek_name}"] = abs_err
        points_df[f"rel_abs_error_{greek_name}"] = rel_abs_err

        metrics = _compute_metrics(y_pred=y_pred, y_ref=y_ref, mape_floor=mape_floor)
        metrics_rows.append(
            {
                "greek": greek_name,
                "n_points": int(y_ref.size),
                **metrics,
            }
        )

        _save_error_hist(
            error=err,
            out_path=fig_dir / f"error_hist_{greek_name}.png",
            greek=greek_name,
        )
        mean_abs, x_edges, y_edges = _build_error_heatmap(
            x_axis=x_axis_values,
            y_axis=y_axis_values,
            values=abs_err,
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=mean_abs,
            x_edges=x_edges,
            y_edges=y_edges,
            x_label=x_axis_label,
            y_label=y_axis_label,
            out_path=fig_dir / f"abs_error_map_{greek_name}.png",
            title=f"{_greek_label(greek_name)}: mean absolute error by zone",
            cbar_label=r"mean $|e|$",
            log_scale=False,
        )
        mean_rel_abs, x_edges2, y_edges2 = _build_error_heatmap(
            x_axis=x_axis_values,
            y_axis=y_axis_values,
            values=rel_abs_err,
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=mean_rel_abs,
            x_edges=x_edges2,
            y_edges=y_edges2,
            x_label=x_axis_label,
            y_label=y_axis_label,
            out_path=fig_dir / f"rel_abs_error_map_{greek_name}.png",
            title=f"{_greek_label(greek_name)}: mean relative absolute error by zone",
            cbar_label=r"mean $|e|_{\mathrm{rel}}$ (log scale)",
            log_scale=True,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    points_path = out_dir / "points_pinn_vs_heston_cf_greeks.csv"
    metrics_path = out_dir / "metrics_by_greek.csv"
    metrics_yaml_path = out_dir / "metrics.yaml"
    execution_path = out_dir / "benchmark_execution.yaml"

    points_df.to_csv(points_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "run_dir": str(loaded.run_dir),
        "checkpoint": str(loaded.checkpoint_path),
        "architecture_config": str(loaded.architecture_config_path),
        "feature_order": feature_order_resolved,
        "device": str(loaded.device),
        "dtype": str(dtype),
        "n_input_points": int(n_rows),
        "n_valid_points": int(np.sum(valid_mask)),
        "failed_reference_points": int(failed_ref),
        "pinn_derivatives_seconds": pinn_seconds,
        "reference_seconds": ref_seconds,
        "pinn_points_per_second": float(n_rows / max(pinn_seconds, 1.0e-12)),
        "reference_points_per_second": float(n_rows / max(ref_seconds, 1.0e-12)),
        "spot_feature": spot_feature,
        "vol_feature": vol_feature,
        "tau_feature": tau_feature,
        "rate_feature": rate_feature,
        "theta_sign": theta_sign,
        "strike": strike,
        "option_type": option_type,
        "cf_integration": {
            "u_min": float(cf_settings.u_min),
            "u_max": float(cf_settings.u_max),
            "n_u": int(cf_settings.n_u),
        },
        "benchmark_metrics": metrics_rows,
        "artifacts": {
            "points_csv": str(points_path),
            "metrics_csv": str(metrics_path),
            "metrics_yaml": str(metrics_yaml_path),
            "figures_dir": str(fig_dir),
        },
    }
    with open(metrics_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    with open(execution_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "timestamp": payload["timestamp"],
                "config_source": str(config_path),
                "global_config": global_cfg,
                "benchmark_config": benchmark_cfg,
                "resolved": {
                    "run_dir": str(loaded.run_dir),
                    "checkpoint": str(loaded.checkpoint_path),
                    "architecture_config": str(loaded.architecture_config_path),
                    "feature_order": feature_order_resolved,
                    "device": str(loaded.device),
                    "dtype": str(dtype),
                },
                "artifacts": payload["artifacts"],
            },
            f,
            sort_keys=False,
        )

    print(f"Config: {config_path}")
    print(f"Run dir: {loaded.run_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Valid points: {int(np.sum(valid_mask))} / {n_rows}")
    print(
        f"Speed PINN={payload['pinn_points_per_second']:.1f} pts/s | "
        f"Heston-CF={payload['reference_points_per_second']:.1f} pts/s"
    )
    for row in metrics_rows:
        print(
            f"{row['greek']}: "
            f"MSE={row['mse']:.3e} | RMSE={row['rmse']:.3e} | "
            f"MAPE={row['mape_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
