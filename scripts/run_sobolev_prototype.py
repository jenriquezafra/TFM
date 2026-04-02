from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
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
from src.greeks.names import build_greek_index_spec
from src.greeks.pinn_adapter import PINNPriceAdapter, load_pinn_price_adapter
from src.pinn.losses import _apply_input_affine, compute_weighted_pinn_loss
from src.pinn.trainer import _load_parquet_matrix, _split_array


DEFAULT_BENCH_CONFIG = PROJECT_ROOT / "configs" / "pinn_greeks_benchmark.yaml"


@dataclass(frozen=True)
class FineTuneArtifacts:
    output_dir: Path
    checkpoints_dir: Path
    training_dir: Path
    benchmark_dir: Path
    benchmark_figures_dir: Path


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data)!r}")
    return data


def _resolve_path(raw: str | Path, *, base_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _parse_dtype(raw: str) -> torch.dtype:
    k = str(raw).strip().lower()
    if k in {"float32", "fp32", "single"}:
        return torch.float32
    if k in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError("dtype must be one of {float32,float64}")


def _resolve_training_config_path(*, run_dir: Path) -> Path:
    execution_path = run_dir / "pipeline_execution.yaml"
    if not execution_path.exists():
        fallback = PROJECT_ROOT / "configs" / "pinn_training.yaml"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            "Could not resolve training config from pipeline_execution.yaml and fallback not found."
        )
    execution = _load_yaml(execution_path)
    cfg_raw = execution.get("config_path")
    if cfg_raw:
        pipeline_cfg_path = _resolve_path(cfg_raw, base_dir=PROJECT_ROOT)
        if pipeline_cfg_path.exists():
            pipeline_cfg = _load_yaml(pipeline_cfg_path)
            train_rel = (
                pipeline_cfg.get("training", {}).get("training_config")
                if isinstance(pipeline_cfg.get("training"), dict)
                else None
            )
            if train_rel:
                train_cfg_path = _resolve_path(train_rel, base_dir=PROJECT_ROOT)
                if train_cfg_path.exists():
                    return train_cfg_path
    fallback = PROJECT_ROOT / "configs" / "pinn_training.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Training config not found.")


def _resolve_collocation_manifest_path(*, run_dir: Path) -> Path:
    summary_path = run_dir / "train" / "metrics" / "train_summary.yaml"
    if summary_path.exists():
        summary = _load_yaml(summary_path)
        manifest_raw = summary.get("collocation_manifest_file")
        if manifest_raw:
            m = _resolve_path(manifest_raw, base_dir=PROJECT_ROOT)
            if m.exists():
                return m

    execution_path = run_dir / "pipeline_execution.yaml"
    if execution_path.exists():
        execution = _load_yaml(execution_path)
        stages = execution.get("stages", {})
        if isinstance(stages, dict):
            train_stage = stages.get("train", {})
            if isinstance(train_stage, dict):
                manifest_raw = train_stage.get("collocation_manifest_file")
                if manifest_raw:
                    m = _resolve_path(manifest_raw, base_dir=PROJECT_ROOT)
                    if m.exists():
                        return m
    raise FileNotFoundError(
        f"Could not resolve collocation manifest for run {run_dir}. "
        "Expected in train_summary or pipeline_execution train stage."
    )


def _build_input_affine_torch(
    *,
    input_scaling: dict,
    feature_order: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    if not isinstance(input_scaling, dict) or not bool(input_scaling.get("enabled", False)):
        return None
    a_raw = list(input_scaling.get("a", []))
    b_raw = list(input_scaling.get("b", []))
    if not a_raw or not b_raw:
        return None
    stats_order = list(input_scaling.get("feature_order", []))
    if stats_order:
        idx = {name: i for i, name in enumerate(stats_order)}
        missing = [name for name in feature_order if name not in idx]
        if missing:
            raise KeyError(f"Missing features in input_scaling.feature_order: {missing}")
        a_vals = [a_raw[idx[name]] for name in feature_order]
        b_vals = [b_raw[idx[name]] for name in feature_order]
    else:
        if len(a_raw) != len(feature_order) or len(b_raw) != len(feature_order):
            raise ValueError("input_scaling dimensions do not match feature_order")
        a_vals = a_raw
        b_vals = b_raw
    return {
        "a": torch.tensor(a_vals, dtype=dtype, device=device).reshape(-1),
        "b": torch.tensor(b_vals, dtype=dtype, device=device).reshape(-1),
    }


def _sample_numpy_batch(x: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    if batch_size >= n:
        return x
    idx = rng.choice(n, size=batch_size, replace=False)
    return x[idx]


def _model_has_nonfinite_params(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


def _load_state_dict_robust(path: Path, *, map_location: torch.device | str) -> dict:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r} in {path}")
    return state


def _build_anchor_grid_from_benchmark_cfg(
    *,
    feature_order: list[str],
    benchmark_cfg: dict,
) -> np.ndarray:
    mode = str(benchmark_cfg.get("source_mode", "surface_grid")).strip().lower()
    if mode != "surface_grid":
        raise ValueError(
            "This prototype expects benchmark.source_mode='surface_grid' to define Greek anchors."
        )
    grid_cfg = benchmark_cfg.get("surface_grid", {})
    if not isinstance(grid_cfg, dict):
        raise ValueError("benchmark.surface_grid must be a dictionary")

    x_feature = str(grid_cfg.get("x_feature", "moneyness"))
    y_feature = str(grid_cfg.get("y_feature", "tau"))
    if x_feature not in feature_order or y_feature not in feature_order:
        raise KeyError(
            f"surface_grid features not in feature_order. x={x_feature}, y={y_feature}, order={feature_order}"
        )

    x_points = int(grid_cfg.get("x_points", 161))
    y_points = int(grid_cfg.get("y_points", 161))
    x_min = float(grid_cfg.get("x_min"))
    x_max = float(grid_cfg.get("x_max"))
    y_min = float(grid_cfg.get("y_min"))
    y_max = float(grid_cfg.get("y_max"))

    if x_points < 2 or y_points < 2:
        raise ValueError("surface_grid x_points,y_points must be >=2")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("surface_grid ranges must satisfy max > min")

    fixed_values = benchmark_cfg.get("fixed_values", {})
    if not isinstance(fixed_values, dict):
        fixed_values = {}

    defaults = {
        "tau": 1.0,
        "moneyness": 1.0,
        "v": 0.04,
        "rho": -0.7,
        "kappa": 2.0,
        "gamma": 0.3,
        "bar_v": 0.04,
        "r": 0.01,
    }
    base = np.zeros(len(feature_order), dtype=np.float64)
    for i, name in enumerate(feature_order):
        if name in fixed_values:
            base[i] = float(fixed_values[name])
        elif name in defaults:
            base[i] = float(defaults[name])
        else:
            base[i] = 0.0

    xv = np.linspace(x_min, x_max, x_points, dtype=np.float64)
    yv = np.linspace(y_min, y_max, y_points, dtype=np.float64)
    xx, yy = np.meshgrid(xv, yv, indexing="ij")

    x_idx = feature_order.index(x_feature)
    y_idx = feature_order.index(y_feature)

    out = np.repeat(base.reshape(1, -1), repeats=xx.size, axis=0)
    out[:, x_idx] = xx.reshape(-1)
    out[:, y_idx] = yy.reshape(-1)
    return out


def _precompute_heston_cf_targets(
    *,
    x_anchor: np.ndarray,
    feature_order: list[str],
    spec: "object",
    spot_feature: str,
    strike: float,
    option_type: str,
    cf_settings: HestonCFGreeksSettings,
) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(feature_order)}
    for req in ("rho", "kappa", "gamma", "bar_v"):
        if req not in idx:
            raise KeyError(f"Missing required Heston parameter feature '{req}' in {feature_order}")
    if spec.idx_tau is None or spec.idx_rate is None or spec.idx_vol is None:
        raise ValueError("spot/vol/tau/rate features must be resolved for Sobolev targets")

    n = x_anchor.shape[0]
    out = {
        "delta": np.full(n, np.nan, dtype=np.float64),
        "gamma": np.full(n, np.nan, dtype=np.float64),
        "vega": np.full(n, np.nan, dtype=np.float64),
        "theta": np.full(n, np.nan, dtype=np.float64),
        "rho": np.full(n, np.nan, dtype=np.float64),
    }
    for i in range(n):
        row = x_anchor[i, :]
        tau = float(row[spec.idx_tau])
        r = float(row[spec.idx_rate])
        v0 = float(row[spec.idx_vol])
        rho = float(row[idx["rho"]])
        kappa = float(row[idx["kappa"]])
        gamma_val = float(row[idx["gamma"]])
        bar_v = float(row[idx["bar_v"]])
        spot_raw = float(row[spec.idx_spot])
        s0 = spot_raw * strike if spot_feature == "moneyness" else spot_raw
        greek_vals = heston_cf_greeks_scalar(
            option_type=option_type,
            S0=s0,
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
        for key in out:
            out[key][i] = float(greek_vals[key])
    valid = np.ones(n, dtype=bool)
    for key in out:
        valid &= np.isfinite(out[key])
    if not np.all(valid):
        x_valid = x_anchor[valid]
        out = {k: v[valid] for k, v in out.items()}
        return {"x_anchor": x_valid, **out}
    return {"x_anchor": x_anchor, **out}


def _greek_scales_from_targets(targets: dict[str, np.ndarray], floor: float = 1.0e-4) -> dict[str, float]:
    scales: dict[str, float] = {}
    for key in ("delta", "vega", "theta", "rho", "gamma"):
        vals = np.abs(np.asarray(targets[key], dtype=np.float64))
        q90 = float(np.nanpercentile(vals, 90.0))
        scales[key] = max(q90, float(floor))
    return scales


def _compute_sobolev_losses(
    *,
    model: torch.nn.Module,
    x_anchor_batch: torch.Tensor,
    target_batch: dict[str, torch.Tensor],
    input_affine: dict[str, torch.Tensor] | None,
    spec: "object",
    spot_feature: str,
    strike: float,
    theta_sign: str,
    scales: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_anchor_batch.detach().clone().requires_grad_(True)
    x_net = _apply_input_affine(x=x, input_affine=input_affine)
    u = model(x_net)
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError(f"Expected model output [N,1], got {tuple(u.shape)}")
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d_spot_raw = grad_u[:, spec.idx_spot]
    if spot_feature == "moneyness":
        delta = d_spot_raw / float(strike)
    else:
        delta = d_spot_raw

    d2_input = torch.autograd.grad(
        outputs=d_spot_raw,
        inputs=x,
        grad_outputs=torch.ones_like(d_spot_raw),
        create_graph=True,
        retain_graph=True,
    )[0][:, spec.idx_spot]
    if spot_feature == "moneyness":
        gamma = d2_input / float(strike * strike)
    else:
        gamma = d2_input

    vega = grad_u[:, spec.idx_vol] if spec.idx_vol is not None else torch.zeros_like(delta)
    theta_raw = grad_u[:, spec.idx_tau] if spec.idx_tau is not None else torch.zeros_like(delta)
    theta = -theta_raw if theta_sign == "minus_dv_dtau" else theta_raw
    rho = grad_u[:, spec.idx_rate] if spec.idx_rate is not None else torch.zeros_like(delta)

    eps = 1.0e-12
    l_delta = torch.mean(((delta - target_batch["delta"]) / (scales["delta"] + eps)) ** 2)
    l_vega = torch.mean(((vega - target_batch["vega"]) / (scales["vega"] + eps)) ** 2)
    l_theta = torch.mean(((theta - target_batch["theta"]) / (scales["theta"] + eps)) ** 2)
    l_rho = torch.mean(((rho - target_batch["rho"]) / (scales["rho"] + eps)) ** 2)
    l_first = 0.25 * (l_delta + l_vega + l_theta + l_rho)
    l_gamma = torch.mean(((gamma - target_batch["gamma"]) / (scales["gamma"] + eps)) ** 2)
    return l_first, l_gamma


def _save_training_loss_plot(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(history["epoch"], history["loss_total"], label="total", linewidth=1.5)
    ax.plot(history["epoch"], history["loss_base"], label="base", linewidth=1.2)
    ax.plot(history["epoch"], history["loss_g1"], label="g1", linewidth=1.0)
    ax.plot(history["epoch"], history["loss_g2"], label="g2", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Sobolev Prototype Fine-Tuning Loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _compute_metrics(y_pred: np.ndarray, y_ref: np.ndarray, mape_floor: float) -> dict[str, float]:
    err = y_pred - y_ref
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    denom = np.maximum(np.abs(y_ref), float(mape_floor))
    mape = float(100.0 * np.mean(np.abs(err) / denom))
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
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
            data = matrix[valid]
            vmin = float(np.nanpercentile(data, 5.0))
            vmax = float(np.nanpercentile(data, 95.0))
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
    cbar.set_label(cbar_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _save_error_hist(error: np.ndarray, out_path: Path, greek: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.hist(error, bins=80, alpha=0.85, color="#2F5D8A")
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("PINN - benchmark")
    ax.set_ylabel("count")
    ax.set_title(f"{greek}: error histogram")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _build_artifact_dirs(*, output_root: Path, run_name: str) -> FineTuneArtifacts:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_root / f"{run_name}_{ts}"
    ckpt = out / "checkpoints"
    trn = out / "training"
    bench = out / "benchmark"
    bench_fig = bench / "figures"
    ckpt.mkdir(parents=True, exist_ok=True)
    trn.mkdir(parents=True, exist_ok=True)
    bench_fig.mkdir(parents=True, exist_ok=True)
    return FineTuneArtifacts(
        output_dir=out,
        checkpoints_dir=ckpt,
        training_dir=trn,
        benchmark_dir=bench,
        benchmark_figures_dir=bench_fig,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prototype Sobolev fine-tuning for PINN Greeks.")
    p.add_argument("--run-dir", default="PINN_mix_scaled_param")
    p.add_argument("--checkpoint-name", default="model_best.pt")
    p.add_argument("--architecture-config", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--output-root", default="outputs/protos/sobolev_mix")

    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument(
        "--switch-epoch",
        type=int,
        default=None,
        help="Epoch where optimizer switches from ADAM to L-BFGS. "
        "If omitted, uses floor(epochs/2). "
        "Use 0 for all L-BFGS, use epochs for all ADAM.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adam-lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--step-size", type=int, default=1000)
    p.add_argument("--gamma-lr", type=float, default=0.5)
    p.add_argument("--lbfgs-lr", type=float, default=0.2)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--lbfgs-history-size", type=int, default=100)
    p.add_argument("--lbfgs-tolerance-grad", type=float, default=1.0e-7)
    p.add_argument("--lbfgs-tolerance-change", type=float, default=1.0e-9)
    p.add_argument("--lbfgs-lr-decay-on-fail", type=float, default=0.5)
    p.add_argument("--lbfgs-min-lr", type=float, default=1.0e-5)
    p.add_argument("--lbfgs-max-failures", type=int, default=25)
    p.add_argument(
        "--lbfgs-line-search-fn",
        default="strong_wolfe",
        choices=["strong_wolfe", "none"],
    )
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--batch-size-collocation", type=int, default=2048)
    p.add_argument("--batch-size-boundary", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.2)

    p.add_argument("--anchor-points", type=int, default=4096)
    p.add_argument("--anchor-batch-size", type=int, default=256)
    p.add_argument("--lambda-g1", type=float, default=0.25)
    p.add_argument("--lambda-g2", type=float, default=0.25)
    p.add_argument("--warmup-epochs", type=int, default=800)
    p.add_argument("--scale-floor", type=float, default=1.0e-4)

    p.add_argument("--spot-feature", default="moneyness")
    p.add_argument("--vol-feature", default="v")
    p.add_argument("--tau-feature", default="tau")
    p.add_argument("--rate-feature", default="r")
    p.add_argument("--theta-sign", default="minus_dv_dtau", choices=["minus_dv_dtau", "dv_dtau"])
    p.add_argument("--strike", type=float, default=1.0)
    p.add_argument("--option-type", default="put", choices=["put", "call"])

    p.add_argument("--benchmark-config", default=str(DEFAULT_BENCH_CONFIG))
    p.add_argument("--benchmark-mape-floor", type=float, default=1.0e-4)
    p.add_argument("--chunk-size-values", type=int, default=4096)
    p.add_argument("--chunk-size-jac", type=int, default=512)
    p.add_argument("--chunk-size-hess", type=int, default=64)
    return p


def run(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.strike <= 0.0:
        raise ValueError("--strike must be > 0")
    if args.anchor_points <= 0:
        raise ValueError("--anchor-points must be > 0")
    if args.anchor_batch_size <= 0:
        raise ValueError("--anchor-batch-size must be > 0")
    if args.switch_epoch is None:
        switch_epoch = int(args.epochs) // 2
    else:
        switch_epoch = int(args.switch_epoch)
    if switch_epoch < 0 or switch_epoch > int(args.epochs):
        raise ValueError("--switch-epoch must be in [0, epochs]")
    if float(args.lbfgs_lr) <= 0.0:
        raise ValueError("--lbfgs-lr must be > 0")
    if int(args.lbfgs_max_failures) < 1:
        raise ValueError("--lbfgs-max-failures must be >= 1")
    if float(args.lbfgs_lr_decay_on_fail) <= 0.0 or float(args.lbfgs_lr_decay_on_fail) >= 1.0:
        raise ValueError("--lbfgs-lr-decay-on-fail must be in (0,1)")
    if float(args.lbfgs_min_lr) <= 0.0:
        raise ValueError("--lbfgs-min-lr must be > 0")

    dtype = _parse_dtype(args.dtype)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    loaded = load_pinn_price_adapter(
        project_root=PROJECT_ROOT,
        run_dir=args.run_dir,
        checkpoint_name=args.checkpoint_name,
        architecture_config_path=args.architecture_config,
        device=args.device,
        dtype=dtype,
        feature_order=None,
    )
    run_dir = loaded.run_dir
    feature_order = list(loaded.feature_order)
    model = loaded.price_fn.model
    model.train()
    device = loaded.device

    artifacts = _build_artifact_dirs(
        output_root=_resolve_path(args.output_root, base_dir=PROJECT_ROOT),
        run_name=run_dir.name,
    )
    torch.save(model.state_dict(), artifacts.checkpoints_dir / "model_init.pt")

    training_cfg_path = _resolve_training_config_path(run_dir=run_dir)
    training_cfg = _load_yaml(training_cfg_path)
    loss_cfg = training_cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}

    manifest_path = _resolve_collocation_manifest_path(run_dir=run_dir)
    manifest = _load_yaml(manifest_path)
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid datasets in manifest: {manifest_path}")
    required_ds = ("interior", "terminal", "lower")
    missing = [k for k in required_ds if k not in datasets]
    if missing:
        raise KeyError(f"Collocation manifest missing datasets: {missing}")

    x_interior = _load_parquet_matrix(
        path=_resolve_path(datasets["interior"], base_dir=PROJECT_ROOT),
        feature_order=feature_order,
    )
    x_terminal = _load_parquet_matrix(
        path=_resolve_path(datasets["terminal"], base_dir=PROJECT_ROOT),
        feature_order=feature_order,
    )
    x_lower = _load_parquet_matrix(
        path=_resolve_path(datasets["lower"], base_dir=PROJECT_ROOT),
        feature_order=feature_order,
    )

    x_int_train, _ = _split_array(x_interior, val_fraction=float(args.val_fraction), seed=int(args.seed))
    x_term_train, _ = _split_array(x_terminal, val_fraction=float(args.val_fraction), seed=int(args.seed) + 1)
    x_low_train, _ = _split_array(x_lower, val_fraction=float(args.val_fraction), seed=int(args.seed) + 2)

    input_affine_t = _build_input_affine_torch(
        input_scaling=loaded.input_scaling,
        feature_order=feature_order,
        device=device,
        dtype=dtype,
    )

    spec = build_greek_index_spec(
        feature_order,
        spot_feature=args.spot_feature,
        vol_feature=args.vol_feature,
        tau_feature=args.tau_feature,
        rate_feature=args.rate_feature,
    )
    if spec.idx_vol is None or spec.idx_tau is None or spec.idx_rate is None:
        raise ValueError("Could not resolve vol/tau/rate feature indices for Sobolev terms.")

    bench_payload = _load_yaml(_resolve_path(args.benchmark_config, base_dir=PROJECT_ROOT))
    bench_cfg = bench_payload.get("benchmark", {})
    if not isinstance(bench_cfg, dict):
        raise ValueError("benchmark section missing in benchmark config.")
    cf_cfg = bench_cfg.get("cf_integration", {})
    if not isinstance(cf_cfg, dict):
        cf_cfg = {}
    cf_settings = HestonCFGreeksSettings(
        u_min=float(cf_cfg.get("u_min", 1.0e-6)),
        u_max=float(cf_cfg.get("u_max", 200.0)),
        n_u=int(cf_cfg.get("n_u", 1200)),
    )

    x_anchor_all = _build_anchor_grid_from_benchmark_cfg(
        feature_order=feature_order,
        benchmark_cfg=bench_cfg,
    )
    if args.anchor_points < x_anchor_all.shape[0]:
        sel = rng.choice(x_anchor_all.shape[0], size=int(args.anchor_points), replace=False)
        x_anchor = x_anchor_all[sel]
    else:
        x_anchor = x_anchor_all

    print(f"Precomputing Heston-CF target Greeks on {x_anchor.shape[0]} anchor points ...")
    target_payload = _precompute_heston_cf_targets(
        x_anchor=x_anchor,
        feature_order=feature_order,
        spec=spec,
        spot_feature=args.spot_feature,
        strike=float(args.strike),
        option_type=str(args.option_type),
        cf_settings=cf_settings,
    )
    x_anchor = np.asarray(target_payload["x_anchor"], dtype=np.float64)
    target_np = {
        key: np.asarray(target_payload[key], dtype=np.float64)
        for key in ("delta", "gamma", "vega", "theta", "rho")
    }
    scales = _greek_scales_from_targets(target_np, floor=float(args.scale_floor))
    print(f"Anchor points after finite filtering: {x_anchor.shape[0]}")
    print(f"Greek normalization scales: {scales}")

    target_t = {
        key: torch.tensor(val, dtype=dtype, device=device)
        for key, val in target_np.items()
    }
    x_anchor_t = torch.tensor(x_anchor, dtype=dtype, device=device)

    adam_epochs = switch_epoch
    lbfgs_epochs = int(args.epochs) - switch_epoch
    adam_optimizer: torch.optim.Optimizer | None = None
    adam_scheduler: torch.optim.lr_scheduler.StepLR | None = None
    if adam_epochs > 0:
        adam_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(args.adam_lr),
            weight_decay=float(args.weight_decay),
        )
        adam_scheduler = torch.optim.lr_scheduler.StepLR(
            adam_optimizer,
            step_size=int(args.step_size),
            gamma=float(args.gamma_lr),
        )
    lbfgs_optimizer: torch.optim.LBFGS | None = None
    if lbfgs_epochs > 0:
        line_search_fn = None if str(args.lbfgs_line_search_fn) == "none" else str(args.lbfgs_line_search_fn)
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=float(args.lbfgs_lr),
            max_iter=int(args.lbfgs_max_iter),
            history_size=int(args.lbfgs_history_size),
            tolerance_grad=float(args.lbfgs_tolerance_grad),
            tolerance_change=float(args.lbfgs_tolerance_change),
            line_search_fn=line_search_fn,
        )

    best_loss = float("inf")
    best_ckpt_path = artifacts.checkpoints_dir / "model_best.pt"
    lbfgs_failures = 0
    stopped_early = False
    stop_reason = ""
    history: list[dict] = []
    t0 = time.perf_counter()
    print(
        f"Optimizer schedule: ADAM epochs [1..{adam_epochs}] "
        f"then L-BFGS epochs [{adam_epochs + 1}..{int(args.epochs)}]."
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        xb_i_np = _sample_numpy_batch(x_int_train, int(args.batch_size_collocation), rng=rng)
        xb_t_np = _sample_numpy_batch(x_term_train, int(args.batch_size_boundary), rng=rng)
        xb_l_np = _sample_numpy_batch(x_low_train, int(args.batch_size_boundary), rng=rng)

        xb_i = torch.tensor(xb_i_np, dtype=dtype, device=device)
        xb_t = torch.tensor(xb_t_np, dtype=dtype, device=device)
        xb_l = torch.tensor(xb_l_np, dtype=dtype, device=device)

        if x_anchor_t.shape[0] > int(args.anchor_batch_size):
            idx_anchor = rng.choice(x_anchor_t.shape[0], size=int(args.anchor_batch_size), replace=False)
            xa = x_anchor_t[idx_anchor]
            target_batch = {k: v[idx_anchor] for k, v in target_t.items()}
        else:
            xa = x_anchor_t
            target_batch = target_t

        ramp = 1.0
        if int(args.warmup_epochs) > 0:
            ramp = min(1.0, float(epoch) / float(args.warmup_epochs))
        lambda_g1_eff = float(args.lambda_g1) * ramp
        lambda_g2_eff = float(args.lambda_g2) * ramp

        optimizer_phase = "adam" if epoch <= adam_epochs else "lbfgs"
        if optimizer_phase == "adam":
            if adam_optimizer is None:
                raise RuntimeError("ADAM phase requested but optimizer was not initialized.")
            base_loss, base_terms = compute_weighted_pinn_loss(
                model=model,
                loss_config=loss_cfg,
                batch_payload={"interior": xb_i, "terminal": xb_t, "lower": xb_l},
                input_affine=input_affine_t,
            )
            g1_loss, g2_loss = _compute_sobolev_losses(
                model=model,
                x_anchor_batch=xa,
                target_batch=target_batch,
                input_affine=input_affine_t,
                spec=spec,
                spot_feature=args.spot_feature,
                strike=float(args.strike),
                theta_sign=args.theta_sign,
                scales=scales,
            )
            total_loss = base_loss + lambda_g1_eff * g1_loss + lambda_g2_eff * g2_loss

            adam_optimizer.zero_grad()
            total_loss.backward()
            adam_optimizer.step()
            if adam_scheduler is not None:
                adam_scheduler.step()

            total_v = float(total_loss.detach().item())
            base_v = float(base_loss.detach().item())
            g1_v = float(g1_loss.detach().item())
            g2_v = float(g2_loss.detach().item())
            lr_now = float(adam_optimizer.param_groups[0]["lr"])
            base_terms_pde = float(base_terms.pde)
            base_terms_term = float(base_terms.term)
            base_terms_low = float(base_terms.low)
        else:
            if lbfgs_optimizer is None:
                raise RuntimeError("L-BFGS phase requested but optimizer was not initialized.")
            # Keep a safe copy so we can rollback if L-BFGS makes parameters non-finite.
            state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            step_stats: dict[str, float] = {}
            step_nonfinite = {"flag": False}

            def closure() -> torch.Tensor:
                lbfgs_optimizer.zero_grad()
                base_loss_c, base_terms_c = compute_weighted_pinn_loss(
                    model=model,
                    loss_config=loss_cfg,
                    batch_payload={"interior": xb_i, "terminal": xb_t, "lower": xb_l},
                    input_affine=input_affine_t,
                )
                g1_loss_c, g2_loss_c = _compute_sobolev_losses(
                    model=model,
                    x_anchor_batch=xa,
                    target_batch=target_batch,
                    input_affine=input_affine_t,
                    spec=spec,
                    spot_feature=args.spot_feature,
                    strike=float(args.strike),
                    theta_sign=args.theta_sign,
                    scales=scales,
                )
                total_loss_c = base_loss_c + lambda_g1_eff * g1_loss_c + lambda_g2_eff * g2_loss_c
                if not torch.isfinite(total_loss_c):
                    step_nonfinite["flag"] = True
                    return torch.full_like(total_loss_c, 1.0e12)
                total_loss_c.backward()

                step_stats["loss_total"] = float(total_loss_c.detach().item())
                step_stats["loss_base"] = float(base_loss_c.detach().item())
                step_stats["loss_g1"] = float(g1_loss_c.detach().item())
                step_stats["loss_g2"] = float(g2_loss_c.detach().item())
                step_stats["loss_base_pde"] = float(base_terms_c.pde)
                step_stats["loss_base_term"] = float(base_terms_c.term)
                step_stats["loss_base_low"] = float(base_terms_c.low)
                return total_loss_c

            loss_return = lbfgs_optimizer.step(closure)
            if not step_stats:
                step_stats["loss_total"] = float(loss_return.detach().item())
                step_stats["loss_base"] = float("nan")
                step_stats["loss_g1"] = float("nan")
                step_stats["loss_g2"] = float("nan")
                step_stats["loss_base_pde"] = float("nan")
                step_stats["loss_base_term"] = float("nan")
                step_stats["loss_base_low"] = float("nan")

            params_finite = not _model_has_nonfinite_params(model)
            losses_finite = (
                np.isfinite(step_stats["loss_total"])
                and np.isfinite(step_stats["loss_base"])
                and np.isfinite(step_stats["loss_g1"])
                and np.isfinite(step_stats["loss_g2"])
            )
            if step_nonfinite["flag"] or (not params_finite) or (not losses_finite):
                model.load_state_dict(state_before)
                lbfgs_failures += 1
                old_lr = float(lbfgs_optimizer.param_groups[0]["lr"])
                new_lr = max(float(args.lbfgs_min_lr), old_lr * float(args.lbfgs_lr_decay_on_fail))
                for pg in lbfgs_optimizer.param_groups:
                    pg["lr"] = new_lr

                base_loss_r, base_terms_r = compute_weighted_pinn_loss(
                    model=model,
                    loss_config=loss_cfg,
                    batch_payload={"interior": xb_i, "terminal": xb_t, "lower": xb_l},
                    input_affine=input_affine_t,
                )
                g1_loss_r, g2_loss_r = _compute_sobolev_losses(
                    model=model,
                    x_anchor_batch=xa,
                    target_batch=target_batch,
                    input_affine=input_affine_t,
                    spec=spec,
                    spot_feature=args.spot_feature,
                    strike=float(args.strike),
                    theta_sign=args.theta_sign,
                    scales=scales,
                )
                total_loss_r = base_loss_r + lambda_g1_eff * g1_loss_r + lambda_g2_eff * g2_loss_r

                total_v = float(total_loss_r.detach().item())
                base_v = float(base_loss_r.detach().item())
                g1_v = float(g1_loss_r.detach().item())
                g2_v = float(g2_loss_r.detach().item())
                base_terms_pde = float(base_terms_r.pde)
                base_terms_term = float(base_terms_r.term)
                base_terms_low = float(base_terms_r.low)
                lr_now = float(lbfgs_optimizer.param_groups[0]["lr"])
                optimizer_phase = "lbfgs_recover"
                print(
                    f"[Sobolev] epoch {epoch:4d}: non-finite detected in L-BFGS step, "
                    f"restored previous params and reduced L-BFGS lr {old_lr:.3e} -> {new_lr:.3e} "
                    f"(failure {lbfgs_failures}/{int(args.lbfgs_max_failures)})."
                )

                if not np.isfinite(total_v):
                    stopped_early = True
                    stop_reason = (
                        "Non-finite loss persisted after rollback in L-BFGS recovery."
                    )
                elif lbfgs_failures >= int(args.lbfgs_max_failures):
                    stopped_early = True
                    stop_reason = (
                        f"Exceeded lbfgs_max_failures={int(args.lbfgs_max_failures)}."
                    )
            else:
                lbfgs_failures = 0
                total_v = step_stats["loss_total"]
                base_v = step_stats["loss_base"]
                g1_v = step_stats["loss_g1"]
                g2_v = step_stats["loss_g2"]
                base_terms_pde = step_stats["loss_base_pde"]
                base_terms_term = step_stats["loss_base_term"]
                base_terms_low = step_stats["loss_base_low"]
                lr_now = float(lbfgs_optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": int(epoch),
                "optimizer_phase": optimizer_phase,
                "lr": lr_now,
                "lambda_g1_effective": lambda_g1_eff,
                "lambda_g2_effective": lambda_g2_eff,
                "loss_total": total_v,
                "loss_base": base_v,
                "loss_g1": g1_v,
                "loss_g2": g2_v,
                "loss_base_pde": base_terms_pde,
                "loss_base_term": base_terms_term,
                "loss_base_low": base_terms_low,
            }
        )
        if np.isfinite(total_v) and total_v < best_loss:
            best_loss = total_v
            torch.save(model.state_dict(), best_ckpt_path)

        if not np.isfinite(total_v):
            stopped_early = True
            stop_reason = "Encountered non-finite total loss."

        if epoch == 1 or epoch % max(1, int(args.log_every)) == 0 or epoch == int(args.epochs):
            elapsed = time.perf_counter() - t0
            mean_epoch = elapsed / float(epoch)
            eta = max(0.0, (int(args.epochs) - epoch) * mean_epoch)
            print(
                f"[Sobolev] epoch {epoch:4d}/{int(args.epochs)} | "
                f"opt={optimizer_phase:5s} | "
                f"lr={lr_now:.3e} | total={total_v:.3e} | base={base_v:.3e} | "
                f"g1={g1_v:.3e} | g2={g2_v:.3e} | "
                f"lambda=({lambda_g1_eff:.3e},{lambda_g2_eff:.3e}) | "
                f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
            )
        if stopped_early:
            print(f"[Sobolev] stopping early at epoch {epoch}: {stop_reason}")
            break

    torch.save(model.state_dict(), artifacts.checkpoints_dir / "model_last.pt")
    history_df = pd.DataFrame(history)
    actual_epochs_completed = int(history_df["epoch"].iloc[-1]) if not history_df.empty else 0
    history_path = artifacts.training_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    loss_plot_path = artifacts.training_dir / "loss_curve.png"
    _save_training_loss_plot(history_df, loss_plot_path)

    def _evaluate_benchmark_once() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[str], np.ndarray]:
        model.eval()
        x_eval_local = _build_anchor_grid_from_benchmark_cfg(
            feature_order=feature_order,
            benchmark_cfg=bench_cfg,
        )
        x_eval_t_local = torch.tensor(x_eval_local, dtype=dtype, device=device)

        price_adapter_local = PINNPriceAdapter(
            model=model,
            input_scaling=loaded.input_scaling,
            feature_order=feature_order,
            dtype=dtype,
            device=device,
        )
        diff_local = derivatives_batch(
            price_adapter_local,
            x_eval_t_local,
            chunk_size_values=int(args.chunk_size_values),
            chunk_size_jac=int(args.chunk_size_jac),
            chunk_size_hess=int(args.chunk_size_hess),
            dtype=dtype,
            device=device,
        )
        jac_local = diff_local.jacobian.detach().cpu()
        hess_local = diff_local.hessian.detach().cpu()
        jac_for_g_local = jac_local
        hess_for_g_local = hess_local
        if args.spot_feature == "moneyness":
            jac_for_g_local, hess_for_g_local = apply_moneyness_to_spot_chain_rule(
                jacobian_wrt_m=jac_for_g_local,
                hessian_wrt_m=hess_for_g_local,
                idx_moneyness=spec.idx_spot,
                strike=float(args.strike),
            )
        greek_map_local = greeks_from_jacobian_hessian(
            jac_for_g_local,
            hess_for_g_local,
            idx_spot=spec.idx_spot,
            idx_vol=spec.idx_vol,
            idx_tau=spec.idx_tau,
            idx_rate=spec.idx_rate,
            theta_is_minus_dv_dtau=(args.theta_sign == "minus_dv_dtau"),
        )
        pred_np_local = {
            k: v.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
            for k, v in greek_map_local.items()
        }

        eval_payload_local = _precompute_heston_cf_targets(
            x_anchor=x_eval_local,
            feature_order=feature_order,
            spec=spec,
            spot_feature=args.spot_feature,
            strike=float(args.strike),
            option_type=str(args.option_type),
            cf_settings=cf_settings,
        )
        x_ref_local = np.asarray(eval_payload_local["x_anchor"], dtype=np.float64)
        ref_np_local = {
            k: np.asarray(eval_payload_local[k], dtype=np.float64)
            for k in ("delta", "gamma", "vega", "theta", "rho")
        }

        if x_ref_local.shape[0] != x_eval_local.shape[0]:
            pred_df_local = pd.DataFrame(x_eval_local, columns=feature_order)
            ref_df_local = pd.DataFrame(x_ref_local, columns=feature_order)
            merge_local = ref_df_local.reset_index().merge(pred_df_local.reset_index(), on=feature_order, how="left")
            pred_idx_local = merge_local["index_y"].to_numpy(dtype=np.int64)
            pred_np_local = {k: v[pred_idx_local] for k, v in pred_np_local.items()}
            x_eval_local = x_ref_local

        compare_local = [g for g in ("delta", "gamma", "vega", "theta", "rho") if g in pred_np_local]
        valid_mask_local = np.ones(x_eval_local.shape[0], dtype=bool)
        for g in compare_local:
            valid_mask_local &= np.isfinite(pred_np_local[g]) & np.isfinite(ref_np_local[g])

        return x_eval_local, pred_np_local, ref_np_local, compare_local, valid_mask_local

    benchmark_model_source = "model_last"
    if _model_has_nonfinite_params(model):
        fallback = best_ckpt_path if best_ckpt_path.exists() else artifacts.checkpoints_dir / "model_init.pt"
        print(
            "Final model has non-finite parameters; "
            f"reloading fallback checkpoint for benchmark: {fallback}"
        )
        model.load_state_dict(_load_state_dict_robust(fallback, map_location=device))
        benchmark_model_source = fallback.name

    x_eval, pred_np, ref_np, compare, valid_mask = _evaluate_benchmark_once()
    if not np.any(valid_mask):
        fallback = best_ckpt_path if best_ckpt_path.exists() else artifacts.checkpoints_dir / "model_init.pt"
        if benchmark_model_source != fallback.name:
            print(
                "No valid points in benchmark with current model; "
                f"retrying benchmark with fallback checkpoint: {fallback}"
            )
            model.load_state_dict(_load_state_dict_robust(fallback, map_location=device))
            benchmark_model_source = fallback.name
            x_eval, pred_np, ref_np, compare, valid_mask = _evaluate_benchmark_once()
    if not np.any(valid_mask):
        raise RuntimeError("No valid points in benchmark evaluation, even after fallback checkpoint.")

    x_valid = x_eval[valid_mask]
    points_df = pd.DataFrame(x_valid, columns=feature_order)
    metrics_rows: list[dict] = []
    n_bins = int(bench_cfg.get("n_bins", 161))
    x_axis = x_valid[:, spec.idx_tau]
    y_axis = x_valid[:, spec.idx_spot]
    x_label = str(args.tau_feature)
    y_label = str(args.spot_feature)

    for g in compare:
        yp = pred_np[g][valid_mask]
        yr = ref_np[g][valid_mask]
        err = yp - yr
        abs_err = np.abs(err)
        rel_abs = abs_err / np.maximum(np.abs(yr), float(args.benchmark_mape_floor))

        points_df[f"pinn_{g}"] = yp
        points_df[f"ref_{g}"] = yr
        points_df[f"error_{g}"] = err
        points_df[f"abs_error_{g}"] = abs_err
        points_df[f"rel_abs_error_{g}"] = rel_abs

        m = _compute_metrics(yp, yr, mape_floor=float(args.benchmark_mape_floor))
        metrics_rows.append({"greek": g, "n_points": int(yr.size), **m})

        _save_error_hist(
            error=err,
            out_path=artifacts.benchmark_figures_dir / f"error_hist_{g}.png",
            greek=g,
        )
        mean_abs, xe, ye = _build_error_heatmap(
            x_axis=x_axis,
            y_axis=y_axis,
            values=abs_err,
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=mean_abs,
            x_edges=xe,
            y_edges=ye,
            x_label=x_label,
            y_label=y_label,
            out_path=artifacts.benchmark_figures_dir / f"abs_error_map_{g}.png",
            title=f"{g}: mean absolute error by zone",
            cbar_label="mean |PINN - benchmark|",
            log_scale=False,
        )
        mean_rel, xe2, ye2 = _build_error_heatmap(
            x_axis=x_axis,
            y_axis=y_axis,
            values=rel_abs,
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=mean_rel,
            x_edges=xe2,
            y_edges=ye2,
            x_label=x_label,
            y_label=y_label,
            out_path=artifacts.benchmark_figures_dir / f"rel_abs_error_map_{g}.png",
            title=f"{g}: mean relative absolute error by zone",
            cbar_label="mean |PINN - benchmark| / max(|benchmark|, floor) (log scale)",
            log_scale=True,
        )

    points_path = artifacts.benchmark_dir / "points_pinn_vs_heston_cf_greeks.csv"
    metrics_csv = artifacts.benchmark_dir / "metrics_by_greek.csv"
    metrics_yaml = artifacts.benchmark_dir / "metrics.yaml"
    points_df.to_csv(points_path, index=False)
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

    summary_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(loaded.checkpoint_path),
        "prototype_output_dir": str(artifacts.output_dir),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(actual_epochs_completed),
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "optimizer": {
            "schedule": "adam_then_lbfgs",
            "switch_epoch": int(switch_epoch),
            "adam": {
                "epochs": int(adam_epochs),
                "lr": float(args.adam_lr),
                "weight_decay": float(args.weight_decay),
                "step_size": int(args.step_size),
                "gamma_lr": float(args.gamma_lr),
            },
            "lbfgs": {
                "epochs": int(lbfgs_epochs),
                "lr": float(args.lbfgs_lr),
                "max_iter": int(args.lbfgs_max_iter),
                "history_size": int(args.lbfgs_history_size),
                "tolerance_grad": float(args.lbfgs_tolerance_grad),
                "tolerance_change": float(args.lbfgs_tolerance_change),
                "line_search_fn": None
                if str(args.lbfgs_line_search_fn) == "none"
                else str(args.lbfgs_line_search_fn),
            },
        },
        "sobolev": {
            "anchor_points": int(x_anchor.shape[0]),
            "anchor_batch_size": int(args.anchor_batch_size),
            "lambda_g1_final": float(args.lambda_g1),
            "lambda_g2_final": float(args.lambda_g2),
            "warmup_epochs": int(args.warmup_epochs),
            "scales": {k: float(v) for k, v in scales.items()},
        },
        "benchmark": {
            "model_source_for_eval": str(benchmark_model_source),
            "points_evaluated": int(x_eval.shape[0]),
            "valid_points": int(np.sum(valid_mask)),
            "metrics": metrics_rows,
            "points_csv": str(points_path),
            "metrics_csv": str(metrics_csv),
            "figures_dir": str(artifacts.benchmark_figures_dir),
        },
        "training_artifacts": {
            "history_csv": str(history_path),
            "loss_curve": str(loss_plot_path),
            "checkpoint_best": str(artifacts.checkpoints_dir / "model_best.pt"),
            "checkpoint_last": str(artifacts.checkpoints_dir / "model_last.pt"),
        },
        "configs": {
            "sobolev_runner_config_path": str(getattr(args, "config_source", "")),
            "training_config_path": str(training_cfg_path),
            "collocation_manifest_path": str(manifest_path),
            "benchmark_config_path": str(_resolve_path(args.benchmark_config, base_dir=PROJECT_ROOT)),
        },
    }
    with open(metrics_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary_payload, f, sort_keys=False)
    with open(artifacts.output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    print("Sobolev prototype finished.")
    print(f"Output dir: {artifacts.output_dir}")
    print(f"Epochs completed: {actual_epochs_completed}/{int(args.epochs)}")
    if stopped_early:
        print(f"Stopped early: {stop_reason}")
    print(f"Loss history: {history_path}")
    print(f"Loss plot: {loss_plot_path}")
    print(f"Benchmark model source: {benchmark_model_source}")
    print(f"Benchmark metrics: {metrics_csv}")
    for row in metrics_rows:
        print(
            f"{row['greek']}: "
            f"MSE={row['mse']:.3e} | RMSE={row['rmse']:.3e} | "
            f"R2={row['r2']:.6f} | MAPE={row['mape_pct']:.3f}%"
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
