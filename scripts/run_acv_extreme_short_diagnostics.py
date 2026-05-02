from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian
from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.greeks.pinn_adapter import load_pinn_price_adapter
from src.pinn.acv_hard_patch import (
    ACVHardPatchPriceAdapter,
    heston_log_pde_residual,
    load_acv_hard_patch_checkpoint,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "acv_extreme_short_diagnostics.yaml"
FEATURE_ORDER = ["tau", "moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
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


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _parse_dtype(raw: str) -> torch.dtype:
    key = str(raw).strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("dtype must be one of {'float64', 'float32'}")


def _resolve_path(raw: str | Path | None, *, base_dir: Path = PROJECT_ROOT) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / path).resolve()


def _resolve_run_dir(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return (PROJECT_ROOT / path).resolve()
    return (PROJECT_ROOT / "outputs" / "pinn" / path).resolve()


def _metric(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "p90_abs_error": float("nan"),
            "p99_abs_error": float("nan"),
            "max_abs_error": float("nan"),
        }
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "p90_abs_error": float(np.percentile(abs_err, 90.0)),
        "p99_abs_error": float(np.percentile(abs_err, 99.0)),
        "max_abs_error": float(np.max(abs_err)),
    }


def _abs_metric(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "p90_abs_error": float("nan"),
            "p99_abs_error": float("nan"),
            "max_abs_error": float("nan"),
        }
    abs_values = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(abs_values)),
        "p90_abs_error": float(np.percentile(abs_values, 90.0)),
        "p99_abs_error": float(np.percentile(abs_values, 99.0)),
        "max_abs_error": float(np.max(abs_values)),
    }


def _cf_settings(cfg: dict) -> HestonCFGreeksSettings:
    cf = cfg.get("cf_integration", {})
    return HestonCFGreeksSettings(
        u_min=float(cf.get("u_min", 1.0e-6)),
        u_max=float(cf.get("u_max", 200.0)),
        n_u=int(cf.get("n_u", 1200)),
    )


def _build_extreme_short_grid(cfg: dict, *, tau_points: int | None, x_points: int | None) -> np.ndarray:
    grid = cfg.get("grid", {})
    n_tau = int(tau_points if tau_points is not None else grid.get("tau_points", 80))
    n_x = int(x_points if x_points is not None else grid.get("x_points", 121))
    if n_tau < 2 or n_x < 2:
        raise ValueError("tau_points and x_points must both be >= 2")

    tau_min = float(grid.get("tau_min", 1.0e-4))
    tau_max = float(grid.get("tau_max", 5.0e-2))
    x_min = float(grid.get("log_moneyness_min", -0.06))
    x_max = float(grid.get("log_moneyness_max", 0.06))
    if tau_min <= 0.0 or tau_max <= tau_min:
        raise ValueError("grid tau bounds must satisfy 0 < tau_min < tau_max")
    if x_max <= x_min:
        raise ValueError("grid log-moneyness bounds must satisfy max > min")

    fixed = dict(DEFAULT_FIXED_VALUES)
    fixed.update(grid.get("fixed_values", {}) or {})
    base = np.array([float(fixed[name]) for name in FEATURE_ORDER], dtype=np.float64)
    tau_axis = np.geomspace(tau_min, tau_max, n_tau)
    x_axis = np.linspace(x_min, x_max, n_x, dtype=np.float64)
    xx, tt = np.meshgrid(x_axis, tau_axis, indexing="ij")
    points = np.repeat(base.reshape(1, -1), repeats=xx.size, axis=0)
    points[:, FEATURE_ORDER.index("tau")] = tt.reshape(-1)
    points[:, FEATURE_ORDER.index("moneyness")] = np.exp(xx.reshape(-1))
    return points


def _region_masks(points: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    regions = cfg.get("regions", {})
    x_abs = np.abs(np.log(points["moneyness"].to_numpy(dtype=np.float64)))
    tau = points["tau"].to_numpy(dtype=np.float64)
    hard_x = float(regions.get("hard_x_abs", 0.03))
    patch_x = float(regions.get("patch_x_abs", 0.06))
    ultra_tau = float(regions.get("ultra_short_tau", 1.0e-3))
    near_zero_tau = float(regions.get("near_zero_tau", 5.0e-3))
    hard_tau = float(regions.get("hard_tau", 5.0e-2))
    return {
        "full_extreme_short": np.ones(points.shape[0], dtype=bool),
        "ultra_short": tau <= ultra_tau,
        "near_zero": tau <= near_zero_tau,
        "hard": (x_abs < hard_x) & (tau < hard_tau),
        "hard_ultra_short": (x_abs < hard_x) & (tau <= ultra_tau),
        "hard_near_zero": (x_abs < hard_x) & (tau <= near_zero_tau),
        "patch": (x_abs < patch_x) & (tau < hard_tau),
    }


def _cf_refs(points: np.ndarray, *, settings: HestonCFGreeksSettings, option_type: str) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    for row in points:
        rows.append(
            heston_cf_greeks_scalar(
                option_type=option_type,
                S0=float(row[1]),
                K=1.0,
                tau=float(row[0]),
                r=float(row[7]),
                rho=float(row[3]),
                kappa=float(row[4]),
                gamma=float(row[5]),
                bar_v=float(row[6]),
                v0=float(row[2]),
                settings=settings,
            )
        )
    keys = ["price", "delta", "gamma", "vega", "theta", "rho"]
    return {key: np.array([item[key] for item in rows], dtype=np.float64) for key in keys}


def _predict_derivative_metrics(
    *,
    price_fn,
    points: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
    chunk_cfg: dict,
) -> dict[str, np.ndarray]:
    diff = derivatives_batch(
        price_fn,
        torch.as_tensor(points, dtype=dtype, device=device),
        chunk_size_values=int(chunk_cfg.get("chunk_size_values", 4096)),
        chunk_size_jac=int(chunk_cfg.get("chunk_size_jac", 512)),
        chunk_size_hess=int(chunk_cfg.get("chunk_size_hess", 64)),
        dtype=dtype,
        device=device,
    )
    greeks = greeks_from_jacobian_hessian(
        diff.jacobian,
        diff.hessian,
        idx_spot=FEATURE_ORDER.index("moneyness"),
        idx_vol=FEATURE_ORDER.index("v"),
        idx_tau=FEATURE_ORDER.index("tau"),
        idx_rate=FEATURE_ORDER.index("r"),
        theta_is_minus_dv_dtau=True,
    )
    return {
        "price": diff.values.detach().cpu().numpy().reshape(-1),
        "delta": greeks["delta"].detach().cpu().numpy().reshape(-1),
        "gamma": greeks["gamma"].detach().cpu().numpy().reshape(-1),
        "vega": greeks["vega"].detach().cpu().numpy().reshape(-1),
        "theta": greeks["theta"].detach().cpu().numpy().reshape(-1),
        "rho": greeks["rho"].detach().cpu().numpy().reshape(-1),
    }


def _pde_residuals(
    *,
    price_fn: Callable[[Tensor], Tensor],
    points: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, points.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), points.shape[0])
        raw = torch.as_tensor(points[start:stop], dtype=dtype, device=device)
        with torch.enable_grad():
            residual, _ = heston_log_pde_residual(price_fn=price_fn, raw=raw, scale_epsilon=None)
        out.append(residual.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, axis=0)


def _baseline_batch_price_fn(loaded) -> Callable[[Tensor], Tensor]:
    def f(raw: Tensor) -> Tensor:
        a = loaded.price_fn.a.to(dtype=raw.dtype, device=raw.device).reshape(1, -1)
        b = loaded.price_fn.b.to(dtype=raw.dtype, device=raw.device).reshape(1, -1)
        return loaded.price_fn.model(a + b * raw)

    return f


def _model_summary(
    *,
    pred: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    pde_residual: np.ndarray | None,
    masks: dict[str, np.ndarray],
) -> dict:
    out: dict[str, dict] = {}
    for region, mask in masks.items():
        out[region] = {"n": int(np.sum(mask))}
        for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
            out[region][key] = _metric(pred[key][mask], refs[key][mask])
        if pde_residual is not None:
            out[region]["pde_residual"] = _abs_metric(pde_residual[mask])
    return out


def _tau_bucket_summary(
    *,
    points: pd.DataFrame,
    pred: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    cfg: dict,
) -> dict:
    buckets = [float(x) for x in cfg.get("tau_buckets", [1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2])]
    buckets = sorted(set(buckets))
    x_abs = np.abs(np.log(points["moneyness"].to_numpy(dtype=np.float64)))
    tau = points["tau"].to_numpy(dtype=np.float64)
    hard_x = float(cfg.get("regions", {}).get("hard_x_abs", 0.03))
    out: dict[str, dict] = {}
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (tau >= lo) & (tau < hi) & (x_abs < hard_x)
        label = f"[{lo:g},{hi:g})"
        out[label] = {"n": int(np.sum(mask))}
        if np.any(mask):
            out[label]["gamma"] = _metric(pred["gamma"][mask], refs["gamma"][mask])
            out[label]["delta"] = _metric(pred["delta"][mask], refs["delta"][mask])
            out[label]["price"] = _metric(pred["price"][mask], refs["price"][mask])
    return out


def run_extreme_short_diagnostics(
    *,
    config_path: Path,
    tau_points: int | None = None,
    x_points: int | None = None,
    no_pde: bool = False,
) -> Path:
    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global", {})
    dtype = _parse_dtype(str(global_cfg.get("dtype", "float64")))
    device_pref = str(global_cfg.get("device", "auto"))
    points = _build_extreme_short_grid(cfg, tau_points=tau_points, x_points=x_points)
    points_df = pd.DataFrame(points, columns=FEATURE_ORDER)
    points_df["log_moneyness"] = np.log(points_df["moneyness"].to_numpy(dtype=np.float64))
    masks = _region_masks(points_df, cfg)

    refs = _cf_refs(
        points,
        settings=_cf_settings(cfg),
        option_type=str(cfg.get("diagnostics", {}).get("option_type", "put")),
    )

    baseline_cfg = cfg.get("baseline", {})
    baseline = load_pinn_price_adapter(
        project_root=PROJECT_ROOT,
        run_dir=str(baseline_cfg.get("run_dir", "PINN_mix_scaled_param")),
        checkpoint_name=str(baseline_cfg.get("checkpoint_name", "model_best.pt")),
        architecture_config_path=baseline_cfg.get("architecture_config"),
        device=device_pref,
        dtype=dtype,
        feature_order=baseline_cfg.get("feature_order"),
    )

    acv_cfg = cfg.get("acv", {})
    acv_run_dir = _resolve_run_dir(acv_cfg.get("run_dir", "acv_hard_patch_control_variate"))
    acv_checkpoint = _resolve_path(
        acv_cfg.get("checkpoint_name", "model_best.pt"),
        base_dir=acv_run_dir / "checkpoints",
    )
    acv_config = _resolve_path(acv_cfg.get("acv_config"), base_dir=acv_run_dir)
    if acv_config is None:
        acv_config = acv_run_dir / "run_config.yaml"
    if acv_checkpoint is None or not acv_checkpoint.exists():
        raise FileNotFoundError(f"ACV checkpoint not found: {acv_checkpoint}")
    loaded_acv = load_acv_hard_patch_checkpoint(
        project_root=PROJECT_ROOT,
        config_path=acv_config,
        checkpoint_path=acv_checkpoint,
        device=device_pref,
        dtype=dtype,
    )
    acv_adapter = ACVHardPatchPriceAdapter(
        model=loaded_acv.model,
        dtype=dtype,
        device=loaded_acv.device,
    )

    baseline_pred = _predict_derivative_metrics(
        price_fn=baseline.price_fn,
        points=points,
        dtype=dtype,
        device=baseline.device,
        chunk_cfg=global_cfg,
    )
    acv_pred = _predict_derivative_metrics(
        price_fn=acv_adapter,
        points=points,
        dtype=dtype,
        device=loaded_acv.device,
        chunk_cfg=global_cfg,
    )

    baseline_pde = None
    acv_pde = None
    if not no_pde and bool(cfg.get("diagnostics", {}).get("compute_pde_residual", True)):
        baseline_pde = _pde_residuals(
            price_fn=_baseline_batch_price_fn(baseline),
            points=points,
            dtype=dtype,
            device=baseline.device,
            chunk_size=int(global_cfg.get("chunk_size_pde", 128)),
        )
        acv_pde = _pde_residuals(
            price_fn=loaded_acv.model,
            points=points,
            dtype=dtype,
            device=loaded_acv.device,
            chunk_size=int(global_cfg.get("chunk_size_pde", 128)),
        )

    output_dir = _resolve_path(
        cfg.get("outputs", {}).get("output_dir", "outputs/pinn/acv_hard_patch_control_variate/extreme_short"),
        base_dir=PROJECT_ROOT,
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = points_df.copy()
    for key, value in refs.items():
        detail[f"{key}_ref"] = value
    for model_name, pred, pde in (
        ("baseline", baseline_pred, baseline_pde),
        ("acv", acv_pred, acv_pde),
    ):
        for key, value in pred.items():
            detail[f"{model_name}_{key}_pred"] = value
            detail[f"{model_name}_{key}_error"] = value - refs[key]
        if pde is not None:
            detail[f"{model_name}_pde_residual"] = pde
    detail_path = output_dir / "extreme_short_diagnostics.csv"
    detail.to_csv(detail_path, index=False)

    summary = {
        "config_path": str(config_path),
        "n_points": int(points.shape[0]),
        "detail_file": str(detail_path),
        "baseline": {
            "run_dir": str(baseline.run_dir),
            "checkpoint": str(baseline.checkpoint_path),
            "metrics": _model_summary(pred=baseline_pred, refs=refs, pde_residual=baseline_pde, masks=masks),
            "tau_buckets_hard_x": _tau_bucket_summary(points=points_df, pred=baseline_pred, refs=refs, cfg=cfg),
        },
        "acv": {
            "run_dir": str(acv_run_dir),
            "checkpoint": str(acv_checkpoint),
            "config": str(acv_config),
            "metrics": _model_summary(pred=acv_pred, refs=refs, pde_residual=acv_pde, masks=masks),
            "tau_buckets_hard_x": _tau_bucket_summary(points=points_df, pred=acv_pred, refs=refs, cfg=cfg),
        },
    }
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"[EXTREME] wrote {summary_path}")
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline vs ACV on an extreme-short tau grid.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tau-points", type=int, default=None, help="Optional tau grid override for smoke runs.")
    parser.add_argument("--x-points", type=int, default=None, help="Optional log-moneyness grid override for smoke runs.")
    parser.add_argument("--no-pde", action="store_true", help="Skip PDE residual diagnostics.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_extreme_short_diagnostics(
        config_path=args.config,
        tau_points=args.tau_points,
        x_points=args.x_points,
        no_pde=bool(args.no_pde),
    )


if __name__ == "__main__":
    main()
