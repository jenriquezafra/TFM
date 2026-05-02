from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian
from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.pinn.acv_hard_patch import (
    ACVHardPatchPriceAdapter,
    heston_log_pde_residual,
    load_acv_hard_patch_checkpoint,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "acv_hard_patch_diagnostics.yaml"
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


def _resolve_run_dir(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return (PROJECT_ROOT / path).resolve()
    return (PROJECT_ROOT / "outputs" / "pinn" / path).resolve()


def _resolve_path(raw: str | Path | None, *, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / path).resolve()


def _build_surface_grid(diagnostics_cfg: dict) -> np.ndarray:
    grid = diagnostics_cfg.get("surface_grid", {})
    x_feature = str(grid.get("x_feature", "moneyness"))
    y_feature = str(grid.get("y_feature", "tau"))
    if x_feature not in FEATURE_ORDER or y_feature not in FEATURE_ORDER:
        raise KeyError(f"surface features must be in {FEATURE_ORDER}")
    x_min, x_max = float(grid.get("x_min", 0.8)), float(grid.get("x_max", 1.2))
    y_min, y_max = float(grid.get("y_min", 0.005)), float(grid.get("y_max", 2.0))
    x_points, y_points = int(grid.get("x_points", 181)), int(grid.get("y_points", 181))
    fixed = dict(DEFAULT_FIXED_VALUES)
    fixed.update(diagnostics_cfg.get("fixed_values", {}) or {})
    base = np.array([float(fixed[name]) for name in FEATURE_ORDER], dtype=np.float64)
    x_axis = np.linspace(x_min, x_max, x_points, dtype=np.float64)
    y_axis = np.linspace(y_min, y_max, y_points, dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    out = np.repeat(base.reshape(1, -1), repeats=xx.size, axis=0)
    out[:, FEATURE_ORDER.index(x_feature)] = xx.reshape(-1)
    out[:, FEATURE_ORDER.index(y_feature)] = yy.reshape(-1)
    return out


def _region_masks(points: pd.DataFrame, diagnostics_cfg: dict) -> dict[str, np.ndarray]:
    hard = diagnostics_cfg.get("hard_region", {})
    eps_m = float(hard.get("epsilon_m", 0.03))
    eps_tau = float(hard.get("epsilon_tau", 0.05))
    m = points["moneyness"].to_numpy(dtype=np.float64)
    tau = points["tau"].to_numpy(dtype=np.float64)
    atm = np.abs(np.log(np.maximum(m, np.finfo(np.float64).tiny))) < eps_m
    short = tau < eps_tau
    return {
        "full": np.ones(points.shape[0], dtype=bool),
        "short_maturity": short,
        "atm": atm,
        "hard": short & atm,
    }


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "p99_abs_error": float("nan"), "max_abs_error": float("nan")}
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "p90_abs_error": float(np.percentile(abs_err, 90.0)),
        "p99_abs_error": float(np.percentile(abs_err, 99.0)),
        "max_abs_error": float(np.max(abs_err)),
    }


def _abs_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "p99_abs_error": float("nan"), "max_abs_error": float("nan")}
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


def _cf_refs(points: np.ndarray, settings: HestonCFGreeksSettings, option_type: str) -> dict[str, np.ndarray]:
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


def _pde_residuals(
    *,
    model,
    points: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        raw = torch.as_tensor(points[start:stop], dtype=dtype, device=device)
        with torch.enable_grad():
            residual, _ = heston_log_pde_residual(price_fn=model, raw=raw, scale_epsilon=None)
        out.append(residual.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, axis=0)


def run_diagnostics(
    *,
    config_path: Path,
    x_points: int | None = None,
    y_points: int | None = None,
    no_pde: bool = False,
) -> Path:
    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global", {})
    diagnostics_cfg = cfg.get("diagnostics", {})
    if x_points is not None or y_points is not None:
        diagnostics_cfg = dict(diagnostics_cfg)
        grid_cfg = dict(diagnostics_cfg.get("surface_grid", {}))
        if x_points is not None:
            grid_cfg["x_points"] = int(x_points)
        if y_points is not None:
            grid_cfg["y_points"] = int(y_points)
        diagnostics_cfg["surface_grid"] = grid_cfg
    if no_pde:
        diagnostics_cfg = dict(diagnostics_cfg)
        diagnostics_cfg["compute_pde_residual"] = False
    dtype = _parse_dtype(str(global_cfg.get("dtype", "float64")))
    device_pref = str(global_cfg.get("device", "auto"))
    run_dir = _resolve_run_dir(global_cfg.get("run_dir", "acv_hard_patch_experimental"))
    checkpoint = _resolve_path(global_cfg.get("checkpoint_name", "model_best.pt"), base_dir=run_dir / "checkpoints")
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"ACV checkpoint not found: {checkpoint}")
    acv_config = _resolve_path(global_cfg.get("acv_config", None), base_dir=run_dir)
    if acv_config is None:
        acv_config = run_dir / "run_config.yaml"
    if not acv_config.exists():
        raise FileNotFoundError(f"ACV config not found: {acv_config}")

    loaded = load_acv_hard_patch_checkpoint(
        project_root=PROJECT_ROOT,
        config_path=acv_config,
        checkpoint_path=checkpoint,
        device=device_pref,
        dtype=dtype,
    )
    adapter = ACVHardPatchPriceAdapter(model=loaded.model, dtype=dtype, device=loaded.device)

    points = _build_surface_grid(diagnostics_cfg)
    points_df = pd.DataFrame(points, columns=FEATURE_ORDER)
    masks = _region_masks(points_df, diagnostics_cfg)

    diff = derivatives_batch(
        adapter,
        torch.as_tensor(points, dtype=dtype, device=loaded.device),
        chunk_size_values=int(global_cfg.get("chunk_size_values", 4096)),
        chunk_size_jac=int(global_cfg.get("chunk_size_jac", 512)),
        chunk_size_hess=int(global_cfg.get("chunk_size_hess", 64)),
        dtype=dtype,
        device=loaded.device,
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
    pred = {
        "price": diff.values.detach().cpu().numpy().reshape(-1),
        "delta": greeks["delta"].detach().cpu().numpy().reshape(-1),
        "gamma": greeks["gamma"].detach().cpu().numpy().reshape(-1),
        "vega": greeks["vega"].detach().cpu().numpy().reshape(-1),
        "theta": greeks["theta"].detach().cpu().numpy().reshape(-1),
        "rho": greeks["rho"].detach().cpu().numpy().reshape(-1),
    }
    refs = _cf_refs(
        points,
        settings=_cf_settings(diagnostics_cfg),
        option_type=str(diagnostics_cfg.get("option_type", "put")),
    )

    compute_pde = bool(diagnostics_cfg.get("compute_pde_residual", True))
    pde = None
    if compute_pde:
        pde = _pde_residuals(
            model=loaded.model,
            points=points,
            device=loaded.device,
            dtype=dtype,
            chunk_size=int(global_cfg.get("chunk_size_pde", 128)),
        )

    output_dir = run_dir / str(diagnostics_cfg.get("output_subdir", "acv_diagnostics"))
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = points_df.copy()
    for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
        detail[f"{key}_pred"] = pred[key]
        detail[f"{key}_ref"] = refs[key]
        detail[f"{key}_error"] = pred[key] - refs[key]
    if pde is not None:
        detail["pde_residual"] = pde
    detail_path = output_dir / "surface_diagnostics.csv"
    detail.to_csv(detail_path, index=False)

    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "acv_config": str(acv_config),
        "detail_file": str(detail_path),
        "metrics": {},
    }
    for region, mask in masks.items():
        summary["metrics"][region] = {"n": int(np.sum(mask))}
        for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
            summary["metrics"][region][key] = _metrics(pred[key][mask], refs[key][mask])
        if pde is not None:
            summary["metrics"][region]["pde_residual"] = _abs_metrics(pde[mask])

    summary_path = output_dir / "metrics.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"[ACV-DIAG] wrote {summary_path}")
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate experimental ACV-HardPatch Greeks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--x-points", type=int, default=None, help="Optional grid x-size override for smoke runs.")
    parser.add_argument("--y-points", type=int, default=None, help="Optional grid y-size override for smoke runs.")
    parser.add_argument("--no-pde", action="store_true", help="Skip PDE residual diagnostics.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_diagnostics(
        config_path=args.config,
        x_points=args.x_points,
        y_points=args.y_points,
        no_pde=bool(args.no_pde),
    )


if __name__ == "__main__":
    main()
