from __future__ import annotations

import argparse
import copy
import os
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_acv_extreme_short_diagnostics import (
    FEATURE_ORDER,
    _abs_metric,
    _baseline_batch_price_fn,
    _build_extreme_short_grid,
    _cf_refs,
    _cf_settings,
    _load_yaml,
    _metric,
    _parse_dtype,
    _pde_residuals,
    _predict_derivative_metrics,
    _region_masks,
    _resolve_path,
)
from src.greeks.pinn_adapter import load_pinn_price_adapter
from src.pinn.acv_hard_patch import ACVHardPatchPriceAdapter, build_acv_hard_patch_model


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "acv_gate_sensitivity.yaml"


def _as_float_list(raw, *, name: str) -> list[float]:
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"sweep.{name} must be a non-empty list")
    return [float(x) for x in raw]


def _model_region_metrics(
    *,
    pred: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    pde_residual: np.ndarray | None,
    masks: dict[str, np.ndarray],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for region, mask in masks.items():
        out[region] = {"n": int(np.sum(mask))}
        for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
            out[region][key] = _metric(pred[key][mask], refs[key][mask])
        if pde_residual is not None:
            out[region]["pde_residual"] = _abs_metric(pde_residual[mask])
    return out


def _flatten_row(
    *,
    gate: dict[str, float],
    metrics: dict[str, dict],
    accepted: bool,
    score: float,
) -> dict[str, float | int | bool]:
    hard = metrics["hard"]
    full = metrics["full_extreme_short"]
    near = metrics["hard_near_zero"]
    row: dict[str, float | int | bool] = {
        **gate,
        "accepted": bool(accepted),
        "score": float(score),
        "full_price_rmse": full["price"]["rmse"],
        "full_gamma_rmse": full["gamma"]["rmse"],
        "full_gamma_p99": full["gamma"]["p99_abs_error"],
        "hard_price_rmse": hard["price"]["rmse"],
        "hard_delta_rmse": hard["delta"]["rmse"],
        "hard_gamma_rmse": hard["gamma"]["rmse"],
        "hard_gamma_p99": hard["gamma"]["p99_abs_error"],
        "near_zero_gamma_rmse": near["gamma"]["rmse"],
        "near_zero_gamma_p99": near["gamma"]["p99_abs_error"],
    }
    if "pde_residual" in hard:
        row["hard_pde_rmse"] = hard["pde_residual"]["rmse"]
    if "pde_residual" in full:
        row["full_pde_rmse"] = full["pde_residual"]["rmse"]
    return row


def _acceptance_and_score(metrics: dict[str, dict], cfg: dict) -> tuple[bool, float]:
    acc = cfg.get("acceptance", {})
    hard = metrics["hard"]
    full = metrics["full_extreme_short"]
    hard_pde = hard.get("pde_residual", {}).get("rmse", 0.0)
    checks = [
        hard["gamma"]["rmse"] <= float(acc.get("hard_gamma_rmse_max", 1.5)),
        hard["gamma"]["p99_abs_error"] <= float(acc.get("hard_gamma_p99_max", 3.0)),
        hard["delta"]["rmse"] <= float(acc.get("hard_delta_rmse_max", 0.03)),
        full["price"]["rmse"] <= float(acc.get("full_price_rmse_max", 0.0047)),
        full["gamma"]["rmse"] <= float(acc.get("full_gamma_rmse_max", 0.535)),
        hard_pde <= float(acc.get("hard_pde_rmse_max", 0.05)),
    ]
    accepted = bool(all(checks))
    penalty = 0.0
    limits = {
        "hard_gamma_rmse": float(acc.get("hard_gamma_rmse_max", 1.5)),
        "hard_gamma_p99": float(acc.get("hard_gamma_p99_max", 3.0)),
        "hard_delta_rmse": float(acc.get("hard_delta_rmse_max", 0.03)),
        "full_price_rmse": float(acc.get("full_price_rmse_max", 0.0047)),
        "full_gamma_rmse": float(acc.get("full_gamma_rmse_max", 0.535)),
        "hard_pde_rmse": float(acc.get("hard_pde_rmse_max", 0.05)),
    }
    values = {
        "hard_gamma_rmse": hard["gamma"]["rmse"],
        "hard_gamma_p99": hard["gamma"]["p99_abs_error"],
        "hard_delta_rmse": hard["delta"]["rmse"],
        "full_price_rmse": full["price"]["rmse"],
        "full_gamma_rmse": full["gamma"]["rmse"],
        "hard_pde_rmse": hard_pde,
    }
    for key, limit in limits.items():
        penalty += max(0.0, values[key] - limit) / max(limit, 1.0e-12)
    score = (
        hard["gamma"]["rmse"]
        + 0.10 * hard["gamma"]["p99_abs_error"]
        + 5.0 * full["price"]["rmse"]
        + 100.0 * penalty
    )
    return accepted, float(score)


def _with_gate(base_cfg: dict, gate: dict[str, float], *, run_name: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("patch", {}).setdefault("gate", {})
    cfg["patch"]["gate"].update(gate)
    cfg.setdefault("training", {})
    cfg["training"]["save_initial_checkpoint_only"] = True
    cfg["training"]["stages"] = []
    cfg.setdefault("outputs", {})
    cfg["outputs"]["run_name"] = run_name
    return cfg


def run_gate_sensitivity(
    *,
    config_path: Path,
    tau_points: int | None = None,
    x_points: int | None = None,
    max_combos: int | None = None,
    no_pde: bool = False,
) -> Path:
    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global", {})
    dtype = _parse_dtype(str(global_cfg.get("dtype", "float64")))
    device_pref = str(global_cfg.get("device", "auto"))
    output_dir = _resolve_path(
        cfg.get("outputs", {}).get("output_dir", "outputs/pinn/acv_hard_patch_control_variate/gate_sensitivity"),
        base_dir=PROJECT_ROOT,
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = _resolve_path(
        cfg.get("acv", {}).get("base_config", "configs/acv_hard_patch_control_variate.yaml"),
        base_dir=PROJECT_ROOT,
    )
    if base_config_path is None or not base_config_path.exists():
        raise FileNotFoundError(f"ACV base config not found: {base_config_path}")
    base_acv_cfg = _load_yaml(base_config_path)

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
    baseline_pred = _predict_derivative_metrics(
        price_fn=baseline.price_fn,
        points=points,
        dtype=dtype,
        device=baseline.device,
        chunk_cfg=global_cfg,
    )
    baseline_pde = None
    if not no_pde and bool(cfg.get("diagnostics", {}).get("compute_pde_residual", True)):
        baseline_pde = _pde_residuals(
            price_fn=_baseline_batch_price_fn(baseline),
            points=points,
            dtype=dtype,
            device=baseline.device,
            chunk_size=int(global_cfg.get("chunk_size_pde", 128)),
        )
    baseline_metrics = _model_region_metrics(
        pred=baseline_pred,
        refs=refs,
        pde_residual=baseline_pde,
        masks=masks,
    )

    sweep = cfg.get("sweep", {})
    combos = list(
        product(
            _as_float_list(sweep.get("x_center", [0.04, 0.05, 0.06, 0.08]), name="x_center"),
            _as_float_list(sweep.get("tau_center", [0.05, 0.08, 0.10]), name="tau_center"),
            _as_float_list(sweep.get("delta_x", [0.01]), name="delta_x"),
            _as_float_list(sweep.get("delta_tau", [0.015]), name="delta_tau"),
        )
    )
    if max_combos is not None:
        combos = combos[: int(max_combos)]
    if not combos:
        raise ValueError("No gate combinations to evaluate")

    rows: list[dict] = []
    metrics_by_gate: dict[str, dict] = {}
    for idx, (x_center, tau_center, delta_x, delta_tau) in enumerate(combos, start=1):
        gate = {
            "x_center": float(x_center),
            "tau_center": float(tau_center),
            "delta_x": float(delta_x),
            "delta_tau": float(delta_tau),
        }
        run_name = (
            f"acv_cv_gate_x{x_center:g}_t{tau_center:g}"
            f"_dx{delta_x:g}_dt{delta_tau:g}"
        ).replace(".", "p")
        acv_cfg = _with_gate(base_acv_cfg, gate, run_name=run_name)
        loaded = build_acv_hard_patch_model(
            project_root=PROJECT_ROOT,
            config=acv_cfg,
            device=device_pref,
            dtype=dtype,
        )
        adapter = ACVHardPatchPriceAdapter(model=loaded.model, dtype=dtype, device=loaded.device)
        pred = _predict_derivative_metrics(
            price_fn=adapter,
            points=points,
            dtype=dtype,
            device=loaded.device,
            chunk_cfg=global_cfg,
        )
        pde = None
        if not no_pde and bool(cfg.get("diagnostics", {}).get("compute_pde_residual", True)):
            pde = _pde_residuals(
                price_fn=loaded.model,
                points=points,
                dtype=dtype,
                device=loaded.device,
                chunk_size=int(global_cfg.get("chunk_size_pde", 128)),
            )
        metrics = _model_region_metrics(pred=pred, refs=refs, pde_residual=pde, masks=masks)
        accepted, score = _acceptance_and_score(metrics, cfg)
        row = _flatten_row(gate=gate, metrics=metrics, accepted=accepted, score=score)
        row["combo_index"] = idx
        rows.append(row)
        key = f"combo_{idx:03d}"
        metrics_by_gate[key] = {"gate": gate, "accepted": accepted, "score": score, "metrics": metrics}
        print(
            "[GATE] "
            f"{idx}/{len(combos)} x={x_center:g} tau={tau_center:g} "
            f"dx={delta_x:g} dt={delta_tau:g} "
            f"hard_gamma={row['hard_gamma_rmse']:.6g} "
            f"accepted={accepted}"
        )

    results = pd.DataFrame(rows).sort_values(
        by=["accepted", "score", "hard_gamma_rmse"],
        ascending=[False, True, True],
    )
    results_path = output_dir / "gate_sensitivity.csv"
    results.to_csv(results_path, index=False)
    best_row = results.iloc[0].to_dict()
    best_gate = {
        "x_center": float(best_row["x_center"]),
        "tau_center": float(best_row["tau_center"]),
        "delta_x": float(best_row["delta_x"]),
        "delta_tau": float(best_row["delta_tau"]),
    }
    best_config = _with_gate(
        base_acv_cfg,
        best_gate,
        run_name=str(cfg.get("outputs", {}).get("best_run_name", "acv_hard_patch_control_variate_best_gate")),
    )
    best_config_path = output_dir / "best_gate_config.yaml"
    with open(best_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)

    summary = {
        "config_path": str(config_path),
        "base_config": str(base_config_path),
        "n_points": int(points.shape[0]),
        "n_combos": int(len(combos)),
        "results_file": str(results_path),
        "best_gate_config": str(best_config_path),
        "baseline": baseline_metrics,
        "best": {
            "gate": best_gate,
            "accepted": bool(best_row["accepted"]),
            "score": float(best_row["score"]),
            "metrics_flat": {
                key: float(value) if isinstance(value, (float, np.floating)) else value
                for key, value in best_row.items()
                if key not in {"accepted"}
            },
        },
        "combos": metrics_by_gate,
    }
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"[GATE] wrote {summary_path}")
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep ACV control-variate gate parameters.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tau-points", type=int, default=None, help="Optional tau grid override for smoke runs.")
    parser.add_argument("--x-points", type=int, default=None, help="Optional log-moneyness grid override for smoke runs.")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional combination cap for smoke runs.")
    parser.add_argument("--no-pde", action="store_true", help="Skip PDE residual metrics.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_gate_sensitivity(
        config_path=args.config,
        tau_points=args.tau_points,
        x_points=args.x_points,
        max_combos=args.max_combos,
        no_pde=bool(args.no_pde),
    )


if __name__ == "__main__":
    main()
