from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.greeks.chain_rule import apply_moneyness_to_spot_chain_rule
from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian
from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.greeks.names import build_greek_index_spec, parse_feature_order
from src.greeks.nn_adapter import FEATURE_ORDER, load_nn_price_adapter
from src.solvers.bs import BS_price_torch


class ANNIVBlackScholesPriceAdapter:
    def __init__(
        self,
        *,
        iv_fn,
        feature_order: Sequence[str],
        strike: float,
        option_type: str,
        spot_feature: str,
        tau_feature: str,
        rate_feature: str,
    ) -> None:
        self.iv_fn = iv_fn
        self.feature_order = list(feature_order)
        self.strike = float(strike)
        self.option_type = str(option_type)
        self.idx_spot = self.feature_order.index(spot_feature)
        self.idx_tau = self.feature_order.index(tau_feature)
        self.idx_rate = self.feature_order.index(rate_feature)
        self.spot_feature = spot_feature

    def __call__(self, x_raw: torch.Tensor) -> torch.Tensor:
        sigma = self.iv_fn(x_raw)
        spot_raw = x_raw[self.idx_spot]
        if self.spot_feature == "moneyness":
            s0 = spot_raw * self.strike
        else:
            s0 = spot_raw
        return BS_price_torch(
            S0=s0,
            K=torch.as_tensor(self.strike, dtype=x_raw.dtype, device=x_raw.device),
            tau=x_raw[self.idx_tau],
            sigma=sigma,
            r=x_raw[self.idx_rate],
            opt_type=self.option_type,
        )


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _parse_dtype(raw: str) -> torch.dtype:
    key = str(raw).strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("dtype must be one of {float64,float32}")


def _parse_feature_order(raw: str | None) -> list[str]:
    if raw is None:
        return list(FEATURE_ORDER)
    return parse_feature_order([part.strip() for part in raw.split(",") if part.strip()])


def _parse_fixed_values(raw: str | None) -> dict[str, float]:
    defaults = {
        "rho": -0.7,
        "kappa": 2.0,
        "gamma": 0.3,
        "bar_v": 0.04,
        "v0": 0.04,
        "r": 0.01,
    }
    if raw is None or not raw.strip():
        return defaults
    out = dict(defaults)
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", maxsplit=1)
        out[key.strip()] = float(value)
    return out


def _build_grid(args: argparse.Namespace, feature_order: list[str]) -> pd.DataFrame:
    fixed = _parse_fixed_values(args.fixed)
    m_vals = np.linspace(float(args.m_min), float(args.m_max), int(args.m_points), dtype=np.float64)
    tau_vals = np.linspace(float(args.tau_min), float(args.tau_max), int(args.tau_points), dtype=np.float64)
    mm, tt = np.meshgrid(m_vals, tau_vals, indexing="ij")

    rows = []
    for m, tau in zip(mm.reshape(-1), tt.reshape(-1)):
        row = dict(fixed)
        row["moneyness"] = float(m)
        row["tau"] = float(tau)
        rows.append({name: float(row[name]) for name in feature_order})
    return pd.DataFrame(rows, columns=feature_order)


def _load_inputs(args: argparse.Namespace, feature_order: list[str]) -> pd.DataFrame:
    if args.input_csv:
        path = _resolve_path(args.input_csv)
        df = pd.read_csv(path)
        missing = [col for col in feature_order if col not in df.columns]
        if missing:
            raise KeyError(f"Input CSV is missing required feature columns: {missing}")
        df = df.loc[:, feature_order].copy()
    else:
        df = _build_grid(args, feature_order)

    if args.max_rows is not None and int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        df = df.iloc[: int(args.max_rows)].reset_index(drop=True)
    return df


def _reference_heston(
    *,
    x_np: np.ndarray,
    feature_order: list[str],
    option_type: str,
    strike: float,
    spot_feature: str,
    settings: HestonCFGreeksSettings,
) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(feature_order)}
    out = {key: np.empty(x_np.shape[0], dtype=np.float64) for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    for i, row in enumerate(x_np):
        spot_raw = float(row[idx[spot_feature]])
        s0 = spot_raw * strike if spot_feature == "moneyness" else spot_raw
        ref = heston_cf_greeks_scalar(
            option_type=option_type,
            S0=s0,
            K=strike,
            tau=float(row[idx["tau"]]),
            r=float(row[idx["r"]]),
            rho=float(row[idx["rho"]]),
            kappa=float(row[idx["kappa"]]),
            gamma=float(row[idx["gamma"]]),
            bar_v=float(row[idx["bar_v"]]),
            v0=float(row[idx["v0"]]),
            settings=settings,
        )
        for key in out:
            out[key][i] = float(ref[key])
    return out


def _metrics(pred: np.ndarray, ref: np.ndarray, *, mape_floor: float) -> dict[str, float]:
    err = pred - ref
    mse = float(np.mean(err**2))
    denom = np.maximum(np.abs(ref), float(mape_floor))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(100.0 * np.mean(np.abs(err) / denom)),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate ANN-IV price and first-order Greeks through Black-Scholes composition."
    )
    p.add_argument("--model-dir", default="latest")
    p.add_argument("--checkpoint-name", default="model_best.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--feature-order", default=None)

    p.add_argument("--input-csv", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--fixed", default=None, help="Comma-separated fixed values, e.g. rho=-0.7,kappa=2.0")
    p.add_argument("--m-min", type=float, default=0.8)
    p.add_argument("--m-max", type=float, default=1.2)
    p.add_argument("--m-points", type=int, default=21)
    p.add_argument("--tau-min", type=float, default=0.05)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--tau-points", type=int, default=21)

    p.add_argument("--spot-feature", default="moneyness")
    p.add_argument("--vol-feature", default="v0")
    p.add_argument("--tau-feature", default="tau")
    p.add_argument("--rate-feature", default="r")
    p.add_argument("--theta-sign", default="minus_dv_dtau", choices=["minus_dv_dtau", "dv_dtau"])
    p.add_argument("--strike", type=float, default=1.0)
    p.add_argument("--option-type", default="put", choices=["put", "call"])

    p.add_argument("--cf-u-min", type=float, default=1.0e-6)
    p.add_argument("--cf-u-max", type=float, default=200.0)
    p.add_argument("--cf-n-u", type=int, default=1200)
    p.add_argument("--mape-floor", type=float, default=1.0e-4)
    p.add_argument("--chunk-size-values", type=int, default=4096)
    p.add_argument("--chunk-size-jac", type=int, default=512)
    p.add_argument("--chunk-size-hess", type=int, default=64)
    p.add_argument("--output-dir", default="outputs/ann_iv_greeks")
    p.add_argument("--run-name", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    dtype = _parse_dtype(args.dtype)
    feature_order = _parse_feature_order(args.feature_order)

    loaded = load_nn_price_adapter(
        project_root=PROJECT_ROOT,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        dtype=dtype,
        feature_order=feature_order,
    )
    df = _load_inputs(args, feature_order)
    x_np = df.to_numpy(dtype=np.float64)
    x_t = torch.from_numpy(x_np).to(device=loaded.device, dtype=dtype)

    price_fn = ANNIVBlackScholesPriceAdapter(
        iv_fn=loaded.price_fn,
        feature_order=feature_order,
        strike=float(args.strike),
        option_type=args.option_type,
        spot_feature=args.spot_feature,
        tau_feature=args.tau_feature,
        rate_feature=args.rate_feature,
    )
    diff = derivatives_batch(
        price_fn,
        x_t,
        chunk_size_values=args.chunk_size_values,
        chunk_size_jac=args.chunk_size_jac,
        chunk_size_hess=args.chunk_size_hess,
        dtype=dtype,
        device=loaded.device,
    )

    jac = diff.jacobian.detach().cpu()
    hess = diff.hessian.detach().cpu()
    spec = build_greek_index_spec(
        feature_order,
        spot_feature=args.spot_feature,
        vol_feature=args.vol_feature,
        tau_feature=args.tau_feature,
        rate_feature=args.rate_feature,
    )
    if args.spot_feature == "moneyness":
        jac, hess = apply_moneyness_to_spot_chain_rule(
            jacobian_wrt_m=jac,
            hessian_wrt_m=hess,
            idx_moneyness=spec.idx_spot,
            strike=float(args.strike),
        )
    pred_greeks = greeks_from_jacobian_hessian(
        jac,
        hess,
        idx_spot=spec.idx_spot,
        idx_vol=spec.idx_vol,
        idx_tau=spec.idx_tau,
        idx_rate=spec.idx_rate,
        theta_is_minus_dv_dtau=(args.theta_sign == "minus_dv_dtau"),
    )
    pred = {
        "price": diff.values.detach().cpu().numpy().reshape(-1),
        **{key: value.detach().cpu().numpy().reshape(-1) for key, value in pred_greeks.items()},
    }

    cf_settings = HestonCFGreeksSettings(
        u_min=float(args.cf_u_min),
        u_max=float(args.cf_u_max),
        n_u=int(args.cf_n_u),
    )
    ref = _reference_heston(
        x_np=x_np,
        feature_order=feature_order,
        option_type=args.option_type,
        strike=float(args.strike),
        spot_feature=args.spot_feature,
        settings=cf_settings,
    )

    out_df = df.copy()
    metric_rows = []
    primary = {"price", "delta", "vega", "theta", "rho"}
    for name in ["price", "delta", "vega", "theta", "rho", "gamma"]:
        if name not in pred or name not in ref:
            continue
        out_df[f"{name}_ann_iv"] = pred[name]
        out_df[f"{name}_heston_cf"] = ref[name]
        out_df[f"{name}_error"] = pred[name] - ref[name]
        metric_rows.append(
            {
                "quantity": name,
                "primary": bool(name in primary),
                **_metrics(pred[name], ref[name], mape_floor=float(args.mape_floor)),
            }
        )

    run_name = args.run_name or loaded.run_dir.name
    out_dir = _resolve_path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / "points_ann_iv_greeks.csv"
    metrics_path = out_dir / "metrics_ann_iv_greeks.csv"
    summary_path = out_dir / "summary.yaml"
    out_df.to_csv(points_path, index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "model_run_dir": str(loaded.run_dir),
                "rows": int(len(out_df)),
                "feature_order": feature_order,
                "primary_quantities": sorted(primary),
                "gamma_note": "Gamma is diagnostic only in the first ANN-IV Sobolev experiment.",
                "points_csv": str(points_path),
                "metrics_csv": str(metrics_path),
            },
            f,
            sort_keys=False,
        )

    print(f"Rows: {len(out_df)}")
    print(f"Saved points: {points_path}")
    print(f"Saved metrics: {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
