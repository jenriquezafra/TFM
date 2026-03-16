from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.greeks.chain_rule import apply_moneyness_to_spot_chain_rule
from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian
from src.greeks.names import DEFAULT_FEATURE_ORDER, build_greek_index_spec, parse_feature_order
from src.greeks.nn_adapter import load_nn_price_adapter


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


def _parse_feature_order(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_FEATURE_ORDER)
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parse_feature_order(parts)


def _parse_feature_vector(raw: str, n_features: int) -> list[float]:
    values = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if len(values) != n_features:
        raise ValueError(
            f"`--features` must contain {n_features} comma-separated values; got {len(values)}"
        )
    return values


def _load_input_dataframe(args: argparse.Namespace, feature_order: list[str]) -> pd.DataFrame:
    if args.input_csv is not None:
        csv_path = Path(args.input_csv)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in feature_order if c not in df.columns]
        if missing:
            raise KeyError(f"Input CSV is missing required feature columns: {missing}")
        return df.loc[:, feature_order].copy()

    if args.features is not None:
        values = _parse_feature_vector(args.features, n_features=len(feature_order))
        return pd.DataFrame([values], columns=feature_order)

    raise ValueError("Provide either --input-csv or --features")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _default_output_path(*, run_name: str, user_output: str | None) -> Path:
    if user_output:
        out = Path(user_output)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        return out

    out_dir = PROJECT_ROOT / "outputs" / "greeks"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"greeks_{_safe_name(run_name)}.csv"


def _add_jacobian_columns(
    out_df: pd.DataFrame,
    jacobian: np.ndarray,
    feature_order: list[str],
) -> None:
    for idx, feature in enumerate(feature_order):
        out_df[f"jac_{feature}"] = jacobian[:, idx]


def _add_hessian_columns(
    out_df: pd.DataFrame,
    hessian: np.ndarray,
    feature_order: list[str],
) -> None:
    n_features = len(feature_order)
    for i in range(n_features):
        fi = feature_order[i]
        for j in range(n_features):
            fj = feature_order[j]
            out_df[f"hess_{fi}__{fj}"] = hessian[:, i, j]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute value/jacobian/hessian/greeks on a trained ANN model."
    )
    parser.add_argument("--model-dir", default="latest", help="Run directory name under outputs/runs")
    parser.add_argument("--checkpoint-name", default="model_best.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])

    parser.add_argument(
        "--feature-order",
        default=None,
        help="Comma-separated feature names. Default is training order.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", default=None, help="CSV with feature columns")
    source.add_argument(
        "--features",
        default=None,
        help="Single point as comma-separated values following --feature-order",
    )

    parser.add_argument("--spot-feature", default="moneyness")
    parser.add_argument("--vol-feature", default=None)
    parser.add_argument("--tau-feature", default="tau")
    parser.add_argument("--rate-feature", default="r")
    parser.add_argument(
        "--theta-sign",
        default="minus_dv_dtau",
        choices=["minus_dv_dtau", "dv_dtau"],
        help="Convention for theta.",
    )

    parser.add_argument(
        "--strike",
        type=float,
        default=None,
        help="If set and spot-feature is moneyness, converts derivatives to spot-space (m=S/K).",
    )

    parser.add_argument("--chunk-size-values", type=int, default=None)
    parser.add_argument("--chunk-size-jac", type=int, default=None)
    parser.add_argument("--chunk-size-hess", type=int, default=32)

    parser.add_argument(
        "--no-hessian-columns",
        action="store_true",
        help="Skip exporting full Hessian matrix columns.",
    )
    parser.add_argument("--output-csv", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

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

    df_in = _load_input_dataframe(args, feature_order=feature_order)
    x_np = df_in.to_numpy(dtype=np.float64)
    x_t = torch.from_numpy(x_np).to(device=loaded.device, dtype=dtype)

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
        feature_order,
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

    out_df = df_in.copy()
    out_df["value"] = values

    jac_np = jacobian.numpy()
    hess_np = hessian.numpy()
    _add_jacobian_columns(out_df, jacobian=jac_np, feature_order=feature_order)

    if not args.no_hessian_columns:
        _add_hessian_columns(out_df, hessian=hess_np, feature_order=feature_order)

    for greek_name, greek_tensor in greek_map.items():
        out_df[greek_name] = greek_tensor.detach().cpu().numpy().reshape(-1)

    if args.strike is not None and args.spot_feature == "moneyness":
        out_df["spot_from_moneyness_strike"] = float(args.strike)

    out_path = _default_output_path(run_name=loaded.run_dir.name, user_output=args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Run dir: {loaded.run_dir}")
    print(f"Device: {loaded.device}")
    print(f"Rows: {len(out_df)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
