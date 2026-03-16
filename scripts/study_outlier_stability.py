from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.model_inference import FEATURE_ORDER, load_model_from_run, predict_iv
from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian, jacobian_batch
from src.greeks.names import build_greek_index_spec
from src.greeks.nn_adapter import LoadedNNPriceAdapter, load_nn_price_adapter


DEFAULT_MONEYNESS_BINS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
DEFAULT_TAU_BINS = [0.05, 0.25, 0.5, 1.0, 2.0, 3.0]
VEGA_FEATURE_PRIORITY = ("v0", "bar_v", "gamma")


@dataclass(frozen=True)
class AnalysisArgs:
    model_dir: str
    checkpoint_name: str
    device: str
    greeks_device: str
    greeks_dtype: torch.dtype
    splits: tuple[str, ...]
    outlier_abs_threshold: float
    outlier_quantile: float
    reference_sample_size: int
    surface_sample_size: int
    chunk_size_values: int
    chunk_size_jac: int
    chunk_size_hess: int
    save_full_eval: bool
    seed: int


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _resolve_data_dir(*, run_dir: Path) -> Path:
    run_cfg = _load_yaml_dict(run_dir / "model_training_copy.yaml")
    cfg_data_dir = run_cfg.get("data", {}).get("dir")
    if cfg_data_dir:
        return PROJECT_ROOT / cfg_data_dir

    default_cfg = _load_yaml_dict(PROJECT_ROOT / "configs" / "model_training.yaml")
    default_data_dir = default_cfg.get("data", {}).get("dir")
    if default_data_dir:
        return PROJECT_ROOT / default_data_dir
    raise FileNotFoundError("Could not resolve data dir from model_training config")


def _resolve_target_col(df: pd.DataFrame) -> str:
    if "IV" in df.columns:
        return "IV"
    return str(df.columns[-1])


def _resolve_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    if all(col in df.columns for col in FEATURE_ORDER):
        return list(FEATURE_ORDER)
    return [c for c in df.columns if c != target_col]


def _parse_splits(raw: str) -> tuple[str, ...]:
    allowed = {"train", "val", "test"}
    out = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not out:
        raise ValueError("splits must not be empty")
    invalid = [x for x in out if x not in allowed]
    if invalid:
        raise ValueError(f"Invalid splits: {invalid}. Allowed: {sorted(allowed)}")
    return out


def _parse_dtype(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("greeks-dtype must be one of {float64, float32}")


def _resolve_floor_iv(data_dir: Path) -> float:
    synth_cfg = _load_yaml_dict(data_dir / "synth_copy.yaml")
    iv_bounds = (
        synth_cfg.get("root_finder", {})
        .get("methods", {})
        .get("brent_iv", {})
        .get("iv_bounds", [1.0e-6, 5.0])
    )
    try:
        return float(iv_bounds[0])
    except Exception:
        return 1.0e-6


def _resolve_eval_bins(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = _load_yaml_dict(run_dir / "model_training_copy.yaml")
    bins_cfg = cfg.get("evaluation", {}).get("bins", {})
    m = np.asarray(bins_cfg.get("moneyness", DEFAULT_MONEYNESS_BINS), dtype=np.float64)
    t = np.asarray(bins_cfg.get("tau", DEFAULT_TAU_BINS), dtype=np.float64)
    if m.size < 2 or np.any(np.diff(m) <= 0):
        m = np.asarray(DEFAULT_MONEYNESS_BINS, dtype=np.float64)
    if t.size < 2 or np.any(np.diff(t) <= 0):
        t = np.asarray(DEFAULT_TAU_BINS, dtype=np.float64)
    return m, t


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def _with_greeks(
    *,
    base_df: pd.DataFrame,
    feature_cols: list[str],
    loaded: LoadedNNPriceAdapter,
    chunk_size_values: int,
    chunk_size_jac: int,
    chunk_size_hess: int,
    greeks_dtype: torch.dtype,
) -> pd.DataFrame:
    if base_df.empty:
        return base_df.copy()

    x_np = base_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
    x_t = torch.from_numpy(x_np).to(device=loaded.device, dtype=greeks_dtype)

    diff = derivatives_batch(
        loaded.price_fn,
        x_t,
        chunk_size_values=chunk_size_values,
        chunk_size_jac=chunk_size_jac,
        chunk_size_hess=chunk_size_hess,
        dtype=greeks_dtype,
        device=loaded.device,
    )
    jac = diff.jacobian.detach().cpu()
    hess = diff.hessian.detach().cpu()

    vega_feature = next((f for f in VEGA_FEATURE_PRIORITY if f in feature_cols), None)
    spec = build_greek_index_spec(
        feature_cols,
        spot_feature="moneyness" if "moneyness" in feature_cols else feature_cols[0],
        vol_feature=vega_feature,
        tau_feature="tau" if "tau" in feature_cols else None,
        rate_feature="r" if "r" in feature_cols else None,
    )

    greek_map = greeks_from_jacobian_hessian(
        jac,
        hess,
        idx_spot=spec.idx_spot,
        idx_vol=spec.idx_vol,
        idx_tau=spec.idx_tau,
        idx_rate=spec.idx_rate,
        theta_is_minus_dv_dtau=True,
    )

    out = base_df.copy()
    for greek_name, greek_tensor in greek_map.items():
        out[greek_name] = greek_tensor.detach().cpu().numpy().reshape(-1)
        out[f"abs_{greek_name}"] = np.abs(out[greek_name].to_numpy(dtype=np.float64))

    abs_cols = [
        c
        for c in ("abs_delta", "abs_gamma", "abs_vega", "abs_theta", "abs_rho")
        if c in out.columns
    ]
    if abs_cols:
        arr = out[abs_cols].to_numpy(dtype=np.float64)
        idx = np.argmax(arr, axis=1)
        out["dominant_greek"] = [abs_cols[i].replace("abs_", "") for i in idx]
    out["vega_feature"] = vega_feature if vega_feature is not None else ""
    return out


def _surface_gradients(
    *,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    loaded: LoadedNNPriceAdapter,
    sample_size: int,
    chunk_size_jac: int,
    greeks_dtype: torch.dtype,
    seed: int,
    moneyness_bins: np.ndarray,
    tau_bins: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_size <= 0 or df_eval.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(df_eval)
    k = min(sample_size, n)
    sample_idx = rng.choice(np.arange(n), size=k, replace=False)
    sample_df = df_eval.iloc[sample_idx].copy().reset_index(drop=True)

    x_np = sample_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
    x_t = torch.from_numpy(x_np).to(device=loaded.device, dtype=greeks_dtype)
    jac = jacobian_batch(
        loaded.price_fn,
        x_t,
        chunk_size=chunk_size_jac,
        dtype=greeks_dtype,
        device=loaded.device,
    ).detach().cpu().numpy()

    sample_df["grad_norm"] = np.linalg.norm(jac, axis=1)
    if "moneyness" in feature_cols:
        j = feature_cols.index("moneyness")
        sample_df["abs_grad_moneyness"] = np.abs(jac[:, j])
    if "tau" in feature_cols:
        j = feature_cols.index("tau")
        sample_df["abs_grad_tau"] = np.abs(jac[:, j])

    if not {"moneyness", "tau"}.issubset(sample_df.columns):
        return sample_df, pd.DataFrame()

    sample_df["m_bin"] = pd.cut(
        sample_df["moneyness"],
        bins=moneyness_bins,
        include_lowest=True,
        right=False,
    )
    sample_df["tau_bin"] = pd.cut(
        sample_df["tau"],
        bins=tau_bins,
        include_lowest=True,
        right=False,
    )
    sample_df["m_bin"] = sample_df["m_bin"].astype(str)
    sample_df["tau_bin"] = sample_df["tau_bin"].astype(str)

    agg_cols = [c for c in ["grad_norm", "abs_grad_moneyness", "abs_grad_tau", "abs_error"] if c in sample_df]
    rows: list[dict[str, Any]] = []
    for (m_bin, tau_bin), grp in sample_df.groupby(["m_bin", "tau_bin"], observed=False):
        if grp.empty:
            continue
        row: dict[str, Any] = {
            "m_bin": str(m_bin),
            "tau_bin": str(tau_bin),
            "n_samples": int(len(grp)),
        }
        for c in agg_cols:
            vals = grp[c].to_numpy(dtype=np.float64)
            row[f"{c}_p50"] = float(np.nanquantile(vals, 0.50))
            row[f"{c}_p90"] = float(np.nanquantile(vals, 0.90))
        rows.append(row)
    region_df = pd.DataFrame(rows).sort_values(["tau_bin", "m_bin"])
    return sample_df, region_df


def _analyze_split(
    *,
    split: str,
    split_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model: torch.nn.Module,
    model_device: torch.device,
    norm_stats: dict | None,
    loaded_adapter: LoadedNNPriceAdapter,
    out_dir: Path,
    floor_iv: float,
    moneyness_bins: np.ndarray,
    tau_bins: np.ndarray,
    cfg: AnalysisArgs,
    seed_offset: int,
) -> dict[str, Any]:
    x = split_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
    y_true = split_df[target_col].to_numpy(dtype=np.float64).reshape(-1)
    y_pred = predict_iv(
        model=model,
        features=x,
        device=model_device,
        batch_size=8192,
        normalization_stats=norm_stats,
    )
    residual = y_pred - y_true
    abs_error = np.abs(residual)
    sq_error = residual**2
    n = len(split_df)

    q_thr = float(np.quantile(abs_error, cfg.outlier_quantile))
    outlier_q_mask = abs_error >= q_thr
    outlier_hard_mask = abs_error > cfg.outlier_abs_threshold
    floor_mask = np.isclose(y_true, floor_iv, atol=0.0, rtol=0.0)

    if outlier_hard_mask.sum() >= 10:
        outlier_mask = outlier_hard_mask
        outlier_mode = f"abs_error_gt_{cfg.outlier_abs_threshold:.4g}"
    else:
        outlier_mask = outlier_q_mask
        outlier_mode = f"quantile_{cfg.outlier_quantile:.4g}"

    region_mask = np.zeros(n, dtype=bool)
    if "tau" in split_df.columns and "moneyness" in split_df.columns:
        tau_vals = split_df["tau"].to_numpy(dtype=np.float64)
        m_vals = split_df["moneyness"].to_numpy(dtype=np.float64)
        region_mask = (tau_vals < 0.25) & (m_vals < 0.8)

    df_eval = split_df.copy()
    df_eval["iv_true"] = y_true
    df_eval["iv_pred"] = y_pred
    df_eval["residual"] = residual
    df_eval["abs_error"] = abs_error
    df_eval["sq_error"] = sq_error
    df_eval["is_outlier"] = outlier_mask
    df_eval["is_outlier_hard"] = outlier_hard_mask
    df_eval["is_outlier_q"] = outlier_q_mask
    df_eval["is_floor_iv"] = floor_mask
    df_eval["is_region_tau_lt_0_25_m_lt_0_8"] = region_mask

    df_out_base = df_eval.loc[outlier_mask].copy()
    df_ref_base = pd.DataFrame()
    non_out_idx = np.flatnonzero(~outlier_mask)
    n_ref = min(cfg.reference_sample_size, len(non_out_idx))
    if n_ref > 0:
        rng = np.random.default_rng(cfg.seed + seed_offset)
        ref_idx = rng.choice(non_out_idx, size=n_ref, replace=False)
        df_ref_base = df_eval.iloc[ref_idx].copy()

    df_out = _with_greeks(
        base_df=df_out_base.reset_index(names="row_in_split"),
        feature_cols=feature_cols,
        loaded=loaded_adapter,
        chunk_size_values=cfg.chunk_size_values,
        chunk_size_jac=cfg.chunk_size_jac,
        chunk_size_hess=cfg.chunk_size_hess,
        greeks_dtype=cfg.greeks_dtype,
    )
    df_ref = _with_greeks(
        base_df=df_ref_base.reset_index(names="row_in_split"),
        feature_cols=feature_cols,
        loaded=loaded_adapter,
        chunk_size_values=cfg.chunk_size_values,
        chunk_size_jac=cfg.chunk_size_jac,
        chunk_size_hess=cfg.chunk_size_hess,
        greeks_dtype=cfg.greeks_dtype,
    )

    compare_rows: list[dict[str, Any]] = []
    greek_mix_df = pd.DataFrame()
    metric_cols = [
        c
        for c in ("abs_delta", "abs_gamma", "abs_vega", "abs_theta", "abs_rho")
        if c in df_out.columns and c in df_ref.columns
    ]
    for m in metric_cols:
        compare_rows.append(
            {
                "metric": m,
                "out_median": float(np.nanmedian(df_out[m].to_numpy(dtype=np.float64))),
                "ref_median": float(np.nanmedian(df_ref[m].to_numpy(dtype=np.float64))),
                "out_p90": float(np.nanquantile(df_out[m].to_numpy(dtype=np.float64), 0.90)),
                "ref_p90": float(np.nanquantile(df_ref[m].to_numpy(dtype=np.float64), 0.90)),
                "spearman_corr_abs_error_outliers": _safe_spearman(
                    pd.Series(df_out["abs_error"]),
                    pd.Series(df_out[m]),
                ),
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    if not compare_df.empty:
        compare_df["median_lift_out_vs_ref"] = compare_df["out_median"] / np.maximum(
            compare_df["ref_median"], 1.0e-15
        )
        compare_df["p90_lift_out_vs_ref"] = compare_df["out_p90"] / np.maximum(
            compare_df["ref_p90"], 1.0e-15
        )
        compare_df = compare_df.sort_values("median_lift_out_vs_ref", ascending=False)

    if "dominant_greek" in df_out.columns and "dominant_greek" in df_ref.columns:
        f_out = df_out["dominant_greek"].value_counts(normalize=True)
        f_ref = df_ref["dominant_greek"].value_counts(normalize=True)
        greek_mix_df = (
            pd.concat([f_out.rename("out_share"), f_ref.rename("ref_share")], axis=1)
            .fillna(0.0)
            .reset_index(names="greek")
        )
        greek_mix_df["lift_out_vs_ref"] = greek_mix_df["out_share"] / np.maximum(
            greek_mix_df["ref_share"], 1.0e-15
        )
        greek_mix_df = greek_mix_df.sort_values("lift_out_vs_ref", ascending=False)

    surface_sample_df, surface_region_df = _surface_gradients(
        df_eval=df_eval,
        feature_cols=feature_cols,
        loaded=loaded_adapter,
        sample_size=cfg.surface_sample_size,
        chunk_size_jac=cfg.chunk_size_jac,
        greeks_dtype=cfg.greeks_dtype,
        seed=cfg.seed + 997 + seed_offset,
        moneyness_bins=moneyness_bins,
        tau_bins=tau_bins,
    )

    mse_total = float(np.mean(sq_error))
    summary = {
        "run_dir": str(out_dir.parent),
        "split": split,
        "n_total": int(n),
        "n_outliers_used": int(outlier_mask.sum()),
        "outlier_mode": outlier_mode,
        "outlier_quantile_threshold": q_thr,
        "n_outliers_hard_abs_error_gt_threshold": int(outlier_hard_mask.sum()),
        "n_outliers_q": int(outlier_q_mask.sum()),
        "n_floor_iv": int(floor_mask.sum()),
        "global_mse": mse_total,
        "global_rmse": float(np.sqrt(mse_total)),
        "global_mae": float(np.mean(abs_error)),
        "error_p99": float(np.quantile(abs_error, 0.99)),
        "error_p999": float(np.quantile(abs_error, 0.999)),
        "mse_contrib_outliers_hard": float(sq_error[outlier_hard_mask].sum() / n),
        "mse_contrib_floor_iv": float(sq_error[floor_mask].sum() / n),
        "mse_contrib_region_tau_lt_0_25_m_lt_0_8": float(sq_error[region_mask].sum() / n),
        "mse_contrib_region_excluding_hard_outliers": float(
            sq_error[region_mask & (~outlier_hard_mask)].sum() / n
        ),
        "reference_non_outliers_n": int(len(df_ref)),
        "vega_feature": str(df_out["vega_feature"].iloc[0]) if "vega_feature" in df_out.columns and len(df_out) else "",
    }
    if "abs_vega" in df_out.columns:
        summary["outlier_abs_vega_p50"] = float(np.nanquantile(df_out["abs_vega"], 0.50))
        summary["outlier_abs_vega_p90"] = float(np.nanquantile(df_out["abs_vega"], 0.90))

    if cfg.save_full_eval:
        df_eval.to_parquet(out_dir / f"{split}_eval_with_errors.parquet", index=False)
    df_out.to_parquet(out_dir / f"{split}_outliers_detailed.parquet", index=False)
    if not df_ref.empty:
        df_ref.to_parquet(out_dir / f"{split}_reference_non_outliers.parquet", index=False)
    if not compare_df.empty:
        compare_df.to_csv(out_dir / f"{split}_outliers_vs_ref_greek_metrics.csv", index=False)
    if not greek_mix_df.empty:
        greek_mix_df.to_csv(out_dir / f"{split}_outliers_vs_ref_dominant_greek.csv", index=False)
    if not surface_sample_df.empty:
        surface_sample_df.to_parquet(out_dir / f"{split}_surface_grad_sample.parquet", index=False)
    if not surface_region_df.empty:
        surface_region_df.to_csv(out_dir / f"{split}_surface_grad_by_region.csv", index=False)

    with open(out_dir / f"{split}_outliers_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    with open(out_dir / f"{split}_outliers_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outlier sensitivity/stability study with autograd greeks."
    )
    parser.add_argument("--model-dir", default="latest")
    parser.add_argument("--checkpoint-name", default="model_best.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--greeks-device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--greeks-dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--splits", default="val,test", help="Comma-separated: train,val,test")
    parser.add_argument("--outlier-abs-threshold", type=float, default=0.05)
    parser.add_argument("--outlier-quantile", type=float, default=0.999)
    parser.add_argument("--reference-sample-size", type=int, default=1200)
    parser.add_argument("--surface-sample-size", type=int, default=5000)
    parser.add_argument("--chunk-size-values", type=int, default=256)
    parser.add_argument("--chunk-size-jac", type=int, default=128)
    parser.add_argument("--chunk-size-hess", type=int, default=32)
    parser.add_argument("--save-full-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args_raw = build_arg_parser().parse_args()
    cfg = AnalysisArgs(
        model_dir=args_raw.model_dir,
        checkpoint_name=args_raw.checkpoint_name,
        device=args_raw.device,
        greeks_device=args_raw.greeks_device,
        greeks_dtype=_parse_dtype(args_raw.greeks_dtype),
        splits=_parse_splits(args_raw.splits),
        outlier_abs_threshold=float(args_raw.outlier_abs_threshold),
        outlier_quantile=float(args_raw.outlier_quantile),
        reference_sample_size=int(args_raw.reference_sample_size),
        surface_sample_size=int(args_raw.surface_sample_size),
        chunk_size_values=int(args_raw.chunk_size_values),
        chunk_size_jac=int(args_raw.chunk_size_jac),
        chunk_size_hess=int(args_raw.chunk_size_hess),
        save_full_eval=bool(args_raw.save_full_eval),
        seed=int(args_raw.seed),
    )
    if cfg.outlier_quantile <= 0.0 or cfg.outlier_quantile >= 1.0:
        raise ValueError("outlier-quantile must be in (0,1)")

    model, model_device, run_dir, _, norm_stats = load_model_from_run(
        project_root=PROJECT_ROOT,
        model_dir=cfg.model_dir,
        checkpoint_name=cfg.checkpoint_name,
        device=cfg.device,
    )
    data_dir = _resolve_data_dir(run_dir=run_dir)
    floor_iv = _resolve_floor_iv(data_dir=data_dir)
    m_bins, t_bins = _resolve_eval_bins(run_dir=run_dir)

    first_split_df = pd.read_parquet(data_dir / f"{cfg.splits[0]}.parquet")
    target_col = _resolve_target_col(first_split_df)
    feature_cols = _resolve_feature_columns(first_split_df, target_col=target_col)

    loaded_adapter = load_nn_price_adapter(
        project_root=PROJECT_ROOT,
        model_dir=run_dir.name,
        checkpoint_name=cfg.checkpoint_name,
        device=cfg.greeks_device,
        dtype=cfg.greeks_dtype,
        feature_order=feature_cols,
    )

    out_dir = run_dir / "outliers_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for i, split in enumerate(cfg.splits):
        split_path = data_dir / f"{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_df = pd.read_parquet(split_path)
        split_target = _resolve_target_col(split_df)
        split_features = _resolve_feature_columns(split_df, target_col=split_target)
        if split_features != feature_cols:
            raise ValueError(
                "Feature columns changed across splits. "
                f"expected={feature_cols} got={split_features}"
            )
        print(f"[study] analyzing split='{split}' | rows={len(split_df):,}")
        summary = _analyze_split(
            split=split,
            split_df=split_df,
            target_col=split_target,
            feature_cols=feature_cols,
            model=model,
            model_device=model_device,
            norm_stats=norm_stats,
            loaded_adapter=loaded_adapter,
            out_dir=out_dir,
            floor_iv=floor_iv,
            moneyness_bins=m_bins,
            tau_bins=t_bins,
            cfg=cfg,
            seed_offset=i * 1000,
        )
        summaries.append(summary)
        print(
            "[study] split done | "
            f"MSE={summary['global_mse']:.6e} | "
            f"hard_outliers={summary['n_outliers_hard_abs_error_gt_threshold']} | "
            f"floor_iv={summary['n_floor_iv']}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "all_splits_outlier_stability_summary.csv", index=False)
    with open(out_dir / "all_splits_outlier_stability_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary_df.to_dict(orient="records"), f, sort_keys=False)

    print("\n[study] completed")
    print(f"Run dir: {run_dir}")
    print(f"Artifacts: {out_dir}")
    print("Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
