from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pinn.model import build_pinn_model
from src.solvers.heston_cos import COS_solver_scalar


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(payload)!r}")
    return payload


def _resolve_device(pref: str) -> torch.device:
    key = str(pref).lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if key in {"cpu", "cuda", "mps"}:
        return torch.device(key)
    raise ValueError(f"Unsupported device '{pref}'")


def _sample_rows(x: np.ndarray, *, n_points: int, seed: int) -> np.ndarray:
    if n_points <= 0 or n_points >= x.shape[0]:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_points, replace=False)
    return x[idx]


def _predict_pinn(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    input_scaling: dict | None,
) -> np.ndarray:
    x_model = x
    if input_scaling is not None and bool(input_scaling.get("enabled", False)):
        a = np.asarray(input_scaling.get("a", []), dtype=np.float32).reshape(1, -1)
        b = np.asarray(input_scaling.get("b", []), dtype=np.float32).reshape(1, -1)
        if a.shape[1] != x.shape[1] or b.shape[1] != x.shape[1]:
            raise ValueError(
                f"input_scaling mismatch: x has {x.shape[1]} features, "
                f"a has {a.shape[1]}, b has {b.shape[1]}."
            )
        x_model = a + b * x

    preds: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, x_model.shape[0], batch_size):
            stop = min(start + batch_size, x_model.shape[0])
            xb = torch.from_numpy(x_model[start:stop]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yb.astype(np.float64, copy=False))
    return np.concatenate(preds, axis=0)


def _predict_cos(
    *,
    x: np.ndarray,
    feature_order: list[str],
    cos_N: int,
    cos_L: float,
    cos_interval_rule: str,
    option_type: str,
    strike: float,
) -> tuple[np.ndarray, int]:
    idx = {name: i for i, name in enumerate(feature_order)}
    required = ["tau", "moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
    missing = [k for k in required if k not in idx]
    if missing:
        raise KeyError(f"Missing required feature(s) for COS benchmark: {missing}")

    out = np.full(shape=x.shape[0], fill_value=np.nan, dtype=np.float64)
    failed = 0
    cos_params = np.array([int(cos_N), float(cos_L)], dtype=np.float64)

    for i in range(x.shape[0]):
        tau = float(x[i, idx["tau"]])
        m = float(x[i, idx["moneyness"]])
        v = float(x[i, idx["v"]])
        rho = float(x[i, idx["rho"]])
        kappa = float(x[i, idx["kappa"]])
        gamma = float(x[i, idx["gamma"]])
        bar_v = float(x[i, idx["bar_v"]])
        r = float(x[i, idx["r"]])

        # In normalized coordinates K=1 by construction, so S0 = moneyness.
        S0 = m * float(strike)
        params_heston = np.array([rho, kappa, gamma, bar_v, v], dtype=np.float64)
        try:
            out[i] = float(
                COS_solver_scalar(
                    params_Heston=params_heston,
                    S0=S0,
                    K=float(strike),
                    tau=tau,
                    r=r,
                    COS_params=cos_params,
                    opt_type=option_type,
                    interval_rule=cos_interval_rule,
                )
            )
        except Exception:
            failed += 1
            out[i] = np.nan

    return out, failed


def _compute_metrics(
    *,
    y_pred: np.ndarray,
    y_ref: np.ndarray,
    mape_floor: float,
) -> dict:
    err = y_pred - y_ref
    abs_err = np.abs(err)
    mse = float(np.mean(err**2))
    denom = np.maximum(np.abs(y_ref), float(mape_floor))
    rel_abs = abs_err / denom
    smape = 2.0 * abs_err / np.maximum(np.abs(y_ref) + np.abs(y_pred), float(mape_floor))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    return {
        "n_points": int(y_ref.size),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "p90_abs_error": float(np.percentile(abs_err, 90.0)),
        "p95_abs_error": float(np.percentile(abs_err, 95.0)),
        "p99_abs_error": float(np.percentile(abs_err, 99.0)),
        "mean_error_bias": float(np.mean(err)),
        "mape_pct": float(100.0 * np.mean(rel_abs)),
        "smape_pct": float(100.0 * np.mean(smape)),
        "r2": r2,
    }


def _save_scatter_plot(*, y_ref: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    lo = float(np.nanmin([np.min(y_ref), np.min(y_pred)]))
    hi = float(np.nanmax([np.max(y_ref), np.max(y_pred)]))
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(y_ref, y_pred, s=8, alpha=0.35, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("COS price")
    ax.set_ylabel("PINN price")
    ax.set_title("PINN vs COS (price parity)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _save_error_distribution_plot(*, abs_err: np.ndarray, out_path: Path) -> None:
    vals = np.sort(abs_err)
    cdf = np.linspace(0.0, 1.0, vals.size, endpoint=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(vals, cdf, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("|PINN - COS|")
    ax.set_ylabel("CDF")
    ax.set_title("Absolute Error Distribution")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_error_heatmaps(
    *,
    x: np.ndarray,
    abs_err: np.ndarray,
    feature_order: list[str],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = {name: i for i, name in enumerate(feature_order)}
    tau = x[:, idx["tau"]]
    moneyness = x[:, idx["moneyness"]]

    tau_edges = np.linspace(float(np.min(tau)), float(np.max(tau)), int(n_bins) + 1)
    m_edges = np.linspace(float(np.min(moneyness)), float(np.max(moneyness)), int(n_bins) + 1)

    tau_idx = np.clip(np.digitize(tau, bins=tau_edges, right=False) - 1, 0, n_bins - 1)
    m_idx = np.clip(np.digitize(moneyness, bins=m_edges, right=False) - 1, 0, n_bins - 1)

    sums = np.zeros((n_bins, n_bins), dtype=np.float64)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(sums, (tau_idx, m_idx), abs_err)
    np.add.at(counts, (tau_idx, m_idx), 1)

    mean_abs = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return mean_abs, counts, tau_edges, m_edges


def _save_heatmap_plot(
    *,
    matrix: np.ndarray,
    tau_edges: np.ndarray,
    m_edges: np.ndarray,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f2f2f2")
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[m_edges[0], m_edges[-1], tau_edges[0], tau_edges[-1]],
        cmap=cmap_obj,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel("moneyness")
    ax.set_ylabel("tau")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PINN prices vs COS prices.")
    parser.add_argument("--run-dir", type=Path, required=True, help="PINN run dir under outputs/pinn/<run_name>.")
    parser.add_argument("--n-points", type=int, default=4000, help="Number of interior points to benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--batch-size", type=int, default=4096, help="PINN inference batch size.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--cos-N", type=int, default=1500, help="COS truncation terms N.")
    parser.add_argument("--cos-L", type=float, default=50.0, help="COS truncation width L.")
    parser.add_argument(
        "--cos-interval-rule",
        type=str,
        default="cumulant_autodiff",
        help="COS interval rule: sqrt_t | cumulant_autodiff",
    )
    parser.add_argument("--option-type", type=str, default="put", help="put|call")
    parser.add_argument("--strike", type=float, default=1.0, help="Strike used by COS benchmark.")
    parser.add_argument("--mape-floor", type=float, default=1.0e-4, help="Denominator floor for MAPE.")
    parser.add_argument("--n-bins", type=int, default=24, help="Bins per axis for heatmaps.")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else (PROJECT_ROOT / args.run_dir)
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    summary_path = run_dir / "train" / "metrics" / "train_summary.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(f"train_summary.yaml not found: {summary_path}")
    summary = _load_yaml(summary_path)

    collocation_manifest_path = Path(summary["collocation_manifest_file"])
    if not collocation_manifest_path.is_absolute():
        collocation_manifest_path = (PROJECT_ROOT / collocation_manifest_path).resolve()
    collocation_manifest = _load_yaml(collocation_manifest_path)
    feature_order = list(collocation_manifest.get("feature_order", []))
    if not feature_order:
        raise ValueError("Collocation manifest missing 'feature_order'.")

    interior_path = Path(collocation_manifest["datasets"]["interior"])
    if not interior_path.is_absolute():
        interior_path = (PROJECT_ROOT / interior_path).resolve()
    interior_df = pd.read_parquet(interior_path)
    x_all = interior_df.loc[:, feature_order].to_numpy(dtype=np.float32)
    x_eval = _sample_rows(x_all, n_points=int(args.n_points), seed=int(args.seed))

    checkpoint_path = Path(summary.get("best_checkpoint", ""))
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "train" / "checkpoints" / "model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found for run: {checkpoint_path}")

    pipeline_plan_path = run_dir / "pipeline_plan.yaml"
    if pipeline_plan_path.exists():
        plan = _load_yaml(pipeline_plan_path)
        arch_cfg_path = Path(plan.get("architecture_config", "configs/pinn_model_architecture.yaml"))
    else:
        arch_cfg_path = Path("configs/pinn_model_architecture.yaml")
    if not arch_cfg_path.is_absolute():
        arch_cfg_path = (PROJECT_ROOT / arch_cfg_path).resolve()
    model_cfg = _load_yaml(arch_cfg_path)

    device = _resolve_device(args.device)
    model = build_pinn_model(model_cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    input_scaling = summary.get("input_scaling", {})
    if not isinstance(input_scaling, dict):
        input_scaling = {}

    out_dir = run_dir / "cos_benchmark"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    y_pinn = _predict_pinn(
        model=model,
        x=x_eval,
        batch_size=int(args.batch_size),
        device=device,
        input_scaling=input_scaling,
    )
    pinn_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    y_cos, n_failed = _predict_cos(
        x=x_eval,
        feature_order=feature_order,
        cos_N=int(args.cos_N),
        cos_L=float(args.cos_L),
        cos_interval_rule=str(args.cos_interval_rule),
        option_type=str(args.option_type),
        strike=float(args.strike),
    )
    cos_seconds = float(time.perf_counter() - t1)

    valid_mask = np.isfinite(y_pinn) & np.isfinite(y_cos)
    x_valid = x_eval[valid_mask]
    y_pinn_valid = y_pinn[valid_mask]
    y_cos_valid = y_cos[valid_mask]
    if x_valid.shape[0] == 0:
        raise RuntimeError("No valid points after COS/PINN filtering.")

    metrics = _compute_metrics(
        y_pred=y_pinn_valid,
        y_ref=y_cos_valid,
        mape_floor=float(args.mape_floor),
    )
    metrics.update(
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "collocation_interior": str(interior_path),
            "n_input_points": int(x_eval.shape[0]),
            "n_valid_points": int(x_valid.shape[0]),
            "n_failed_cos": int(n_failed),
            "feature_order": feature_order,
            "input_scaling_enabled": bool(input_scaling.get("enabled", False)),
            "input_scaling_method": input_scaling.get("method"),
            "pinn_inference_seconds": pinn_seconds,
            "cos_inference_seconds": cos_seconds,
            "pinn_points_per_second": float(x_eval.shape[0] / max(pinn_seconds, 1.0e-12)),
            "cos_points_per_second": float(x_eval.shape[0] / max(cos_seconds, 1.0e-12)),
            "cos_params": {
                "N": int(args.cos_N),
                "L": float(args.cos_L),
                "interval_rule": str(args.cos_interval_rule),
                "option_type": str(args.option_type),
                "strike": float(args.strike),
            },
        }
    )

    out_df = pd.DataFrame(x_valid, columns=feature_order)
    out_df["price_pinn"] = y_pinn_valid
    out_df["price_cos"] = y_cos_valid
    out_df["error"] = y_pinn_valid - y_cos_valid
    out_df["abs_error"] = np.abs(out_df["error"].to_numpy())
    out_df["rel_abs_error"] = out_df["abs_error"] / np.maximum(
        np.abs(out_df["price_cos"].to_numpy()),
        float(args.mape_floor),
    )

    points_csv = out_dir / "points_pinn_vs_cos.csv"
    out_df.to_csv(points_csv, index=False)

    metrics_yaml = out_dir / "metrics.yaml"
    with open(metrics_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    metrics_csv = out_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    abs_err = np.abs(out_df["error"].to_numpy(dtype=np.float64))
    _save_scatter_plot(
        y_ref=y_cos_valid,
        y_pred=y_pinn_valid,
        out_path=fig_dir / "scatter_pinn_vs_cos.png",
    )
    _save_error_distribution_plot(
        abs_err=abs_err,
        out_path=fig_dir / "abs_error_cdf.png",
    )
    mean_abs, counts, tau_edges, m_edges = _build_error_heatmaps(
        x=x_valid,
        abs_err=abs_err,
        feature_order=feature_order,
        n_bins=int(args.n_bins),
    )
    _save_heatmap_plot(
        matrix=mean_abs,
        tau_edges=tau_edges,
        m_edges=m_edges,
        out_path=fig_dir / "abs_error_map_m_tau.png",
        title="PINN vs COS | Mean Absolute Error by Zone",
        cbar_label="mean |PINN - COS|",
        cmap="magma",
    )
    _save_heatmap_plot(
        matrix=counts.astype(np.float64),
        tau_edges=tau_edges,
        m_edges=m_edges,
        out_path=fig_dir / "counts_map_m_tau.png",
        title="Benchmark Samples per Zone",
        cbar_label="count",
        cmap="viridis",
    )

    print("COS benchmark completed")
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Valid points: {x_valid.shape[0]} / {x_eval.shape[0]}")
    print(
        f"MAE={metrics['mae']:.3e} | RMSE={metrics['rmse']:.3e} | "
        f"MAPE={metrics['mape_pct']:.3f}% | MaxAE={metrics['max_abs_error']:.3e}"
    )
    print(
        f"Speed PINN={metrics['pinn_points_per_second']:.1f} pts/s | "
        f"COS={metrics['cos_points_per_second']:.1f} pts/s"
    )


if __name__ == "__main__":
    main()
