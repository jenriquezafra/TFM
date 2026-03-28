from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.calibration.de_solver import run_de_calibration
from src.calibration.model_inference import (
    build_features_from_theta,
    load_model_from_run,
    predict_iv,
)
from src.calibration.objective_func import build_market_inputs


DEFAULT_PARAM_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0"]


@dataclass(frozen=True)
class CalibrationRunArtifacts:
    model_bucket_dir: Path
    calibration_id: int
    output_dir: Path
    summary_yaml: Path
    summary_json: Path
    market_quotes_copy: Path
    residual_heatmap_png: Path | None
    parameter_error_table: Path | None
    parameter_error_abs_bar_png: Path | None
    parameter_error_abs_heatmap_png: Path | None
    parameter_error_rel_heatmap_png: Path | None
    quotes_comparison_parquet: Path | None
    quotes_comparison_csv: Path | None
    config_source_copy: Path
    config_used: Path
    curvature_json: Path | None
    hessian_theta_star_csv: Path | None
    jacobian_theta_star_png: Path | None
    hessian_theta_star_png: Path | None


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def read_quotes_table(path: Path, file_format: str = "auto") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Quotes file not found: {path}")

    fmt = file_format.lower()
    if fmt == "auto":
        if path.suffix.lower() in {".parquet", ".pq"}:
            fmt = "parquet"
        elif path.suffix.lower() in {".csv", ".txt"}:
            fmt = "csv"
        else:
            raise ValueError(
                f"Could not infer file format from suffix '{path.suffix}'. "
                "Set data.format to 'csv' or 'parquet'."
            )

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported quotes format '{file_format}'")


def as_bounds(values, *, name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"Bounds for '{name}' must be [lower, upper]")
    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise ValueError(f"Invalid bounds for '{name}': {lower} >= {upper}")
    return lower, upper


def _next_calibration_id(model_bucket_dir: Path) -> int:
    """
    Return next integer id for folders named Calibration_<id>.
    """
    pattern = re.compile(r"^Calibration_(\d+)$")
    max_id = 0
    for child in model_bucket_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def _save_residual_heatmap(quotes_df: pd.DataFrame, output_path: Path) -> Path | None:
    """
    Save residual heatmap over (moneyness, tau).
    If required columns are missing or no valid cells exist, returns None.
    """
    required = {"moneyness", "tau", "residual"}
    if not required.issubset(quotes_df.columns):
        return None

    pivot = quotes_df.pivot_table(
        index="tau",
        columns="moneyness",
        values="residual",
        aggfunc="mean",
    )
    if pivot.empty:
        return None

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    z_vals = pivot.to_numpy(dtype=float)
    valid = np.isfinite(z_vals)
    if not np.any(valid):
        return None

    max_abs = float(np.nanmax(np.abs(z_vals[valid])))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0e-8

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(
        z_vals,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
    )

    ax.set_title("Residual Heatmap (IV_pred - IV_market)")
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Tau")
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in x_vals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{y:.3f}" for y in y_vals])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _theta_from_raw(theta_raw, *, param_order: list[str], context: str) -> np.ndarray:
    if isinstance(theta_raw, dict):
        missing = [name for name in param_order if name not in theta_raw]
        if missing:
            raise ValueError(f"{context} dict missing keys: {missing}")
        return np.array([float(theta_raw[name]) for name in param_order], dtype=np.float64)

    theta = np.asarray(theta_raw, dtype=np.float64).reshape(-1)
    if theta.size != len(param_order):
        raise ValueError(
            f"{context} list must have same size as parameter_order. "
            f"Got {theta.size} vs {len(param_order)}."
        )
    return theta


def _load_theta_true_from_file(*, truth_file: Path, param_order: list[str]) -> np.ndarray:
    if not truth_file.exists():
        raise FileNotFoundError(f"Synthetic truth file not found: {truth_file}")

    suffix = truth_file.suffix.lower()
    if suffix == ".json":
        with open(truth_file, "r") as f:
            payload = json.load(f)
    else:
        with open(truth_file, "r") as f:
            payload = yaml.safe_load(f)

    if isinstance(payload, dict):
        if "theta_true" in payload:
            return _theta_from_raw(
                payload["theta_true"],
                param_order=param_order,
                context=f"{truth_file}",
            )
        if "synthetic_truth" in payload and isinstance(payload["synthetic_truth"], dict):
            if "theta_true" in payload["synthetic_truth"]:
                return _theta_from_raw(
                    payload["synthetic_truth"]["theta_true"],
                    param_order=param_order,
                    context=f"{truth_file}:synthetic_truth.theta_true",
                )
        if all(name in payload for name in param_order):
            return _theta_from_raw(payload, param_order=param_order, context=f"{truth_file}")

    return _theta_from_raw(payload, param_order=param_order, context=f"{truth_file}")


def _parse_theta_true(
    config: dict,
    param_order: list[str],
    *,
    quotes_df: pd.DataFrame | None = None,
    quotes_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Parse optional synthetic ground-truth parameters from:
    1) config.synthetic_truth.theta_true
    2) quotes columns (rho,kappa,gamma,bar_v,v0) if present and constant
    3) config.synthetic_truth.truth_file
    4) sidecar file near quotes: <quotes>.truth.{yaml|yml|json} or <stem>_truth.{yaml|yml|json}
    """
    synth_cfg = config.get("synthetic_truth", {})
    theta_raw = synth_cfg.get("theta_true", None)
    if theta_raw is not None:
        return _theta_from_raw(
            theta_raw,
            param_order=param_order,
            context="synthetic_truth.theta_true",
        ), "synthetic_truth.theta_true"

    if quotes_df is not None and all(name in quotes_df.columns for name in param_order):
        values = []
        is_constant = True
        for name in param_order:
            uniq = pd.unique(quotes_df[name].dropna())
            if len(uniq) != 1:
                is_constant = False
                break
            values.append(float(uniq[0]))
        if is_constant:
            return np.array(values, dtype=np.float64), "quotes_columns"

    truth_file_raw = synth_cfg.get("truth_file", None)
    if truth_file_raw:
        base_dir = project_root if project_root is not None else Path.cwd()
        truth_file = resolve_path(truth_file_raw, base_dir=base_dir)
        theta = _load_theta_true_from_file(truth_file=truth_file, param_order=param_order)
        return theta, f"synthetic_truth.truth_file:{truth_file}"

    if quotes_path is not None:
        candidates = [
            quotes_path.with_suffix(".truth.yaml"),
            quotes_path.with_suffix(".truth.yml"),
            quotes_path.with_suffix(".truth.json"),
            quotes_path.parent / f"{quotes_path.stem}_truth.yaml",
            quotes_path.parent / f"{quotes_path.stem}_truth.yml",
            quotes_path.parent / f"{quotes_path.stem}_truth.json",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            theta = _load_theta_true_from_file(truth_file=candidate, param_order=param_order)
            return theta, f"quotes_sidecar:{candidate}"

    return None, None


def _save_parameter_error_artifacts(
    *,
    theta_hat: np.ndarray,
    theta_true: np.ndarray,
    param_order: list[str],
    out_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Save parameter calibration error artifacts (absolute/relative).
    """
    abs_error = np.abs(theta_hat - theta_true)
    denom = np.maximum(np.abs(theta_true), 1.0e-12)
    rel_error = abs_error / denom

    table = pd.DataFrame(
        {
            "parameter": param_order,
            "theta_true": theta_true,
            "theta_hat": theta_hat,
            "abs_error": abs_error,
            "rel_error": rel_error,
        }
    )
    table_path = out_dir / "parameter_errors.csv"
    table.to_csv(table_path, index=False)

    # Bar chart of absolute parameter errors
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(param_order, abs_error, color="#2E6EA5")
    ax.set_title("Absolute Calibration Error by Parameter")
    ax.set_ylabel("|theta_hat - theta_true|")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    abs_bar_path = out_dir / "parameter_error_abs_bar.png"
    fig.savefig(abs_bar_path, dpi=220)
    plt.close(fig)

    # 1xN heatmap for absolute errors
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    data_abs = abs_error.reshape(1, -1)
    im = ax.imshow(data_abs, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([0])
    ax.set_yticklabels(["abs_error"])
    ax.set_xticks(np.arange(len(param_order)))
    ax.set_xticklabels(param_order)
    ax.set_title("Parameter Absolute Errors")
    for j, val in enumerate(abs_error):
        ax.text(j, 0, f"{val:.3e}", ha="center", va="center", color="black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute error")
    fig.tight_layout()
    abs_hm_path = out_dir / "parameter_error_abs_heatmap.png"
    fig.savefig(abs_hm_path, dpi=220)
    plt.close(fig)

    # 1xN heatmap for relative errors
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    data_rel = rel_error.reshape(1, -1)
    im = ax.imshow(data_rel, aspect="auto", cmap="PuRd")
    ax.set_yticks([0])
    ax.set_yticklabels(["rel_error"])
    ax.set_xticks(np.arange(len(param_order)))
    ax.set_xticklabels(param_order)
    ax.set_title("Parameter Relative Errors")
    for j, val in enumerate(rel_error):
        ax.text(j, 0, f"{val:.2%}", ha="center", va="center", color="black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative error")
    fig.tight_layout()
    rel_hm_path = out_dir / "parameter_error_rel_heatmap.png"
    fig.savefig(rel_hm_path, dpi=220)
    plt.close(fig)

    return table_path, abs_bar_path, abs_hm_path, rel_hm_path


def _run_auto_curvature(
    *,
    project_root: Path,
    summary_json: Path,
    out_dir: Path,
    regularization: str,
    lambda_reg: float,
    out_cfg: dict,
) -> dict[str, str] | None:
    """
    Optionally compute curvature/Hessian right after calibration using the exact summary
    just written for this run, keeping calibration and curvature aligned.
    """
    if not bool(out_cfg.get("auto_eval_curvature", True)):
        return None

    eval_script = project_root / "scripts" / "eval_calibration_curvature.py"
    if not eval_script.exists():
        print(
            "[WARN] Auto-curvature requested but script not found: "
            f"{eval_script}"
        )
        return None

    cmd = [
        sys.executable,
        str(eval_script),
        "--summary-path",
        str(summary_json),
        "--theta-key",
        "theta_star",
        "--regularization",
        str(regularization),
        "--lambda-reg",
        str(float(lambda_reg)),
    ]

    curvature_dtype = out_cfg.get("curvature_dtype", None)
    if curvature_dtype is not None:
        cmd.extend(["--dtype", str(curvature_dtype)])

    curvature_device = out_cfg.get("curvature_device", None)
    if curvature_device is not None:
        cmd.extend(["--device", str(curvature_device)])

    if bool(out_cfg.get("curvature_no_plots", False)):
        cmd.append("--no-plots")

    proc = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(
            "[WARN] Auto-curvature failed. "
            f"returncode={proc.returncode}. "
            "Calibration outputs were still saved."
        )
        if proc.stdout.strip():
            print("[WARN] Auto-curvature stdout:")
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print("[WARN] Auto-curvature stderr:")
            print(proc.stderr.strip())
        return None

    curvature_json = out_dir / "curvature_theta_star.json"
    hessian_csv = out_dir / "hessian_theta_star.csv"
    jacobian_plot = out_dir / "jacobian_theta_star.png"
    hessian_plot = out_dir / "hessian_theta_star.png"

    artifacts: dict[str, str] = {}
    if curvature_json.exists():
        artifacts["json"] = str(curvature_json)
    if hessian_csv.exists():
        artifacts["hessian_csv"] = str(hessian_csv)
    if jacobian_plot.exists():
        artifacts["jacobian_plot"] = str(jacobian_plot)
    if hessian_plot.exists():
        artifacts["hessian_plot"] = str(hessian_plot)

    if artifacts:
        print("Auto-curvature finished for theta_star.")
        return artifacts
    return None


def run_calibration_from_config(
    *,
    project_root: Path,
    config_path: Path,
    quotes_override: Path | None = None,
    model_dir_override: str | None = None,
    output_dir_override: Path | None = None,
    tag_override: str | None = None,
    maxiter_override: int | None = None,
    popsize_override: int | None = None,
    theta_true_override: list[float] | tuple[float, ...] | np.ndarray | None = None,
    truth_file_override: Path | None = None,
) -> CalibrationRunArtifacts:
    cfg_path = resolve_path(config_path, base_dir=project_root)
    cfg = load_yaml(cfg_path)
    configured_param_order = list(
        cfg.get("calibration", {}).get("parameter_order", DEFAULT_PARAM_ORDER)
    )

    if theta_true_override is not None and truth_file_override is not None:
        raise ValueError("Use either theta_true_override or truth_file_override, not both.")

    if theta_true_override is not None:
        theta_override = [float(x) for x in np.asarray(theta_true_override, dtype=np.float64).reshape(-1)]
        if len(theta_override) != len(configured_param_order):
            raise ValueError(
                "theta_true_override must have same size as calibration.parameter_order. "
                f"Got {len(theta_override)} vs {len(configured_param_order)}."
            )
        cfg.setdefault("synthetic_truth", {})
        cfg["synthetic_truth"]["theta_true"] = theta_override
        cfg["synthetic_truth"]["truth_file"] = None

    if truth_file_override is not None:
        cfg.setdefault("synthetic_truth", {})
        cfg["synthetic_truth"]["theta_true"] = None
        cfg["synthetic_truth"]["truth_file"] = str(resolve_path(truth_file_override, base_dir=project_root))

    meta_cfg = cfg.get("meta", {})
    seed_default = meta_cfg.get("seed", 42)

    model_cfg = cfg.get("model", {})
    model_dir = model_dir_override or model_cfg.get("model_dir", "latest")
    checkpoint_name = model_cfg.get("checkpoint", "model_best.pt")
    model_device_pref = model_cfg.get("device", "auto")

    model, model_device, run_dir, _, normalization_stats = load_model_from_run(
        project_root=project_root,
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        device=model_device_pref,
    )
    print(f"Loaded model from: {run_dir}")
    print(f"Device: {model_device}")

    data_cfg = cfg.get("data", {})
    quotes_path_cfg = data_cfg.get("market_quotes_path", "data/market/market_quotes.csv")
    quotes_path_raw = quotes_override if quotes_override is not None else quotes_path_cfg
    quotes_path = resolve_path(quotes_path_raw, base_dir=project_root)
    quotes_format = data_cfg.get("format", "auto")
    quotes_df = read_quotes_table(quotes_path, file_format=quotes_format)
    if quotes_df.empty:
        raise ValueError(f"Quotes table is empty: {quotes_path}")

    col_cfg = data_cfg.get("columns", {})
    col_m = col_cfg.get("moneyness", "moneyness")
    col_tau = col_cfg.get("tau", "tau")
    col_r = col_cfg.get("r", "r")
    col_iv = col_cfg.get("iv_market", "iv_market")
    col_w = col_cfg.get("weight", "weight")

    for required_col in (col_m, col_tau, col_r, col_iv):
        if required_col not in quotes_df.columns:
            raise KeyError(
                f"Column '{required_col}' not found in quotes file {quotes_path}. "
                f"Available columns: {list(quotes_df.columns)}"
            )

    if col_w in quotes_df.columns:
        weights = quotes_df[col_w].to_numpy(dtype=np.float64)
    else:
        default_weight = float(data_cfg.get("default_weight", 1.0))
        weights = np.full(len(quotes_df), fill_value=default_weight, dtype=np.float64)

    market_inputs = build_market_inputs(
        moneyness=quotes_df[col_m].to_numpy(dtype=np.float64),
        tau=quotes_df[col_tau].to_numpy(dtype=np.float64),
        r=quotes_df[col_r].to_numpy(dtype=np.float64),
        iv_market=quotes_df[col_iv].to_numpy(dtype=np.float64),
        weights=weights,
    )
    print(f"Loaded {market_inputs.n_quotes} market quotes from: {quotes_path}")

    cal_cfg = cfg.get("calibration", {})
    param_order = list(cal_cfg.get("parameter_order", DEFAULT_PARAM_ORDER))
    bounds_cfg = cal_cfg.get("bounds", {})
    bounds = [as_bounds(bounds_cfg[name], name=name) for name in param_order]

    reg_cfg = cal_cfg.get("regularization", {})
    reg_type = str(reg_cfg.get("type", "l2"))
    lambda_reg = float(reg_cfg.get("lambda", 0.0))

    de_cfg = cal_cfg.get("de", {})
    de_maxiter = int(maxiter_override if maxiter_override is not None else de_cfg.get("maxiter", 300))
    de_popsize = int(popsize_override if popsize_override is not None else de_cfg.get("popsize", 10))
    de_tol = float(de_cfg.get("tol", 0.01))
    mutation_raw = de_cfg.get("mutation", [0.5, 1.0])
    if not isinstance(mutation_raw, (list, tuple)) or len(mutation_raw) != 2:
        raise ValueError("calibration.de.mutation must be [min, max]")
    de_mutation = (float(mutation_raw[0]), float(mutation_raw[1]))
    de_recombination = float(de_cfg.get("recombination", 0.7))
    de_seed = de_cfg.get("seed", seed_default)
    de_seed = None if de_seed is None else int(de_seed)
    de_polish = bool(de_cfg.get("polish", False))

    print("Running DE calibration...")
    result = run_de_calibration(
        model=model,
        market_inputs=market_inputs,
        bounds=bounds,
        lambda_reg=lambda_reg,
        regularization=reg_type,
        normalization_stats=normalization_stats,
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=de_tol,
        mutation=de_mutation,
        recombination=de_recombination,
        seed=de_seed,
        polish=de_polish,
        device=model_device,
    )

    theta_star = np.asarray(result.x, dtype=np.float64).reshape(-1)
    if theta_star.size != len(param_order):
        raise RuntimeError(
            f"Expected {len(param_order)} calibrated params; got {theta_star.size}"
        )

    inf_cfg = cfg.get("inference", {})
    infer_batch_size = int(inf_cfg.get("batch_size", 4096))
    features_star = build_features_from_theta(
        theta_star,
        moneyness=market_inputs.moneyness,
        tau=market_inputs.tau,
        r=market_inputs.r,
    )
    iv_pred = predict_iv(
        model=model,
        features=features_star,
        device=model_device,
        batch_size=infer_batch_size,
        normalization_stats=normalization_stats,
    )

    residual = iv_pred - market_inputs.iv_market
    weighted_sq = market_inputs.weights * (residual**2)

    out_cfg = cfg.get("outputs", {})
    out_root_raw = output_dir_override if output_dir_override is not None else out_cfg.get(
        "dir", "outputs/calibration"
    )
    out_root = resolve_path(out_root_raw, base_dir=project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_bucket_name = run_dir.name
    model_bucket_dir = out_root / model_bucket_name
    model_bucket_dir.mkdir(parents=True, exist_ok=True)

    calibration_id = _next_calibration_id(model_bucket_dir)
    out_dir = model_bucket_dir / f"Calibration_{calibration_id}"
    out_dir.mkdir(parents=True, exist_ok=False)

    configured_tag = str(out_cfg.get("tag", "")).strip()
    run_tag = str(tag_override).strip() if tag_override is not None else configured_tag
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    config_source_copy = out_dir / "calibration_config_source.yaml"
    shutil.copy(cfg_path, config_source_copy)

    # Persist a self-contained effective config snapshot with resolved runtime choices.
    used_cfg = dict(cfg)
    used_cfg.setdefault("meta", {})
    used_cfg["meta"]["executed_at"] = ts
    if run_tag:
        used_cfg["meta"]["tag"] = run_tag
    used_cfg.setdefault("model", {})
    used_cfg["model"]["resolved_run_dir"] = str(run_dir)
    used_cfg.setdefault("data", {})
    used_cfg["data"]["resolved_market_quotes_path"] = str(quotes_path)
    used_cfg.setdefault("calibration", {})
    used_cfg["calibration"].setdefault("de", {})
    used_cfg["calibration"]["de"]["maxiter"] = de_maxiter
    used_cfg["calibration"]["de"]["popsize"] = de_popsize
    used_cfg["calibration"]["de"]["tol"] = de_tol
    used_cfg["calibration"]["de"]["mutation"] = [float(de_mutation[0]), float(de_mutation[1])]
    used_cfg["calibration"]["de"]["recombination"] = de_recombination
    used_cfg["calibration"]["de"]["seed"] = de_seed
    used_cfg["calibration"]["de"]["polish"] = de_polish

    config_used = out_dir / "calibration_config_used.yaml"
    with open(config_used, "w") as f:
        yaml.safe_dump(used_cfg, f, sort_keys=False)

    # Copy the exact market quotes file used for this run.
    quotes_suffix = quotes_path.suffix if quotes_path.suffix else ".csv"
    market_quotes_copy = out_dir / f"market_quotes_input{quotes_suffix}"
    shutil.copy2(quotes_path, market_quotes_copy)

    summary = {
        "timestamp": ts,
        "model_bucket": model_bucket_name,
        "calibration_id": calibration_id,
        "run_tag": run_tag,
        "quotes_file": str(quotes_path),
        "market_quotes_copy": str(market_quotes_copy),
        "model_run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "device": str(model_device),
        "n_quotes": int(market_inputs.n_quotes),
        "parameter_order": param_order,
        "theta_star": [float(x) for x in theta_star],
        "objective_fun": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "weighted_mse": float(np.mean(weighted_sq)),
        "de_settings": {
            "maxiter": de_maxiter,
            "popsize": de_popsize,
            "tol": de_tol,
            "mutation": list(de_mutation),
            "recombination": de_recombination,
            "seed": de_seed,
            "polish": de_polish,
        },
        "regularization": {
            "type": reg_type,
            "lambda": lambda_reg,
        },
        "bounds": {name: [float(low), float(high)] for name, (low, high) in zip(param_order, bounds)},
    }

    # Optional synthetic ground-truth parameter diagnostics.
    parameter_error_table = None
    parameter_error_abs_bar_path = None
    parameter_error_abs_heatmap_path = None
    parameter_error_rel_heatmap_path = None
    theta_true, theta_true_source = _parse_theta_true(
        cfg,
        param_order,
        quotes_df=quotes_df,
        quotes_path=quotes_path,
        project_root=project_root,
    )
    if theta_true is not None:
        (
            parameter_error_table,
            parameter_error_abs_bar_path,
            parameter_error_abs_heatmap_path,
            parameter_error_rel_heatmap_path,
        ) = _save_parameter_error_artifacts(
            theta_hat=theta_star,
            theta_true=theta_true,
            param_order=param_order,
            out_dir=out_dir,
        )
        abs_error = np.abs(theta_star - theta_true)
        denom = np.maximum(np.abs(theta_true), 1.0e-12)
        rel_error = abs_error / denom
        summary["theta_true"] = [float(x) for x in theta_true]
        if theta_true_source is not None:
            summary["theta_true_source"] = theta_true_source
        summary["parameter_abs_error"] = {
            name: float(err) for name, err in zip(param_order, abs_error)
        }
        summary["parameter_rel_error"] = {
            name: float(err) for name, err in zip(param_order, rel_error)
        }
        summary["parameter_error_artifacts"] = {
            "table_csv": str(parameter_error_table),
            "abs_bar_png": str(parameter_error_abs_bar_path),
            "abs_heatmap_png": str(parameter_error_abs_heatmap_path),
            "rel_heatmap_png": str(parameter_error_rel_heatmap_path),
        }

    quotes_out = pd.DataFrame(
        {
            "moneyness": market_inputs.moneyness,
            "tau": market_inputs.tau,
            "r": market_inputs.r,
            "iv_market": market_inputs.iv_market,
            "iv_pred": iv_pred,
            "residual": residual,
            "residual_abs": np.abs(residual),
            "residual_sq": residual**2,
            "weight": market_inputs.weights,
            "weighted_sq_error": weighted_sq,
        }
    )

    quotes_parquet_path = None
    quotes_csv_path = None
    residual_heatmap_path = None
    if bool(out_cfg.get("save_quotes_parquet", True)):
        quotes_parquet_path = out_dir / "quotes_comparison.parquet"
        quotes_out.to_parquet(quotes_parquet_path, index=False)
    if bool(out_cfg.get("save_quotes_csv", False)):
        quotes_csv_path = out_dir / "quotes_comparison.csv"
        quotes_out.to_csv(quotes_csv_path, index=False)
    if bool(out_cfg.get("save_residual_heatmap", False)):
        residual_heatmap_path = _save_residual_heatmap(
            quotes_out,
            out_dir / "residual_heatmap.png",
        )
        if residual_heatmap_path is not None:
            summary["residual_heatmap"] = str(residual_heatmap_path)

    summary_yaml = out_dir / "summary.yaml"
    summary_json = out_dir / "summary.json"
    with open(summary_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    curvature_artifacts = _run_auto_curvature(
        project_root=project_root,
        summary_json=summary_json,
        out_dir=out_dir,
        regularization=reg_type,
        lambda_reg=lambda_reg,
        out_cfg=out_cfg,
    )
    if curvature_artifacts is not None:
        summary["curvature_theta_star_artifacts"] = curvature_artifacts
        with open(summary_yaml, "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Calibration finished. Success={bool(result.success)}")
    print(f"Best objective: {float(result.fun):.8e}")
    print(f"Best theta ({', '.join(param_order)}): {np.array2string(theta_star, precision=6)}")
    print(f"Saved outputs to: {out_dir}")

    return CalibrationRunArtifacts(
        model_bucket_dir=model_bucket_dir,
        calibration_id=calibration_id,
        output_dir=out_dir,
        summary_yaml=summary_yaml,
        summary_json=summary_json,
        market_quotes_copy=market_quotes_copy,
        residual_heatmap_png=residual_heatmap_path,
        parameter_error_table=parameter_error_table,
        parameter_error_abs_bar_png=parameter_error_abs_bar_path,
        parameter_error_abs_heatmap_png=parameter_error_abs_heatmap_path,
        parameter_error_rel_heatmap_png=parameter_error_rel_heatmap_path,
        quotes_comparison_parquet=quotes_parquet_path,
        quotes_comparison_csv=quotes_csv_path,
        config_source_copy=config_source_copy,
        config_used=config_used,
        curvature_json=(
            Path(curvature_artifacts["json"])
            if curvature_artifacts is not None and "json" in curvature_artifacts
            else None
        ),
        hessian_theta_star_csv=(
            Path(curvature_artifacts["hessian_csv"])
            if curvature_artifacts is not None and "hessian_csv" in curvature_artifacts
            else None
        ),
        jacobian_theta_star_png=(
            Path(curvature_artifacts["jacobian_plot"])
            if curvature_artifacts is not None and "jacobian_plot" in curvature_artifacts
            else None
        ),
        hessian_theta_star_png=(
            Path(curvature_artifacts["hessian_plot"])
            if curvature_artifacts is not None and "hessian_plot" in curvature_artifacts
            else None
        ),
    )
