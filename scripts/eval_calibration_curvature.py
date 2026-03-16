from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "outputs" / ".mplconfig")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration.model_inference import FEATURE_ORDER, load_model_from_run, resolve_device
from src.greeks.core import derivatives_point


CALIBRATION_FOLDER_PATTERN = re.compile(r"^Calibration_(\d+)$")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_summary(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Summary file has invalid content: {path}")
    return data


def _parse_dtype(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("dtype must be one of {float64, float32}")


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _extract_calibration_id(folder_name: str) -> int | None:
    match = CALIBRATION_FOLDER_PATTERN.match(folder_name)
    if match is None:
        return None
    return int(match.group(1))


def _load_calibration_output_root(project_root: Path) -> Path:
    cfg_path = project_root / "configs" / "calibration.yaml"
    if not cfg_path.exists():
        return project_root / "outputs" / "calibration"
    cfg = _load_yaml(cfg_path)
    raw = cfg.get("outputs", {}).get("dir", "outputs/calibration")
    return _resolve_path(raw, base_dir=project_root)


def _list_summary_candidates(calibration_root: Path) -> list[Path]:
    out: list[Path] = []
    if not calibration_root.exists():
        return out
    for summary in calibration_root.rglob("summary.json"):
        if summary.is_file():
            out.append(summary)
    if out:
        return out
    for summary in calibration_root.rglob("summary.yaml"):
        if summary.is_file():
            out.append(summary)
    return out


def _resolve_summary_path(
    *,
    project_root: Path,
    calibration_root: Path,
    summary_path_arg: str | None,
    calibration_dir_arg: str | None,
    model_dir_arg: str | None,
    calibration_id_arg: int | None,
) -> Path:
    if summary_path_arg:
        summary_path = _resolve_path(summary_path_arg, base_dir=project_root)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        return summary_path

    if calibration_dir_arg and calibration_dir_arg.strip().lower() != "latest":
        cal_dir = _resolve_path(calibration_dir_arg, base_dir=project_root)
        if not cal_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {cal_dir}")
        if cal_dir.is_file():
            return cal_dir
        summary_json = cal_dir / "summary.json"
        summary_yaml = cal_dir / "summary.yaml"
        if summary_json.exists():
            return summary_json
        if summary_yaml.exists():
            return summary_yaml
        raise FileNotFoundError(f"No summary file found in calibration dir: {cal_dir}")

    if model_dir_arg:
        model_bucket = calibration_root / model_dir_arg
        if not model_bucket.exists():
            raise FileNotFoundError(
                f"Model bucket '{model_dir_arg}' not found under calibration root: {calibration_root}"
            )
        if calibration_id_arg is not None:
            candidate_folders = [
                p
                for p in model_bucket.iterdir()
                if p.is_dir() and _extract_calibration_id(p.name) == int(calibration_id_arg)
            ]
            if not candidate_folders:
                raise FileNotFoundError(
                    f"Calibration id {calibration_id_arg} not found in bucket: {model_bucket}"
                )
            cal_dir = candidate_folders[0]
            summary_json = cal_dir / "summary.json"
            summary_yaml = cal_dir / "summary.yaml"
            if summary_json.exists():
                return summary_json
            if summary_yaml.exists():
                return summary_yaml
            raise FileNotFoundError(f"Summary not found for calibration dir: {cal_dir}")

        candidates: list[tuple[int, float, Path]] = []
        for child in model_bucket.iterdir():
            if not child.is_dir():
                continue
            cal_id = _extract_calibration_id(child.name)
            if cal_id is None:
                continue
            summary_json = child / "summary.json"
            summary_yaml = child / "summary.yaml"
            if summary_json.exists():
                summary_path = summary_json
            elif summary_yaml.exists():
                summary_path = summary_yaml
            else:
                continue
            candidates.append((cal_id, summary_path.stat().st_mtime, summary_path))
        if not candidates:
            raise FileNotFoundError(f"No calibration summaries found in: {model_bucket}")
        # Prefer highest calibration id, then latest mtime.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    all_candidates = _list_summary_candidates(calibration_root)
    if not all_candidates:
        raise FileNotFoundError(f"No calibration summaries found under: {calibration_root}")
    all_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_candidates[0]


def _resolve_market_table_path(summary: dict[str, Any], summary_path: Path) -> Path:
    preferred_keys = ("market_quotes_copy", "quotes_file")
    for key in preferred_keys:
        raw = summary.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = _resolve_path(path, base_dir=summary_path.parent)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not resolve market quotes file from summary keys "
        f"{preferred_keys}. Summary: {summary_path}"
    )


def _read_quotes_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    # Fallback: try parquet, then CSV.
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _parse_theta_override(raw: str, expected_size: int) -> np.ndarray:
    values = [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    theta = np.asarray(values, dtype=np.float64).reshape(-1)
    if theta.size != expected_size:
        raise ValueError(
            f"--theta must provide {expected_size} comma-separated values; got {theta.size}"
        )
    return theta


def _resolve_feature_order(normalization_stats: dict[str, Any] | None) -> list[str]:
    if normalization_stats and bool(normalization_stats.get("enabled", False)):
        names = normalization_stats.get("feature_names", None)
        if isinstance(names, list) and names:
            return [str(x).strip() for x in names]
    return list(FEATURE_ORDER)


def _resolve_model_dir_name(
    *,
    project_root: Path,
    summary: dict[str, Any],
    summary_path: Path,
    model_dir_override: str | None,
) -> str:
    runs_root = project_root / "outputs" / "runs"
    if model_dir_override:
        return str(model_dir_override)

    raw_candidates: list[str] = []
    model_run_dir_raw = summary.get("model_run_dir", None)
    if model_run_dir_raw:
        raw_candidates.append(Path(str(model_run_dir_raw)).name)

    model_bucket_raw = summary.get("model_bucket", None)
    if model_bucket_raw:
        raw_candidates.append(str(model_bucket_raw).strip())

    # Most robust fallback: calibration folder structure is usually
    # outputs/calibration/<model_dir>/Calibration_<id>/summary.*
    raw_candidates.append(summary_path.parent.parent.name)

    candidates: list[str] = []
    for item in raw_candidates:
        name = str(item).strip()
        if not name or name in candidates:
            continue
        candidates.append(name)

    for name in candidates:
        if (runs_root / name).exists():
            return name

    if candidates:
        return candidates[0]
    return "latest"


def _build_objective_fn(
    *,
    model: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    feature_order: list[str],
    parameter_order: list[str],
    moneyness: np.ndarray,
    tau: np.ndarray,
    rate: np.ndarray,
    iv_market: np.ndarray,
    weights: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    regularization: str,
    lambda_reg: float,
) -> tuple[callable, dict[str, Any]]:
    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_quotes = int(iv_market.size)
    if n_quotes <= 0:
        raise ValueError("No quotes available to build objective")

    param_to_idx = {name: i for i, name in enumerate(parameter_order)}
    missing_params = [name for name in parameter_order if name not in feature_order]
    if missing_params:
        raise KeyError(
            f"Parameter(s) {missing_params} not found in model feature order={feature_order}"
        )

    required_market_features = ["moneyness", "tau", "r"]
    missing_market = [name for name in required_market_features if name not in feature_order]
    if missing_market:
        raise KeyError(
            f"Market feature(s) {missing_market} not found in model feature order={feature_order}"
        )

    m_t = torch.as_tensor(moneyness, dtype=dtype, device=device).reshape(-1)
    tau_t = torch.as_tensor(tau, dtype=dtype, device=device).reshape(-1)
    r_t = torch.as_tensor(rate, dtype=dtype, device=device).reshape(-1)
    iv_t = torch.as_tensor(iv_market, dtype=dtype, device=device).reshape(-1)
    w_t = torch.as_tensor(weights, dtype=dtype, device=device).reshape(-1)

    if normalization_stats and bool(normalization_stats.get("enabled", False)):
        x_mean = torch.as_tensor(
            normalization_stats.get("x_mean", []), dtype=dtype, device=device
        ).reshape(-1)
        x_std = torch.as_tensor(
            normalization_stats.get("x_std", []), dtype=dtype, device=device
        ).reshape(-1)
        if x_mean.numel() != len(feature_order) or x_std.numel() != len(feature_order):
            raise ValueError(
                "Normalization stats shape mismatch with feature order. "
                f"Expected {len(feature_order)}, got mean={x_mean.numel()}, std={x_std.numel()}."
            )
        x_std = torch.clamp(x_std, min=1.0e-12)

        normalize_target = bool(normalization_stats.get("normalize_target", False))
        y_mean = torch.as_tensor(
            float(normalization_stats.get("y_mean", 0.0)), dtype=dtype, device=device
        )
        y_std_scalar = float(normalization_stats.get("y_std", 1.0))
        if abs(y_std_scalar) < 1.0e-12:
            y_std_scalar = 1.0
        y_std = torch.as_tensor(y_std_scalar, dtype=dtype, device=device)
    else:
        x_mean = torch.zeros(len(feature_order), dtype=dtype, device=device)
        x_std = torch.ones(len(feature_order), dtype=dtype, device=device)
        normalize_target = False
        y_mean = torch.zeros((), dtype=dtype, device=device)
        y_std = torch.ones((), dtype=dtype, device=device)

    def _regularization(theta: torch.Tensor) -> torch.Tensor:
        reg = regularization.strip().lower()
        if reg == "none":
            return torch.zeros((), dtype=theta.dtype, device=theta.device)
        if reg == "l1":
            return torch.sum(torch.abs(theta))
        if reg == "l2":
            return torch.linalg.norm(theta, ord=2)
        if reg == "l2_squared":
            return torch.sum(theta**2)
        raise ValueError(
            "regularization must be one of {'none','l1','l2','l2_squared'}; "
            f"got '{regularization}'"
        )

    def objective(theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape={tuple(theta.shape)}")
        if theta.numel() != len(parameter_order):
            raise ValueError(
                "theta size mismatch. "
                f"Expected {len(parameter_order)}, got {theta.numel()}"
            )

        cols: list[torch.Tensor] = []
        for name in feature_order:
            if name in param_to_idx:
                cols.append(theta[param_to_idx[name]].expand(n_quotes))
            elif name == "moneyness":
                cols.append(m_t)
            elif name == "tau":
                cols.append(tau_t)
            elif name == "r":
                cols.append(r_t)
            else:
                raise KeyError(
                    f"Feature '{name}' cannot be built from theta+market inputs. "
                    "Supported non-parameter features are {'moneyness','tau','r'}."
                )

        x_raw = torch.stack(cols, dim=1)
        x_norm = (x_raw - x_mean.unsqueeze(0)) / x_std.unsqueeze(0)
        y_norm = model(x_norm).reshape(-1)
        y_pred = y_norm * y_std + y_mean if normalize_target else y_norm

        residual = y_pred - iv_t
        data_term = torch.sum(w_t * (residual**2))
        reg_term = _regularization(theta)
        return data_term + float(lambda_reg) * reg_term

    debug = {
        "n_quotes": n_quotes,
        "feature_order": feature_order,
        "parameter_order": parameter_order,
        "regularization": regularization,
        "lambda_reg": float(lambda_reg),
    }
    return objective, debug


def _default_output_json_path(summary_path: Path, theta_label: str) -> Path:
    return summary_path.parent / f"curvature_{theta_label}.json"


def _default_output_hessian_csv_path(summary_path: Path, theta_label: str) -> Path:
    return summary_path.parent / f"hessian_{theta_label}.csv"


def _default_output_jacobian_plot_path(summary_path: Path, theta_label: str) -> Path:
    return summary_path.parent / f"jacobian_{theta_label}.png"


def _default_output_hessian_plot_path(summary_path: Path, theta_label: str) -> Path:
    return summary_path.parent / f"hessian_{theta_label}.png"


def _save_jacobian_plot(
    *,
    jacobian: np.ndarray,
    parameter_order: list[str],
    output_path: Path,
    title: str,
) -> Path:
    colors = ["#2E6EA5" if x >= 0.0 else "#B44A3A" for x in jacobian]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(parameter_order, jacobian, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("dJ / dtheta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _save_hessian_plot(
    *,
    hessian: np.ndarray,
    parameter_order: list[str],
    output_path: Path,
    title: str,
) -> Path:
    def _fmt(v: float) -> str:
        av = abs(v)
        if av >= 1000.0 or (av > 0.0 and av < 1.0e-2):
            return f"{v:.2e}"
        return f"{v:.3f}"

    max_abs = float(np.nanmax(np.abs(hessian)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0e-12

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(
        hessian,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(parameter_order)))
    ax.set_yticks(np.arange(len(parameter_order)))
    ax.set_xticklabels(parameter_order, rotation=45, ha="right")
    ax.set_yticklabels(parameter_order)

    # Overlay each Hessian entry to ease direct reading from the heatmap.
    n_rows, n_cols = hessian.shape
    threshold = 0.45 * max_abs
    for i in range(n_rows):
        for j in range(n_cols):
            val = float(hessian[i, j])
            txt_color = "white" if abs(val) >= threshold else "black"
            ax.text(
                j,
                i,
                _fmt(val),
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("d2J / dtheta_i dtheta_j")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate calibration objective value/jacobian/hessian at theta* "
            "(or another theta) for a saved calibration run."
        )
    )
    parser.add_argument("--summary-path", default=None, help="Path to calibration summary.{json|yaml}")
    parser.add_argument(
        "--calibration-dir",
        default="latest",
        help="Path to calibration folder (contains summary), or 'latest'",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional calibration model bucket under outputs/calibration (e.g. MIX_v05).",
    )
    parser.add_argument(
        "--calibration-id",
        type=int,
        default=None,
        help="Optional Calibration_<id> within --model-dir bucket. If omitted, latest id is used.",
    )
    parser.add_argument(
        "--theta-key",
        default="theta_star",
        choices=["theta_star", "theta_true"],
        help="Summary key to use when --theta is not provided.",
    )
    parser.add_argument(
        "--theta",
        default=None,
        help="Optional manual theta as comma-separated values in parameter_order.",
    )
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=None,
        help="Optional override for regularization lambda.",
    )
    parser.add_argument(
        "--regularization",
        default=None,
        choices=["none", "l1", "l2", "l2_squared"],
        help="Optional override for regularization type.",
    )
    parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    parser.add_argument("--output-hessian-csv", default=None, help="Optional Hessian CSV path.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--output-jacobian-plot", default=None, help="Optional Jacobian plot path.")
    parser.add_argument("--output-hessian-plot", default=None, help="Optional Hessian heatmap path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    calibration_root = _load_calibration_output_root(PROJECT_ROOT)

    summary_path = _resolve_summary_path(
        project_root=PROJECT_ROOT,
        calibration_root=calibration_root,
        summary_path_arg=args.summary_path,
        calibration_dir_arg=args.calibration_dir,
        model_dir_arg=args.model_dir,
        calibration_id_arg=args.calibration_id,
    )
    summary = _load_summary(summary_path)

    parameter_order = [str(x) for x in summary.get("parameter_order", [])]
    if not parameter_order:
        raise KeyError(f"'parameter_order' not found in summary: {summary_path}")

    if args.theta:
        theta_np = _parse_theta_override(args.theta, expected_size=len(parameter_order))
        theta_label = "theta_manual"
        theta_source = "cli_override"
    else:
        if args.theta_key not in summary or summary.get(args.theta_key) is None:
            raise KeyError(
                f"'{args.theta_key}' not available in summary. "
                "Use --theta or choose another --theta-key."
            )
        theta_np = np.asarray(summary[args.theta_key], dtype=np.float64).reshape(-1)
        if theta_np.size != len(parameter_order):
            raise ValueError(
                f"{args.theta_key} size mismatch: expected {len(parameter_order)}, got {theta_np.size}"
            )
        theta_label = args.theta_key
        theta_source = f"summary:{args.theta_key}"

    regularization = (
        args.regularization
        if args.regularization is not None
        else str(summary.get("regularization", {}).get("type", "l2"))
    )
    lambda_reg = (
        float(args.lambda_reg)
        if args.lambda_reg is not None
        else float(summary.get("regularization", {}).get("lambda", 0.0))
    )

    quotes_path = _resolve_market_table_path(summary, summary_path)
    quotes_df = _read_quotes_table(quotes_path)

    cfg_used_path = summary_path.parent / "calibration_config_used.yaml"
    if cfg_used_path.exists():
        cfg_used = _load_yaml(cfg_used_path)
    else:
        cfg_used = {}

    data_cfg = cfg_used.get("data", {})
    col_cfg = data_cfg.get("columns", {})
    col_m = str(col_cfg.get("moneyness", "moneyness"))
    col_tau = str(col_cfg.get("tau", "tau"))
    col_r = str(col_cfg.get("r", "r"))
    col_iv = str(col_cfg.get("iv_market", "iv_market"))
    col_w = str(col_cfg.get("weight", "weight"))
    default_weight = float(data_cfg.get("default_weight", 1.0))

    for col in (col_m, col_tau, col_r, col_iv):
        if col not in quotes_df.columns:
            raise KeyError(
                f"Required column '{col}' not found in quotes file {quotes_path}. "
                f"Columns={list(quotes_df.columns)}"
            )

    moneyness = quotes_df[col_m].to_numpy(dtype=np.float64)
    tau = quotes_df[col_tau].to_numpy(dtype=np.float64)
    rate = quotes_df[col_r].to_numpy(dtype=np.float64)
    iv_market = quotes_df[col_iv].to_numpy(dtype=np.float64)
    if col_w in quotes_df.columns:
        weights = quotes_df[col_w].to_numpy(dtype=np.float64)
    else:
        weights = np.full(iv_market.size, fill_value=default_weight, dtype=np.float64)

    model_dir = _resolve_model_dir_name(
        project_root=PROJECT_ROOT,
        summary=summary,
        summary_path=summary_path,
        model_dir_override=args.model_dir,
    )
    checkpoint_name = str(summary.get("checkpoint", "model_best.pt"))

    resolved_device = resolve_device(args.device)
    if resolved_device.type == "mps" and dtype == torch.float64:
        # float64 second-order autodiff is not reliable on MPS.
        resolved_device = torch.device("cpu")

    model, _, run_dir, _, normalization_stats = load_model_from_run(
        project_root=PROJECT_ROOT,
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        device=str(resolved_device),
    )

    feature_order = _resolve_feature_order(normalization_stats)
    objective_fn, debug = _build_objective_fn(
        model=model,
        dtype=dtype,
        device=resolved_device,
        feature_order=feature_order,
        parameter_order=parameter_order,
        moneyness=moneyness,
        tau=tau,
        rate=rate,
        iv_market=iv_market,
        weights=weights,
        normalization_stats=normalization_stats,
        regularization=regularization,
        lambda_reg=lambda_reg,
    )

    point = torch.as_tensor(theta_np, dtype=dtype, device=resolved_device)
    diff = derivatives_point(
        objective_fn,
        point,
        dtype=dtype,
        device=resolved_device,
    )

    value = float(diff.value.detach().cpu().item())
    jac = diff.jacobian.detach().cpu().numpy().astype(np.float64, copy=False)
    hess = diff.hessian.detach().cpu().numpy().astype(np.float64, copy=False)

    hess_sym = 0.5 * (hess + hess.T)
    eigvals = np.linalg.eigvalsh(hess_sym)
    grad_norm = float(np.linalg.norm(jac, ord=2))
    min_abs_eig = float(np.min(np.abs(eigvals)))
    max_abs_eig = float(np.max(np.abs(eigvals)))
    cond_abs = float(max_abs_eig / min_abs_eig) if min_abs_eig > 0.0 else None
    symmetry_error = float(np.max(np.abs(hess - hess.T)))

    output = {
        "summary_path": str(summary_path),
        "calibration_dir": str(summary_path.parent),
        "model_run_dir": str(run_dir),
        "quotes_path": str(quotes_path),
        "device": str(resolved_device),
        "dtype": str(dtype).replace("torch.", ""),
        "theta_source": theta_source,
        "theta_label": theta_label,
        "parameter_order": parameter_order,
        "theta": [float(x) for x in theta_np],
        "objective_value": value,
        "jacobian": [float(x) for x in jac],
        "hessian": [[float(x) for x in row] for row in hess],
        "diagnostics": {
            "grad_l2_norm": grad_norm,
            "hessian_symmetry_max_abs_diff": symmetry_error,
            "hessian_eigvals_sym": [float(x) for x in eigvals],
            "hessian_eig_min": float(np.min(eigvals)),
            "hessian_eig_max": float(np.max(eigvals)),
            "hessian_is_psd": bool(float(np.min(eigvals)) >= -1.0e-10),
            "hessian_cond_abs": cond_abs,
        },
        "objective_debug": debug,
        "artifacts": {},
    }

    if args.output_json:
        output_json = _resolve_path(args.output_json, base_dir=PROJECT_ROOT)
    else:
        output_json = _default_output_json_path(summary_path, theta_label)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if args.output_hessian_csv:
        output_hessian_csv = _resolve_path(args.output_hessian_csv, base_dir=PROJECT_ROOT)
    else:
        output_hessian_csv = _default_output_hessian_csv_path(summary_path, theta_label)
    output_hessian_csv.parent.mkdir(parents=True, exist_ok=True)
    hess_df = pd.DataFrame(hess, index=parameter_order, columns=parameter_order)
    hess_df.to_csv(output_hessian_csv, index=True)

    output["artifacts"]["json"] = str(output_json)
    output["artifacts"]["hessian_csv"] = str(output_hessian_csv)

    jacobian_plot_path = None
    hessian_plot_path = None
    if not args.no_plots:
        if args.output_jacobian_plot:
            jacobian_plot_path = _resolve_path(args.output_jacobian_plot, base_dir=PROJECT_ROOT)
        else:
            jacobian_plot_path = _default_output_jacobian_plot_path(summary_path, theta_label)
        if args.output_hessian_plot:
            hessian_plot_path = _resolve_path(args.output_hessian_plot, base_dir=PROJECT_ROOT)
        else:
            hessian_plot_path = _default_output_hessian_plot_path(summary_path, theta_label)

        _save_jacobian_plot(
            jacobian=jac,
            parameter_order=parameter_order,
            output_path=jacobian_plot_path,
            title=f"Jacobian at {theta_label}",
        )
        _save_hessian_plot(
            hessian=hess,
            parameter_order=parameter_order,
            output_path=hessian_plot_path,
            title=f"Hessian at {theta_label}",
        )
        output["artifacts"]["jacobian_plot"] = str(jacobian_plot_path)
        output["artifacts"]["hessian_plot"] = str(hessian_plot_path)

        # Re-write JSON including generated plot paths.
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    print(f"Summary: {summary_path}")
    print(f"Run dir: {run_dir}")
    print(f"Theta source: {theta_source}")
    print(f"Objective J(theta): {value:.10e}")
    print(f"||grad J||_2: {grad_norm:.10e}")
    print(f"min eig(H_sym): {float(np.min(eigvals)):.10e}")
    print(f"max eig(H_sym): {float(np.max(eigvals)):.10e}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved Hessian CSV: {output_hessian_csv}")
    if jacobian_plot_path is not None:
        print(f"Saved Jacobian plot: {jacobian_plot_path}")
    if hessian_plot_path is not None:
        print(f"Saved Hessian plot: {hessian_plot_path}")


if __name__ == "__main__":
    main()
