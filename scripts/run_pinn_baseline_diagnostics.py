from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

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
from src.greeks.core import derivatives_batch, greeks_from_jacobian_hessian, values_batch
from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.greeks.names import build_greek_index_spec, parse_feature_order
from src.greeks.pinn_adapter import DEFAULT_PINN_FEATURE_ORDER, load_pinn_price_adapter
from src.pinn.global_acv_pinn import head_greeks_x_to_financial
from src.pinn.losses import (
    compute_heston_pde_derivative_residual,
    compute_heston_pde_residual,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pinn_baseline_diagnostics.yaml"
PINN_PDE_FEATURE_ORDER = ["tau", "moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
PINN_LOG_PDE_FEATURE_ORDER = ["tau", "log_moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
DEFAULT_FIXED_VALUES = {
    "tau": 1.0,
    "moneyness": 1.0,
    "log_moneyness": 0.0,
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
    raise ValueError("dtype must be one of {float64, float32}")


def _parse_feature_order(raw: str | list[str] | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return parse_feature_order(raw.split(","), fallback=DEFAULT_PINN_FEATURE_ORDER)
    if isinstance(raw, list):
        return parse_feature_order(raw, fallback=DEFAULT_PINN_FEATURE_ORDER)
    raise ValueError("feature_order must be null, comma-separated string, or list")


def _none_if_empty(raw: str | None) -> str | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt.lower() in {"", "none", "null"}:
        return None
    return txt


def _resolve_defaults(feature_order: list[str], fixed_values: dict | None) -> np.ndarray:
    fixed = fixed_values or {}
    out = np.zeros(len(feature_order), dtype=np.float64)
    for i, name in enumerate(feature_order):
        if name in fixed:
            out[i] = float(fixed[name])
        elif name == "log_moneyness" and "moneyness" in fixed:
            m = float(fixed["moneyness"])
            if m <= 0.0:
                raise ValueError("fixed_values.moneyness must be > 0 when feature_order uses log_moneyness.")
            out[i] = float(np.log(m))
        elif name in DEFAULT_FIXED_VALUES:
            out[i] = float(DEFAULT_FIXED_VALUES[name])
        else:
            out[i] = 0.0
    return out


def _storage_feature_for_grid(display_feature: str, feature_order: list[str]) -> str:
    if display_feature in feature_order:
        return display_feature
    if display_feature == "moneyness" and "log_moneyness" in feature_order:
        return "log_moneyness"
    raise KeyError(
        f"surface_grid feature '{display_feature}' cannot be resolved for feature_order={feature_order}."
    )


def _grid_values_for_storage(display_feature: str, storage_feature: str, values: np.ndarray) -> np.ndarray:
    if display_feature == "moneyness" and storage_feature == "log_moneyness":
        if (values <= 0.0).any():
            raise ValueError("surface_grid moneyness values must be > 0 for log_moneyness models.")
        return np.log(values)
    return values


def _is_log_moneyness_spot(spot_feature: str) -> bool:
    return str(spot_feature).strip().lower() in {"log_moneyness", "log-moneyness", "x"}


def _build_surface_grid(*, feature_order: list[str], diagnostics_cfg: dict) -> np.ndarray:
    grid_cfg = diagnostics_cfg.get("surface_grid", {})
    if not isinstance(grid_cfg, dict):
        raise ValueError("diagnostics.surface_grid must be a dictionary")

    x_feature = str(grid_cfg.get("x_feature", "moneyness"))
    y_feature = str(grid_cfg.get("y_feature", "tau"))
    if x_feature == y_feature:
        raise ValueError("surface_grid x_feature and y_feature must be different")
    x_storage_feature = _storage_feature_for_grid(x_feature, feature_order)
    y_storage_feature = _storage_feature_for_grid(y_feature, feature_order)
    if x_storage_feature == y_storage_feature:
        raise ValueError("surface_grid x_feature and y_feature resolve to the same model coordinate")

    x_min = float(grid_cfg.get("x_min"))
    x_max = float(grid_cfg.get("x_max"))
    y_min = float(grid_cfg.get("y_min"))
    y_max = float(grid_cfg.get("y_max"))
    x_points = int(grid_cfg.get("x_points", 181))
    y_points = int(grid_cfg.get("y_points", 181))
    if x_points < 2 or y_points < 2:
        raise ValueError("surface_grid x_points and y_points must be >= 2")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("surface_grid ranges must satisfy max > min")

    base = _resolve_defaults(feature_order, diagnostics_cfg.get("fixed_values"))
    idx_x = feature_order.index(x_storage_feature)
    idx_y = feature_order.index(y_storage_feature)
    x_axis = np.linspace(x_min, x_max, x_points, dtype=np.float64)
    y_axis = np.linspace(y_min, y_max, y_points, dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    xx_storage = _grid_values_for_storage(x_feature, x_storage_feature, xx)
    yy_storage = _grid_values_for_storage(y_feature, y_storage_feature, yy)

    out = np.repeat(base.reshape(1, -1), repeats=xx.size, axis=0)
    out[:, idx_x] = xx_storage.reshape(-1)
    out[:, idx_y] = yy_storage.reshape(-1)
    return out


def _compute_metrics(y_pred: np.ndarray, y_ref: np.ndarray, *, mape_floor: float) -> dict[str, float]:
    err = y_pred - y_ref
    abs_err = np.abs(err)
    mse = float(np.mean(err**2))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    denom = np.maximum(np.abs(y_ref), float(mape_floor))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "mape_pct": float(100.0 * np.mean(abs_err / denom)),
        "p90_abs_error": float(np.percentile(abs_err, 90.0)),
        "p99_abs_error": float(np.percentile(abs_err, 99.0)),
        "max_abs_error": float(np.max(abs_err)),
    }


def _compute_error_metrics(y_pred: np.ndarray, y_ref: np.ndarray) -> dict[str, float]:
    err = y_pred - y_ref
    abs_err = np.abs(err)
    mse = float(np.mean(err**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_err)),
        "p90_abs_error": float(np.percentile(abs_err, 90.0)),
        "p99_abs_error": float(np.percentile(abs_err, 99.0)),
        "max_abs_error": float(np.max(abs_err)),
    }


def _compute_absolute_metrics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mape_pct": float("nan"),
            "p90_abs_error": float("nan"),
            "p99_abs_error": float("nan"),
            "max_abs_error": float("nan"),
        }
    abs_values = np.abs(finite)
    mse = float(np.mean(finite**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_values)),
        "r2": float("nan"),
        "mape_pct": float("nan"),
        "p90_abs_error": float(np.percentile(abs_values, 90.0)),
        "p99_abs_error": float(np.percentile(abs_values, 99.0)),
        "max_abs_error": float(np.max(abs_values)),
    }


def _compute_stabilized_relative_metrics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean_stabilized_rel_error": float("nan"),
            "p90_stabilized_rel_error": float("nan"),
            "p99_stabilized_rel_error": float("nan"),
            "max_stabilized_rel_error": float("nan"),
        }
    return {
        "mean_stabilized_rel_error": float(np.mean(finite)),
        "p90_stabilized_rel_error": float(np.percentile(finite, 90.0)),
        "p99_stabilized_rel_error": float(np.percentile(finite, 99.0)),
        "max_stabilized_rel_error": float(np.max(finite)),
    }


def _region_masks(
    *,
    points: pd.DataFrame,
    spot_feature: str,
    tau_feature: str,
    epsilon_m: float,
    epsilon_tau: float,
) -> dict[str, np.ndarray]:
    if spot_feature not in points.columns or tau_feature not in points.columns:
        raise KeyError(f"Missing region features spot={spot_feature}, tau={tau_feature}")
    m = points[spot_feature].to_numpy(dtype=np.float64)
    tau = points[tau_feature].to_numpy(dtype=np.float64)
    atm = np.abs(np.log(np.maximum(m, np.finfo(np.float64).tiny))) < float(epsilon_m)
    short = tau < float(epsilon_tau)
    hard = short & atm
    return {
        "full": np.ones(points.shape[0], dtype=bool),
        "non_hard": ~hard,
        "short_maturity": short,
        "atm": atm,
        "hard": hard,
    }


def _build_heatmap(
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = float(np.min(x_axis)), float(np.max(x_axis))
    y_min, y_max = float(np.min(y_axis)), float(np.max(y_axis))
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
    plot_matrix = matrix
    norm = None
    if log_scale:
        valid = np.isfinite(matrix) & (matrix > 0.0)
        if np.any(valid):
            vals = matrix[valid]
            vmin = max(float(np.percentile(vals, 5.0)), float(np.finfo(np.float64).tiny))
            vmax = float(np.percentile(vals, 95.0))
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


def _predict_shifted(
    *,
    price_fn: Callable[[torch.Tensor], torch.Tensor],
    x_base: torch.Tensor,
    idx: int,
    shift: float,
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    x_shifted = x_base.detach().clone()
    x_shifted[:, idx] += float(shift)
    return values_batch(
        price_fn,
        x_shifted,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    ).reshape(-1)


def _finite_difference_consistency(
    *,
    price_fn: Callable[[torch.Tensor], torch.Tensor],
    x_eval: np.ndarray,
    feature_order: list[str],
    spec,
    spot_feature: str,
    strike: float,
    theta_sign: str,
    fd_cfg: dict,
    dtype: torch.dtype,
    device: torch.device,
    ad_greeks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[dict]]:
    if not bool(fd_cfg.get("enabled", True)):
        return pd.DataFrame(), []

    max_points = int(fd_cfg.get("max_points", 2048))
    seed = int(fd_cfg.get("seed", 42))
    n = x_eval.shape[0]
    if max_points > 0 and max_points < n:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        rows = np.arange(n)

    x_np = x_eval[rows].copy()
    x_t = torch.from_numpy(x_np).to(device=device, dtype=dtype)
    value_chunk = int(fd_cfg.get("chunk_size_values", 4096))

    out = pd.DataFrame(x_np, columns=feature_order)
    out["row_index"] = rows
    metrics_rows: list[dict] = []

    def add_metric(name: str, fd_values: torch.Tensor, ad_values: np.ndarray) -> None:
        fd_np = fd_values.detach().cpu().numpy().astype(np.float64, copy=False)
        ad_np = np.asarray(ad_values[rows], dtype=np.float64)
        finite = np.isfinite(fd_np) & np.isfinite(ad_np)
        out[f"fd_{name}"] = fd_np
        out[f"ad_{name}"] = ad_np
        out[f"ad_minus_fd_{name}"] = ad_np - fd_np
        if np.any(finite):
            metrics_rows.append(
                {
                    "greek": name,
                    "n_points": int(np.sum(finite)),
                    **_compute_error_metrics(ad_np[finite], fd_np[finite]),
                }
            )

    h_spot = float(fd_cfg.get("spot_step", 1.0e-4))
    vp = _predict_shifted(
        price_fn=price_fn,
        x_base=x_t,
        idx=spec.idx_spot,
        shift=h_spot,
        chunk_size=value_chunk,
        dtype=dtype,
        device=device,
    )
    vm = _predict_shifted(
        price_fn=price_fn,
        x_base=x_t,
        idx=spec.idx_spot,
        shift=-h_spot,
        chunk_size=value_chunk,
        dtype=dtype,
        device=device,
    )
    v0 = values_batch(
        price_fn,
        x_t,
        chunk_size=value_chunk,
        dtype=dtype,
        device=device,
    ).reshape(-1)
    first_spot_fd = (vp - vm) / (2.0 * h_spot)
    second_spot_fd = (vp - 2.0 * v0 + vm) / (h_spot * h_spot)
    if _is_log_moneyness_spot(spot_feature):
        x_coord = x_t[:, spec.idx_spot]
        delta_fd = torch.exp(-x_coord) * first_spot_fd / float(strike)
        gamma_fd = torch.exp(-2.0 * x_coord) * (second_spot_fd - first_spot_fd) / (float(strike) ** 2)
    else:
        spot_scale = 1.0 / float(strike) if spot_feature == "moneyness" else 1.0
        delta_fd = first_spot_fd * spot_scale
        gamma_fd = second_spot_fd * spot_scale * spot_scale
    add_metric("delta", delta_fd, ad_greeks["delta"])
    add_metric("gamma", gamma_fd, ad_greeks["gamma"])

    if spec.idx_vol is not None and "vega" in ad_greeks:
        h_vol = float(fd_cfg.get("vol_step", 1.0e-4))
        vp = _predict_shifted(
            price_fn=price_fn,
            x_base=x_t,
            idx=spec.idx_vol,
            shift=h_vol,
            chunk_size=value_chunk,
            dtype=dtype,
            device=device,
        )
        vm = _predict_shifted(
            price_fn=price_fn,
            x_base=x_t,
            idx=spec.idx_vol,
            shift=-h_vol,
            chunk_size=value_chunk,
            dtype=dtype,
            device=device,
        )
        add_metric("vega", (vp - vm) / (2.0 * h_vol), ad_greeks["vega"])

    if spec.idx_tau is not None and "theta" in ad_greeks:
        h_tau = float(fd_cfg.get("tau_step", 1.0e-4))
        valid = x_np[:, spec.idx_tau] > h_tau
        if np.any(valid):
            x_tau = x_t[valid]
            vp = _predict_shifted(
                price_fn=price_fn,
                x_base=x_tau,
                idx=spec.idx_tau,
                shift=h_tau,
                chunk_size=value_chunk,
                dtype=dtype,
                device=device,
            )
            vm = _predict_shifted(
                price_fn=price_fn,
                x_base=x_tau,
                idx=spec.idx_tau,
                shift=-h_tau,
                chunk_size=value_chunk,
                dtype=dtype,
                device=device,
            )
            theta_fd = (vp - vm) / (2.0 * h_tau)
            if theta_sign == "minus_dv_dtau":
                theta_fd = -theta_fd
            theta_full = np.full(x_np.shape[0], np.nan, dtype=np.float64)
            theta_full[valid] = theta_fd.detach().cpu().numpy()
            add_metric("theta", torch.from_numpy(theta_full).to(dtype=dtype), ad_greeks["theta"])

    if spec.idx_rate is not None and "rho" in ad_greeks:
        h_rate = float(fd_cfg.get("rate_step", 1.0e-4))
        vp = _predict_shifted(
            price_fn=price_fn,
            x_base=x_t,
            idx=spec.idx_rate,
            shift=h_rate,
            chunk_size=value_chunk,
            dtype=dtype,
            device=device,
        )
        vm = _predict_shifted(
            price_fn=price_fn,
            x_base=x_t,
            idx=spec.idx_rate,
            shift=-h_rate,
            chunk_size=value_chunk,
            dtype=dtype,
            device=device,
        )
        add_metric("rho", (vp - vm) / (2.0 * h_rate), ad_greeks["rho"])

    return out, metrics_rows


def _compute_multi_output_head_greeks(
    *,
    model: torch.nn.Module,
    x_eval: np.ndarray,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    spot_feature: str,
    spot_index: int,
    strike: float,
    dtype: torch.dtype,
    device: torch.device,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    forward_all = getattr(model, "forward_all", None)
    if not callable(forward_all):
        return {}
    if chunk_size <= 0:
        raise ValueError("head-greek chunk_size must be > 0")

    values: list[np.ndarray] = []
    model.eval()
    for start in range(0, x_eval.shape[0], chunk_size):
        stop = min(start + chunk_size, x_eval.shape[0])
        x_raw = torch.from_numpy(x_eval[start:stop]).to(device=device, dtype=dtype)
        x_model = input_a.to(dtype=dtype, device=device).view(1, -1) + input_b.to(
            dtype=dtype,
            device=device,
        ).view(1, -1) * x_raw
        with torch.no_grad():
            heads = forward_all(x_model)
        values.append(heads.detach().cpu().numpy().astype(np.float64, copy=False))
    out = np.concatenate(values, axis=0)
    if out.ndim != 2 or out.shape[1] < 4:
        return {}

    delta = out[:, 1]
    gamma = out[:, 2]
    if spot_feature == "moneyness":
        delta = delta / strike
        gamma = gamma / (strike**2)
    elif _is_log_moneyness_spot(spot_feature):
        x_coord = torch.as_tensor(x_eval[:, spot_index], dtype=torch.float64)
        delta_t, gamma_t = head_greeks_x_to_financial(
            u_x=torch.as_tensor(delta, dtype=torch.float64),
            u_xx=torch.as_tensor(gamma, dtype=torch.float64),
            x=x_coord,
            strike=strike,
        )
        delta = delta_t.detach().cpu().numpy()
        gamma = gamma_t.detach().cpu().numpy()
    return {
        "delta": delta.astype(np.float64, copy=False),
        "gamma": gamma.astype(np.float64, copy=False),
        "vega": out[:, 3].astype(np.float64, copy=False),
    }


def _compute_pde_derivative_residuals(
    *,
    model: torch.nn.Module,
    x_eval: np.ndarray,
    input_affine: dict,
    coordinate: str,
    dtype: torch.dtype,
    device: torch.device,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    enabled = bool(cfg.get("enabled", False))
    n = x_eval.shape[0]
    out_dx = np.full(n, np.nan, dtype=np.float64)
    out_dv = np.full(n, np.nan, dtype=np.float64)
    if not enabled:
        return out_dx, out_dv

    chunk_size = int(cfg.get("chunk_size", 128))
    if chunk_size <= 0:
        raise ValueError("diagnostics.derivative_residual.chunk_size must be > 0")

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        x_batch = torch.from_numpy(x_eval[start:stop]).to(device=device, dtype=dtype)
        with torch.enable_grad():
            _, residual_dx, residual_dv = compute_heston_pde_derivative_residual(
                model=model,
                x_interior=x_batch,
                input_affine=input_affine,
                coordinate=coordinate,
            )
        out_dx[start:stop] = residual_dx.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        out_dv[start:stop] = residual_dv.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    return out_dx, out_dv


def _surface_shape_from_config(diagnostics_cfg: dict) -> tuple[int, int] | None:
    grid_cfg = diagnostics_cfg.get("surface_grid", {})
    if not isinstance(grid_cfg, dict):
        return None
    x_points = int(grid_cfg.get("x_points", 181))
    y_points = int(grid_cfg.get("y_points", 181))
    if x_points <= 1 or y_points <= 1:
        return None
    return x_points, y_points


def _smoothness_rows(
    *,
    points_df: pd.DataFrame,
    diagnostics_cfg: dict,
    variables: list[str],
) -> list[dict]:
    shape = _surface_shape_from_config(diagnostics_cfg)
    if shape is None:
        return []
    x_points, y_points = shape
    if len(points_df) != x_points * y_points:
        return []

    rows: list[dict] = []
    for variable in variables:
        col = f"pinn_{variable}" if variable != "price" else "pinn_price"
        if col not in points_df.columns:
            continue
        grid = points_df[col].to_numpy(dtype=np.float64).reshape(x_points, y_points)
        for axis_name, diff in (
            ("surface_x", np.diff(grid, axis=0)),
            ("surface_y", np.diff(grid, axis=1)),
        ):
            finite = np.abs(diff[np.isfinite(diff)])
            if finite.size == 0:
                continue
            rows.append(
                {
                    "check": "smoothness_neighbor_jump",
                    "variable": variable,
                    "axis": axis_name,
                    "n_points": int(finite.size),
                    "mean_abs_jump": float(np.mean(finite)),
                    "p90_abs_jump": float(np.percentile(finite, 90.0)),
                    "p99_abs_jump": float(np.percentile(finite, 99.0)),
                    "max_abs_jump": float(np.max(finite)),
                }
            )
    return rows


def _no_reference_diagnostics(
    *,
    points_df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    diagnostics_cfg: dict,
    option_type: str,
    variables: list[str],
) -> list[dict]:
    cfg = diagnostics_cfg.get("no_reference", {})
    if not isinstance(cfg, dict):
        cfg = {}
    tol = float(cfg.get("tolerance", 1.0e-8))
    rows: list[dict] = []

    def add_violation_rows(*, check: str, variable: str, values: np.ndarray) -> None:
        finite_all = np.isfinite(values)
        for region, mask in masks.items():
            region_values = values[mask]
            finite = finite_all[mask]
            n_finite = int(np.sum(finite))
            violations = np.isfinite(region_values) & (region_values > 0.0)
            rows.append(
                {
                    "check": check,
                    "region": region,
                    "variable": variable,
                    "n_points": n_finite,
                    "n_violations": int(np.sum(violations)),
                    "violation_rate": (
                        float(np.sum(violations) / n_finite) if n_finite > 0 else float("nan")
                    ),
                    "max_violation": (
                        float(np.nanmax(region_values[violations])) if np.any(violations) else 0.0
                    ),
                }
            )

    if "pinn_price" in points_df.columns:
        add_violation_rows(
            check="nonnegative_price",
            variable="price",
            values=np.maximum(-points_df["pinn_price"].to_numpy(dtype=np.float64) - tol, 0.0),
        )

    if "pinn_delta" in points_df.columns:
        delta = points_df["pinn_delta"].to_numpy(dtype=np.float64)
        if option_type == "put":
            add_violation_rows(
                check="put_delta_upper_bound_delta_le_0",
                variable="delta",
                values=np.maximum(delta - tol, 0.0),
            )
            add_violation_rows(
                check="put_delta_lower_bound_delta_ge_minus_1",
                variable="delta",
                values=np.maximum(-1.0 - delta - tol, 0.0),
            )
        elif option_type == "call":
            add_violation_rows(
                check="call_delta_lower_bound_delta_ge_0",
                variable="delta",
                values=np.maximum(-delta - tol, 0.0),
            )
            add_violation_rows(
                check="call_delta_upper_bound_delta_le_1",
                variable="delta",
                values=np.maximum(delta - 1.0 - tol, 0.0),
            )

    if "pinn_gamma" in points_df.columns:
        add_violation_rows(
            check="convexity_gamma_ge_0",
            variable="gamma",
            values=np.maximum(-points_df["pinn_gamma"].to_numpy(dtype=np.float64) - tol, 0.0),
        )

    for variable in variables:
        col = f"pinn_{variable}" if variable != "price" else "pinn_price"
        if col not in points_df.columns:
            continue
        values = points_df[col].to_numpy(dtype=np.float64)
        for region, mask in masks.items():
            finite = np.isfinite(values[mask])
            rows.append(
                {
                    "check": "finite_values",
                    "region": region,
                    "variable": variable,
                    "n_points": int(mask.sum()),
                    "n_violations": int(mask.sum() - np.sum(finite)),
                    "violation_rate": (
                        float((mask.sum() - np.sum(finite)) / mask.sum()) if mask.sum() > 0 else float("nan")
                    ),
                    "max_violation": 0.0,
                }
            )

    for row in _smoothness_rows(
        points_df=points_df,
        diagnostics_cfg=diagnostics_cfg,
        variables=variables,
    ):
        row.setdefault("region", "full")
        row.setdefault("n_violations", 0)
        row.setdefault("violation_rate", float("nan"))
        row.setdefault("max_violation", float("nan"))
        rows.append(row)

    return rows


def _boundary_condition_rows(
    *,
    price_fn: Callable[[torch.Tensor], torch.Tensor],
    feature_order: list[str],
    diagnostics_cfg: dict,
    spot_feature: str,
    tau_feature: str,
    rate_feature: str,
    strike: float,
    option_type: str,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict]:
    cfg = diagnostics_cfg.get("boundary_checks", {})
    if not isinstance(cfg, dict):
        cfg = {}
    if not bool(cfg.get("enabled", True)):
        return []
    if spot_feature not in feature_order or tau_feature not in feature_order or rate_feature not in feature_order:
        return []

    fixed = _resolve_defaults(feature_order, diagnostics_cfg.get("fixed_values"))
    grid_cfg = diagnostics_cfg.get("surface_grid", {})
    n_points = int(cfg.get("n_points", grid_cfg.get("x_points", 181) if isinstance(grid_cfg, dict) else 181))
    if n_points < 2:
        return []
    tol = float(cfg.get("tolerance", 1.0e-4))
    chunk_size = int(cfg.get("chunk_size_values", 4096))

    idx_spot = feature_order.index(spot_feature)
    idx_tau = feature_order.index(tau_feature)
    idx_rate = feature_order.index(rate_feature)
    x_min = float(grid_cfg.get("x_min", 0.0)) if isinstance(grid_cfg, dict) else 0.0
    x_max = float(grid_cfg.get("x_max", 2.0)) if isinstance(grid_cfg, dict) else 2.0
    y_min = float(grid_cfg.get("y_min", 0.0)) if isinstance(grid_cfg, dict) else 0.0
    y_max = float(grid_cfg.get("y_max", 3.0)) if isinstance(grid_cfg, dict) else 3.0

    rows: list[dict] = []
    log_spot = _is_log_moneyness_spot(spot_feature)

    terminal = np.repeat(fixed.reshape(1, -1), repeats=n_points, axis=0)
    terminal[:, idx_tau] = 0.0
    terminal_spot_axis = np.linspace(x_min, x_max, n_points, dtype=np.float64)
    if log_spot:
        terminal_spot_axis = np.maximum(terminal_spot_axis, np.finfo(np.float64).tiny)
        terminal[:, idx_spot] = np.log(terminal_spot_axis)
    else:
        terminal[:, idx_spot] = terminal_spot_axis
    terminal_t = torch.from_numpy(terminal).to(device=device, dtype=dtype)
    pred_terminal = values_batch(
        price_fn,
        terminal_t,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    ).detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    if log_spot:
        s_terminal = terminal_spot_axis * strike
    else:
        s_terminal = terminal[:, idx_spot] * strike if spot_feature == "moneyness" else terminal[:, idx_spot]
    if option_type == "call":
        target_terminal = np.maximum(s_terminal - strike, 0.0)
    else:
        target_terminal = np.maximum(strike - s_terminal, 0.0)
    err_terminal = pred_terminal - target_terminal
    abs_terminal = np.abs(err_terminal)
    rows.append(
        {
            "check": "terminal_payoff_boundary",
            "region": "terminal",
            "variable": "price",
            "n_points": int(n_points),
            "n_violations": int(np.sum(abs_terminal > tol)),
            "violation_rate": float(np.mean(abs_terminal > tol)),
            "max_violation": float(np.max(abs_terminal)),
            "rmse": float(np.sqrt(np.mean(err_terminal**2))),
            "mae": float(np.mean(abs_terminal)),
            "p99_abs_error": float(np.percentile(abs_terminal, 99.0)),
        }
    )

    lower = np.repeat(fixed.reshape(1, -1), repeats=n_points, axis=0)
    lower_moneyness = float(cfg.get("lower_moneyness", 0.0))
    if log_spot:
        lower_moneyness = float(cfg.get("lower_moneyness", max(x_min, 1.0e-4)))
        lower_moneyness = max(lower_moneyness, np.finfo(np.float64).tiny)
        lower[:, idx_spot] = np.log(lower_moneyness)
        lower_spot = np.full(n_points, lower_moneyness * strike, dtype=np.float64)
    else:
        lower[:, idx_spot] = lower_moneyness
        lower_spot = lower[:, idx_spot] * strike if spot_feature == "moneyness" else lower[:, idx_spot]
    lower[:, idx_tau] = np.linspace(max(0.0, y_min), max(0.0, y_max), n_points, dtype=np.float64)
    lower_t = torch.from_numpy(lower).to(device=device, dtype=dtype)
    pred_lower = values_batch(
        price_fn,
        lower_t,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    ).detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    r_lower = lower[:, idx_rate]
    tau_lower = lower[:, idx_tau]
    target_lower = np.zeros_like(pred_lower)
    if option_type == "put":
        target_lower = np.maximum(strike * np.exp(-r_lower * tau_lower) - lower_spot, 0.0)
    err_lower = pred_lower - target_lower
    abs_lower = np.abs(err_lower)
    rows.append(
        {
            "check": "lower_spot_boundary",
            "region": "lower_spot",
            "variable": "price",
            "n_points": int(n_points),
            "n_violations": int(np.sum(abs_lower > tol)),
            "violation_rate": float(np.mean(abs_lower > tol)),
            "max_violation": float(np.max(abs_lower)),
            "rmse": float(np.sqrt(np.mean(err_lower**2))),
            "mae": float(np.mean(abs_lower)),
            "p99_abs_error": float(np.percentile(abs_lower, 99.0)),
        }
    )
    return rows


def _reference_heston(
    *,
    x_eval: np.ndarray,
    feature_order: list[str],
    spec,
    spot_feature: str,
    strike: float,
    option_type: str,
    cf_settings: HestonCFGreeksSettings,
) -> tuple[dict[str, np.ndarray], int, float]:
    idx = {name: i for i, name in enumerate(feature_order)}
    for req in ("rho", "kappa", "gamma", "bar_v"):
        if req not in idx:
            raise KeyError(f"Required Heston feature '{req}' missing in feature_order={feature_order}")
    if spec.idx_vol is None or spec.idx_tau is None or spec.idx_rate is None:
        raise ValueError("Heston benchmark requires vol, tau and rate features.")

    ref = {
        key: np.full(x_eval.shape[0], np.nan, dtype=np.float64)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho")
    }
    failed = 0
    t0 = time.perf_counter()
    for i, row in enumerate(x_eval):
        spot_raw = float(row[spec.idx_spot])
        if _is_log_moneyness_spot(spot_feature):
            s0 = float(np.exp(spot_raw) * strike)
        else:
            s0 = spot_raw * strike if spot_feature == "moneyness" else spot_raw
        try:
            vals = heston_cf_greeks_scalar(
                option_type=option_type,
                S0=s0,
                K=strike,
                tau=float(row[spec.idx_tau]),
                r=float(row[spec.idx_rate]),
                rho=float(row[idx["rho"]]),
                kappa=float(row[idx["kappa"]]),
                gamma=float(row[idx["gamma"]]),
                bar_v=float(row[idx["bar_v"]]),
                v0=float(row[spec.idx_vol]),
                settings=cf_settings,
            )
            for key in ref:
                ref[key][i] = float(vals[key])
        except Exception:
            failed += 1
    return ref, failed, float(time.perf_counter() - t0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline PINN diagnostics for price and Greeks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to diagnostics YAML config.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    cfg = _load_yaml(config_path)
    global_cfg = cfg.get("global", {})
    diagnostics_cfg = cfg.get("diagnostics", {})
    if not isinstance(global_cfg, dict) or not isinstance(diagnostics_cfg, dict):
        raise ValueError("config must contain global and diagnostics dictionaries")

    dtype = _parse_dtype(global_cfg.get("dtype", "float64"))
    loaded = load_pinn_price_adapter(
        project_root=PROJECT_ROOT,
        run_dir=str(global_cfg.get("run_dir", "latest")),
        checkpoint_name=str(global_cfg.get("checkpoint_name", "model_best.pt")),
        architecture_config_path=global_cfg.get("architecture_config"),
        device=str(global_cfg.get("device", "auto")),
        dtype=dtype,
        feature_order=_parse_feature_order(global_cfg.get("feature_order")),
    )
    feature_order = list(loaded.feature_order)
    x_eval = _build_surface_grid(feature_order=feature_order, diagnostics_cfg=diagnostics_cfg)

    spot_feature = str(global_cfg.get("spot_feature", "moneyness"))
    vol_feature = _none_if_empty(global_cfg.get("vol_feature", "v"))
    tau_feature = _none_if_empty(global_cfg.get("tau_feature", "tau"))
    rate_feature = _none_if_empty(global_cfg.get("rate_feature", "r"))
    theta_sign = str(global_cfg.get("theta_sign", "minus_dv_dtau"))
    strike = float(global_cfg.get("strike", 1.0))
    if strike <= 0.0:
        raise ValueError("strike must be > 0")

    spec = build_greek_index_spec(
        feature_order,
        spot_feature=spot_feature,
        vol_feature=vol_feature,
        tau_feature=tau_feature,
        rate_feature=rate_feature,
    )

    x_t = torch.from_numpy(x_eval).to(device=loaded.device, dtype=dtype)
    t0 = time.perf_counter()
    diff = derivatives_batch(
        loaded.price_fn,
        x_t,
        chunk_size_values=int(global_cfg.get("chunk_size_values", 4096)),
        chunk_size_jac=int(global_cfg.get("chunk_size_jac", 512)),
        chunk_size_hess=int(global_cfg.get("chunk_size_hess", 64)),
        dtype=dtype,
        device=loaded.device,
    )
    pinn_seconds = float(time.perf_counter() - t0)

    values = diff.values.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    jacobian = diff.jacobian.detach().cpu()
    hessian = diff.hessian.detach().cpu()
    jac_for_greeks = jacobian
    hess_for_greeks = hessian
    if spot_feature == "moneyness":
        jac_for_greeks, hess_for_greeks = apply_moneyness_to_spot_chain_rule(
            jacobian_wrt_m=jac_for_greeks,
            hessian_wrt_m=hess_for_greeks,
            idx_moneyness=spec.idx_spot,
            strike=strike,
        )
    elif _is_log_moneyness_spot(spot_feature):
        x_coord = torch.as_tensor(x_eval[:, spec.idx_spot], dtype=jacobian.dtype, device=jacobian.device)
        jac_for_greeks = jacobian.clone()
        hess_for_greeks = hessian.clone()
        u_x = jacobian[:, spec.idx_spot]
        u_xx = hessian[:, spec.idx_spot, spec.idx_spot]
        jac_for_greeks[:, spec.idx_spot] = torch.exp(-x_coord) * u_x / float(strike)
        hess_for_greeks[:, spec.idx_spot, spec.idx_spot] = (
            torch.exp(-2.0 * x_coord) * (u_xx - u_x) / (float(strike) ** 2)
        )
    greek_map = greeks_from_jacobian_hessian(
        jac_for_greeks,
        hess_for_greeks,
        idx_spot=spec.idx_spot,
        idx_vol=spec.idx_vol,
        idx_tau=spec.idx_tau,
        idx_rate=spec.idx_rate,
        theta_is_minus_dv_dtau=(theta_sign == "minus_dv_dtau"),
    )
    pinn_greeks = {
        key: tensor.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        for key, tensor in greek_map.items()
    }
    head_greeks = _compute_multi_output_head_greeks(
        model=loaded.price_fn.model,
        x_eval=x_eval,
        input_a=loaded.price_fn.a,
        input_b=loaded.price_fn.b,
        spot_feature=spot_feature,
        spot_index=spec.idx_spot,
        strike=strike,
        dtype=dtype,
        device=loaded.device,
        chunk_size=int(global_cfg.get("chunk_size_values", 4096)),
    )

    pde_residual = np.full(x_eval.shape[0], np.nan, dtype=np.float64)
    pde_residual_dx_log_m = np.full(x_eval.shape[0], np.nan, dtype=np.float64)
    pde_residual_dv = np.full(x_eval.shape[0], np.nan, dtype=np.float64)
    pde_coordinate = "log_moneyness" if feature_order == PINN_LOG_PDE_FEATURE_ORDER else "moneyness"
    pde_supported = feature_order in (PINN_PDE_FEATURE_ORDER, PINN_LOG_PDE_FEATURE_ORDER)
    if pde_supported:
        x_pde = torch.from_numpy(x_eval).to(device=loaded.device, dtype=dtype)
        pde_input_affine = {"a": loaded.price_fn.a, "b": loaded.price_fn.b}
        residual_t = compute_heston_pde_residual(
            model=loaded.price_fn.model,
            x_interior=x_pde,
            input_affine=pde_input_affine,
            coordinate=pde_coordinate,
        )
        pde_residual = residual_t.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        pde_residual_dx_log_m, pde_residual_dv = _compute_pde_derivative_residuals(
            model=loaded.price_fn.model,
            x_eval=x_eval,
            input_affine=pde_input_affine,
            coordinate=pde_coordinate,
            dtype=dtype,
            device=loaded.device,
            cfg=diagnostics_cfg.get("derivative_residual", {}),
        )

    cf_cfg = diagnostics_cfg.get("cf_integration", {})
    if not isinstance(cf_cfg, dict):
        raise ValueError("diagnostics.cf_integration must be a dictionary")
    cf_settings = HestonCFGreeksSettings(
        u_min=float(cf_cfg.get("u_min", 1.0e-6)),
        u_max=float(cf_cfg.get("u_max", 200.0)),
        n_u=int(cf_cfg.get("n_u", 1200)),
    )
    ref, failed_ref, ref_seconds = _reference_heston(
        x_eval=x_eval,
        feature_order=feature_order,
        spec=spec,
        spot_feature=spot_feature,
        strike=strike,
        option_type=str(diagnostics_cfg.get("option_type", "put")),
        cf_settings=cf_settings,
    )

    points_df = pd.DataFrame(x_eval, columns=feature_order)
    if "log_moneyness" in points_df.columns and "moneyness" not in points_df.columns:
        points_df["moneyness"] = np.exp(points_df["log_moneyness"].to_numpy(dtype=np.float64))
    points_df["pinn_price"] = values
    points_df["ref_price"] = ref["price"]
    points_df["error_price"] = points_df["pinn_price"] - points_df["ref_price"]
    points_df["abs_error_price"] = np.abs(points_df["error_price"])
    points_df["pde_residual"] = pde_residual
    points_df["pde_residual_dx_log_m"] = pde_residual_dx_log_m
    points_df["pde_residual_dv"] = pde_residual_dv
    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        if greek in pinn_greeks:
            points_df[f"pinn_{greek}"] = pinn_greeks[greek]
            points_df[f"ref_{greek}"] = ref[greek]
            points_df[f"error_{greek}"] = points_df[f"pinn_{greek}"] - points_df[f"ref_{greek}"]
            points_df[f"abs_error_{greek}"] = np.abs(points_df[f"error_{greek}"])
            points_df[f"stabilized_rel_error_{greek}"] = (
                points_df[f"abs_error_{greek}"]
                / (float(diagnostics_cfg.get("stabilized_relative_floor", 1.0)) + np.abs(points_df[f"ref_{greek}"]))
            )
        if greek in head_greeks:
            points_df[f"head_{greek}"] = head_greeks[greek]
            points_df[f"head_error_{greek}"] = points_df[f"head_{greek}"] - ref[greek]
            points_df[f"head_abs_error_{greek}"] = np.abs(points_df[f"head_error_{greek}"])
            if greek in pinn_greeks:
                points_df[f"head_minus_autodiff_{greek}"] = (
                    points_df[f"head_{greek}"] - points_df[f"pinn_{greek}"]
                )

    hard_cfg = diagnostics_cfg.get("hard_region", {})
    region_spot_feature = "moneyness" if "moneyness" in points_df.columns else spot_feature
    masks = _region_masks(
        points=points_df,
        spot_feature=region_spot_feature,
        tau_feature=str(tau_feature),
        epsilon_m=float(hard_cfg.get("epsilon_m", 0.03)),
        epsilon_tau=float(hard_cfg.get("epsilon_tau", 0.05)),
    )
    metrics_rows: list[dict] = []
    mape_floor = float(diagnostics_cfg.get("mape_floor", 1.0e-4))
    variables = ["price"] + [g for g in ("delta", "gamma", "vega", "theta", "rho") if g in pinn_greeks]
    for region, mask in masks.items():
        for variable in variables:
            pred = points_df.loc[mask, f"pinn_{variable}"].to_numpy(dtype=np.float64)
            target = points_df.loc[mask, f"ref_{variable}"].to_numpy(dtype=np.float64)
            finite = np.isfinite(pred) & np.isfinite(target)
            if not np.any(finite):
                continue
            row = {
                "region": region,
                "variable": variable,
                "n_points": int(np.sum(finite)),
                **_compute_metrics(pred[finite], target[finite], mape_floor=mape_floor),
            }
            if variable != "price":
                rel_col = f"stabilized_rel_error_{variable}"
                row.update(_compute_stabilized_relative_metrics(points_df.loc[mask, rel_col]))
            metrics_rows.append(row)

        for greek in ("delta", "gamma", "vega"):
            head_col = f"head_{greek}"
            if head_col not in points_df.columns:
                continue
            pred = points_df.loc[mask, head_col].to_numpy(dtype=np.float64)
            target = points_df.loc[mask, f"ref_{greek}"].to_numpy(dtype=np.float64)
            finite = np.isfinite(pred) & np.isfinite(target)
            if not np.any(finite):
                continue
            metrics_rows.append(
                {
                    "region": region,
                    "variable": f"head_{greek}",
                    "n_points": int(np.sum(finite)),
                    **_compute_metrics(pred[finite], target[finite], mape_floor=mape_floor),
                }
            )
            if f"pinn_{greek}" in points_df.columns:
                diff_head_auto = (
                    points_df.loc[mask, head_col].to_numpy(dtype=np.float64)
                    - points_df.loc[mask, f"pinn_{greek}"].to_numpy(dtype=np.float64)
                )
                finite_diff = np.isfinite(diff_head_auto)
                if np.any(finite_diff):
                    metrics_rows.append(
                        {
                            "region": region,
                            "variable": f"head_minus_autodiff_{greek}",
                            "n_points": int(np.sum(finite_diff)),
                            **_compute_absolute_metrics(diff_head_auto[finite_diff]),
                        }
                    )

        for residual_col in ("pde_residual", "pde_residual_dx_log_m", "pde_residual_dv"):
            residual = points_df.loc[mask, residual_col].to_numpy(dtype=np.float64)
            finite_residual = np.isfinite(residual)
            if not np.any(finite_residual):
                continue
            metrics_rows.append(
                {
                    "region": region,
                    "variable": residual_col,
                    "n_points": int(np.sum(finite_residual)),
                    **_compute_absolute_metrics(residual[finite_residual]),
                }
            )

    fd_df, fd_metrics_rows = _finite_difference_consistency(
        price_fn=loaded.price_fn,
        x_eval=x_eval,
        feature_order=feature_order,
        spec=spec,
        spot_feature=spot_feature,
        strike=strike,
        theta_sign=theta_sign,
        fd_cfg=diagnostics_cfg.get("finite_difference", {}),
        dtype=dtype,
        device=loaded.device,
        ad_greeks=pinn_greeks,
    )

    output_subdir = str(diagnostics_cfg.get("output_subdir", "baseline_diagnostics")).strip()
    out_dir = loaded.run_dir / "greeks" / output_subdir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    points_path = out_dir / "points_baseline_diagnostics.csv"
    metrics_path = out_dir / "metrics_by_region.csv"
    fd_path = out_dir / "finite_difference_consistency.csv"
    fd_metrics_path = out_dir / "finite_difference_metrics.csv"
    no_ref_path = out_dir / "no_reference_diagnostics.csv"
    report_path = out_dir / "sanity_report.yaml"
    execution_path = out_dir / "diagnostics_execution.yaml"

    no_ref_rows = _no_reference_diagnostics(
        points_df=points_df,
        masks=masks,
        diagnostics_cfg=diagnostics_cfg,
        option_type=str(diagnostics_cfg.get("option_type", "put")).strip().lower(),
        variables=variables,
    )
    no_ref_rows.extend(
        _boundary_condition_rows(
            price_fn=loaded.price_fn,
            feature_order=feature_order,
            diagnostics_cfg=diagnostics_cfg,
            spot_feature=spot_feature,
            tau_feature=str(tau_feature),
            rate_feature=str(rate_feature),
            strike=strike,
            option_type=str(diagnostics_cfg.get("option_type", "put")).strip().lower(),
            dtype=dtype,
            device=loaded.device,
        )
    )

    points_df.to_csv(points_path, index=False)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(no_ref_rows).to_csv(no_ref_path, index=False)
    if not fd_df.empty:
        fd_df.to_csv(fd_path, index=False)
        pd.DataFrame(fd_metrics_rows).to_csv(fd_metrics_path, index=False)

    n_bins = int(diagnostics_cfg.get("n_bins", 181))
    x_axis_values = points_df[str(tau_feature)].to_numpy(dtype=np.float64)
    y_axis_feature = "moneyness" if "moneyness" in points_df.columns else spot_feature
    y_axis_values = points_df[y_axis_feature].to_numpy(dtype=np.float64)
    for variable in ("price", "delta", "gamma", "vega", "theta", "rho"):
        col = f"abs_error_{variable}"
        if col not in points_df.columns:
            continue
        matrix, x_edges, y_edges = _build_heatmap(
            x_axis=x_axis_values,
            y_axis=y_axis_values,
            values=points_df[col].to_numpy(dtype=np.float64),
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=matrix,
            x_edges=x_edges,
            y_edges=y_edges,
            x_label=str(tau_feature),
            y_label=y_axis_feature,
            out_path=fig_dir / f"abs_error_map_{variable}.png",
            title=f"{variable}: absolute error baseline diagnostic",
            cbar_label="mean absolute error",
            log_scale=(variable != "price"),
        )

    if np.isfinite(pde_residual).any():
        matrix, x_edges, y_edges = _build_heatmap(
            x_axis=x_axis_values,
            y_axis=y_axis_values,
            values=np.abs(pde_residual),
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=matrix,
            x_edges=x_edges,
            y_edges=y_edges,
            x_label=str(tau_feature),
            y_label=y_axis_feature,
            out_path=fig_dir / "pde_residual_abs_map.png",
            title="PDE residual absolute value baseline diagnostic",
            cbar_label="mean |PDE residual|",
            log_scale=True,
        )

    for residual_col, filename, title in (
        (
            "pde_residual_dx_log_m",
            "pde_residual_dx_log_m_abs_map.png",
            "PDE residual derivative wrt log-moneyness",
        ),
        (
            "pde_residual_dv",
            "pde_residual_dv_abs_map.png",
            "PDE residual derivative wrt variance",
        ),
    ):
        residual_values = points_df[residual_col].to_numpy(dtype=np.float64)
        if not np.isfinite(residual_values).any():
            continue
        matrix, x_edges, y_edges = _build_heatmap(
            x_axis=x_axis_values,
            y_axis=y_axis_values,
            values=np.abs(residual_values),
            n_bins=n_bins,
        )
        _save_heatmap(
            matrix=matrix,
            x_edges=x_edges,
            y_edges=y_edges,
            x_label=str(tau_feature),
            y_label=y_axis_feature,
            out_path=fig_dir / filename,
            title=title,
            cbar_label=f"mean |{residual_col}|",
            log_scale=True,
        )

    if pde_coordinate == "log_moneyness":
        implemented_residual = (
            "U_tau - 0.5*v*U_xx - (r-0.5*v)*U_x - rho*gamma*v*U_xv "
            "- 0.5*gamma^2*v*U_vv - kappa*(bar_v-v)*U_v + r*U"
        )
    else:
        implemented_residual = (
            "u_tau - 0.5*v*m^2*u_mm - rho*gamma*v*m*u_mv "
            "- 0.5*gamma^2*v*u_vv - r*m*u_m - kappa*(bar_v-v)*u_v + r*u"
        )

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "experiment_id": str(global_cfg.get("experiment_id", "baseline_pinn")),
        "pde_convention": {
            "time_variable": "tau = T - t",
            "calendar_theta": "theta = -dV/dtau when theta_sign='minus_dv_dtau'",
            "coordinate": pde_coordinate,
            "implemented_residual": implemented_residual,
            "pde_feature_order_supported": [PINN_PDE_FEATURE_ORDER, PINN_LOG_PDE_FEATURE_ORDER],
            "pde_residual_computed": bool(pde_supported),
            "derivative_residual_computed": bool(np.isfinite(pde_residual_dx_log_m).any()),
        },
        "greek_definitions": {
            "delta": (
                f"dV/d{spot_feature}; converted to spot S with strike={strike}. "
                "For log_moneyness, Delta=e^-x U_x / K."
            ),
            "gamma": (
                f"d2V/d{spot_feature}2; converted to spot S with strike={strike}. "
                "For log_moneyness, Gamma=e^-2x (U_xx-U_x) / K^2."
            ),
            "vega": f"dV/d{vol_feature}; current convention is sensitivity to variance v",
            "theta": "calendar-time theta = -dV/dtau" if theta_sign == "minus_dv_dtau" else "dV/dtau",
            "rho": f"dV/d{rate_feature}",
            "multi_output_heads": (
                "head_delta/head_gamma/head_vega are reported when the model exposes forward_all(); "
                "they are diagnostics only and are not COS-trained labels."
            ),
        },
        "regions": {
            "epsilon_m": float(hard_cfg.get("epsilon_m", 0.03)),
            "epsilon_tau": float(hard_cfg.get("epsilon_tau", 0.05)),
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        },
        "run": {
            "run_dir": str(loaded.run_dir),
            "checkpoint": str(loaded.checkpoint_path),
            "architecture_config": str(loaded.architecture_config_path),
            "feature_order": feature_order,
            "device": str(loaded.device),
            "dtype": str(dtype),
            "multi_output_heads_computed": bool(head_greeks),
        },
        "timing": {
            "pinn_derivatives_seconds": pinn_seconds,
            "reference_seconds": ref_seconds,
            "pinn_points_per_second": float(x_eval.shape[0] / max(pinn_seconds, 1.0e-12)),
            "reference_points_per_second": float(x_eval.shape[0] / max(ref_seconds, 1.0e-12)),
        },
        "reference": {
            "option_type": str(diagnostics_cfg.get("option_type", "put")),
            "failed_reference_points": int(failed_ref),
            "cf_integration": {
                "u_min": float(cf_settings.u_min),
                "u_max": float(cf_settings.u_max),
                "n_u": int(cf_settings.n_u),
            },
        },
        "artifacts": {
            "points_csv": str(points_path),
            "metrics_by_region_csv": str(metrics_path),
            "no_reference_diagnostics_csv": str(no_ref_path),
            "finite_difference_csv": str(fd_path) if not fd_df.empty else None,
            "finite_difference_metrics_csv": str(fd_metrics_path) if fd_metrics_rows else None,
            "figures_dir": str(fig_dir),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False)
    with open(execution_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "timestamp": report["timestamp"],
                "config_source": str(config_path),
                "global_config": global_cfg,
                "diagnostics_config": diagnostics_cfg,
                "resolved": report["run"],
                "artifacts": report["artifacts"],
            },
            f,
            sort_keys=False,
        )

    print(f"Config: {config_path}")
    print(f"Run dir: {loaded.run_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Points: {x_eval.shape[0]} | Reference failures: {failed_ref}")
    print(
        f"Speed PINN={report['timing']['pinn_points_per_second']:.1f} pts/s | "
        f"Heston-CF={report['timing']['reference_points_per_second']:.1f} pts/s"
    )
    metrics_df = pd.DataFrame(metrics_rows)
    for variable in ("price", "delta", "gamma", "vega", "theta"):
        row = metrics_df[(metrics_df["region"] == "hard") & (metrics_df["variable"] == variable)]
        if not row.empty:
            item = row.iloc[0]
            print(
                f"hard {variable}: RMSE={item['rmse']:.3e} | "
                f"p99_abs={item['p99_abs_error']:.3e} | n={int(item['n_points'])}"
            )


if __name__ == "__main__":
    main()
