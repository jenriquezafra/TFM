from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.greeks.names import feature_index
from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.solvers.bs import BS_greeks_np


DEFAULT_ANN_IV_FEATURE_COLUMNS = [
    "rho",
    "kappa",
    "gamma",
    "bar_v",
    "v0",
    "moneyness",
    "tau",
    "r",
]

DEFAULT_ANN_IV_DERIVATIVE_COLUMNS = [
    "d_iv_dm",
    "d_iv_dtau",
    "d_iv_dr",
    "d_iv_dv0",
]

DERIVATIVE_COLUMN_TO_FEATURE = {
    "d_iv_dm": "moneyness",
    "d_iv_dtau": "tau",
    "d_iv_dr": "r",
    "d_iv_dv0": "v0",
}


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _invalid_row(row: pd.Series, *, iv_column: str, feature_columns: Sequence[str]) -> str | None:
    for col in list(feature_columns) + [iv_column]:
        value = float(row[col])
        if not np.isfinite(value):
            return f"non_finite_{col}"
    if float(row["moneyness"]) <= 0.0:
        return "non_positive_moneyness"
    if float(row["tau"]) <= 0.0:
        return "non_positive_tau"
    if float(row[iv_column]) <= 0.0:
        return "non_positive_iv"
    return None


def compute_ann_iv_sobolev_targets(
    df: pd.DataFrame,
    *,
    iv_column: str = "IV",
    feature_columns: Sequence[str] = DEFAULT_ANN_IV_FEATURE_COLUMNS,
    option_type: str = "put",
    strike: float = 1.0,
    cf_settings: HestonCFGreeksSettings | None = None,
    vega_floor: float = 1.0e-8,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    """
    Build first-order Sobolev labels for an ANN whose output is implied volatility.

    If sigma_IV solves

        V_BS(m, tau, r, sigma_IV) = V_Heston(m, tau, r, v0, theta),

    then implicit differentiation gives

        d sigma_IV / dq = (d_q V_Heston - partial_q V_BS) / Vega_BS

    for q in {m, tau, r, v0}. Theta follows the trading convention used by
    src.greeks.heston_cf_greeks: theta = dV/dt = -dV/dtau.
    """
    if strike <= 0.0:
        raise ValueError("strike must be > 0")
    if vega_floor <= 0.0:
        raise ValueError("vega_floor must be > 0")

    feature_columns = [str(c) for c in feature_columns]
    required = set(feature_columns + [iv_column])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    cfg = cf_settings or HestonCFGreeksSettings()
    rows: list[dict[str, object]] = []
    for source_index, row in df.iterrows():
        out = {col: float(row[col]) for col in feature_columns}
        out[iv_column] = float(row[iv_column])
        out["source_index"] = int(source_index) if isinstance(source_index, (int, np.integer)) else str(source_index)
        out["valid"] = False
        out["invalid_reason"] = ""

        invalid_reason = _invalid_row(row, iv_column=iv_column, feature_columns=feature_columns)
        if invalid_reason is not None:
            out["invalid_reason"] = invalid_reason
            if keep_invalid:
                rows.append(out)
            continue

        try:
            sigma = float(row[iv_column])
            m = float(row["moneyness"])
            tau = float(row["tau"])
            r = float(row["r"])
            s0 = m * float(strike)

            heston = heston_cf_greeks_scalar(
                option_type=option_type,
                S0=s0,
                K=float(strike),
                tau=tau,
                r=r,
                rho=float(row["rho"]),
                kappa=float(row["kappa"]),
                gamma=float(row["gamma"]),
                bar_v=float(row["bar_v"]),
                v0=float(row["v0"]),
                settings=cfg,
            )
            bs = BS_greeks_np(
                S0=s0,
                K=float(strike),
                tau=tau,
                sigma=sigma,
                r=r,
                opt_type=option_type,
            )

            bs_vega = _scalar(bs["vega_sigma"])
            if (not np.isfinite(bs_vega)) or abs(bs_vega) <= float(vega_floor):
                out["invalid_reason"] = "bs_vega_floor"
                if keep_invalid:
                    rows.append(out)
                continue

            bs_delta = _scalar(bs["delta"])
            bs_theta = _scalar(bs["theta"])
            bs_rho = _scalar(bs["rho"])
            bs_price = _scalar(bs["price"])

            # m = S/K, so dV/dm = K dV/dS.
            out["d_iv_dm"] = float(float(strike) * (heston["delta"] - bs_delta) / bs_vega)
            out["d_iv_dtau"] = float((bs_theta - heston["theta"]) / bs_vega)
            out["d_iv_dr"] = float((heston["rho"] - bs_rho) / bs_vega)
            out["d_iv_dv0"] = float(heston["vega"] / bs_vega)
            out["bs_vega_sigma"] = float(bs_vega)
            out["price_heston_cf"] = float(heston["price"])
            out["price_bs_iv"] = float(bs_price)
            out["price_abs_diff_cf_bs"] = float(abs(heston["price"] - bs_price))
            out["valid"] = bool(
                np.all(
                    np.isfinite(
                        [
                            out["d_iv_dm"],
                            out["d_iv_dtau"],
                            out["d_iv_dr"],
                            out["d_iv_dv0"],
                            out["bs_vega_sigma"],
                        ]
                    )
                )
            )
            if not out["valid"]:
                out["invalid_reason"] = "non_finite_target"
        except Exception as exc:
            out["invalid_reason"] = f"{type(exc).__name__}: {exc}"

        if keep_invalid or bool(out["valid"]):
            rows.append(out)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if not keep_invalid and "valid" in result.columns:
        result = result.loc[result["valid"]].reset_index(drop=True)
    return result


def robust_derivative_scales(
    df: pd.DataFrame,
    derivative_columns: Sequence[str] = DEFAULT_ANN_IV_DERIVATIVE_COLUMNS,
    *,
    quantile: float = 0.90,
    floor: float = 1.0e-8,
) -> dict[str, float]:
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be in (0, 1]")
    scales: dict[str, float] = {}
    for col in derivative_columns:
        if col not in df.columns:
            raise KeyError(f"Missing derivative column '{col}'")
        vals = np.abs(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            scales[str(col)] = float(floor)
            continue
        scale = float(np.quantile(vals, quantile))
        scales[str(col)] = max(scale, float(floor))
    return scales


def derivative_columns_to_indices(
    feature_order: Sequence[str],
    derivative_columns: Sequence[str],
) -> list[int]:
    indices: list[int] = []
    for col in derivative_columns:
        feature = DERIVATIVE_COLUMN_TO_FEATURE.get(str(col))
        if feature is None:
            raise KeyError(f"Unsupported derivative column '{col}'")
        idx = feature_index(feature_order, feature, required=True)
        if idx is None:
            raise KeyError(f"Could not resolve feature '{feature}'")
        indices.append(int(idx))
    return indices


def sobolev_derivative_loss(
    *,
    model: torch.nn.Module,
    x_model: torch.Tensor,
    target_derivatives_raw: torch.Tensor,
    derivative_indices: Sequence[int],
    x_std: torch.Tensor | Sequence[float],
    y_std: float | torch.Tensor = 1.0,
    derivative_scales: torch.Tensor | Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first-order Sobolev loss in raw derivative units.

    `x_model` must be the tensor actually passed to the network. If features are
    normalized as x_model=(x_raw-mu)/x_std and the network output is normalized,
    the raw derivative is

        d y_raw / d x_raw_j = y_std / x_std_j * d y_model / d x_model_j.
    """
    if x_model.ndim != 2:
        raise ValueError(f"x_model must be [B,D], got shape={tuple(x_model.shape)}")
    if target_derivatives_raw.ndim != 2:
        raise ValueError(
            "target_derivatives_raw must be [B,K], "
            f"got shape={tuple(target_derivatives_raw.shape)}"
        )

    idx = torch.as_tensor(list(derivative_indices), dtype=torch.long, device=x_model.device)
    if idx.numel() != target_derivatives_raw.shape[1]:
        raise ValueError(
            "derivative_indices length must match target derivative columns. "
            f"Got {idx.numel()} vs {target_derivatives_raw.shape[1]}"
        )

    x_req = x_model.detach().clone().requires_grad_(True)
    y_model = model(x_req)
    if y_model.ndim == 1:
        y_model = y_model.reshape(-1, 1)
    if y_model.ndim != 2 or y_model.shape[1] != 1:
        raise ValueError(f"Expected scalar model output [B,1], got {tuple(y_model.shape)}")

    grad_model = torch.autograd.grad(
        outputs=y_model,
        inputs=x_req,
        grad_outputs=torch.ones_like(y_model),
        create_graph=True,
        retain_graph=True,
    )[0]

    x_std_t = torch.as_tensor(x_std, dtype=x_model.dtype, device=x_model.device)
    if x_std_t.ndim != 1 or x_std_t.numel() != x_model.shape[1]:
        raise ValueError("x_std must be a 1D tensor with one value per feature")
    y_std_t = torch.as_tensor(y_std, dtype=x_model.dtype, device=x_model.device)
    raw_scale = y_std_t / torch.clamp(x_std_t[idx], min=1.0e-12)
    pred_derivatives_raw = grad_model.index_select(dim=1, index=idx) * raw_scale.reshape(1, -1)

    target = target_derivatives_raw.to(dtype=x_model.dtype, device=x_model.device)
    if derivative_scales is None:
        scales = torch.ones(idx.numel(), dtype=x_model.dtype, device=x_model.device)
    else:
        scales = torch.as_tensor(derivative_scales, dtype=x_model.dtype, device=x_model.device)
    scales = torch.clamp(scales.reshape(1, -1), min=1.0e-12)
    loss = torch.mean(((pred_derivatives_raw - target) / scales) ** 2)
    return loss, pred_derivatives_raw


__all__ = [
    "DEFAULT_ANN_IV_FEATURE_COLUMNS",
    "DEFAULT_ANN_IV_DERIVATIVE_COLUMNS",
    "DERIVATIVE_COLUMN_TO_FEATURE",
    "compute_ann_iv_sobolev_targets",
    "robust_derivative_scales",
    "derivative_columns_to_indices",
    "sobolev_derivative_loss",
]
