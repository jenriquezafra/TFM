from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from src.models.normalization import denormalize_target, normalize_features

Regularization = Literal["none", "l1", "l2", "l2_squared"]


@dataclass(frozen=True)
class MarketInputs:
    """
    Container for observed market data used during calibration.

    Expected shape for each array: (N,)
    """

    moneyness: np.ndarray
    tau: np.ndarray
    r: np.ndarray
    iv_market: np.ndarray
    weights: np.ndarray

    @property
    def n_quotes(self) -> int:
        return int(self.iv_market.size)


def _as_1d_float_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"'{name}' cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"'{name}' contains non-finite values")
    return array


def build_market_inputs(
    *,
    moneyness: Sequence[float] | np.ndarray,
    tau: Sequence[float] | np.ndarray,
    r: Sequence[float] | np.ndarray,
    iv_market: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> MarketInputs:
    """
    Build validated market data for calibration.

    Inputs
    - moneyness: observed S0/K values, shape (N,)
    - tau: observed maturities, shape (N,)
    - r: observed rates, shape (N,)
    - iv_market: observed implied vols, shape (N,)
    - weights: optional quote weights, shape (N,), defaults to 1

    Output
    - MarketInputs object with validated numpy arrays.
    """

    moneyness_np = _as_1d_float_array(moneyness, name="moneyness")
    tau_np = _as_1d_float_array(tau, name="tau")
    r_np = _as_1d_float_array(r, name="r")
    iv_market_np = _as_1d_float_array(iv_market, name="iv_market")

    n = iv_market_np.size
    if moneyness_np.size != n or tau_np.size != n or r_np.size != n:
        raise ValueError(
            "moneyness, tau, r, iv_market must have the same length. "
            f"Received lengths: {moneyness_np.size}, {tau_np.size}, {r_np.size}, {n}"
        )
    if np.any(tau_np <= 0.0):
        raise ValueError("tau must be strictly positive")

    if weights is None:
        weights_np = np.ones(n, dtype=np.float64)
    else:
        weights_np = _as_1d_float_array(weights, name="weights")
        if weights_np.size != n:
            raise ValueError(
                "weights must have the same length as iv_market. "
                f"Received lengths: {weights_np.size} and {n}"
            )
        if np.any(weights_np < 0.0):
            raise ValueError("weights must be non-negative")

    return MarketInputs(
        moneyness=moneyness_np,
        tau=tau_np,
        r=r_np,
        iv_market=iv_market_np,
        weights=weights_np,
    )


def _theta_to_matrix(
    theta: Sequence[float] | np.ndarray,
    *,
    n_model_params: int,
) -> tuple[np.ndarray, bool]:
    """
    Normalize theta into shape (n_model_params, S).

    Returns (theta_matrix, is_single_candidate).
    """

    theta_np = np.asarray(theta, dtype=np.float64)
    if theta_np.ndim == 1:
        if theta_np.size != n_model_params:
            raise ValueError(
                f"theta must have size={n_model_params}; got {theta_np.size}"
            )
        return theta_np.reshape(n_model_params, 1), True

    if theta_np.ndim == 2:
        if theta_np.shape[0] == n_model_params:
            return theta_np, False
        if theta_np.shape[1] == n_model_params:
            return theta_np.T, False
        raise ValueError(
            "2D theta must have shape (n_model_params, S) or (S, n_model_params). "
            f"Received shape={theta_np.shape} with n_model_params={n_model_params}."
        )

    raise ValueError(
        "theta must be 1D or 2D array-like. "
        f"Received ndim={theta_np.ndim}."
    )


def _build_features(theta_matrix: np.ndarray, market_inputs: MarketInputs) -> np.ndarray:
    """
    Build ANN features [rho, kappa, gamma, bar_v, v0, moneyness, tau, r].
    """

    s_candidates = theta_matrix.shape[1]
    n_quotes = market_inputs.n_quotes

    theta_rows = np.repeat(theta_matrix.T, repeats=n_quotes, axis=0)
    m_col = np.tile(market_inputs.moneyness, reps=s_candidates)
    tau_col = np.tile(market_inputs.tau, reps=s_candidates)
    r_col = np.tile(market_inputs.r, reps=s_candidates)

    return np.column_stack((theta_rows, m_col, tau_col, r_col)).astype(np.float32, copy=False)


def _regularization_term(
    theta_matrix: np.ndarray,
    *,
    regularization: Regularization,
) -> np.ndarray:
    if regularization == "none":
        return np.zeros(theta_matrix.shape[1], dtype=np.float64)
    if regularization == "l1":
        return np.sum(np.abs(theta_matrix), axis=0)
    if regularization == "l2":
        return np.linalg.norm(theta_matrix, ord=2, axis=0)
    if regularization == "l2_squared":
        return np.sum(theta_matrix**2, axis=0)
    raise ValueError(
        "regularization must be one of {'none', 'l1', 'l2', 'l2_squared'}; "
        f"got '{regularization}'"
    )


def _resolve_device(model: torch.nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def calibration_objective_vectorized(
    theta: Sequence[float] | np.ndarray,
    *,
    model: torch.nn.Module,
    market_inputs: MarketInputs,
    normalization_stats: dict | None = None,
    lambda_reg: float = 0.0,
    regularization: Regularization = "l2",
    n_model_params: int = 5,
    device: str | torch.device | None = None,
    invalid_value: float = 1.0e12,
) -> np.ndarray:
    """
    Vectorized objective J(theta) used by Differential Evolution.

    Inputs
    - theta:
      1) Single candidate of shape (n_model_params,)
      2) Population of shape (n_model_params, S) or (S, n_model_params)
    - model: trained ANN forward model (8 -> 1 IV)
    - market_inputs: validated market quote arrays
    - lambda_reg: regularization strength
    - regularization: 'none' | 'l1' | 'l2' | 'l2_squared'
    - n_model_params: number of calibrated model parameters (default 5 for Heston)
    - device: torch device for inference (default: model device)
    - invalid_value: penalty returned when non-finite values appear

    Output
    - Objective values as numpy array with shape (S,).
    """

    theta_matrix, _ = _theta_to_matrix(theta, n_model_params=n_model_params)
    if not np.all(np.isfinite(theta_matrix)):
        return np.full(theta_matrix.shape[1], fill_value=invalid_value, dtype=np.float64)

    features = _build_features(theta_matrix, market_inputs)
    features = normalize_features(features, normalization_stats)
    model_device = _resolve_device(model, device)

    x_tensor = torch.from_numpy(features).to(model_device)
    model.eval()
    with torch.inference_mode():
        pred_tensor = model(x_tensor)

    pred_flat = np.asarray(pred_tensor.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    pred_flat = denormalize_target(pred_flat, normalization_stats)
    expected_size = theta_matrix.shape[1] * market_inputs.n_quotes
    if pred_flat.size != expected_size:
        raise ValueError(
            "Model output size mismatch. "
            f"Expected {expected_size} predictions, got {pred_flat.size}."
        )

    pred_surface = pred_flat.reshape(theta_matrix.shape[1], market_inputs.n_quotes)
    residual = pred_surface - market_inputs.iv_market.reshape(1, -1)
    weighted_sq_error = market_inputs.weights.reshape(1, -1) * (residual**2)
    data_term = np.sum(weighted_sq_error, axis=1)

    reg_term = _regularization_term(theta_matrix, regularization=regularization)
    objective = data_term + float(lambda_reg) * reg_term
    objective[~np.isfinite(objective)] = float(invalid_value)
    return objective


def calibration_objective(
    theta: Sequence[float] | np.ndarray,
    *,
    model: torch.nn.Module,
    market_inputs: MarketInputs,
    normalization_stats: dict | None = None,
    lambda_reg: float = 0.0,
    regularization: Regularization = "l2",
    n_model_params: int = 5,
    device: str | torch.device | None = None,
    invalid_value: float = 1.0e12,
) -> float:
    """
    Scalar objective J(theta) for a single candidate.

    Inputs
    - theta: shape (n_model_params,)
    - model, market_inputs, lambda_reg, regularization: same as vectorized version

    Output
    - Single float with objective value.
    """

    values = calibration_objective_vectorized(
        theta,
        model=model,
        market_inputs=market_inputs,
        normalization_stats=normalization_stats,
        lambda_reg=lambda_reg,
        regularization=regularization,
        n_model_params=n_model_params,
        device=device,
        invalid_value=invalid_value,
    )
    return float(values[0])
