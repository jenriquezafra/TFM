from __future__ import annotations
from typing import Mapping, Sequence

import torch
from torch import Tensor

def _as_1d_tensor(
    x: Tensor | Sequence[float],
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim != 1:
        raise ValueError(f"`{name}` must be 1D, got shape={tuple(t.shape)}")
    return t

def _check_grad_hess_shapes(jacobian: Tensor, hessian: Tensor) -> None:
    if jacobian.ndim not in (1, 2):
        raise ValueError(
            f"`jacobian` must be [D] or [N,D], got shape={tuple(jacobian.shape)}"
        )
    if hessian.ndim not in (2, 3):
        raise ValueError(
            f"`hessian` must be [D,D] or [N,D,D], got shape={tuple(hessian.shape)}"
        )
    if jacobian.ndim == 1 and hessian.ndim != 2:
        raise ValueError("If `jacobian` is [D], `hessian` must be [D,D].")
    if jacobian.ndim == 2 and hessian.ndim != 3:
        raise ValueError("If `jacobian` is [N,D], `hessian` must be [N,D,D].")
    d = jacobian.shape[-1]
    if hessian.shape[-1] != d or hessian.shape[-2] != d:
        raise ValueError(
            f"Dimension mismatch: jacobian has D={d}, hessian has shape={tuple(hessian.shape)}"
        )
    if jacobian.ndim == 2 and jacobian.shape[0] != hessian.shape[0]:
        raise ValueError("Batch size mismatch between `jacobian` and `hessian`.")


def apply_input_linear_chain_rule(
        jacobian: Tensor,
        hessian: Tensor, 
        input_scales: Tensor | Sequence[float],
) -> tuple[Tensor, Tensor]:
    """
    Transform derivatives from z-space to x-space when z = Ax+b (A diag)

    If input_scales[i] = dz_i/dx_i, then:
        dV/dx_i = dV/dz_i *input_scales[i]
        d2V/dxidxj  = d2V/dzidzj * input_scales[i] * input_scales[j]
    """
    _check_grad_hess_shapes(jacobian, hessian)
    scales = _as_1d_tensor(
        input_scales,
        name="input_scales", 
        dtype=jacobian.dtype,
        device=jacobian.device,
    )

    d = jacobian.shape[-1]
    if scales.numel() != d:
        raise ValueError(f"`input_scales` must have length {d}, got {scales.numel()}")

    jac_out = jacobian * scales  # [D] or [N,D]
    scale_outer = scales[:, None] * scales[None, :]  # [D,D]

    if hessian.ndim == 2:
        hess_out = hessian * scale_outer
    else:
        hess_out = hessian * scale_outer.unsqueeze(0)

    return jac_out, hess_out


def apply_output_linear_chain_rule(
    value: Tensor,
    jacobian: Tensor,
    hessian: Tensor,
    *,
    y_scale: float = 1.0,
    y_shift: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Transform derivatives when y = y_scale * y_net + y_shift.
    """
    s = float(y_scale)
    b = float(y_shift)
    value_out = value * s + b
    jac_out = jacobian * s
    hess_out = hessian * s
    return value_out, jac_out, hess_out


def normalization_params_from_stats(
    stats: Mapping[str, object] | None,
    *,
    feature_order: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[Tensor, float, float]:
    """
    Returns:
      x_std: Tensor [D]
      y_std: float
      y_mean: float
    """
    if not stats or not bool(stats.get("enabled", False)):
        if feature_order is None:
            raise ValueError("`feature_order` is required when stats are missing/disabled.")
        d = len(feature_order)
        return torch.ones(d, dtype=dtype, device=device), 1.0, 0.0

    x_std_raw = list(stats.get("x_std", []))
    if not x_std_raw:
        raise ValueError("Normalization stats missing `x_std`.")

    if feature_order is None:
        x_std = torch.tensor(x_std_raw, dtype=dtype, device=device)
    else:
        stats_names = list(stats.get("feature_names", []))
        if not stats_names:
            raise ValueError("Stats missing `feature_names`; cannot reorder to `feature_order`.")
        idx = {name: i for i, name in enumerate(stats_names)}
        missing = [name for name in feature_order if name not in idx]
        if missing:
            raise ValueError(f"Features not found in stats: {missing}")
        x_std = torch.tensor(
            [x_std_raw[idx[name]] for name in feature_order],
            dtype=dtype,
            device=device,
        )

    y_std = float(stats.get("y_std", 1.0))
    y_mean = float(stats.get("y_mean", 0.0))
    normalize_target = bool(stats.get("normalize_target", False))
    if not normalize_target:
        y_std, y_mean = 1.0, 0.0

    eps = 1e-12
    x_std = torch.clamp(x_std, min=eps)
    if abs(y_std) < eps:
        y_std = 1.0

    return x_std, y_std, y_mean


def network_space_to_raw_space(
    value_net: Tensor,
    jacobian_net: Tensor,
    hessian_net: Tensor,
    *,
    x_std: Tensor | Sequence[float],
    y_std: float = 1.0,
    y_mean: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Convert derivatives from network space to raw space when:
      x_net = (x_raw - mu) / x_std
      y_raw = y_std * y_net + y_mean
    """
    _check_grad_hess_shapes(jacobian_net, hessian_net)

    value_raw, jac_raw, hess_raw = apply_output_linear_chain_rule(
        value_net, jacobian_net, hessian_net, y_scale=y_std, y_shift=y_mean
    )

    x_std_t = _as_1d_tensor(
        x_std, name="x_std", dtype=jacobian_net.dtype, device=jacobian_net.device
    )
    input_scales = 1.0 / x_std_t  # dz/dx where z=x_net, x=x_raw
    jac_raw, hess_raw = apply_input_linear_chain_rule(jac_raw, hess_raw, input_scales)

    return value_raw, jac_raw, hess_raw


def moneyness_to_spot_scales(
    *,
    dim: int,
    idx_moneyness: int,
    strike: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Build coordinate scales for m = S/K (K constant):
      dm/dS = 1/K
    """
    if dim <= 0:
        raise ValueError("`dim` must be > 0.")
    if not (0 <= idx_moneyness < dim):
        raise ValueError(f"`idx_moneyness` must be in [0, {dim-1}]")
    if strike <= 0:
        raise ValueError("`strike` must be > 0.")

    scales = torch.ones(dim, dtype=dtype, device=device)
    scales[idx_moneyness] = 1.0 / float(strike)
    return scales


def apply_moneyness_to_spot_chain_rule(
    jacobian_wrt_m: Tensor,
    hessian_wrt_m: Tensor,
    *,
    idx_moneyness: int,
    strike: float,
) -> tuple[Tensor, Tensor]:
    """
    Convert derivatives from coordinates with m=S/K to coordinates with S.
    Keeps tensor shape unchanged; the idx_moneyness coordinate now means spot.
    """
    _check_grad_hess_shapes(jacobian_wrt_m, hessian_wrt_m)
    d = jacobian_wrt_m.shape[-1]
    scales = moneyness_to_spot_scales(
        dim=d,
        idx_moneyness=idx_moneyness,
        strike=strike,
        dtype=jacobian_wrt_m.dtype,
        device=jacobian_wrt_m.device,
    )
    return apply_input_linear_chain_rule(jacobian_wrt_m, hessian_wrt_m, scales)


def log_moneyness_delta_gamma_to_moneyness(
    *,
    u_x: Tensor,
    u_xx: Tensor,
    x: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Convert derivatives with respect to x=log(m) to derivatives with respect
    to moneyness m:
      dU/dm = exp(-x) U_x
      d2U/dm2 = exp(-2x) (U_xx - U_x)
    """
    x_t = torch.as_tensor(x, dtype=u_x.dtype, device=u_x.device)
    delta_m = torch.exp(-x_t) * u_x
    gamma_m = torch.exp(-2.0 * x_t) * (u_xx - u_x)
    return delta_m, gamma_m


__all__ = [
    "apply_input_linear_chain_rule",
    "apply_output_linear_chain_rule",
    "normalization_params_from_stats",
    "network_space_to_raw_space",
    "moneyness_to_spot_scales",
    "apply_moneyness_to_spot_chain_rule",
    "log_moneyness_delta_gamma_to_moneyness",
]
