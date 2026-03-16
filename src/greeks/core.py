from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.func import hessian, jacrev, vmap


PriceFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class PointDerivatives:
    value: Tensor
    jacobian: Tensor
    hessian: Tensor


@dataclass(frozen=True)
class BatchDerivatives:
    values: Tensor
    jacobian: Tensor
    hessian: Tensor


def _as_1d_float_tensor(
    x: Tensor | list[float] | tuple[float, ...],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    x_t = torch.as_tensor(x, dtype=dtype, device=device)
    if x_t.ndim != 1:
        raise ValueError(f"`x` must be 1D [D], got shape={tuple(x_t.shape)}")
    if not torch.is_floating_point(x_t):
        x_t = x_t.to(torch.float64)
    return x_t


def _as_2d_float_tensor(
    x: Tensor | list[list[float]],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    x_t = torch.as_tensor(x, dtype=dtype, device=device)
    if x_t.ndim != 2:
        raise ValueError(f"`x_batch` must be 2D [N, D], got shape={tuple(x_t.shape)}")
    if not torch.is_floating_point(x_t):
        x_t = x_t.to(torch.float64)
    return x_t


def _to_scalar(y: Tensor) -> Tensor:
    if y.ndim == 0:
        return y
    if y.ndim == 1 and y.numel() == 1:
        return y.reshape(())
    raise ValueError(
        f"`price_fn` must return a scalar Tensor. Got shape={tuple(y.shape)}."
    )


def _scalar_price_fn(price_fn: PriceFn) -> PriceFn:
    def f(x: Tensor) -> Tensor:
        return _to_scalar(price_fn(x))

    return f


def _chunked_apply(
    op: Callable[[Tensor], Tensor],
    x_batch: Tensor,
    chunk_size: int | None,
) -> Tensor:
    if chunk_size is None or chunk_size <= 0 or x_batch.shape[0] <= chunk_size:
        return op(x_batch)

    outs: list[Tensor] = []
    n = x_batch.shape[0]
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        outs.append(op(x_batch[start:stop]))
    return torch.cat(outs, dim=0)


def value_point(
    price_fn: PriceFn,
    x: Tensor | list[float] | tuple[float, ...],
    *,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    f = _scalar_price_fn(price_fn)
    x_t = _as_1d_float_tensor(x, dtype=dtype, device=device)
    return f(x_t)


def jacobian_point(
    price_fn: PriceFn,
    x: Tensor | list[float] | tuple[float, ...],
    *,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    f = _scalar_price_fn(price_fn)
    x_t = _as_1d_float_tensor(x, dtype=dtype, device=device)
    return jacrev(f)(x_t)


def hessian_point(
    price_fn: PriceFn,
    x: Tensor | list[float] | tuple[float, ...],
    *,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    f = _scalar_price_fn(price_fn)
    x_t = _as_1d_float_tensor(x, dtype=dtype, device=device)
    return hessian(f)(x_t)


def derivatives_point(
    price_fn: PriceFn,
    x: Tensor | list[float] | tuple[float, ...],
    *,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> PointDerivatives:
    f = _scalar_price_fn(price_fn)
    x_t = _as_1d_float_tensor(x, dtype=dtype, device=device)
    return PointDerivatives(
        value=f(x_t),
        jacobian=jacrev(f)(x_t),
        hessian=hessian(f)(x_t),
    )


def values_batch(
    price_fn: PriceFn,
    x_batch: Tensor | list[list[float]],
    *,
    chunk_size: int | None = None,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    f = _scalar_price_fn(price_fn)
    xb = _as_2d_float_tensor(x_batch, dtype=dtype, device=device)
    op = vmap(f)
    return _chunked_apply(op, xb, chunk_size)


def jacobian_batch(
    price_fn: PriceFn,
    x_batch: Tensor | list[list[float]],
    *,
    chunk_size: int | None = None,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    f = _scalar_price_fn(price_fn)
    xb = _as_2d_float_tensor(x_batch, dtype=dtype, device=device)
    op = vmap(jacrev(f))
    return _chunked_apply(op, xb, chunk_size)


def hessian_batch(
    price_fn: PriceFn,
    x_batch: Tensor | list[list[float]],
    *,
    chunk_size: int | None = 32,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    f = _scalar_price_fn(price_fn)
    xb = _as_2d_float_tensor(x_batch, dtype=dtype, device=device)
    op = vmap(hessian(f))
    return _chunked_apply(op, xb, chunk_size)


def derivatives_batch(
    price_fn: PriceFn,
    x_batch: Tensor | list[list[float]],
    *,
    chunk_size_values: int | None = None,
    chunk_size_jac: int | None = None,
    chunk_size_hess: int | None = 32,
    dtype: torch.dtype | None = torch.float64,
    device: torch.device | str | None = None,
) -> BatchDerivatives:
    xb = _as_2d_float_tensor(x_batch, dtype=dtype, device=device)
    return BatchDerivatives(
        values=values_batch(
            price_fn,
            xb,
            chunk_size=chunk_size_values,
            dtype=dtype,
            device=device,
        ),
        jacobian=jacobian_batch(
            price_fn,
            xb,
            chunk_size=chunk_size_jac,
            dtype=dtype,
            device=device,
        ),
        hessian=hessian_batch(
            price_fn,
            xb,
            chunk_size=chunk_size_hess,
            dtype=dtype,
            device=device,
        ),
    )


def greeks_from_jacobian_hessian(
    jacobian: Tensor,
    hess: Tensor,
    *,
    idx_spot: int,
    idx_vol: int | None = None,
    idx_tau: int | None = None,
    idx_rate: int | None = None,
    theta_is_minus_dv_dtau: bool = True,
) -> dict[str, Tensor]:
    if jacobian.ndim not in (1, 2):
        raise ValueError(f"`jacobian` must be [D] or [N,D], got {tuple(jacobian.shape)}")
    if hess.ndim not in (2, 3):
        raise ValueError(f"`hess` must be [D,D] or [N,D,D], got {tuple(hess.shape)}")

    if jacobian.ndim == 1 and hess.ndim != 2:
        raise ValueError("If `jacobian` is [D], `hess` must be [D,D]")
    if jacobian.ndim == 2 and hess.ndim != 3:
        raise ValueError("If `jacobian` is [N,D], `hess` must be [N,D,D]")

    out: dict[str, Tensor] = {
        "delta": jacobian[..., idx_spot],
        "gamma": hess[..., idx_spot, idx_spot],
    }

    if idx_vol is not None:
        out["vega"] = jacobian[..., idx_vol]

    if idx_tau is not None:
        theta_raw = jacobian[..., idx_tau]
        out["theta"] = -theta_raw if theta_is_minus_dv_dtau else theta_raw

    if idx_rate is not None:
        out["rho"] = jacobian[..., idx_rate]

    return out


__all__ = [
    "PriceFn",
    "PointDerivatives",
    "BatchDerivatives",
    "value_point",
    "jacobian_point",
    "hessian_point",
    "derivatives_point",
    "values_batch",
    "jacobian_batch",
    "hessian_batch",
    "derivatives_batch",
    "greeks_from_jacobian_hessian",
]
