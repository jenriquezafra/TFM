# Solver of the Black-Scholes model for vanilla options

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm


def _validate_option_type(opt_type: str) -> str:
    opt = str(opt_type).strip().lower()
    if opt not in {"call", "put"}:
        raise ValueError("opt_type must be one of {'call', 'put'}")
    return opt


def _as_np_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _np_d1_d2(S0, K, tau, sigma, r) -> tuple[np.ndarray, np.ndarray]:
    S0 = _as_np_float_array(S0)
    K = _as_np_float_array(K)
    tau = _as_np_float_array(tau)
    sigma = _as_np_float_array(sigma)
    r = _as_np_float_array(r)

    if np.any(S0 <= 0.0):
        raise ValueError("S0 must be > 0")
    if np.any(K <= 0.0):
        raise ValueError("K must be > 0")
    if np.any(tau <= 0.0):
        raise ValueError("tau must be > 0")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be > 0")

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


def BS_price_np(S0, K, tau, sigma, r, opt_type: str = "put") -> np.ndarray:
    """
    Vectorized Black--Scholes price.

    Theta conventions in this module follow the usual trading convention:
    theta = dV/dt = -dV/dtau.
    """
    opt = _validate_option_type(opt_type)
    S0 = _as_np_float_array(S0)
    K = _as_np_float_array(K)
    tau = _as_np_float_array(tau)
    sigma = _as_np_float_array(sigma)
    r = _as_np_float_array(r)
    d1, d2 = _np_d1_d2(S0, K, tau, sigma, r)
    disc = np.exp(-r * tau)
    if opt == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def BS_greeks_np(S0, K, tau, sigma, r, opt_type: str = "put") -> dict[str, np.ndarray]:
    """
    Vectorized Black--Scholes price and Greeks.

    Returned keys:
      price, delta, gamma, vega_sigma, rho, theta

    `vega_sigma` is dV/dsigma per one volatility unit, not per volatility
    percentage point. `theta` is the calendar-time convention dV/dt=-dV/dtau.
    """
    opt = _validate_option_type(opt_type)
    S0 = _as_np_float_array(S0)
    K = _as_np_float_array(K)
    tau = _as_np_float_array(tau)
    sigma = _as_np_float_array(sigma)
    r = _as_np_float_array(r)
    d1, d2 = _np_d1_d2(S0, K, tau, sigma, r)
    sqrt_tau = np.sqrt(tau)
    disc = np.exp(-r * tau)
    pdf_d1 = norm.pdf(d1)
    price = BS_price_np(S0=S0, K=K, tau=tau, sigma=sigma, r=r, opt_type=opt)
    gamma = pdf_d1 / (S0 * sigma * sqrt_tau)
    vega_sigma = S0 * pdf_d1 * sqrt_tau

    if opt == "call":
        delta = norm.cdf(d1)
        rho = K * tau * disc * norm.cdf(d2)
        theta = -(S0 * pdf_d1 * sigma) / (2.0 * sqrt_tau) - r * K * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        rho = -K * tau * disc * norm.cdf(-d2)
        theta = -(S0 * pdf_d1 * sigma) / (2.0 * sqrt_tau) + r * K * disc * norm.cdf(-d2)

    return {
        "price": np.asarray(price, dtype=np.float64),
        "delta": np.asarray(delta, dtype=np.float64),
        "gamma": np.asarray(gamma, dtype=np.float64),
        "vega_sigma": np.asarray(vega_sigma, dtype=np.float64),
        "rho": np.asarray(rho, dtype=np.float64),
        "theta": np.asarray(theta, dtype=np.float64),
    }


def _torch_norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.as_tensor(2.0, dtype=x.dtype, device=x.device))))


def BS_price_torch(
    S0: torch.Tensor,
    K: torch.Tensor | float,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
    opt_type: str = "put",
) -> torch.Tensor:
    """
    Differentiable Black--Scholes price for scalar or batched tensors.
    """
    opt = _validate_option_type(opt_type)
    K_t = torch.as_tensor(K, dtype=S0.dtype, device=S0.device)
    tau_t = torch.clamp(tau, min=torch.as_tensor(1.0e-12, dtype=S0.dtype, device=S0.device))
    sigma_t = torch.clamp(sigma, min=torch.as_tensor(1.0e-12, dtype=S0.dtype, device=S0.device))
    sqrt_tau = torch.sqrt(tau_t)
    d1 = (torch.log(S0 / K_t) + (r + 0.5 * sigma_t**2) * tau_t) / (sigma_t * sqrt_tau)
    d2 = d1 - sigma_t * sqrt_tau
    disc = torch.exp(-r * tau_t)
    if opt == "call":
        return S0 * _torch_norm_cdf(d1) - K_t * disc * _torch_norm_cdf(d2)
    return K_t * disc * _torch_norm_cdf(-d2) - S0 * _torch_norm_cdf(-d1)


def BS_solver(S0, K, tau, sigma, r, t0=0, opt_type="put"):
    """
    Pointwise, not vectors
     
     :param S0: spot proce
     :param K: strike price
     :param tau: time to maturity
     :param sigma: volatility
     :param r: risk-free rate 
     :param t0: actual time
     :param opt_type: option type
     """

    # some formatting
    S0 = np.float64(S0)
    K = np.float64(K)

    # some constants
    T = tau + t0

    # compute d1 and d2
    d1 = (np.log(S0 / K) + (r+0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # compute the option price
    if opt_type == "call":
        V = S0*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    elif opt_type == "put":
        V = K*np.exp(-r*tau)*norm.cdf(-d2) - S0*norm.cdf(-d1) 
    return V
