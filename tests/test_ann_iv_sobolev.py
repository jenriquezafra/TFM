from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.sobolev.ann_iv import compute_ann_iv_sobolev_targets, sobolev_derivative_loss
from src.solvers.bs import BS_greeks_np, BS_price_np


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def test_black_scholes_greeks_match_finite_differences() -> None:
    s0 = 1.07
    k = 1.0
    tau = 0.8
    sigma = 0.32
    r = 0.03
    h = 1.0e-5
    greeks = BS_greeks_np(S0=s0, K=k, tau=tau, sigma=sigma, r=r, opt_type="put")

    p_s_plus = _scalar(BS_price_np(S0=s0 + h, K=k, tau=tau, sigma=sigma, r=r, opt_type="put"))
    p_s = _scalar(BS_price_np(S0=s0, K=k, tau=tau, sigma=sigma, r=r, opt_type="put"))
    p_s_minus = _scalar(BS_price_np(S0=s0 - h, K=k, tau=tau, sigma=sigma, r=r, opt_type="put"))
    delta_fd = (p_s_plus - p_s_minus) / (2.0 * h)
    gamma_fd = (p_s_plus - 2.0 * p_s + p_s_minus) / (h * h)

    p_vol_plus = _scalar(BS_price_np(S0=s0, K=k, tau=tau, sigma=sigma + h, r=r, opt_type="put"))
    p_vol_minus = _scalar(BS_price_np(S0=s0, K=k, tau=tau, sigma=sigma - h, r=r, opt_type="put"))
    vega_fd = (p_vol_plus - p_vol_minus) / (2.0 * h)

    p_r_plus = _scalar(BS_price_np(S0=s0, K=k, tau=tau, sigma=sigma, r=r + h, opt_type="put"))
    p_r_minus = _scalar(BS_price_np(S0=s0, K=k, tau=tau, sigma=sigma, r=r - h, opt_type="put"))
    rho_fd = (p_r_plus - p_r_minus) / (2.0 * h)

    p_tau_plus = _scalar(BS_price_np(S0=s0, K=k, tau=tau + h, sigma=sigma, r=r, opt_type="put"))
    p_tau_minus = _scalar(BS_price_np(S0=s0, K=k, tau=tau - h, sigma=sigma, r=r, opt_type="put"))
    d_price_dtau_fd = (p_tau_plus - p_tau_minus) / (2.0 * h)

    assert np.isclose(_scalar(greeks["delta"]), delta_fd, rtol=1.0e-6, atol=1.0e-8)
    assert np.isclose(_scalar(greeks["gamma"]), gamma_fd, rtol=2.0e-4, atol=2.0e-6)
    assert np.isclose(_scalar(greeks["vega_sigma"]), vega_fd, rtol=1.0e-6, atol=1.0e-8)
    assert np.isclose(_scalar(greeks["rho"]), rho_fd, rtol=1.0e-6, atol=1.0e-8)
    assert np.isclose(-_scalar(greeks["theta"]), d_price_dtau_fd, rtol=1.0e-6, atol=1.0e-8)


def _iv_from_heston_cf(row: dict[str, float], *, settings: HestonCFGreeksSettings) -> float:
    price = heston_cf_greeks_scalar(
        option_type="put",
        S0=float(row["moneyness"]),
        K=1.0,
        tau=float(row["tau"]),
        r=float(row["r"]),
        rho=float(row["rho"]),
        kappa=float(row["kappa"]),
        gamma=float(row["gamma"]),
        bar_v=float(row["bar_v"]),
        v0=float(row["v0"]),
        settings=settings,
    )["price"]

    def residual(sigma: float) -> float:
        return _scalar(
            BS_price_np(
                S0=float(row["moneyness"]),
                K=1.0,
                tau=float(row["tau"]),
                sigma=sigma,
                r=float(row["r"]),
                opt_type="put",
            )
        ) - float(price)

    return float(brentq(residual, 1.0e-5, 4.0, xtol=1.0e-10))


def test_implicit_iv_sobolev_targets_match_bump_and_invert() -> None:
    settings = HestonCFGreeksSettings(u_min=1.0e-6, u_max=160.0, n_u=900)
    row = {
        "rho": -0.5,
        "kappa": 1.5,
        "gamma": 0.25,
        "bar_v": 0.04,
        "v0": 0.04,
        "moneyness": 1.05,
        "tau": 1.0,
        "r": 0.02,
    }
    row["IV"] = _iv_from_heston_cf(row, settings=settings)
    targets = compute_ann_iv_sobolev_targets(
        pd.DataFrame([row]),
        iv_column="IV",
        cf_settings=settings,
        vega_floor=1.0e-8,
    )
    assert len(targets) == 1

    bumps = {
        "moneyness": ("d_iv_dm", 1.0e-4),
        "tau": ("d_iv_dtau", 1.0e-4),
        "r": ("d_iv_dr", 1.0e-5),
        "v0": ("d_iv_dv0", 1.0e-5),
    }
    for feature, (target_col, h) in bumps.items():
        plus = dict(row)
        minus = dict(row)
        plus[feature] += h
        minus[feature] -= h
        iv_plus = _iv_from_heston_cf(plus, settings=settings)
        iv_minus = _iv_from_heston_cf(minus, settings=settings)
        fd = (iv_plus - iv_minus) / (2.0 * h)
        assert np.isclose(float(targets.loc[0, target_col]), fd, rtol=5.0e-3, atol=5.0e-4)


def test_sobolev_derivative_loss_uses_raw_derivative_units() -> None:
    class ToyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (2.0 * x[:, 0:1]) - (4.0 * x[:, 2:3])

    x_model = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    x_std = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float32)
    y_std = 3.0
    target = torch.tensor([[3.0, -3.0], [3.0, -3.0]], dtype=torch.float32)

    loss, pred = sobolev_derivative_loss(
        model=ToyModel(),
        x_model=x_model,
        target_derivatives_raw=target,
        derivative_indices=[0, 2],
        x_std=x_std,
        y_std=y_std,
        derivative_scales=[1.0, 1.0],
    )
    assert torch.allclose(pred, target, atol=1.0e-7)
    assert float(loss.item()) == 0.0
