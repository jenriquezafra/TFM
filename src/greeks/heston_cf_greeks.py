from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class HestonCFGreeksSettings:
    u_min: float = 1.0e-6
    u_max: float = 200.0
    n_u: int = 1200


def _validate_settings(settings: HestonCFGreeksSettings) -> None:
    if settings.n_u < 8:
        raise ValueError("n_u must be >= 8")
    if settings.u_min <= 0.0:
        raise ValueError("u_min must be > 0")
    if settings.u_max <= settings.u_min:
        raise ValueError("u_max must be > u_min")


def _heston_log_chf_with_derivatives(
    z: np.ndarray,
    *,
    tau: float,
    r: float,
    rho: float,
    kappa: float,
    gamma: float,
    bar_v: float,
    v0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      log_phi(z), dlog_phi/dr, dlog_phi/dv0, dlog_phi/dtau
    for the characteristic function of X=log(S_T/S_0) under Heston.
    """
    if gamma == 0.0:
        raise ValueError("gamma must be non-zero")

    zc = np.asarray(z, dtype=np.complex128)
    gamma2 = float(gamma) ** 2

    a = kappa - gamma * rho * 1j * zc
    d = np.sqrt(a * a + gamma2 * (zc * zc + 1j * zc))
    g = (a - d) / (a + d)
    cc = a - d
    exp_term = np.exp(-d * tau)

    one = 1.0 + 0.0j
    denom = one - g * exp_term
    q = ((one - exp_term) / denom) * cc
    log_ratio = np.log((one - g * exp_term) / (one - g))

    log_phi = (
        1j * zc * tau * r
        + (v0 / gamma2) * q
        + (kappa * bar_v / gamma2) * (tau * cc - 2.0 * log_ratio)
    )

    dlog_dr = 1j * zc * tau
    dlog_dv0 = q / gamma2

    d_q_dtau = cc * d * exp_term * (one - g) / (denom * denom)
    d_log_term_dtau = (g * d * exp_term) / denom
    dlog_dtau = (
        1j * zc * r
        + (v0 / gamma2) * d_q_dtau
        + (kappa * bar_v / gamma2) * (cc - 2.0 * d_log_term_dtau)
    )
    return log_phi, dlog_dr, dlog_dv0, dlog_dtau


def _trapz_real(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(np.real(y), x))


def heston_cf_call_greeks_scalar(
    *,
    S0: float,
    K: float,
    tau: float,
    r: float,
    rho: float,
    kappa: float,
    gamma: float,
    bar_v: float,
    v0: float,
    settings: HestonCFGreeksSettings | None = None,
) -> dict[str, float]:
    """
    Semi-analytical Heston call price + Greeks from characteristic-function
    probabilities (Pi1, Pi2) and analytic derivatives under the integral sign.
    """
    cfg = settings or HestonCFGreeksSettings()
    _validate_settings(cfg)

    if S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if K <= 0.0:
        raise ValueError("K must be > 0")
    if tau < 0.0:
        raise ValueError("tau must be >= 0")

    u = np.linspace(float(cfg.u_min), float(cfg.u_max), int(cfg.n_u), dtype=np.float64)
    m = float(np.log(S0 / K))
    exp_ium = np.exp(1j * u * m)
    inv_iu = 1.0 / (1j * u)

    z2 = u.astype(np.complex128)        # for Pi2
    z1 = z2 - 1j                        # for Pi1 numerator
    z0 = np.asarray([-1j], dtype=np.complex128)  # Pi1 denominator

    log_phi_2, dlog2_dr, dlog2_dv0, dlog2_dtau = _heston_log_chf_with_derivatives(
        z2,
        tau=tau,
        r=r,
        rho=rho,
        kappa=kappa,
        gamma=gamma,
        bar_v=bar_v,
        v0=v0,
    )
    log_phi_1n, dlog1n_dr, dlog1n_dv0, dlog1n_dtau = _heston_log_chf_with_derivatives(
        z1,
        tau=tau,
        r=r,
        rho=rho,
        kappa=kappa,
        gamma=gamma,
        bar_v=bar_v,
        v0=v0,
    )
    log_phi_10, dlog10_dr, dlog10_dv0, dlog10_dtau = _heston_log_chf_with_derivatives(
        z0,
        tau=tau,
        r=r,
        rho=rho,
        kappa=kappa,
        gamma=gamma,
        bar_v=bar_v,
        v0=v0,
    )

    phi_2 = np.exp(log_phi_2)
    phi_1n = np.exp(log_phi_1n)
    phi_10 = np.exp(log_phi_10[0])

    A2 = phi_2
    A1 = phi_1n / phi_10

    dA2_dr = A2 * dlog2_dr
    dA2_dv0 = A2 * dlog2_dv0
    dA2_dtau = A2 * dlog2_dtau

    dlog10_dr_s = dlog10_dr[0]
    dlog10_dv0_s = dlog10_dv0[0]
    dlog10_dtau_s = dlog10_dtau[0]
    dA1_dr = A1 * (dlog1n_dr - dlog10_dr_s)
    dA1_dv0 = A1 * (dlog1n_dv0 - dlog10_dv0_s)
    dA1_dtau = A1 * (dlog1n_dtau - dlog10_dtau_s)

    Pi1 = 0.5 + (1.0 / pi) * _trapz_real(exp_ium * A1 * inv_iu, u)
    Pi2 = 0.5 + (1.0 / pi) * _trapz_real(exp_ium * A2 * inv_iu, u)

    Pi1_s = (1.0 / (pi * S0)) * _trapz_real(exp_ium * A1, u)
    Pi2_s = (1.0 / (pi * S0)) * _trapz_real(exp_ium * A2, u)

    factor_ss = 1.0 / (pi * S0 * S0)
    Pi1_ss = factor_ss * _trapz_real(exp_ium * A1 * (1j * u - 1.0), u)
    Pi2_ss = factor_ss * _trapz_real(exp_ium * A2 * (1j * u - 1.0), u)

    Pi1_v0 = (1.0 / pi) * _trapz_real(exp_ium * dA1_dv0 * inv_iu, u)
    Pi2_v0 = (1.0 / pi) * _trapz_real(exp_ium * dA2_dv0 * inv_iu, u)

    Pi1_r = (1.0 / pi) * _trapz_real(exp_ium * dA1_dr * inv_iu, u)
    Pi2_r = (1.0 / pi) * _trapz_real(exp_ium * dA2_dr * inv_iu, u)

    Pi1_tau = (1.0 / pi) * _trapz_real(exp_ium * dA1_dtau * inv_iu, u)
    Pi2_tau = (1.0 / pi) * _trapz_real(exp_ium * dA2_dtau * inv_iu, u)

    disc = float(np.exp(-r * tau))
    price = S0 * Pi1 - K * disc * Pi2
    delta = Pi1 + S0 * Pi1_s - K * disc * Pi2_s
    gamma_out = 2.0 * Pi1_s + S0 * Pi1_ss - K * disc * Pi2_ss
    vega_v0 = S0 * Pi1_v0 - K * disc * Pi2_v0
    rho_out = S0 * Pi1_r + K * tau * disc * Pi2 - K * disc * Pi2_r
    dC_dtau = S0 * Pi1_tau + K * r * disc * Pi2 - K * disc * Pi2_tau
    theta = -dC_dtau

    return {
        "price": float(np.real(price)),
        "delta": float(np.real(delta)),
        "gamma": float(np.real(gamma_out)),
        "vega": float(np.real(vega_v0)),
        "rho": float(np.real(rho_out)),
        "theta": float(np.real(theta)),
    }


def heston_cf_greeks_scalar(
    *,
    option_type: str,
    S0: float,
    K: float,
    tau: float,
    r: float,
    rho: float,
    kappa: float,
    gamma: float,
    bar_v: float,
    v0: float,
    settings: HestonCFGreeksSettings | None = None,
) -> dict[str, float]:
    """
    Semi-analytical Heston Greeks for call/put.
    Put Greeks are derived by put-call parity from call Greeks.
    """
    opt = str(option_type).strip().lower()
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be one of {'call', 'put'}")

    call = heston_cf_call_greeks_scalar(
        S0=S0,
        K=K,
        tau=tau,
        r=r,
        rho=rho,
        kappa=kappa,
        gamma=gamma,
        bar_v=bar_v,
        v0=v0,
        settings=settings,
    )
    if opt == "call":
        return call

    disc = float(np.exp(-r * tau))
    out = dict(call)
    out["price"] = float(call["price"] - S0 + K * disc)
    out["delta"] = float(call["delta"] - 1.0)
    out["gamma"] = float(call["gamma"])
    out["vega"] = float(call["vega"])
    out["rho"] = float(call["rho"] - tau * K * disc)
    out["theta"] = float(call["theta"] + r * K * disc)
    return out


__all__ = [
    "HestonCFGreeksSettings",
    "heston_cf_call_greeks_scalar",
    "heston_cf_greeks_scalar",
]
