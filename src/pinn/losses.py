from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PINNLossTerms:
    pde: float
    term: float
    low: float
    no_arbitrage: float

    @property
    def total(self) -> float:
        return self.pde + self.term + self.low + self.no_arbitrage


def _get_weight(loss_config: dict, key: str, default: float) -> float:
    weights = loss_config.get("weights", {})
    return float(weights.get(key, default))


def _gradient(*, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad_out = torch.ones_like(y)
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
    )[0]


def compute_weighted_pinn_loss(
    *,
    model: nn.Module,
    loss_config: dict,
    batch_payload: dict,
) -> tuple[torch.Tensor, PINNLossTerms]:
    """
    Compute weighted PINN objective:
      L = w_pde * L_pde + w_term * L_term + w_low * L_low (+ optional no-arbitrage term).
    """

    x_interior = batch_payload["interior"]
    x_terminal = batch_payload["terminal"]
    x_lower = batch_payload["lower"]

    if x_interior.ndim != 2 or x_terminal.ndim != 2 or x_lower.ndim != 2:
        raise ValueError("All PINN batch tensors must be 2D [batch, features].")
    if x_interior.shape[1] < 8 or x_terminal.shape[1] < 8 or x_lower.shape[1] < 8:
        raise ValueError("Expected at least 8 input features: [tau,m,v,rho,kappa,gamma,bar_v,r].")

    x_interior = x_interior.requires_grad_(True)
    u_interior = model(x_interior)
    if u_interior.ndim != 2 or u_interior.shape[1] != 1:
        raise ValueError(f"Expected model output shape [N,1], got {tuple(u_interior.shape)}.")

    grads = _gradient(y=u_interior, x=x_interior)
    u_tau = grads[:, 0:1]
    u_m = grads[:, 1:2]
    u_v = grads[:, 2:3]

    grad_u_m = _gradient(y=u_m, x=x_interior)
    grad_u_v = _gradient(y=u_v, x=x_interior)
    u_mm = grad_u_m[:, 1:2]
    u_mv = grad_u_m[:, 2:3]
    u_vv = grad_u_v[:, 2:3]

    tau = x_interior[:, 0:1]
    m = x_interior[:, 1:2]
    v = x_interior[:, 2:3]
    rho = x_interior[:, 3:4]
    kappa = x_interior[:, 4:5]
    gamma = x_interior[:, 5:6]
    bar_v = x_interior[:, 6:7]
    r = x_interior[:, 7:8]
    del tau  # kept in layout for consistency; PDE here is tau-forward.

    residual = (
        u_tau
        - 0.5 * v * (m**2) * u_mm
        - rho * gamma * v * m * u_mv
        - 0.5 * (gamma**2) * v * u_vv
        - r * m * u_m
        - kappa * (bar_v - v) * u_v
        + r * u_interior
    )
    l_pde = torch.mean(residual**2)

    u_terminal = model(x_terminal)
    m_terminal = x_terminal[:, 1:2]
    payoff = torch.clamp(1.0 - m_terminal, min=0.0)
    l_term = torch.mean((u_terminal - payoff) ** 2)

    u_lower = model(x_lower)
    tau_lower = x_lower[:, 0:1]
    r_lower = x_lower[:, 7:8]
    lower_target = torch.exp(-r_lower * tau_lower)
    l_low = torch.mean((u_lower - lower_target) ** 2)

    w_pde = _get_weight(loss_config, "pde", 1.0)
    w_term = _get_weight(
        loss_config,
        "term",
        _get_weight(loss_config, "terminal", 1.0),
    )
    w_low = _get_weight(
        loss_config,
        "low",
        _get_weight(loss_config, "boundary", 1.0),
    )
    w_no_arb = _get_weight(loss_config, "no_arbitrage", 0.0)

    no_arb_term = torch.zeros_like(l_pde)
    total = w_pde * l_pde + w_term * l_term + w_low * l_low + w_no_arb * no_arb_term

    terms = PINNLossTerms(
        pde=float((w_pde * l_pde).detach().item()),
        term=float((w_term * l_term).detach().item()),
        low=float((w_low * l_low).detach().item()),
        no_arbitrage=float((w_no_arb * no_arb_term).detach().item()),
    )
    return total, terms
