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
    dpde: float = 0.0
    greek_delta: float = 0.0
    greek_gamma: float = 0.0
    greek_vega: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.pde
            + self.dpde
            + self.term
            + self.low
            + self.no_arbitrage
            + self.greek_delta
            + self.greek_gamma
            + self.greek_vega
        )


def _apply_input_affine(
    *,
    x: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if input_affine is None:
        return x

    a_raw = input_affine.get("a")
    b_raw = input_affine.get("b")
    if a_raw is None or b_raw is None:
        raise KeyError("input_affine must include keys 'a' and 'b'.")

    a = torch.as_tensor(a_raw, dtype=x.dtype, device=x.device).reshape(-1)
    b = torch.as_tensor(b_raw, dtype=x.dtype, device=x.device).reshape(-1)
    if a.numel() != x.shape[1] or b.numel() != x.shape[1]:
        raise ValueError(
            "input_affine shape mismatch: "
            f"x has {x.shape[1]} features, "
            f"a has {a.numel()}, b has {b.numel()}."
        )
    return x * b.view(1, -1) + a.view(1, -1)


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


def compute_heston_pde_residual(
    *,
    model: nn.Module,
    x_interior: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if x_interior.ndim != 2:
        raise ValueError(f"x_interior must be 2D [batch, features], got {tuple(x_interior.shape)}")
    if x_interior.shape[1] < 8:
        raise ValueError("Expected at least 8 input features: [tau,m,v,rho,kappa,gamma,bar_v,r].")

    x_interior = x_interior.requires_grad_(True)
    x_net = _apply_input_affine(x=x_interior, input_affine=input_affine)
    u_interior = model(x_net)
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

    m = x_interior[:, 1:2]
    v = x_interior[:, 2:3]
    rho = x_interior[:, 3:4]
    kappa = x_interior[:, 4:5]
    gamma = x_interior[:, 5:6]
    bar_v = x_interior[:, 6:7]
    r = x_interior[:, 7:8]

    residual = (
        u_tau
        - 0.5 * v * (m**2) * u_mm
        - rho * gamma * v * m * u_mv
        - 0.5 * (gamma**2) * v * u_vv
        - r * m * u_m
        - kappa * (bar_v - v) * u_v
        + r * u_interior
    )
    return residual


def compute_heston_pde_derivative_residual(
    *,
    model: nn.Module,
    x_interior: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return PDE residual and first derivatives of the residual with respect to
    log-moneyness x=log(m) and variance v.

    The model and PDE still receive raw financial coordinates. The x derivative
    is obtained by chain rule from the raw moneyness derivative:
      dN/dx = m * dN/dm.
    """

    residual = compute_heston_pde_residual(
        model=model,
        x_interior=x_interior,
        input_affine=input_affine,
    )
    grad_residual = _gradient(y=residual, x=x_interior)
    m = x_interior[:, 1:2]
    d_residual_dx = m * grad_residual[:, 1:2]
    d_residual_dv = grad_residual[:, 2:3]
    return residual, d_residual_dx, d_residual_dv


def compute_weighted_pinn_loss(
    *,
    model: nn.Module,
    loss_config: dict,
    batch_payload: dict,
    input_affine: dict[str, torch.Tensor] | None = None,
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

    residual = compute_heston_pde_residual(
        model=model,
        x_interior=x_interior,
        input_affine=input_affine,
    )
    l_pde = torch.mean(residual**2)

    u_terminal = model(_apply_input_affine(x=x_terminal, input_affine=input_affine))
    m_terminal = x_terminal[:, 1:2]
    payoff = torch.clamp(1.0 - m_terminal, min=0.0)
    l_term = torch.mean((u_terminal - payoff) ** 2)

    u_lower = model(_apply_input_affine(x=x_lower, input_affine=input_affine))
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
    w_dpde = _get_weight(
        loss_config,
        "dpde",
        _get_weight(loss_config, "derivative_pde", 0.0),
    )
    w_greek_delta = _get_weight(
        loss_config,
        "greek_delta",
        _get_weight(loss_config, "delta_consistency", 0.0),
    )
    w_greek_gamma = _get_weight(
        loss_config,
        "greek_gamma",
        _get_weight(loss_config, "gamma_consistency", 0.0),
    )
    w_greek_vega = _get_weight(
        loss_config,
        "greek_vega",
        _get_weight(loss_config, "vega_consistency", 0.0),
    )

    no_arb_term = torch.zeros_like(l_pde)
    if w_dpde > 0.0:
        grad_residual = _gradient(y=residual, x=x_interior)
        m_interior = x_interior[:, 1:2]
        d_residual_dx = m_interior * grad_residual[:, 1:2]
        d_residual_dv = grad_residual[:, 2:3]
        l_dpde = torch.mean(d_residual_dx**2 + d_residual_dv**2)
    else:
        l_dpde = torch.zeros_like(l_pde)

    needs_greek_heads = any(
        weight > 0.0 for weight in (w_greek_delta, w_greek_gamma, w_greek_vega)
    )
    if needs_greek_heads:
        forward_all = getattr(model, "forward_all", None)
        if not callable(forward_all):
            raise ValueError(
                "Greek-consistency weights require a model with forward_all(), "
                "for example greek_consistency.enabled=true."
            )
        x_net_for_heads = _apply_input_affine(x=x_interior, input_affine=input_affine)
        outputs = forward_all(x_net_for_heads)
        if outputs.ndim != 2 or outputs.shape[1] < 4:
            raise ValueError(
                "Greek-consistency model must return at least four heads: "
                "[price, delta, gamma, vega]."
            )
        price_head = outputs[:, 0:1]
        delta_head = outputs[:, 1:2]
        gamma_head = outputs[:, 2:3]
        vega_head = outputs[:, 3:4]

        grad_price = _gradient(y=price_head, x=x_interior)
        price_m = grad_price[:, 1:2]
        price_v = grad_price[:, 2:3]
        grad_delta_head = _gradient(y=delta_head, x=x_interior)
        delta_head_m = grad_delta_head[:, 1:2]

        l_greek_delta = torch.mean((delta_head - price_m) ** 2)
        l_greek_gamma = torch.mean((gamma_head - delta_head_m) ** 2)
        l_greek_vega = torch.mean((vega_head - price_v) ** 2)
    else:
        l_greek_delta = torch.zeros_like(l_pde)
        l_greek_gamma = torch.zeros_like(l_pde)
        l_greek_vega = torch.zeros_like(l_pde)

    total = (
        w_pde * l_pde
        + w_dpde * l_dpde
        + w_term * l_term
        + w_low * l_low
        + w_no_arb * no_arb_term
        + w_greek_delta * l_greek_delta
        + w_greek_gamma * l_greek_gamma
        + w_greek_vega * l_greek_vega
    )

    terms = PINNLossTerms(
        pde=float((w_pde * l_pde).detach().item()),
        dpde=float((w_dpde * l_dpde).detach().item()),
        term=float((w_term * l_term).detach().item()),
        low=float((w_low * l_low).detach().item()),
        no_arbitrage=float((w_no_arb * no_arb_term).detach().item()),
        greek_delta=float((w_greek_delta * l_greek_delta).detach().item()),
        greek_gamma=float((w_greek_gamma * l_greek_gamma).detach().item()),
        greek_vega=float((w_greek_vega * l_greek_vega).detach().item()),
    )
    return total, terms
