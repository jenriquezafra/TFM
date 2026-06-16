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
    right: float = 0.0
    v_zero: float = 0.0
    dpde: float = 0.0
    greek_delta: float = 0.0
    greek_gamma: float = 0.0
    greek_vega: float = 0.0
    curvature_bulk: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.pde
            + self.dpde
            + self.term
            + self.low
            + self.right
            + self.v_zero
            + self.no_arbitrage
            + self.greek_delta
            + self.greek_gamma
            + self.greek_vega
            + self.curvature_bulk
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


def _nested_dict(config: dict, key: str) -> dict:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _learned_log_var_term(
    *,
    name: str,
    base_loss: torch.Tensor,
    loss_config: dict,
    learned_log_vars: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    cfg = _nested_dict(loss_config, "learned_weights")
    if not bool(cfg.get("enabled", False)) or learned_log_vars is None or name not in learned_log_vars:
        return base_loss

    min_log_var = float(cfg.get("min_log_var", cfg.get("min_log_weight", -6.0)))
    max_log_var = float(cfg.get("max_log_var", cfg.get("max_log_weight", 6.0)))
    if min_log_var > max_log_var:
        raise ValueError(
            "loss.learned_weights min_log_var must be <= max_log_var. "
            f"Got {min_log_var} > {max_log_var}."
        )

    log_var = torch.clamp(learned_log_vars[name], min=min_log_var, max=max_log_var)
    total = torch.exp(-log_var) * base_loss + log_var

    prior_cfg = cfg.get("prior", {})
    prior_cfg = prior_cfg if isinstance(prior_cfg, dict) else {}
    prior_strengths = cfg.get("prior_strengths", prior_cfg.get("term_strengths", {}))
    prior_strengths = prior_strengths if isinstance(prior_strengths, dict) else {}
    prior_targets = cfg.get("prior_targets", prior_cfg.get("term_targets", {}))
    prior_targets = prior_targets if isinstance(prior_targets, dict) else {}
    prior_strength = float(
        prior_strengths.get(
            name,
            prior_cfg.get("strength", cfg.get("prior_strength", 0.0)),
        )
    )
    if prior_strength > 0.0:
        prior_target = float(
            prior_targets.get(
                name,
                prior_cfg.get("target", cfg.get("prior_target", 0.0)),
            )
        )
        raw_log_var = learned_log_vars[name]
        target = torch.as_tensor(prior_target, dtype=raw_log_var.dtype, device=raw_log_var.device)
        total = total + prior_strength * (raw_log_var - target) ** 2
    return total


def _coordinate_key(coordinate: str) -> str:
    key = str(coordinate).strip().lower()
    if key in {"moneyness", "m", "raw"}:
        return "moneyness"
    if key in {"log_moneyness", "log-moneyness", "x"}:
        return "log_moneyness"
    raise ValueError("coordinate must be 'moneyness' or 'log_moneyness'.")


def _moneyness_from_input(x: torch.Tensor, coordinate: str) -> torch.Tensor:
    if _coordinate_key(coordinate) == "log_moneyness":
        return torch.exp(x[:, 1:2])
    return x[:, 1:2]


def _log_moneyness_from_input(x: torch.Tensor, coordinate: str) -> torch.Tensor:
    if _coordinate_key(coordinate) == "log_moneyness":
        return x[:, 1:2]
    m = torch.clamp(x[:, 1:2], min=torch.finfo(x.dtype).eps)
    return torch.log(m)


def _gradient(*, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad_out = torch.ones_like(y)
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
    )[0]


def _heston_pde_residual_and_scale(
    *,
    model: nn.Module,
    x_interior: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None = None,
    coordinate: str = "moneyness",
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_interior.ndim != 2:
        raise ValueError(f"x_interior must be 2D [batch, features], got {tuple(x_interior.shape)}")
    if x_interior.shape[1] < 8:
        raise ValueError("Expected at least 8 input features: [tau,spot_coord,v,rho,kappa,gamma,bar_v,r].")

    x_interior = x_interior.requires_grad_(True)
    x_net = _apply_input_affine(x=x_interior, input_affine=input_affine)
    u_interior = model(x_net)
    if u_interior.ndim != 2 or u_interior.shape[1] != 1:
        raise ValueError(f"Expected model output shape [N,1], got {tuple(u_interior.shape)}.")

    grads = _gradient(y=u_interior, x=x_interior)
    u_tau = grads[:, 0:1]
    u_s = grads[:, 1:2]
    u_v = grads[:, 2:3]

    grad_u_s = _gradient(y=u_s, x=x_interior)
    grad_u_v = _gradient(y=u_v, x=x_interior)
    u_ss = grad_u_s[:, 1:2]
    u_sv = grad_u_s[:, 2:3]
    u_vv = grad_u_v[:, 2:3]

    v = x_interior[:, 2:3]
    rho = x_interior[:, 3:4]
    kappa = x_interior[:, 4:5]
    gamma = x_interior[:, 5:6]
    bar_v = x_interior[:, 6:7]
    r = x_interior[:, 7:8]

    coordinate_key = _coordinate_key(coordinate)
    if coordinate_key == "moneyness":
        m = x_interior[:, 1:2]
        diffusion_s = 0.5 * v * (m**2) * u_ss
        mixed = rho * gamma * v * m * u_sv
        diffusion_v = 0.5 * (gamma**2) * v * u_vv
        drift_s = r * m * u_s
        drift_v = kappa * (bar_v - v) * u_v
        discount = r * u_interior
        residual = (
            u_tau
            - diffusion_s
            - mixed
            - diffusion_v
            - drift_s
            - drift_v
            + discount
        )
        scale = (
            torch.abs(u_tau)
            + torch.abs(diffusion_s)
            + torch.abs(mixed)
            + torch.abs(diffusion_v)
            + torch.abs(drift_s)
            + torch.abs(drift_v)
            + torch.abs(discount)
        )
        return residual, scale

    diffusion_x = 0.5 * v * u_ss
    drift_x = (r - 0.5 * v) * u_s
    mixed = rho * gamma * v * u_sv
    diffusion_v = 0.5 * (gamma**2) * v * u_vv
    drift_v = kappa * (bar_v - v) * u_v
    discount = r * u_interior
    residual = (
        u_tau
        - diffusion_x
        - drift_x
        - mixed
        - diffusion_v
        - drift_v
        + discount
    )
    scale = (
        torch.abs(u_tau)
        + torch.abs(diffusion_x)
        + torch.abs(drift_x)
        + torch.abs(mixed)
        + torch.abs(diffusion_v)
        + torch.abs(drift_v)
        + torch.abs(discount)
    )
    return residual, scale


def _normalize_pde_residual(
    residual: torch.Tensor,
    scale: torch.Tensor,
    *,
    loss_config: dict,
) -> torch.Tensor:
    pde_cfg = _nested_dict(loss_config, "pde")
    norm_cfg = pde_cfg.get("normalization", {})
    norm_cfg = norm_cfg if isinstance(norm_cfg, dict) else {}
    if not bool(norm_cfg.get("enabled", False)):
        return residual

    mode = str(norm_cfg.get("mode", "sum_abs_terms")).strip().lower()
    if mode not in {"sum_abs_terms", "local_terms"}:
        raise ValueError(
            "loss.pde.normalization.mode must be 'sum_abs_terms' or 'local_terms'."
        )
    floor = float(norm_cfg.get("floor", norm_cfg.get("eps", 1.0e-3)))
    if floor <= 0.0:
        raise ValueError("loss.pde.normalization.floor must be > 0.")
    denominator = torch.clamp(scale, min=torch.as_tensor(floor, dtype=scale.dtype, device=scale.device))
    return residual / denominator


def compute_heston_pde_residual(
    *,
    model: nn.Module,
    x_interior: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None = None,
    coordinate: str = "moneyness",
) -> torch.Tensor:
    residual, _scale = _heston_pde_residual_and_scale(
        model=model,
        x_interior=x_interior,
        input_affine=input_affine,
        coordinate=coordinate,
    )
    return residual


def compute_heston_v_zero_residual(
    *,
    model: nn.Module,
    x_v_zero: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None = None,
    coordinate: str = "moneyness",
) -> torch.Tensor:
    if x_v_zero.ndim != 2:
        raise ValueError(f"x_v_zero must be 2D [batch, features], got {tuple(x_v_zero.shape)}")
    x_v_zero = x_v_zero.requires_grad_(True)
    u = model(_apply_input_affine(x=x_v_zero, input_affine=input_affine))
    grads = _gradient(y=u, x=x_v_zero)
    u_tau = grads[:, 0:1]
    u_s = grads[:, 1:2]
    u_v = grads[:, 2:3]
    kappa = x_v_zero[:, 4:5]
    bar_v = x_v_zero[:, 6:7]
    r = x_v_zero[:, 7:8]
    if _coordinate_key(coordinate) == "log_moneyness":
        return u_tau - r * u_s - kappa * bar_v * u_v + r * u
    m = x_v_zero[:, 1:2]
    return u_tau - r * m * u_s - kappa * bar_v * u_v + r * u


def _price_derivatives(
    *,
    model: nn.Module,
    x: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None,
    coordinate: str = "moneyness",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return price, first spot-coordinate derivative, v derivative and financial
    curvature term. For log-moneyness the last component is G=U_xx-U_x.
    """
    x = x.requires_grad_(True)
    u = model(_apply_input_affine(x=x, input_affine=input_affine))
    grads = _gradient(y=u, x=x)
    u_s = grads[:, 1:2]
    u_v = grads[:, 2:3]
    grad_u_s = _gradient(y=u_s, x=x)
    u_ss = grad_u_s[:, 1:2]
    if _coordinate_key(coordinate) == "log_moneyness":
        return u, u_s, u_v, u_ss - u_s
    return u, u_s, u_v, u_ss


def _kink_gate(x: torch.Tensor, loss_config: dict, *, coordinate: str = "moneyness") -> torch.Tensor:
    cfg = _nested_dict(loss_config, "kink_bulk")
    eps_l = float(cfg.get("ell_epsilon", cfg.get("epsilon", 1.0e-8)))
    c = float(cfg.get("c", 2.0))
    delta_x = float(cfg.get("delta_x", 0.05))
    delta_tau = float(cfg.get("delta_tau", 0.02))
    tau_c = float(cfg.get("tau_c", 0.15))
    log_m = _log_moneyness_from_input(x, coordinate)
    tau = torch.clamp(x[:, 0:1], min=0.0)
    v = torch.clamp(x[:, 2:3], min=0.0)
    ell = c * torch.sqrt(v * tau + torch.as_tensor(eps_l, dtype=x.dtype, device=x.device))
    gate_x = torch.sigmoid((ell - torch.abs(log_m)) / delta_x)
    gate_tau = torch.sigmoid((tau_c - tau) / delta_tau)
    return gate_x * gate_tau


def _weighted_mean_square(values: torch.Tensor, weights: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    denom = torch.sum(weights) + torch.as_tensor(eps, dtype=values.dtype, device=values.device)
    return torch.sum(weights * values**2) / denom


def compute_heston_pde_derivative_residual(
    *,
    model: nn.Module,
    x_interior: torch.Tensor,
    input_affine: dict[str, torch.Tensor] | None = None,
    coordinate: str = "moneyness",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return PDE residual and first derivatives of the residual with respect to
    log-moneyness x=log(m) and variance v.

    When the second input coordinate is raw moneyness, the x derivative is
    obtained by chain rule: dR/dx = m dR/dm. When the second coordinate is
    already x=log(m), the derivative is the direct autograd derivative.
    """

    coordinate_key = _coordinate_key(coordinate)
    residual, _scale = _heston_pde_residual_and_scale(
        model=model,
        x_interior=x_interior,
        input_affine=input_affine,
        coordinate=coordinate_key,
    )
    grad_residual = _gradient(y=residual, x=x_interior)
    if coordinate_key == "log_moneyness":
        d_residual_dx = grad_residual[:, 1:2]
    else:
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
    learned_log_vars: dict[str, torch.Tensor] | None = None,
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
        raise ValueError("Expected at least 8 input features: [tau,spot_coord,v,rho,kappa,gamma,bar_v,r].")

    coordinate = str(_nested_dict(loss_config, "pde").get("coordinate", "moneyness"))
    coordinate_key = _coordinate_key(coordinate)
    raw_residual, residual_scale = _heston_pde_residual_and_scale(
        model=model,
        x_interior=x_interior,
        input_affine=input_affine,
        coordinate=coordinate_key,
    )
    residual = _normalize_pde_residual(
        raw_residual,
        residual_scale,
        loss_config=loss_config,
    )
    kink_bulk_enabled = bool(_nested_dict(loss_config, "kink_bulk").get("enabled", False))
    if kink_bulk_enabled:
        gate = _kink_gate(x_interior, loss_config, coordinate=coordinate_key)
        l_pde_kink = _weighted_mean_square(residual, gate)
        l_pde_bulk = _weighted_mean_square(residual, 1.0 - gate)
        l_pde = l_pde_kink + l_pde_bulk
    else:
        l_pde_kink = torch.zeros_like(torch.mean(residual**2))
        l_pde_bulk = torch.zeros_like(torch.mean(residual**2))
        l_pde = torch.mean(residual**2)

    u_terminal = model(_apply_input_affine(x=x_terminal, input_affine=input_affine))
    m_terminal = _moneyness_from_input(x_terminal, coordinate_key)
    payoff = torch.clamp(1.0 - m_terminal, min=0.0)
    l_term = torch.mean((u_terminal - payoff) ** 2)

    u_lower = model(_apply_input_affine(x=x_lower, input_affine=input_affine))
    tau_lower = x_lower[:, 0:1]
    r_lower = x_lower[:, 7:8]
    m_lower = _moneyness_from_input(x_lower, coordinate_key)
    lower_target = torch.clamp(torch.exp(-r_lower * tau_lower) - m_lower, min=0.0)
    l_low = torch.mean((u_lower - lower_target) ** 2)

    w_pde = _get_weight(loss_config, "pde", 1.0)
    w_pde_kink = _get_weight(loss_config, "pde_kink", w_pde)
    w_pde_bulk = _get_weight(loss_config, "pde_bulk", w_pde)
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
    w_right = _get_weight(loss_config, "right", _get_weight(loss_config, "upper", 0.0))
    w_v_zero = _get_weight(loss_config, "v_zero", 0.0)
    w_left_delta = _get_weight(loss_config, "left_delta", 0.0)
    w_left_gamma = _get_weight(loss_config, "left_gamma", 0.0)
    w_right_delta = _get_weight(loss_config, "right_delta", 0.0)
    w_right_gamma = _get_weight(loss_config, "right_gamma", 0.0)
    w_curvature_bulk = _get_weight(loss_config, "curvature_bulk", 0.0)
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
    if w_no_arb > 0.0:
        u_noarb, u_s_noarb, _u_v_noarb, u_curv_noarb = _price_derivatives(
            model=model,
            x=x_interior,
            input_affine=input_affine,
            coordinate=coordinate_key,
        )
        tau_noarb = x_interior[:, 0:1]
        m_noarb = _moneyness_from_input(x_interior, coordinate_key)
        r_noarb = x_interior[:, 7:8]
        lower_bound = torch.clamp(torch.exp(-r_noarb * tau_noarb) - m_noarb, min=0.0)
        upper_bound = torch.exp(-r_noarb * tau_noarb)
        noarb_cfg = _nested_dict(loss_config, "no_arbitrage")
        beta_delta = float(noarb_cfg.get("beta_delta", 1.0))
        beta_gamma = float(noarb_cfg.get("beta_gamma", 1.0))
        price_bounds = torch.relu(lower_bound - u_noarb) ** 2 + torch.relu(u_noarb - upper_bound) ** 2
        if coordinate_key == "log_moneyness":
            delta_bounds = torch.relu(u_s_noarb) ** 2 + torch.relu(-m_noarb - u_s_noarb) ** 2
        else:
            delta_bounds = torch.relu(u_s_noarb) ** 2 + torch.relu(-1.0 - u_s_noarb) ** 2
        convexity = torch.relu(-u_curv_noarb) ** 2
        no_arb_term = torch.mean(price_bounds + beta_delta * delta_bounds + beta_gamma * convexity)
    if w_dpde > 0.0:
        grad_residual = _gradient(y=residual, x=x_interior)
        if coordinate_key == "log_moneyness":
            d_residual_dx = grad_residual[:, 1:2]
        else:
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

    l_low_extra = torch.zeros_like(l_pde)
    if w_left_delta > 0.0 or w_left_gamma > 0.0:
        m_left = _moneyness_from_input(x_lower, coordinate_key)
        _u_left, u_s_left, _u_v_left, u_curv_left = _price_derivatives(
            model=model,
            x=x_lower,
            input_affine=input_affine,
            coordinate=coordinate_key,
        )
        if w_left_delta > 0.0:
            left_delta_target = -m_left if coordinate_key == "log_moneyness" else -torch.ones_like(m_left)
            l_low_extra = l_low_extra + w_left_delta * torch.mean((u_s_left - left_delta_target) ** 2)
        if w_left_gamma > 0.0:
            l_low_extra = l_low_extra + w_left_gamma * torch.mean(u_curv_left**2)

    l_right = torch.zeros_like(l_pde)
    l_right_extra = torch.zeros_like(l_pde)
    if "right" in batch_payload and (w_right > 0.0 or w_right_delta > 0.0 or w_right_gamma > 0.0):
        x_right = batch_payload["right"]
        u_right, u_s_right, _u_v_right, u_curv_right = _price_derivatives(
            model=model,
            x=x_right,
            input_affine=input_affine,
            coordinate=coordinate_key,
        )
        if w_right > 0.0:
            l_right = l_right + torch.mean(u_right**2)
        if w_right_delta > 0.0:
            l_right_extra = l_right_extra + w_right_delta * torch.mean(u_s_right**2)
        if w_right_gamma > 0.0:
            l_right_extra = l_right_extra + w_right_gamma * torch.mean(u_curv_right**2)

    l_v_zero = torch.zeros_like(l_pde)
    if "v_zero" in batch_payload and w_v_zero > 0.0:
        l_v_zero = torch.mean(
            compute_heston_v_zero_residual(
                model=model,
                x_v_zero=batch_payload["v_zero"],
                input_affine=input_affine,
                coordinate=coordinate_key,
            )
            ** 2
        )

    l_curvature_bulk = torch.zeros_like(l_pde)
    if w_curvature_bulk > 0.0:
        u_bulk, u_s_bulk, _u_v_bulk, u_curv_bulk = _price_derivatives(
            model=model,
            x=x_interior,
            input_affine=input_affine,
            coordinate=coordinate_key,
        )
        del u_bulk, u_s_bulk, _u_v_bulk
        grad_curv = _gradient(y=u_curv_bulk, x=x_interior)
        curv_x = grad_curv[:, 1:2]
        gate = _kink_gate(x_interior, loss_config, coordinate=coordinate_key)
        l_curvature_bulk = _weighted_mean_square(curv_x, 1.0 - gate)

    if kink_bulk_enabled:
        pde_base = w_pde_kink * l_pde_kink + w_pde_bulk * l_pde_bulk
    else:
        pde_base = w_pde * l_pde
    dpde_base = w_dpde * l_dpde
    term_base = w_term * l_term
    low_base = w_low * l_low + l_low_extra
    right_base = w_right * l_right + l_right_extra
    v_zero_base = w_v_zero * l_v_zero
    no_arb_base = w_no_arb * no_arb_term
    greek_delta_base = w_greek_delta * l_greek_delta
    greek_gamma_base = w_greek_gamma * l_greek_gamma
    greek_vega_base = w_greek_vega * l_greek_vega
    curvature_bulk_base = w_curvature_bulk * l_curvature_bulk

    pde_weighted = _learned_log_var_term(
        name="pde",
        base_loss=pde_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    dpde_weighted = _learned_log_var_term(
        name="dpde",
        base_loss=dpde_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    term_weighted = _learned_log_var_term(
        name="term",
        base_loss=term_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    low_weighted = _learned_log_var_term(
        name="low",
        base_loss=low_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    right_weighted = _learned_log_var_term(
        name="right",
        base_loss=right_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    v_zero_weighted = _learned_log_var_term(
        name="v_zero",
        base_loss=v_zero_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    no_arb_weighted = _learned_log_var_term(
        name="no_arbitrage",
        base_loss=no_arb_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    greek_delta_weighted = _learned_log_var_term(
        name="greek_delta",
        base_loss=greek_delta_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    greek_gamma_weighted = _learned_log_var_term(
        name="greek_gamma",
        base_loss=greek_gamma_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    greek_vega_weighted = _learned_log_var_term(
        name="greek_vega",
        base_loss=greek_vega_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )
    curvature_bulk_weighted = _learned_log_var_term(
        name="curvature_bulk",
        base_loss=curvature_bulk_base,
        loss_config=loss_config,
        learned_log_vars=learned_log_vars,
    )

    total = (
        pde_weighted
        + dpde_weighted
        + term_weighted
        + low_weighted
        + right_weighted
        + v_zero_weighted
        + no_arb_weighted
        + greek_delta_weighted
        + greek_gamma_weighted
        + greek_vega_weighted
        + curvature_bulk_weighted
    )

    terms = PINNLossTerms(
        pde=float(pde_weighted.detach().item()),
        dpde=float(dpde_weighted.detach().item()),
        term=float(term_weighted.detach().item()),
        low=float(low_weighted.detach().item()),
        right=float(right_weighted.detach().item()),
        v_zero=float(v_zero_weighted.detach().item()),
        no_arbitrage=float(no_arb_weighted.detach().item()),
        greek_delta=float(greek_delta_weighted.detach().item()),
        greek_gamma=float(greek_gamma_weighted.detach().item()),
        greek_vega=float(greek_vega_weighted.detach().item()),
        curvature_bulk=float(curvature_bulk_weighted.detach().item()),
    )
    return total, terms
