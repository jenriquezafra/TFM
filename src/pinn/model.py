from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Activation '{name}' is not supported")


def _init_linear_weights(module: nn.Module, *, initialization: str) -> None:
    init = str(initialization).lower()
    for layer in module.modules():
        if not isinstance(layer, nn.Linear):
            continue
        if init == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
        elif init == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight)
        elif init == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight)
        else:
            raise ValueError(f"Initialization '{initialization}' is not supported")
        nn.init.zeros_(layer.bias)


def _build_backbone(architecture_config: dict, *, input_dim: int) -> nn.Sequential:
    hidden_cfg = architecture_config.get("hidden", {})
    output_cfg = architecture_config.get("output", {})

    hidden_dims = hidden_cfg.get("dims", [256, 256, 256, 256])
    output_dim = int(output_cfg.get("dim", 1))
    activation_name = str(hidden_cfg.get("activation", "tanh"))
    dropout_rate = float(hidden_cfg.get("dropout_rate", 0.0))
    initialization = str(hidden_cfg.get("initialization", "xavier_uniform"))
    use_batch_norm = bool(hidden_cfg.get("batch_norm", False))

    dims: list[int] = [input_dim] + [int(dim) for dim in hidden_dims]
    if len(dims) < 2:
        raise ValueError("hidden.dims must include at least one hidden layer")

    act = _get_activation(activation_name)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(act.__class__())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
    layers.append(nn.Linear(dims[-1], output_dim))

    backbone = nn.Sequential(*layers)
    _init_linear_weights(backbone, initialization=initialization)
    return backbone


class PINNPricer(nn.Module):
    """
    Minimal PINN backbone (MLP) for price regression.
    """

    def __init__(self, architecture_config: dict, *, network_input_dim: int | None = None):
        super().__init__()
        self.architecture_config = architecture_config

        input_cfg = architecture_config.get("input", {})
        raw_input_dim = int(input_cfg.get("dim", 8))
        self.raw_input_dim = raw_input_dim
        self.network_input_dim = raw_input_dim if network_input_dim is None else int(network_input_dim)
        self.backbone = _build_backbone(architecture_config, input_dim=self.network_input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        return self.backbone(x)


class MultiOutputGreekPINNPricer(PINNPricer):
    """
    Experimental multi-output PINN for Greek consistency.

    The network emits four raw heads:
      [price, delta_m, gamma_mm, vega_v]

    `forward()` intentionally returns only the price head [N,1] so existing
    PDE losses, adapters, and diagnostics continue to treat the model as a
    scalar pricer. Training losses can call `forward_all()` to access the Greek
    heads and enforce no-label consistency against autodiff derivatives.
    """

    def __init__(self, architecture_config: dict):
        output_dim = int(architecture_config.get("output", {}).get("dim", 4))
        if output_dim != 4:
            raise ValueError(
                "MultiOutputGreekPINNPricer requires output.dim=4 "
                "for [price, delta, gamma, vega] heads."
            )
        super().__init__(architecture_config=architecture_config)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        out = self.backbone(x)
        if out.ndim != 2 or out.shape[1] != 4:
            raise ValueError(f"Expected multi-output shape [N,4], got {tuple(out.shape)}")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[:, 0:1]


class BoundaryLayerPINNPricer(PINNPricer):
    """
    PINN with a boundary-layer input transform only:
      raw [tau,m,v,rho,kappa,gamma,bar_v,r]
      -> [z,log_m,sqrt_tau_eps,log_tau_eps,v,rho,kappa,gamma,bar_v,r]

    The output remains unconstrained: V = N(phi(raw)).
    """

    def __init__(self, architecture_config: dict):
        boundary_cfg = architecture_config.get("boundary_layer", {})
        if not isinstance(boundary_cfg, dict):
            raise ValueError("architecture.boundary_layer must be a dictionary when provided.")
        if not bool(boundary_cfg.get("enabled", False)):
            raise ValueError("BoundaryLayerPINNPricer requires boundary_layer.enabled=true")

        input_dim = int(architecture_config.get("input", {}).get("dim", 8))
        self.boundary_layer_epsilon = float(boundary_cfg.get("epsilon", 1.0e-6))
        if self.boundary_layer_epsilon <= 0.0:
            raise ValueError("boundary_layer.epsilon must be > 0")
        self.log_moneyness_floor = float(boundary_cfg.get("log_moneyness_floor", 1.0e-6))
        if self.log_moneyness_floor <= 0.0:
            raise ValueError("boundary_layer.log_moneyness_floor must be > 0")
        self.boundary_z_clip = boundary_cfg.get("z_clip", 50.0)
        if self.boundary_z_clip is not None:
            self.boundary_z_clip = float(self.boundary_z_clip)
            if self.boundary_z_clip <= 0.0:
                raise ValueError("boundary_layer.z_clip must be > 0 when set")

        # z, log_m, sqrt_tau_eps, log_tau_eps, v, rho, kappa, gamma, bar_v, r
        super().__init__(architecture_config, network_input_dim=10)

        self.register_buffer("input_affine_a", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("input_affine_b", torch.ones(input_dim, dtype=torch.float32))

    def configure_input_affine(self, input_affine: dict | None) -> None:
        if input_affine is None:
            self.input_affine_a.zero_()
            self.input_affine_b.fill_(1.0)
            return

        a = torch.as_tensor(
            input_affine.get("a"),
            dtype=self.input_affine_a.dtype,
            device=self.input_affine_a.device,
        )
        b = torch.as_tensor(
            input_affine.get("b"),
            dtype=self.input_affine_b.dtype,
            device=self.input_affine_b.device,
        )
        if a.numel() != self.input_affine_a.numel() or b.numel() != self.input_affine_b.numel():
            raise ValueError(
                "boundary-layer input affine dimension mismatch: "
                f"expected {self.input_affine_a.numel()}, got a={a.numel()} b={b.numel()}"
            )
        if torch.any(torch.abs(b) < 1.0e-12):
            raise ValueError("boundary-layer input affine contains near-zero scale values.")
        self.input_affine_a.copy_(a.reshape_as(self.input_affine_a))
        self.input_affine_b.copy_(b.reshape_as(self.input_affine_b))

    def _raw_inputs(self, x_net: torch.Tensor) -> torch.Tensor:
        a = self.input_affine_a.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        b = self.input_affine_b.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        return (x_net - a) / b

    def _boundary_layer_features(self, raw: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        moneyness = torch.clamp(raw[:, 1:2], min=self.log_moneyness_floor)
        tau_eps = tau + torch.as_tensor(
            self.boundary_layer_epsilon,
            dtype=raw.dtype,
            device=raw.device,
        )
        log_m = torch.log(moneyness)
        sqrt_tau_eps = torch.sqrt(tau_eps)
        z = log_m / sqrt_tau_eps
        if self.boundary_z_clip is not None:
            z = torch.clamp(z, min=-self.boundary_z_clip, max=self.boundary_z_clip)
        log_tau_eps = torch.log(tau_eps)
        return torch.cat(
            [
                z,
                log_m,
                sqrt_tau_eps,
                log_tau_eps,
                raw[:, 2:3],  # v
                raw[:, 3:4],  # rho
                raw[:, 4:5],  # kappa
                raw[:, 5:6],  # gamma
                raw[:, 6:7],  # bar_v
                raw[:, 7:8],  # r
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        raw = self._raw_inputs(x)
        return self.backbone(self._boundary_layer_features(raw))


class FeatureMapPINNPricer(PINNPricer):
    """
    PINN with explicit feature maps for ablation studies.

    The trainer stores and batches physical coordinates. By default the second
    coordinate is moneyness m, but `feature_map.input_coordinate=log_moneyness`
    makes it x=log(m). This module recovers those raw coordinates from the
    input affine transform and feeds either log-moneyness or kink-adapted
    features to the backbone. Autograd therefore still differentiates with
    respect to the physical PDE coordinates, not detached engineered features.
    """

    def __init__(self, architecture_config: dict):
        fmap_cfg = architecture_config.get("feature_map", {})
        if not isinstance(fmap_cfg, dict):
            raise ValueError("architecture.feature_map must be a dictionary when provided.")
        if not bool(fmap_cfg.get("enabled", False)):
            raise ValueError("FeatureMapPINNPricer requires feature_map.enabled=true")

        input_dim = int(architecture_config.get("input", {}).get("dim", 8))
        if input_dim < 8:
            raise ValueError("FeatureMapPINNPricer expects at least 8 physical inputs.")

        self.feature_map_mode = str(fmap_cfg.get("mode", "log_moneyness")).strip().lower()
        if self.feature_map_mode not in {"log_moneyness", "kink_adapted"}:
            raise ValueError("feature_map.mode must be 'log_moneyness' or 'kink_adapted'.")

        self.input_coordinate = str(
            fmap_cfg.get("input_coordinate", fmap_cfg.get("coordinate_space", "moneyness"))
        ).strip().lower()
        if self.input_coordinate not in {"moneyness", "m", "log_moneyness", "log-moneyness", "x"}:
            raise ValueError("feature_map.input_coordinate must be 'moneyness' or 'log_moneyness'.")

        self.moneyness_floor = float(fmap_cfg.get("moneyness_floor", 1.0e-6))
        if self.moneyness_floor <= 0.0:
            raise ValueError("feature_map.moneyness_floor must be > 0.")

        self.tau_epsilon = float(fmap_cfg.get("tau_epsilon", 1.0e-6))
        if self.tau_epsilon <= 0.0:
            raise ValueError("feature_map.tau_epsilon must be > 0.")

        self.q_epsilon = float(fmap_cfg.get("q_epsilon", 1.0e-8))
        if self.q_epsilon <= 0.0:
            raise ValueError("feature_map.q_epsilon must be > 0.")

        self.q_clip = float(fmap_cfg.get("q_clip", 10.0))
        if self.q_clip <= 0.0:
            raise ValueError("feature_map.q_clip must be > 0.")

        # log_moneyness: [tau, x, v, rho, kappa, gamma, bar_v, r]
        # kink_adapted:  [tau, x, sqrt(tau+eps), q_clip, v, rho, kappa, gamma, bar_v, r]
        network_input_dim = 8 if self.feature_map_mode == "log_moneyness" else 10
        super().__init__(architecture_config, network_input_dim=network_input_dim)

        self.register_buffer("input_affine_a", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("input_affine_b", torch.ones(input_dim, dtype=torch.float32))

    def configure_input_affine(self, input_affine: dict | None) -> None:
        if input_affine is None:
            self.input_affine_a.zero_()
            self.input_affine_b.fill_(1.0)
            return

        a = torch.as_tensor(
            input_affine.get("a"),
            dtype=self.input_affine_a.dtype,
            device=self.input_affine_a.device,
        )
        b = torch.as_tensor(
            input_affine.get("b"),
            dtype=self.input_affine_b.dtype,
            device=self.input_affine_b.device,
        )
        if a.numel() != self.input_affine_a.numel() or b.numel() != self.input_affine_b.numel():
            raise ValueError(
                "feature-map input affine dimension mismatch: "
                f"expected {self.input_affine_a.numel()}, got a={a.numel()} b={b.numel()}"
            )
        if torch.any(torch.abs(b) < 1.0e-12):
            raise ValueError("feature-map input affine contains near-zero scale values.")
        self.input_affine_a.copy_(a.reshape_as(self.input_affine_a))
        self.input_affine_b.copy_(b.reshape_as(self.input_affine_b))

    def _raw_inputs(self, x_net: torch.Tensor) -> torch.Tensor:
        a = self.input_affine_a.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        b = self.input_affine_b.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        return (x_net - a) / b

    def _feature_map(self, raw: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        if self.input_coordinate in {"log_moneyness", "log-moneyness", "x"}:
            x = raw[:, 1:2]
            m = torch.exp(x)
        else:
            m = torch.clamp(raw[:, 1:2], min=self.moneyness_floor)
            x = torch.log(m)
        v = torch.clamp(raw[:, 2:3], min=0.0)

        if self.feature_map_mode == "log_moneyness":
            return torch.cat([tau, x, raw[:, 2:8]], dim=1)

        tau_eps = tau + torch.as_tensor(self.tau_epsilon, dtype=raw.dtype, device=raw.device)
        sqrt_tau_eps = torch.sqrt(tau_eps)
        q_denom = torch.sqrt(v * tau + torch.as_tensor(self.q_epsilon, dtype=raw.dtype, device=raw.device))
        q = x / q_denom
        q_clip = self.q_clip * torch.tanh(q / self.q_clip)
        return torch.cat(
            [
                tau,
                x,
                sqrt_tau_eps,
                q_clip,
                raw[:, 2:3],  # v
                raw[:, 3:4],  # rho
                raw[:, 4:5],  # kappa
                raw[:, 5:6],  # gamma
                raw[:, 6:7],  # bar_v
                raw[:, 7:8],  # r
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        raw = self._raw_inputs(x)
        return self.backbone(self._feature_map(raw))


class BoundedPricePINNPricer(FeatureMapPINNPricer):
    """
    Pricing-only PINN with kink-adapted inputs and hard no-arbitrage price bounds.

    For a normalized put with m=S/K, the model outputs
      L(tau,m,r) + (U(tau,r)-L(tau,m,r)) y(tau,m,...),
    where y is constrained to [0,1]. This enforces non-negativity and the
    standard put upper/lower price bounds by construction, while a temporal
    gate anchors tau=0 to the payoff.
    """

    def __init__(self, architecture_config: dict):
        bounded_cfg = architecture_config.get("bounded_price", {})
        if not isinstance(bounded_cfg, dict):
            raise ValueError("architecture.bounded_price must be a dictionary when provided.")
        if not bool(bounded_cfg.get("enabled", False)):
            raise ValueError("BoundedPricePINNPricer requires bounded_price.enabled=true")

        super().__init__(architecture_config=architecture_config)

        input_dim = int(architecture_config.get("input", {}).get("dim", 8))
        self.tau_index = int(bounded_cfg.get("tau_index", 0))
        self.spot_index = int(bounded_cfg.get("spot_index", 1))
        self.r_index = int(bounded_cfg.get("r_index", 7))
        for name, value in (
            ("bounded_price.tau_index", self.tau_index),
            ("bounded_price.spot_index", self.spot_index),
            ("bounded_price.r_index", self.r_index),
        ):
            if not (0 <= value < input_dim):
                raise ValueError(f"{name} must be in [0,{input_dim - 1}]")

        self.option_type = str(bounded_cfg.get("option_type", "put")).strip().lower()
        if self.option_type not in {"put", "call"}:
            raise ValueError("bounded_price.option_type must be one of {'put', 'call'}")

        self.strike = float(bounded_cfg.get("strike", 1.0))
        if self.strike <= 0.0:
            raise ValueError("bounded_price.strike must be > 0")

        self.sigmoid_temperature = float(bounded_cfg.get("sigmoid_temperature", 2.0))
        if self.sigmoid_temperature <= 0.0:
            raise ValueError("bounded_price.sigmoid_temperature must be > 0")

        self.payoff_smoothing = float(bounded_cfg.get("payoff_smoothing", 1.0e-4))
        if self.payoff_smoothing < 0.0:
            raise ValueError("bounded_price.payoff_smoothing must be >= 0")

        self.lower_smoothing = float(bounded_cfg.get("lower_smoothing", self.payoff_smoothing))
        if self.lower_smoothing < 0.0:
            raise ValueError("bounded_price.lower_smoothing must be >= 0")

        self.min_width = float(bounded_cfg.get("min_width", 1.0e-10))
        if self.min_width <= 0.0:
            raise ValueError("bounded_price.min_width must be > 0")

        self.time_gate = str(bounded_cfg.get("time_gate", "rational_tau")).strip().lower()
        if self.time_gate not in {"rational_tau", "one_minus_exp"}:
            raise ValueError("bounded_price.time_gate must be 'rational_tau' or 'one_minus_exp'")

        self.time_gate_tau = float(bounded_cfg.get("time_gate_tau", 2.0e-2))
        if self.time_gate_tau <= 0.0:
            raise ValueError("bounded_price.time_gate_tau must be > 0")

        right_cfg = bounded_cfg.get("right_boundary", {})
        if not isinstance(right_cfg, dict):
            raise ValueError("bounded_price.right_boundary must be a dictionary when provided.")
        self.right_boundary_enabled = bool(right_cfg.get("enabled", False))
        self.right_boundary_moneyness = float(
            right_cfg.get("moneyness", right_cfg.get("right_moneyness", 3.0))
        )
        if self.right_boundary_enabled and self.right_boundary_moneyness <= 0.0:
            raise ValueError("bounded_price.right_boundary.moneyness must be > 0")
        self.right_boundary_start = float(
            right_cfg.get("start_moneyness", max(0.0, self.right_boundary_moneyness - 1.0))
        )
        if self.right_boundary_enabled and self.right_boundary_start >= self.right_boundary_moneyness:
            raise ValueError(
                "bounded_price.right_boundary.start_moneyness must be smaller than moneyness"
            )
        self.right_boundary_power = float(right_cfg.get("power", 2.0))
        if self.right_boundary_enabled and self.right_boundary_power <= 0.0:
            raise ValueError("bounded_price.right_boundary.power must be > 0")
        self.right_boundary_mode = str(right_cfg.get("mode", "smoothstep")).strip().lower()
        if self.right_boundary_mode not in {"linear", "smoothstep"}:
            raise ValueError("bounded_price.right_boundary.mode must be 'linear' or 'smoothstep'")

    def _positive_part(self, signed: torch.Tensor, smoothing: float) -> torch.Tensor:
        if smoothing <= 0.0:
            return torch.clamp(signed, min=0.0)
        eps = torch.as_tensor(smoothing, dtype=signed.dtype, device=signed.device)
        eps = torch.clamp(eps, min=torch.finfo(signed.dtype).eps)
        return eps * F.softplus(signed / eps)

    def _time_gate(self, tau: torch.Tensor) -> torch.Tensor:
        tau_nonnegative = torch.clamp(tau, min=0.0)
        tau0 = torch.as_tensor(self.time_gate_tau, dtype=tau.dtype, device=tau.device)
        if self.time_gate == "one_minus_exp":
            return 1.0 - torch.exp(-tau_nonnegative / tau0)
        return tau_nonnegative / (tau_nonnegative + tau0)

    def _moneyness_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        spot = raw[:, self.spot_index : self.spot_index + 1]
        if self.input_coordinate in {"log_moneyness", "log-moneyness", "x"}:
            return torch.exp(spot)
        return torch.clamp(spot, min=0.0)

    def _right_boundary_gate(self, moneyness: torch.Tensor) -> torch.Tensor:
        if not self.right_boundary_enabled:
            return torch.ones_like(moneyness)
        start = torch.as_tensor(
            self.right_boundary_start,
            dtype=moneyness.dtype,
            device=moneyness.device,
        )
        right = torch.as_tensor(
            self.right_boundary_moneyness,
            dtype=moneyness.dtype,
            device=moneyness.device,
        )
        width = torch.clamp(right - start, min=torch.finfo(moneyness.dtype).eps)
        t = torch.clamp((right - moneyness) / width, min=0.0, max=1.0)
        if self.right_boundary_mode == "smoothstep":
            t = t * t * (3.0 - 2.0 * t)
        return t**self.right_boundary_power

    def _bounds_and_payoff(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = torch.clamp(raw[:, self.tau_index : self.tau_index + 1], min=0.0)
        r = raw[:, self.r_index : self.r_index + 1]
        m = self._moneyness_from_raw(raw)
        spot = self.strike * m
        discounted_strike = self.strike * torch.exp(-r * tau)

        if self.option_type == "call":
            lower_signed = spot - discounted_strike
            payoff_signed = spot - self.strike
            upper = spot
        else:
            lower_signed = discounted_strike - spot
            payoff_signed = self.strike - spot
            upper = discounted_strike

        lower = self._positive_part(lower_signed, self.lower_smoothing)
        payoff = self._positive_part(payoff_signed, self.payoff_smoothing)
        lower = torch.minimum(lower, upper)
        return lower, upper, payoff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        raw = self._raw_inputs(x)
        raw_price = self.backbone(self._feature_map(raw))
        if raw_price.ndim != 2 or raw_price.shape[1] != 1:
            raise ValueError(f"Expected backbone output shape [N,1], got {tuple(raw_price.shape)}")

        lower, upper, payoff = self._bounds_and_payoff(raw)
        width = torch.clamp(upper - lower, min=0.0)
        width_safe = torch.clamp(
            width,
            min=torch.as_tensor(self.min_width, dtype=raw.dtype, device=raw.device),
        )

        y_terminal = torch.clamp((payoff - lower) / width_safe, min=0.0, max=1.0)
        y_model = torch.sigmoid(raw_price / self.sigmoid_temperature)
        tau = raw[:, self.tau_index : self.tau_index + 1]
        gate = self._time_gate(tau)
        right_gate = self._right_boundary_gate(self._moneyness_from_raw(raw))
        y = right_gate * ((1.0 - gate) * y_terminal + gate * y_model)
        return lower + width * y


class PayoffAwarePINNPricer(PINNPricer):
    """
    PINN ansatz:
      V(tau, m, ...) = g_eps(m) + a(tau) * N(tau, m, ...)

    The forward input remains the scaled network input used by the existing
    trainer. The raw financial coordinates are recovered through affine
    buffers configured after training-set scaling is computed.

    Experimental mode:
      payoff_smoothing_mode='adaptive_sqrt_v_tau' replaces the fixed softplus
      width by epsilon(tau,v)=c*sqrt(v*tau+eps0), optionally clipped. This is a
      research variant for short-maturity Greek diagnostics, not the default
      production payoff-aware formulation.
    """

    def __init__(self, architecture_config: dict):
        ansatz_cfg = architecture_config.get("payoff_aware", {})
        if not isinstance(ansatz_cfg, dict):
            raise ValueError("architecture.payoff_aware must be a dictionary when provided.")

        input_dim = int(architecture_config.get("input", {}).get("dim", 8))
        boundary_cfg = ansatz_cfg.get("boundary_layer", {})
        if not isinstance(boundary_cfg, dict):
            raise ValueError("payoff_aware.boundary_layer must be a dictionary when provided.")
        self.boundary_layer_enabled = bool(boundary_cfg.get("enabled", False))
        self.boundary_layer_epsilon = float(boundary_cfg.get("epsilon", 1.0e-6))
        if self.boundary_layer_epsilon <= 0.0:
            raise ValueError("payoff_aware.boundary_layer.epsilon must be > 0")
        self.log_moneyness_floor = float(boundary_cfg.get("log_moneyness_floor", 1.0e-6))
        if self.log_moneyness_floor <= 0.0:
            raise ValueError("payoff_aware.boundary_layer.log_moneyness_floor must be > 0")
        self.boundary_z_clip = boundary_cfg.get("z_clip", 50.0)
        if self.boundary_z_clip is not None:
            self.boundary_z_clip = float(self.boundary_z_clip)
            if self.boundary_z_clip <= 0.0:
                raise ValueError("payoff_aware.boundary_layer.z_clip must be > 0 when set")

        # z, log_m, sqrt_tau_eps, log_tau_eps, v, rho, kappa, gamma, bar_v, r
        network_input_dim = 10 if self.boundary_layer_enabled else input_dim
        super().__init__(architecture_config, network_input_dim=network_input_dim)

        self.tau_index = int(ansatz_cfg.get("tau_index", 0))
        self.spot_index = int(ansatz_cfg.get("spot_index", 1))
        if not (0 <= self.tau_index < input_dim):
            raise ValueError(f"payoff_aware.tau_index must be in [0,{input_dim - 1}]")
        if not (0 <= self.spot_index < input_dim):
            raise ValueError(f"payoff_aware.spot_index must be in [0,{input_dim - 1}]")

        self.option_type = str(ansatz_cfg.get("option_type", "put")).strip().lower()
        if self.option_type not in {"put", "call"}:
            raise ValueError("payoff_aware.option_type must be one of {'put', 'call'}")

        self.strike = float(ansatz_cfg.get("strike", 1.0))
        if self.strike <= 0.0:
            raise ValueError("payoff_aware.strike must be > 0")

        self.payoff_smoothing = float(ansatz_cfg.get("payoff_smoothing", 1.0e-3))
        if self.payoff_smoothing < 0.0:
            raise ValueError("payoff_aware.payoff_smoothing must be >= 0")

        self.payoff_smoothing_mode = (
            str(ansatz_cfg.get("payoff_smoothing_mode", "fixed")).strip().lower()
        )
        if self.payoff_smoothing_mode not in {"fixed", "adaptive_sqrt_v_tau"}:
            raise ValueError(
                "payoff_aware.payoff_smoothing_mode must be 'fixed' or "
                "'adaptive_sqrt_v_tau'"
            )
        self.vol_index = int(ansatz_cfg.get("vol_index", 2))
        if not (0 <= self.vol_index < input_dim):
            raise ValueError(f"payoff_aware.vol_index must be in [0,{input_dim - 1}]")
        self.payoff_smoothing_scale = float(ansatz_cfg.get("payoff_smoothing_scale", 1.0))
        if self.payoff_smoothing_scale < 0.0:
            raise ValueError("payoff_aware.payoff_smoothing_scale must be >= 0")
        self.payoff_smoothing_epsilon = float(ansatz_cfg.get("payoff_smoothing_epsilon", 1.0e-10))
        if self.payoff_smoothing_epsilon <= 0.0:
            raise ValueError("payoff_aware.payoff_smoothing_epsilon must be > 0")
        self.payoff_smoothing_min = float(ansatz_cfg.get("payoff_smoothing_min", 0.0))
        if self.payoff_smoothing_min < 0.0:
            raise ValueError("payoff_aware.payoff_smoothing_min must be >= 0")
        max_raw = ansatz_cfg.get("payoff_smoothing_max")
        self.payoff_smoothing_max = None if max_raw is None else float(max_raw)
        if self.payoff_smoothing_max is not None and self.payoff_smoothing_max <= 0.0:
            raise ValueError("payoff_aware.payoff_smoothing_max must be > 0 when set")
        if (
            self.payoff_smoothing_max is not None
            and self.payoff_smoothing_min > self.payoff_smoothing_max
        ):
            raise ValueError("payoff_aware.payoff_smoothing_min must be <= payoff_smoothing_max")

        self.time_factor = str(ansatz_cfg.get("time_factor", "linear_tau")).strip().lower()
        if self.time_factor not in {"linear_tau", "sqrt_tau_eps"}:
            raise ValueError("payoff_aware.time_factor must be 'linear_tau' or 'sqrt_tau_eps'")

        self.time_epsilon = float(ansatz_cfg.get("time_epsilon", 1.0e-6))
        if self.time_epsilon <= 0.0:
            raise ValueError("payoff_aware.time_epsilon must be > 0")

        self.register_buffer("input_affine_a", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("input_affine_b", torch.ones(input_dim, dtype=torch.float32))

    def configure_input_affine(self, input_affine: dict | None) -> None:
        if input_affine is None:
            self.input_affine_a.zero_()
            self.input_affine_b.fill_(1.0)
            return

        a = torch.as_tensor(
            input_affine.get("a"),
            dtype=self.input_affine_a.dtype,
            device=self.input_affine_a.device,
        )
        b = torch.as_tensor(
            input_affine.get("b"),
            dtype=self.input_affine_b.dtype,
            device=self.input_affine_b.device,
        )
        if a.numel() != self.input_affine_a.numel() or b.numel() != self.input_affine_b.numel():
            raise ValueError(
                "payoff-aware input affine dimension mismatch: "
                f"expected {self.input_affine_a.numel()}, got a={a.numel()} b={b.numel()}"
            )
        if torch.any(torch.abs(b) < 1.0e-12):
            raise ValueError("payoff-aware input affine contains near-zero scale values.")
        self.input_affine_a.copy_(a.reshape_as(self.input_affine_a))
        self.input_affine_b.copy_(b.reshape_as(self.input_affine_b))

    def _raw_inputs(self, x_net: torch.Tensor) -> torch.Tensor:
        a = self.input_affine_a.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        b = self.input_affine_b.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        return (x_net - a) / b

    def _payoff_smoothing_width(
        self,
        *,
        tau: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        if self.payoff_smoothing_mode == "fixed":
            return torch.as_tensor(
                self.payoff_smoothing,
                dtype=tau.dtype,
                device=tau.device,
            )

        tau_nonnegative = torch.clamp(tau, min=0.0)
        variance_nonnegative = torch.clamp(variance, min=0.0)
        eps0 = torch.as_tensor(
            self.payoff_smoothing_epsilon,
            dtype=tau.dtype,
            device=tau.device,
        )
        width = self.payoff_smoothing_scale * torch.sqrt(
            variance_nonnegative * tau_nonnegative + eps0
        )
        if self.payoff_smoothing_min > 0.0:
            width = torch.clamp(width, min=self.payoff_smoothing_min)
        if self.payoff_smoothing_max is not None:
            width = torch.clamp(width, max=self.payoff_smoothing_max)
        return width

    def _smooth_payoff(
        self,
        *,
        tau: torch.Tensor,
        moneyness: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        s = self.strike * moneyness
        if self.payoff_smoothing_mode == "fixed" and self.payoff_smoothing <= 0.0:
            if self.option_type == "call":
                return torch.clamp(s - self.strike, min=0.0)
            return torch.clamp(self.strike - s, min=0.0)

        eps = self._payoff_smoothing_width(tau=tau, variance=variance)
        eps = torch.clamp(eps, min=torch.finfo(s.dtype).eps)
        signed = (s - self.strike) if self.option_type == "call" else (self.strike - s)
        return eps * F.softplus(signed / eps)

    def _time_multiplier(self, tau: torch.Tensor) -> torch.Tensor:
        tau_nonnegative = torch.clamp(tau, min=0.0)
        if self.time_factor == "linear_tau":
            return tau_nonnegative
        eps = torch.as_tensor(self.time_epsilon, dtype=tau.dtype, device=tau.device)
        return torch.sqrt(tau_nonnegative + eps) - torch.sqrt(eps)

    def _boundary_layer_features(self, raw: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        moneyness = torch.clamp(raw[:, 1:2], min=self.log_moneyness_floor)
        tau_eps = tau + torch.as_tensor(
            self.boundary_layer_epsilon,
            dtype=raw.dtype,
            device=raw.device,
        )
        log_m = torch.log(moneyness)
        sqrt_tau_eps = torch.sqrt(tau_eps)
        z = log_m / sqrt_tau_eps
        if self.boundary_z_clip is not None:
            z = torch.clamp(z, min=-self.boundary_z_clip, max=self.boundary_z_clip)
        log_tau_eps = torch.log(tau_eps)
        return torch.cat(
            [
                z,
                log_m,
                sqrt_tau_eps,
                log_tau_eps,
                raw[:, 2:3],  # v
                raw[:, 3:4],  # rho
                raw[:, 4:5],  # kappa
                raw[:, 5:6],  # gamma
                raw[:, 6:7],  # bar_v
                raw[:, 7:8],  # r
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        raw = self._raw_inputs(x)
        tau = raw[:, self.tau_index : self.tau_index + 1]
        moneyness = raw[:, self.spot_index : self.spot_index + 1]
        variance = raw[:, self.vol_index : self.vol_index + 1]
        network_input = self._boundary_layer_features(raw) if self.boundary_layer_enabled else x
        payoff = self._smooth_payoff(
            tau=tau,
            moneyness=moneyness,
            variance=variance,
        )
        return payoff + self._time_multiplier(tau) * self.backbone(network_input)


def build_pinn_model(architecture_config: dict) -> PINNPricer:
    """
    Helper used by trainer/pipeline to instantiate the model.
    """
    global_acv_cfg = architecture_config.get("global_acv", {})
    if isinstance(global_acv_cfg, dict) and bool(global_acv_cfg.get("enabled", False)):
        from src.pinn.global_acv_pinn import GlobalACVResidualPINN

        return GlobalACVResidualPINN(architecture_config=architecture_config)
    greek_cfg = architecture_config.get("greek_consistency", {})
    if isinstance(greek_cfg, dict) and bool(greek_cfg.get("enabled", False)):
        return MultiOutputGreekPINNPricer(architecture_config=architecture_config)
    bounded_cfg = architecture_config.get("bounded_price", {})
    if isinstance(bounded_cfg, dict) and bool(bounded_cfg.get("enabled", False)):
        return BoundedPricePINNPricer(architecture_config=architecture_config)
    ansatz_cfg = architecture_config.get("payoff_aware", {})
    if isinstance(ansatz_cfg, dict) and bool(ansatz_cfg.get("enabled", False)):
        return PayoffAwarePINNPricer(architecture_config=architecture_config)
    feature_map_cfg = architecture_config.get("feature_map", {})
    if isinstance(feature_map_cfg, dict) and bool(feature_map_cfg.get("enabled", False)):
        return FeatureMapPINNPricer(architecture_config=architecture_config)
    boundary_cfg = architecture_config.get("boundary_layer", {})
    if isinstance(boundary_cfg, dict) and bool(boundary_cfg.get("enabled", False)):
        return BoundaryLayerPINNPricer(architecture_config=architecture_config)
    return PINNPricer(architecture_config=architecture_config)
