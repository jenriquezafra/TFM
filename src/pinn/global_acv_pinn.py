from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor


LOG_FEATURE_ORDER = ["tau", "log_moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]


def _normal_cdf(x: Tensor) -> Tensor:
    two = torch.as_tensor(2.0, dtype=x.dtype, device=x.device)
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(two)))


def _normal_pdf(x: Tensor) -> Tensor:
    two_pi = torch.as_tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device)
    return torch.exp(-0.5 * x**2) / torch.sqrt(two_pi)


def _activation_factory(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key == "tanh":
        return nn.Tanh
    if key == "silu":
        return nn.SiLU
    if key == "gelu":
        return nn.GELU
    if key == "elu":
        return nn.ELU
    if key == "relu":
        return nn.ReLU
    raise ValueError("activation must be one of {'tanh', 'silu', 'gelu', 'elu', 'relu'}")


def _build_residual_mlp(
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
    initialization: str,
    final_init_scale: float,
) -> nn.Sequential:
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer.")
    if output_dim != 4:
        raise ValueError("GlobalACVResidualPINN requires output_dim=4.")

    act_cls = _activation_factory(activation)
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
    layers: list[nn.Module] = []
    init_key = str(initialization).strip().lower()

    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        linear = nn.Linear(in_dim, out_dim)
        if init_key == "xavier_uniform":
            nn.init.xavier_uniform_(linear.weight)
        elif init_key == "xavier_normal":
            nn.init.xavier_normal_(linear.weight)
        elif init_key == "kaiming_uniform":
            nn.init.kaiming_uniform_(linear.weight)
        elif init_key == "kaiming_normal":
            nn.init.kaiming_normal_(linear.weight)
        else:
            raise ValueError(f"Initialization '{initialization}' is not supported.")
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(act_cls())

    final = nn.Linear(dims[-2], dims[-1])
    if float(final_init_scale) == 0.0:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    else:
        nn.init.xavier_uniform_(final.weight)
        final.weight.data.mul_(float(final_init_scale))
        nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


def head_greeks_x_to_financial(
    *,
    u_x: Tensor,
    u_xx: Tensor,
    x: Tensor,
    strike: float = 1.0,
) -> tuple[Tensor, Tensor]:
    if strike <= 0.0:
        raise ValueError("strike must be > 0.")
    x_t = torch.as_tensor(x, dtype=u_x.dtype, device=u_x.device)
    delta = torch.exp(-x_t) * u_x / float(strike)
    gamma = torch.exp(-2.0 * x_t) * (u_xx - u_x) / (float(strike) ** 2)
    return delta, gamma


class GlobalACVResidualPINN(nn.Module):
    """
    Global ACV-PINN residual ansatz for log-moneyness inputs.

    The trainable residual network emits [R, R_x, R_xx, R_v]. The public
    forward remains scalar-price only, while forward_all exposes [U, U_x,
    U_xx, U_v] so the existing no-label Greek consistency loss can be reused.
    """

    def __init__(self, architecture_config: dict) -> None:
        super().__init__()
        self.architecture_config = dict(architecture_config)

        input_cfg = architecture_config.get("input", {})
        input_dim = int(input_cfg.get("dim", 8))
        if input_dim != 8:
            raise ValueError("GlobalACVResidualPINN expects input.dim=8.")
        self.raw_input_dim = input_dim
        self.network_input_dim = input_dim

        cfg = architecture_config.get("global_acv", {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            raise ValueError("GlobalACVResidualPINN requires global_acv.enabled=true.")

        feature_order = input_cfg.get("features", LOG_FEATURE_ORDER)
        if list(feature_order) != LOG_FEATURE_ORDER:
            raise ValueError(
                "GlobalACVResidualPINN expects input.features="
                f"{LOG_FEATURE_ORDER}, got {feature_order}."
            )

        self.option_type = str(cfg.get("option_type", "put")).strip().lower()
        if self.option_type != "put":
            raise ValueError("GlobalACVResidualPINN currently supports only put options.")
        self.strike = float(cfg.get("strike", 1.0))
        if self.strike <= 0.0:
            raise ValueError("global_acv.strike must be > 0.")

        self.q_epsilon = float(cfg.get("q_epsilon", 1.0e-8))
        self.bs_epsilon = float(cfg.get("bs_epsilon", 1.0e-12))
        self.terminal_tau = float(cfg.get("terminal_tau", 0.0))
        self.z_clip = cfg.get("z_clip", 50.0)
        self.z_clip = None if self.z_clip is None else float(self.z_clip)
        if self.q_epsilon <= 0.0 or self.bs_epsilon <= 0.0:
            raise ValueError("global_acv.q_epsilon and global_acv.bs_epsilon must be > 0.")
        if self.z_clip is not None and self.z_clip <= 0.0:
            raise ValueError("global_acv.z_clip must be > 0 when provided.")

        self.time_factor = str(cfg.get("time_factor", "linear_tau")).strip().lower()
        if self.time_factor not in {"linear_tau", "one_minus_exp"}:
            raise ValueError("global_acv.time_factor must be 'linear_tau' or 'one_minus_exp'.")
        self.time_lambda = float(cfg.get("time_lambda", 1.0))
        if self.time_lambda <= 0.0:
            raise ValueError("global_acv.time_lambda must be > 0.")

        fourier_cfg = cfg.get("fourier", {})
        fourier_cfg = fourier_cfg if isinstance(fourier_cfg, dict) else {}
        self.fourier_frequencies = int(fourier_cfg.get("frequencies", cfg.get("fourier_frequencies", 6)))
        if self.fourier_frequencies < 0:
            raise ValueError("global_acv.fourier.frequencies must be >= 0.")

        hidden_cfg = architecture_config.get("hidden", {})
        output_cfg = architecture_config.get("output", {})
        output_dim = int(output_cfg.get("dim", 4))
        hidden_dims = hidden_cfg.get("dims", [256, 256, 256, 256])
        activation = str(hidden_cfg.get("activation", "tanh"))
        initialization = str(hidden_cfg.get("initialization", "xavier_uniform"))
        final_init_scale = float(cfg.get("final_init_scale", 0.0))

        # [tau, x, z, q, v, rho, kappa, gamma, bar_v, r] plus sin/cos(z).
        base_feature_dim = 10
        residual_input_dim = base_feature_dim + 2 * self.fourier_frequencies
        self.residual_net = _build_residual_mlp(
            input_dim=residual_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            initialization=initialization,
            final_init_scale=final_init_scale,
        )

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
                "global-acv input affine dimension mismatch: "
                f"expected {self.input_affine_a.numel()}, got a={a.numel()} b={b.numel()}."
            )
        if torch.any(torch.abs(b) < 1.0e-12):
            raise ValueError("global-acv input affine contains near-zero scale values.")
        self.input_affine_a.copy_(a.reshape_as(self.input_affine_a))
        self.input_affine_b.copy_(b.reshape_as(self.input_affine_b))

    def _raw_inputs(self, x_net: Tensor) -> Tensor:
        a = self.input_affine_a.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        b = self.input_affine_b.to(dtype=x_net.dtype, device=x_net.device).view(1, -1)
        return (x_net - a) / b

    def _time_multiplier(self, tau: Tensor) -> Tensor:
        tau_nonnegative = torch.clamp(tau, min=0.0)
        if self.time_factor == "linear_tau":
            return tau_nonnegative
        lam = torch.as_tensor(self.time_lambda, dtype=tau.dtype, device=tau.device)
        return 1.0 - torch.exp(-lam * tau_nonnegative)

    def local_bs_put(self, raw: Tensor) -> Tensor:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        x = raw[:, 1:2]
        m = torch.exp(x)
        v = torch.clamp(raw[:, 2:3], min=0.0)
        r = raw[:, 7:8]

        sigma_tau = torch.sqrt(v * tau + torch.as_tensor(self.bs_epsilon, dtype=raw.dtype, device=raw.device))
        d1 = (x + (r + 0.5 * v) * tau) / sigma_tau
        d2 = d1 - sigma_tau
        put = torch.exp(-r * tau) * _normal_cdf(-d2) - m * _normal_cdf(-d1)
        payoff = torch.clamp(1.0 - m, min=0.0)
        if self.terminal_tau <= 0.0:
            return torch.where(tau <= 0.0, payoff, put)
        return torch.where(tau <= self.terminal_tau, payoff, put)

    def _local_bs_derivatives(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        x = raw[:, 1:2]
        m = torch.exp(x)
        v = torch.clamp(raw[:, 2:3], min=0.0)
        r = raw[:, 7:8]

        y = torch.sqrt(v * tau + torch.as_tensor(self.bs_epsilon, dtype=raw.dtype, device=raw.device))
        a = x + (r + 0.5 * v) * tau
        d1 = a / y
        d2 = d1 - y
        n1 = _normal_pdf(d1)
        n2 = _normal_pdf(d2)
        n_big_1 = _normal_cdf(-d1)
        disc = torch.exp(-r * tau)

        inv_y = 1.0 / y
        inv_y2 = inv_y**2
        bs_x = -disc * n2 * inv_y - m * n_big_1 + m * n1 * inv_y
        bs_xx = disc * d2 * n2 * inv_y2 - m * n_big_1 + 2.0 * m * n1 * inv_y - m * d1 * n1 * inv_y2

        dy_dv = 0.5 * tau * inv_y
        d1_dv = 0.5 * tau * inv_y - a * 0.5 * tau * (inv_y**3)
        d2_dv = d1_dv - dy_dv
        bs_v = -disc * n2 * d2_dv + m * n1 * d1_dv

        return bs_x, bs_xx, bs_v

    def residual_features(self, raw: Tensor) -> Tensor:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        x = raw[:, 1:2]
        v = torch.clamp(raw[:, 2:3], min=0.0)
        q = torch.sqrt(v * tau + torch.as_tensor(self.q_epsilon, dtype=raw.dtype, device=raw.device))
        z = x / q
        if self.z_clip is not None:
            z = torch.clamp(z, min=-self.z_clip, max=self.z_clip)
        return torch.cat(
            [
                tau,
                x,
                z,
                q,
                raw[:, 2:3],
                raw[:, 3:4],
                raw[:, 4:5],
                raw[:, 5:6],
                raw[:, 6:7],
                raw[:, 7:8],
            ],
            dim=1,
        )

    def _augment_features(self, features: Tensor) -> Tensor:
        if self.fourier_frequencies == 0:
            return features
        z = features[:, 2:3]
        freqs = torch.pow(
            torch.as_tensor(2.0, dtype=features.dtype, device=features.device),
            torch.arange(self.fourier_frequencies, dtype=features.dtype, device=features.device),
        ).view(1, -1)
        angles = z * freqs
        return torch.cat([features, torch.sin(angles), torch.cos(angles)], dim=1)

    def forward_residual_all(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}.")
        raw = self._raw_inputs(x)
        out = self.residual_net(self._augment_features(self.residual_features(raw)))
        if out.ndim != 2 or out.shape[1] != 4:
            raise ValueError(f"Expected residual heads shape [N,4], got {tuple(out.shape)}.")
        return out

    def forward_all(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}.")
        raw = self._raw_inputs(x)
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        residual_heads = self.residual_net(self._augment_features(self.residual_features(raw)))
        r_head = residual_heads[:, 0:1]
        rx_head = residual_heads[:, 1:2]
        rxx_head = residual_heads[:, 2:3]
        rv_head = residual_heads[:, 3:4]

        multiplier = self._time_multiplier(tau)
        price = self.local_bs_put(raw) + multiplier * r_head
        bs_x, bs_xx, bs_v = self._local_bs_derivatives(raw)
        return torch.cat(
            [
                price,
                bs_x + multiplier * rx_head,
                bs_xx + multiplier * rxx_head,
                bs_v + multiplier * rv_head,
            ],
            dim=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_all(x)[:, 0:1]


__all__ = [
    "GlobalACVResidualPINN",
    "LOG_FEATURE_ORDER",
    "head_greeks_x_to_financial",
]
