from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from src.greeks.pinn_adapter import (
    DEFAULT_PINN_FEATURE_ORDER,
    LoadedPINNPriceAdapter,
    load_pinn_price_adapter,
)


RAW_FEATURE_ORDER = ["tau", "moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
LOG_FEATURE_ORDER = ["tau", "log_moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.as_tensor(2.0, dtype=x.dtype, device=x.device))))


def raw_to_log_independent(raw: Tensor, *, m_floor: float = 1.0e-12) -> Tensor:
    if raw.ndim != 2 or raw.shape[1] < 8:
        raise ValueError(f"raw must have shape [N,8], got {tuple(raw.shape)}")
    m = torch.clamp(raw[:, 1:2], min=float(m_floor))
    return torch.cat([raw[:, 0:1], torch.log(m), raw[:, 2:8]], dim=1)


def log_to_raw(log_inputs: Tensor) -> Tensor:
    if log_inputs.ndim != 2 or log_inputs.shape[1] < 8:
        raise ValueError(f"log_inputs must have shape [N,8], got {tuple(log_inputs.shape)}")
    return torch.cat([log_inputs[:, 0:1], torch.exp(log_inputs[:, 1:2]), log_inputs[:, 2:8]], dim=1)


class FourierResidualMLP(nn.Module):
    """
    Experimental residual net for ACV-HardPatch.

    Fourier features are applied only to z=x/sqrt(v*tau+eps), because this is
    the local boundary-layer coordinate we want to enrich without making all
    Heston parameters high-frequency inputs.
    """

    def __init__(
        self,
        *,
        base_input_dim: int = 10,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        activation: str = "tanh",
        fourier_frequencies: int = 6,
        final_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if base_input_dim != 10:
            raise ValueError("ACV residual base input is fixed at 10 features.")
        if fourier_frequencies < 0:
            raise ValueError("fourier_frequencies must be >= 0")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        self.fourier_frequencies = int(fourier_frequencies)
        self.base_input_dim = int(base_input_dim)
        in_dim = self.base_input_dim + 2 * self.fourier_frequencies

        act_key = str(activation).strip().lower()
        if act_key == "tanh":
            act_factory: Callable[[], nn.Module] = nn.Tanh
        elif act_key == "gelu":
            act_factory = nn.GELU
        elif act_key == "silu":
            act_factory = nn.SiLU
        elif act_key == "elu":
            act_factory = nn.ELU
        else:
            raise ValueError("activation must be one of {'tanh', 'gelu', 'silu', 'elu'}")

        dims = [in_dim, *[int(x) for x in hidden_dims], 1]
        layers: list[nn.Module] = []
        for din, dout in zip(dims[:-2], dims[1:-1]):
            linear = nn.Linear(din, dout)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(act_factory())

        final = nn.Linear(dims[-2], dims[-1])
        if float(final_init_scale) == 0.0:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            nn.init.xavier_uniform_(final.weight)
            final.weight.data.mul_(float(final_init_scale))
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def _augment(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.base_input_dim:
            raise ValueError(
                f"features must have shape [N,{self.base_input_dim}], got {tuple(features.shape)}"
            )
        if self.fourier_frequencies == 0:
            return features

        z = features[:, 0:1]
        freqs = torch.pow(
            torch.as_tensor(2.0, dtype=features.dtype, device=features.device),
            torch.arange(self.fourier_frequencies, dtype=features.dtype, device=features.device),
        ).view(1, -1)
        angles = z * freqs
        return torch.cat([features, torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self._augment(features))


class ACVHardPatchModel(nn.Module):
    """
    Experimental Asymptotic Control Variate hard-region patch.

    The frozen global baseline V_B is blended with a local singularity-aware
    patch:

      V = (1 - chi) V_B + chi (P_locBS + tau * R_theta)

    This class is intentionally not the default PINN model. It exists for the
    hard ATM-short Greek experiment and should be treated as experimental.
    """

    def __init__(
        self,
        *,
        baseline_model: nn.Module,
        baseline_input_a: Tensor,
        baseline_input_b: Tensor,
        residual_net: FourierResidualMLP,
        x_center: float = 0.06,
        tau_center: float = 0.08,
        delta_x: float = 0.01,
        delta_tau: float = 0.015,
        q_epsilon: float = 1.0e-10,
        bs_epsilon: float = 1.0e-12,
        tau_epsilon: float = 1.0e-8,
        terminal_tau: float = 0.0,
        z_clip: float | None = 50.0,
    ) -> None:
        super().__init__()
        if x_center <= 0.0 or tau_center <= 0.0:
            raise ValueError("x_center and tau_center must be > 0")
        if delta_x <= 0.0 or delta_tau <= 0.0:
            raise ValueError("delta_x and delta_tau must be > 0")
        if q_epsilon <= 0.0 or bs_epsilon <= 0.0 or tau_epsilon <= 0.0:
            raise ValueError("q_epsilon, bs_epsilon and tau_epsilon must be > 0")

        self.baseline_model = baseline_model
        for param in self.baseline_model.parameters():
            param.requires_grad_(False)
        self.baseline_model.eval()

        self.residual_net = residual_net
        self.x_center = float(x_center)
        self.tau_center = float(tau_center)
        self.delta_x = float(delta_x)
        self.delta_tau = float(delta_tau)
        self.q_epsilon = float(q_epsilon)
        self.bs_epsilon = float(bs_epsilon)
        self.tau_epsilon = float(tau_epsilon)
        self.terminal_tau = float(terminal_tau)
        self.z_clip = None if z_clip is None else float(z_clip)

        self.register_buffer("baseline_input_a", baseline_input_a.detach().clone().reshape(1, -1))
        self.register_buffer("baseline_input_b", baseline_input_b.detach().clone().reshape(1, -1))

    def train(self, mode: bool = True) -> "ACVHardPatchModel":
        super().train(mode)
        self.baseline_model.eval()
        return self

    def baseline_price(self, raw: Tensor) -> Tensor:
        x_net = self.baseline_input_a.to(dtype=raw.dtype, device=raw.device) + (
            self.baseline_input_b.to(dtype=raw.dtype, device=raw.device) * raw
        )
        return self.baseline_model(x_net)

    def gate(self, raw: Tensor) -> Tensor:
        y = raw_to_log_independent(raw)
        tau = torch.clamp(y[:, 0:1], min=0.0)
        x = y[:, 1:2]
        gate_x = torch.sigmoid((self.x_center - torch.abs(x)) / self.delta_x)
        gate_tau = torch.sigmoid((self.tau_center - tau) / self.delta_tau)
        return gate_x * gate_tau

    def local_bs_put(self, raw: Tensor) -> Tensor:
        y = raw_to_log_independent(raw)
        tau = torch.clamp(y[:, 0:1], min=0.0)
        x = y[:, 1:2]
        m = torch.exp(x)
        v = torch.clamp(y[:, 2:3], min=0.0)
        r = y[:, 7:8]

        vtau = torch.clamp(v * tau, min=0.0)
        sigma_tau = torch.sqrt(vtau + torch.as_tensor(self.bs_epsilon, dtype=raw.dtype, device=raw.device))
        d1 = (x + (r + 0.5 * v) * tau) / sigma_tau
        d2 = d1 - sigma_tau
        put = torch.exp(-r * tau) * _normal_cdf(-d2) - m * _normal_cdf(-d1)
        payoff = torch.clamp(1.0 - m, min=0.0)
        if self.terminal_tau <= 0.0:
            return torch.where(tau <= 0.0, payoff, put)
        return torch.where(tau <= self.terminal_tau, payoff, put)

    def residual_features(self, raw: Tensor) -> Tensor:
        y = raw_to_log_independent(raw)
        tau = torch.clamp(y[:, 0:1], min=0.0)
        x = y[:, 1:2]
        v = torch.clamp(y[:, 2:3], min=0.0)
        q = torch.sqrt(
            v * tau + torch.as_tensor(self.q_epsilon, dtype=raw.dtype, device=raw.device)
        )
        z = x / q
        if self.z_clip is not None:
            z = torch.clamp(z, min=-self.z_clip, max=self.z_clip)
        log_tau = torch.log(tau + torch.as_tensor(self.tau_epsilon, dtype=raw.dtype, device=raw.device))
        return torch.cat(
            [
                z,
                q,
                log_tau,
                x,
                y[:, 2:3],
                y[:, 3:4],
                y[:, 4:5],
                y[:, 5:6],
                y[:, 6:7],
                y[:, 7:8],
            ],
            dim=1,
        )

    def patch_price(self, raw: Tensor) -> Tensor:
        tau = torch.clamp(raw[:, 0:1], min=0.0)
        return self.local_bs_put(raw) + tau * self.residual_net(self.residual_features(raw))

    def forward(self, raw: Tensor) -> Tensor:
        if raw.ndim != 2 or raw.shape[1] < 8:
            raise ValueError(f"raw must have shape [N,8], got {tuple(raw.shape)}")
        chi = self.gate(raw)
        base = self.baseline_price(raw)
        patch = self.patch_price(raw)
        return (1.0 - chi) * base + chi * patch


def _gradient(*, y: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def heston_log_pde_residual(
    *,
    price_fn: Callable[[Tensor], Tensor],
    raw: Tensor,
    scale_epsilon: float | None = None,
) -> tuple[Tensor, Tensor | None]:
    """
    Heston residual in independent variables (tau, x=log(m), v).

    Returned residual uses the normalized put-price convention used by the
    existing PINN code:

      N[V] = V_tau - [0.5 v V_xx + (r - 0.5v)V_x + rho gamma v V_xv
                      + 0.5 gamma^2 v V_vv + kappa(bar_v-v)V_v - rV]
    """

    y = raw_to_log_independent(raw).detach().requires_grad_(True)
    raw_y = log_to_raw(y)
    value = price_fn(raw_y)
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"price_fn must return [N,1], got {tuple(value.shape)}")

    grads = _gradient(y=value, x=y)
    v_tau = grads[:, 0:1]
    v_x = grads[:, 1:2]
    v_v = grads[:, 2:3]
    grad_v_x = _gradient(y=v_x, x=y)
    grad_v_v = _gradient(y=v_v, x=y)
    v_xx = grad_v_x[:, 1:2]
    v_xv = grad_v_x[:, 2:3]
    v_vv = grad_v_v[:, 2:3]

    v = y[:, 2:3]
    rho = y[:, 3:4]
    kappa = y[:, 4:5]
    gamma = y[:, 5:6]
    bar_v = y[:, 6:7]
    r = y[:, 7:8]

    generator = (
        0.5 * v * v_xx
        + (r - 0.5 * v) * v_x
        + rho * gamma * v * v_xv
        + 0.5 * (gamma**2) * v * v_vv
        + kappa * (bar_v - v) * v_v
        - r * value
    )
    residual = v_tau - generator
    scale = None
    if scale_epsilon is not None:
        tau = torch.clamp(y[:, 0:1], min=0.0)
        scale = 1.0 + 1.0 / torch.sqrt(
            torch.clamp(v * tau, min=0.0)
            + torch.as_tensor(float(scale_epsilon), dtype=raw.dtype, device=raw.device)
        )
    return residual, scale


def normalized_pde_loss(
    *,
    model: ACVHardPatchModel,
    raw: Tensor,
    target: str = "final",
    scale_epsilon: float = 1.0e-8,
    huber_beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    if target == "final":
        price_fn = model
    elif target == "patch":
        price_fn = model.patch_price
    else:
        raise ValueError("target must be one of {'final', 'patch'}")

    residual, scale = heston_log_pde_residual(
        price_fn=price_fn,
        raw=raw,
        scale_epsilon=scale_epsilon,
    )
    assert scale is not None
    scaled = residual / scale
    loss = F.smooth_l1_loss(
        scaled,
        torch.zeros_like(scaled),
        beta=float(huber_beta),
        reduction="mean",
    )
    return loss, residual


def terminal_loss(*, model: ACVHardPatchModel, raw_terminal: Tensor) -> Tensor:
    raw = raw_terminal.clone()
    raw[:, 0] = 0.0
    pred = model(raw)
    payoff = torch.clamp(1.0 - raw[:, 1:2], min=0.0)
    return torch.mean((pred - payoff) ** 2)


def interface_loss(
    *,
    model: ACVHardPatchModel,
    raw_interface: Tensor,
    lambda_x: float = 0.1,
    lambda_v: float = 0.1,
) -> Tensor:
    y = raw_to_log_independent(raw_interface).detach().requires_grad_(True)
    raw_y = log_to_raw(y)
    pred = model(raw_y)
    base = model.baseline_price(raw_y)
    grad_pred = _gradient(y=pred, x=y)
    grad_base = _gradient(y=base, x=y)
    value = torch.mean((pred - base) ** 2)
    x_part = torch.mean((grad_pred[:, 1:2] - grad_base[:, 1:2]) ** 2)
    v_part = torch.mean((grad_pred[:, 2:3] - grad_base[:, 2:3]) ** 2)
    return value + float(lambda_x) * x_part + float(lambda_v) * v_part


def baseline_distill_loss(*, model: ACVHardPatchModel, raw_patch: Tensor) -> Tensor:
    pred = model.patch_price(raw_patch)
    with torch.no_grad():
        target = model.baseline_price(raw_patch)
    return torch.mean((pred - target) ** 2)


def global_replay_loss(*, model: ACVHardPatchModel, raw_replay: Tensor) -> Tensor:
    pred = model(raw_replay)
    with torch.no_grad():
        target = model.baseline_price(raw_replay)
    return torch.mean((pred - target) ** 2)


def price_label_loss(
    *,
    model: ACVHardPatchModel,
    raw: Tensor,
    ref_price: Tensor,
    alpha: float = 0.5,
    floor: float = 1.0e-4,
) -> Tensor:
    pred = model(raw)
    tau = torch.clamp(raw[:, 0:1], min=0.0)
    v = torch.clamp(raw[:, 2:3], min=0.0)
    weight = 1.0 / torch.pow(torch.as_tensor(floor, dtype=raw.dtype, device=raw.device) + v * tau, alpha)
    return torch.mean(weight * (pred - ref_price) ** 2)


def stencil_price_loss(
    *,
    model: ACVHardPatchModel,
    raw_stencil: Tensor,
    ref_price: Tensor,
    h_x: Tensor,
    mode: str = "price",
    epsilon: float = 1.0e-12,
) -> Tensor:
    if raw_stencil.ndim != 3 or raw_stencil.shape[1] != 5 or raw_stencil.shape[2] < 8:
        raise ValueError(f"raw_stencil must have shape [N,5,8], got {tuple(raw_stencil.shape)}")
    if ref_price.shape != raw_stencil.shape[:2]:
        raise ValueError(
            f"ref_price must have shape {tuple(raw_stencil.shape[:2])}, got {tuple(ref_price.shape)}"
        )

    n = raw_stencil.shape[0]
    pred = model(raw_stencil.reshape(n * 5, raw_stencil.shape[2])).reshape(n, 5)
    mode_key = str(mode).strip().lower()
    if mode_key == "price":
        center = raw_stencil[:, 2, :]
        tau = torch.clamp(center[:, 0], min=0.0)
        v = torch.clamp(center[:, 2], min=0.0)
        g_scale = 1.0 / torch.sqrt(v * tau + torch.as_tensor(epsilon, dtype=pred.dtype, device=pred.device))
        denom = torch.square(torch.square(h_x.reshape(-1)) * g_scale + epsilon).view(-1, 1)
        return torch.mean(torch.square(pred - ref_price) / denom)
    if mode_key == "curvature":
        h2 = torch.square(h_x.reshape(-1))
        pred_curv = (pred[:, 3] - 2.0 * pred[:, 2] + pred[:, 1]) / h2
        ref_curv = (ref_price[:, 3] - 2.0 * ref_price[:, 2] + ref_price[:, 1]) / h2
        return torch.mean(torch.square((pred_curv - ref_curv) / (1.0 + torch.abs(ref_curv))))
    raise ValueError("mode must be one of {'price', 'curvature'}")


@dataclass(frozen=True)
class LoadedACVHardPatch:
    model: ACVHardPatchModel
    baseline: LoadedPINNPriceAdapter
    device: torch.device
    dtype: torch.dtype


def build_acv_hard_patch_model(
    *,
    project_root: Path,
    config: dict,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
) -> LoadedACVHardPatch:
    baseline_cfg = config.get("baseline", {})
    baseline = load_pinn_price_adapter(
        project_root=project_root,
        run_dir=str(baseline_cfg.get("run_dir", "PINN_mix_scaled_param")),
        checkpoint_name=str(baseline_cfg.get("checkpoint_name", "model_best.pt")),
        architecture_config_path=baseline_cfg.get("architecture_config"),
        device=device,
        dtype=dtype,
        feature_order=baseline_cfg.get("feature_order"),
    )
    if list(baseline.feature_order) != DEFAULT_PINN_FEATURE_ORDER:
        raise ValueError(
            "ACV-HardPatch currently expects baseline feature order "
            f"{DEFAULT_PINN_FEATURE_ORDER}, got {baseline.feature_order}"
        )

    model_cfg = config.get("model", {})
    residual_cfg = model_cfg.get("residual", {})
    residual = FourierResidualMLP(
        hidden_dims=residual_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=str(residual_cfg.get("activation", "tanh")),
        fourier_frequencies=int(residual_cfg.get("fourier_frequencies", 6)),
        final_init_scale=float(residual_cfg.get("final_init_scale", 0.0)),
    )
    patch_cfg = config.get("patch", {})
    gate_cfg = patch_cfg.get("gate", {})
    scale_cfg = patch_cfg.get("scales", {})
    model = ACVHardPatchModel(
        baseline_model=baseline.price_fn.model,
        baseline_input_a=baseline.price_fn.a,
        baseline_input_b=baseline.price_fn.b,
        residual_net=residual,
        x_center=float(gate_cfg.get("x_center", 0.06)),
        tau_center=float(gate_cfg.get("tau_center", 0.08)),
        delta_x=float(gate_cfg.get("delta_x", 0.01)),
        delta_tau=float(gate_cfg.get("delta_tau", 0.015)),
        q_epsilon=float(scale_cfg.get("q_epsilon", 1.0e-10)),
        bs_epsilon=float(scale_cfg.get("bs_epsilon", 1.0e-12)),
        tau_epsilon=float(scale_cfg.get("tau_epsilon", 1.0e-8)),
        terminal_tau=float(scale_cfg.get("terminal_tau", 0.0)),
        z_clip=scale_cfg.get("z_clip", 50.0),
    )
    model.to(device=baseline.device, dtype=dtype)
    return LoadedACVHardPatch(
        model=model,
        baseline=baseline,
        device=baseline.device,
        dtype=dtype,
    )


def load_acv_hard_patch_checkpoint(
    *,
    project_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
) -> LoadedACVHardPatch:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML dictionary in {config_path}")
    loaded = build_acv_hard_patch_model(
        project_root=project_root,
        config=config,
        device=device,
        dtype=dtype,
    )
    state = torch.load(checkpoint_path, map_location=loaded.device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")
    loaded.model.load_state_dict(state, strict=True)
    loaded.model.eval()
    return loaded


class ACVHardPatchPriceAdapter:
    def __init__(self, *, model: ACVHardPatchModel, dtype: torch.dtype, device: torch.device) -> None:
        self.model = model.to(device=device, dtype=dtype)
        self.model.eval()
        self.dtype = dtype
        self.device = device
        self.feature_order = list(RAW_FEATURE_ORDER)

    def __call__(self, x_raw: Tensor) -> Tensor:
        x = torch.as_tensor(x_raw, dtype=self.dtype, device=self.device)
        if x.ndim != 1:
            raise ValueError(f"x_raw must be 1D [D], got shape={tuple(x.shape)}")
        return self.model(x.unsqueeze(0)).reshape(())


__all__ = [
    "ACVHardPatchModel",
    "ACVHardPatchPriceAdapter",
    "FourierResidualMLP",
    "LoadedACVHardPatch",
    "RAW_FEATURE_ORDER",
    "LOG_FEATURE_ORDER",
    "baseline_distill_loss",
    "build_acv_hard_patch_model",
    "global_replay_loss",
    "heston_log_pde_residual",
    "interface_loss",
    "load_acv_hard_patch_checkpoint",
    "log_to_raw",
    "normalized_pde_loss",
    "price_label_loss",
    "raw_to_log_independent",
    "stencil_price_loss",
    "terminal_loss",
]
