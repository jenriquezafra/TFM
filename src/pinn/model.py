from __future__ import annotations

import torch
import torch.nn as nn


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


class PINNPricer(nn.Module):
    """
    Minimal PINN backbone (MLP) for price regression.
    """

    def __init__(self, architecture_config: dict):
        super().__init__()
        self.architecture_config = architecture_config

        input_cfg = architecture_config.get("input", {})
        hidden_cfg = architecture_config.get("hidden", {})
        output_cfg = architecture_config.get("output", {})

        input_dim = int(input_cfg.get("dim", 8))
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

        self.backbone = nn.Sequential(*layers)
        _init_linear_weights(self.backbone, initialization=initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got shape {tuple(x.shape)}")
        return self.backbone(x)


def build_pinn_model(architecture_config: dict) -> PINNPricer:
    """
    Helper used by trainer/pipeline to instantiate the model.
    """
    return PINNPricer(architecture_config=architecture_config)
