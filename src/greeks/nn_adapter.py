from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import yaml
from torch import Tensor

from src.greeks.names import parse_feature_order
from src.models.ANN_pricer import ANN
from src.models.normalization import load_normalization_stats_from_run


FEATURE_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _normalize_run_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _list_available_run_dirs(project_root: Path) -> list[Path]:
    runs_dir = project_root / "outputs" / "runs"
    if not runs_dir.exists():
        return []
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _resolve_run_dir(project_root: Path, model_dir: str) -> Path:
    runs = _list_available_run_dirs(project_root)
    runs_dir = project_root / "outputs" / "runs"
    if not runs:
        raise FileNotFoundError(f"No run directories found under {runs_dir}")

    if model_dir == "latest":
        return runs[0]

    run_dir = runs_dir / model_dir
    if run_dir.exists():
        return run_dir

    target_norm = _normalize_run_name(model_dir)
    matches = [p for p in runs if _normalize_run_name(p.name) == target_norm]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(p.name for p in matches)
        raise FileNotFoundError(
            f"Model '{model_dir}' is ambiguous. Candidates: {options}"
        )

    available_preview = ", ".join(p.name for p in runs[:10])
    raise FileNotFoundError(
        f"Run directory not found for '{model_dir}'. Latest available: {available_preview}"
    )


def _resolve_device(preferred: str = "auto") -> torch.device:
    pref = preferred.lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available")
        return torch.device("mps")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' is not available")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of {'auto', 'cpu', 'mps', 'cuda'}")


def _load_model_from_run(
    *,
    project_root: Path,
    model_dir: str,
    checkpoint_name: str,
    device: str,
) -> tuple[ANN, torch.device, Path, dict, dict | None]:
    run_dir = _resolve_run_dir(project_root, model_dir)

    model_cfg_path = run_dir / "model_architecture_copy.yaml"
    if not model_cfg_path.exists():
        model_cfg_path = project_root / "configs" / "model_architecture.yaml"
    model_cfg = _load_yaml(model_cfg_path)

    model = ANN(
        input_dim=model_cfg["input"]["dim"],
        hidden_dims=model_cfg["hidden"]["dims"],
        output_dim=model_cfg["output"]["dim"],
        activation=model_cfg["hidden"]["activation"],
        dropout_rate=model_cfg["hidden"]["dropout_rate"],
        initialization=model_cfg["hidden"]["initialization"],
    )

    model_device = _resolve_device(device)
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=model_device)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state': {ckpt_path}")

    model.load_state_dict(ckpt["model_state"])
    model.to(model_device)
    model.eval()

    normalization_stats = load_normalization_stats_from_run(run_dir)
    return model, model_device, run_dir, model_cfg, normalization_stats


@dataclass(frozen=True)
class LoadedNNPriceAdapter:
    price_fn: "NNPriceAdapter"
    run_dir: Path
    feature_order: list[str]
    normalization_stats: dict | None
    device: torch.device


class NNPriceAdapter:
    """
    Wrap a trained ANN so it behaves as a scalar, differentiable function:
      x_raw (shape [D]) -> y_raw scalar.

    The wrapper applies input normalization and target denormalization inside the
    graph, so jacobians/hessians from src.greeks.core are in raw-space units.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        normalization_stats: dict | None,
        feature_order: Sequence[str],
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        self.feature_order = parse_feature_order(feature_order, fallback=FEATURE_ORDER)
        self.input_dim = len(self.feature_order)
        self.dtype = dtype

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        if self.device.type == "mps" and self.dtype == torch.float64:
            raise ValueError(
                "float64 on MPS is not supported reliably. Use device='cpu' "
                "or dtype=torch.float32."
            )

        self.model = model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self.x_mean, self.x_std, self.y_mean, self.y_std = self._build_norm_tensors(
            normalization_stats=normalization_stats
        )

    def _build_norm_tensors(
        self,
        *,
        normalization_stats: dict | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not normalization_stats or not bool(normalization_stats.get("enabled", False)):
            x_mean = torch.zeros(self.input_dim, dtype=self.dtype, device=self.device)
            x_std = torch.ones(self.input_dim, dtype=self.dtype, device=self.device)
            y_mean = torch.zeros((), dtype=self.dtype, device=self.device)
            y_std = torch.ones((), dtype=self.dtype, device=self.device)
            return x_mean, x_std, y_mean, y_std

        x_mean_raw = list(normalization_stats.get("x_mean", []))
        x_std_raw = list(normalization_stats.get("x_std", []))
        if not x_mean_raw or not x_std_raw:
            raise ValueError("Normalization stats must include x_mean and x_std")

        stats_feature_names = list(normalization_stats.get("feature_names", []))
        if stats_feature_names:
            idx = {name: i for i, name in enumerate(stats_feature_names)}
            missing = [name for name in self.feature_order if name not in idx]
            if missing:
                raise ValueError(f"Missing feature(s) in normalization stats: {missing}")
            x_mean = torch.tensor(
                [x_mean_raw[idx[name]] for name in self.feature_order],
                dtype=self.dtype,
                device=self.device,
            )
            x_std = torch.tensor(
                [x_std_raw[idx[name]] for name in self.feature_order],
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if len(x_mean_raw) != self.input_dim or len(x_std_raw) != self.input_dim:
                raise ValueError(
                    "Normalization stats dimension mismatch with feature_order. "
                    f"Expected {self.input_dim}, got mean={len(x_mean_raw)} std={len(x_std_raw)}"
                )
            x_mean = torch.tensor(x_mean_raw, dtype=self.dtype, device=self.device)
            x_std = torch.tensor(x_std_raw, dtype=self.dtype, device=self.device)

        x_std = torch.clamp(x_std, min=1.0e-12)

        if bool(normalization_stats.get("normalize_target", False)):
            y_mean = torch.tensor(
                float(normalization_stats.get("y_mean", 0.0)),
                dtype=self.dtype,
                device=self.device,
            )
            y_std_scalar = float(normalization_stats.get("y_std", 1.0))
            if abs(y_std_scalar) < 1.0e-12:
                y_std_scalar = 1.0
            y_std = torch.tensor(y_std_scalar, dtype=self.dtype, device=self.device)
        else:
            y_mean = torch.zeros((), dtype=self.dtype, device=self.device)
            y_std = torch.ones((), dtype=self.dtype, device=self.device)

        return x_mean, x_std, y_mean, y_std

    def __call__(self, x_raw: Tensor) -> Tensor:
        x = torch.as_tensor(x_raw, dtype=self.dtype, device=self.device)
        if x.ndim != 1:
            raise ValueError(f"x_raw must be 1D [D], got shape={tuple(x.shape)}")
        if x.numel() != self.input_dim:
            raise ValueError(
                f"x_raw has wrong size. Expected {self.input_dim}, got {x.numel()}"
            )

        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.model(x_norm.unsqueeze(0)).reshape(())
        y_raw = y_norm * self.y_std + self.y_mean
        return y_raw



def load_nn_price_adapter(
    *,
    project_root: Path,
    model_dir: str = "latest",
    checkpoint_name: str = "model_best.pt",
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    feature_order: Sequence[str] | None = None,
) -> LoadedNNPriceAdapter:
    feature_order_list = parse_feature_order(feature_order, fallback=FEATURE_ORDER)

    model, model_device, run_dir, model_cfg, normalization_stats = _load_model_from_run(
        project_root=project_root,
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    input_dim = int(model_cfg["input"]["dim"])
    if input_dim != len(feature_order_list):
        raise ValueError(
            "Feature order size does not match model input_dim. "
            f"input_dim={input_dim}, len(feature_order)={len(feature_order_list)}"
        )

    target_device = model_device
    if target_device.type == "mps" and dtype == torch.float64:
        # MPS float64 is not reliable for second-order autodiff.
        target_device = torch.device("cpu")

    adapter = NNPriceAdapter(
        model=model,
        normalization_stats=normalization_stats,
        feature_order=feature_order_list,
        dtype=dtype,
        device=target_device,
    )

    return LoadedNNPriceAdapter(
        price_fn=adapter,
        run_dir=run_dir,
        feature_order=feature_order_list,
        normalization_stats=normalization_stats,
        device=adapter.device,
    )


__all__ = [
    "FEATURE_ORDER",
    "NNPriceAdapter",
    "LoadedNNPriceAdapter",
    "load_nn_price_adapter",
]
