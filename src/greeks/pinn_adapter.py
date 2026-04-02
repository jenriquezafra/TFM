from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import yaml
from torch import Tensor

from src.greeks.names import parse_feature_order
from src.pinn.model import build_pinn_model


DEFAULT_PINN_FEATURE_ORDER = ["tau", "moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_path(raw: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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


def _resolve_run_dir(project_root: Path, run_dir: str) -> Path:
    pinn_root = project_root / "outputs" / "pinn"
    if not pinn_root.exists():
        raise FileNotFoundError(f"PINN outputs directory not found: {pinn_root}")

    if run_dir == "latest":
        candidates = [p for p in pinn_root.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No PINN run directories found under {pinn_root}")
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    candidate = Path(run_dir)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    by_name = pinn_root / run_dir
    if by_name.exists():
        return by_name

    raise FileNotFoundError(
        f"PINN run directory not found for '{run_dir}'. "
        f"Checked absolute path and {by_name}"
    )


def _resolve_architecture_config_path(
    *,
    project_root: Path,
    run_dir: Path,
    explicit_path: str | None,
) -> Path:
    if explicit_path is not None:
        path = _resolve_path(explicit_path, base_dir=project_root)
        if not path.exists():
            raise FileNotFoundError(f"Architecture config not found: {path}")
        return path

    plan_path = run_dir / "pipeline_plan.yaml"
    if plan_path.exists():
        plan = _load_yaml(plan_path)
        cfg_raw = plan.get("architecture_config")
        if cfg_raw:
            cfg_path = _resolve_path(cfg_raw, base_dir=project_root)
            if cfg_path.exists():
                return cfg_path

    execution_path = run_dir / "pipeline_execution.yaml"
    if execution_path.exists():
        execution = _load_yaml(execution_path)
        pipeline_cfg_raw = execution.get("config_path")
        if pipeline_cfg_raw:
            pipeline_cfg_path = _resolve_path(pipeline_cfg_raw, base_dir=project_root)
            if pipeline_cfg_path.exists():
                pipeline_cfg = _load_yaml(pipeline_cfg_path)
                model_cfg = pipeline_cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    cfg_rel = model_cfg.get("architecture_config")
                    if cfg_rel:
                        cfg_path = _resolve_path(cfg_rel, base_dir=project_root)
                        if cfg_path.exists():
                            return cfg_path

    default_cfg = project_root / "configs" / "pinn_model_architecture.yaml"
    if default_cfg.exists():
        return default_cfg

    raise FileNotFoundError("Could not resolve PINN architecture config path.")


def _resolve_checkpoint_path(
    *,
    project_root: Path,
    run_dir: Path,
    checkpoint_name: str,
) -> Path:
    ckpt_raw = Path(checkpoint_name)
    if ckpt_raw.is_absolute():
        if not ckpt_raw.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_raw}")
        return ckpt_raw

    if len(ckpt_raw.parts) > 1:
        ckpt_path = _resolve_path(ckpt_raw, base_dir=project_root)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    ckpt_path = run_dir / "train" / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _load_train_summary(run_dir: Path) -> dict:
    path = run_dir / "train" / "metrics" / "train_summary.yaml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def _resolve_collocation_manifest_path(
    *,
    project_root: Path,
    run_dir: Path,
    train_summary: dict,
) -> Path | None:
    summary_path = train_summary.get("collocation_manifest_file")
    if summary_path:
        path = _resolve_path(summary_path, base_dir=project_root)
        if path.exists():
            return path

    execution_path = run_dir / "pipeline_execution.yaml"
    if execution_path.exists():
        execution = _load_yaml(execution_path)
        stages = execution.get("stages", {})
        if isinstance(stages, dict):
            train_stage = stages.get("train", {})
            if isinstance(train_stage, dict):
                colloc_path = train_stage.get("collocation_manifest_file")
                if colloc_path:
                    path = _resolve_path(colloc_path, base_dir=project_root)
                    if path.exists():
                        return path

    return None


def _resolve_feature_order(
    *,
    project_root: Path,
    run_dir: Path,
    train_summary: dict,
    user_feature_order: Sequence[str] | None,
) -> list[str]:
    if user_feature_order is not None:
        return parse_feature_order(user_feature_order, fallback=DEFAULT_PINN_FEATURE_ORDER)

    scaling = train_summary.get("input_scaling", {})
    if isinstance(scaling, dict):
        scaling_order = scaling.get("feature_order")
        if isinstance(scaling_order, list) and scaling_order:
            return parse_feature_order(scaling_order, fallback=DEFAULT_PINN_FEATURE_ORDER)

    colloc_manifest_path = _resolve_collocation_manifest_path(
        project_root=project_root,
        run_dir=run_dir,
        train_summary=train_summary,
    )
    if colloc_manifest_path is not None:
        colloc = _load_yaml(colloc_manifest_path)
        colloc_order = colloc.get("feature_order")
        if isinstance(colloc_order, list) and colloc_order:
            return parse_feature_order(colloc_order, fallback=DEFAULT_PINN_FEATURE_ORDER)

    return list(DEFAULT_PINN_FEATURE_ORDER)


@dataclass(frozen=True)
class LoadedPINNPriceAdapter:
    price_fn: "PINNPriceAdapter"
    run_dir: Path
    checkpoint_path: Path
    architecture_config_path: Path
    feature_order: list[str]
    input_scaling: dict
    device: torch.device


class PINNPriceAdapter:
    """
    Wrap a trained PINN so it behaves as a scalar, differentiable function:
      x_raw (shape [D]) -> y_raw scalar.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        input_scaling: dict | None,
        feature_order: Sequence[str],
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        self.feature_order = parse_feature_order(feature_order, fallback=DEFAULT_PINN_FEATURE_ORDER)
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

        self.a, self.b = self._build_input_affine_tensors(input_scaling=input_scaling)

    def _build_input_affine_tensors(
        self,
        *,
        input_scaling: dict | None,
    ) -> tuple[Tensor, Tensor]:
        if not input_scaling or not bool(input_scaling.get("enabled", False)):
            a = torch.zeros(self.input_dim, dtype=self.dtype, device=self.device)
            b = torch.ones(self.input_dim, dtype=self.dtype, device=self.device)
            return a, b

        a_raw = list(input_scaling.get("a", []))
        b_raw = list(input_scaling.get("b", []))
        if not a_raw or not b_raw:
            raise ValueError("Input scaling must include non-empty 'a' and 'b'.")

        stats_feature_order = list(input_scaling.get("feature_order", []))
        if stats_feature_order:
            idx = {name: i for i, name in enumerate(stats_feature_order)}
            missing = [name for name in self.feature_order if name not in idx]
            if missing:
                raise ValueError(f"Missing feature(s) in input_scaling.feature_order: {missing}")
            a_vals = [a_raw[idx[name]] for name in self.feature_order]
            b_vals = [b_raw[idx[name]] for name in self.feature_order]
        else:
            if len(a_raw) != self.input_dim or len(b_raw) != self.input_dim:
                raise ValueError(
                    "Input scaling dimension mismatch with feature_order. "
                    f"Expected {self.input_dim}, got a={len(a_raw)} b={len(b_raw)}"
                )
            a_vals = a_raw
            b_vals = b_raw

        a = torch.tensor(a_vals, dtype=self.dtype, device=self.device)
        b = torch.tensor(b_vals, dtype=self.dtype, device=self.device)
        return a, b

    def __call__(self, x_raw: Tensor) -> Tensor:
        x = torch.as_tensor(x_raw, dtype=self.dtype, device=self.device)
        if x.ndim != 1:
            raise ValueError(f"x_raw must be 1D [D], got shape={tuple(x.shape)}")
        if x.numel() != self.input_dim:
            raise ValueError(
                f"x_raw has wrong size. Expected {self.input_dim}, got {x.numel()}"
            )

        x_model = self.a + self.b * x
        y_raw = self.model(x_model.unsqueeze(0)).reshape(())
        return y_raw


def load_pinn_price_adapter(
    *,
    project_root: Path,
    run_dir: str = "latest",
    checkpoint_name: str = "model_best.pt",
    architecture_config_path: str | None = None,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    feature_order: Sequence[str] | None = None,
) -> LoadedPINNPriceAdapter:
    resolved_run_dir = _resolve_run_dir(project_root, run_dir)
    resolved_arch_cfg = _resolve_architecture_config_path(
        project_root=project_root,
        run_dir=resolved_run_dir,
        explicit_path=architecture_config_path,
    )
    resolved_ckpt = _resolve_checkpoint_path(
        project_root=project_root,
        run_dir=resolved_run_dir,
        checkpoint_name=checkpoint_name,
    )
    train_summary = _load_train_summary(resolved_run_dir)
    resolved_feature_order = _resolve_feature_order(
        project_root=project_root,
        run_dir=resolved_run_dir,
        train_summary=train_summary,
        user_feature_order=feature_order,
    )
    input_scaling = train_summary.get("input_scaling", {})
    if not isinstance(input_scaling, dict):
        input_scaling = {}

    model_cfg = _load_yaml(resolved_arch_cfg)
    model = build_pinn_model(model_cfg)

    model_device = _resolve_device(device)
    if device.lower() == "auto" and model_device.type == "mps" and dtype == torch.float64:
        model_device = torch.device("cpu")
    state = torch.load(resolved_ckpt, map_location=model_device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")
    model.load_state_dict(state)
    model.to(model_device)
    model.eval()

    price_fn = PINNPriceAdapter(
        model=model,
        input_scaling=input_scaling,
        feature_order=resolved_feature_order,
        dtype=dtype,
        device=model_device,
    )
    return LoadedPINNPriceAdapter(
        price_fn=price_fn,
        run_dir=resolved_run_dir,
        checkpoint_path=resolved_ckpt,
        architecture_config_path=resolved_arch_cfg,
        feature_order=list(resolved_feature_order),
        input_scaling=input_scaling,
        device=model_device,
    )


__all__ = [
    "DEFAULT_PINN_FEATURE_ORDER",
    "PINNPriceAdapter",
    "LoadedPINNPriceAdapter",
    "load_pinn_price_adapter",
]
