from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from src.models.ANN_pricer import ANN


FEATURE_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]
HESTON_PARAM_COUNT = 5


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _normalize_run_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def list_available_run_dirs(project_root: Path) -> list[Path]:
    runs_dir = project_root / "outputs" / "runs"
    if not runs_dir.exists():
        return []
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def resolve_run_dir(project_root: Path, model_dir: str) -> Path:
    runs = list_available_run_dirs(project_root)
    runs_dir = project_root / "outputs" / "runs"
    if not runs:
        raise FileNotFoundError(f"No run directories found under {runs_dir}")

    if model_dir == "latest":
        return runs[0]

    run_dir = runs_dir / model_dir
    if not run_dir.exists():
        # tolerate case/style differences, e.g. AdamV05 -> ADAM_v05
        target_norm = _normalize_run_name(model_dir)
        matches = [p for p in runs if _normalize_run_name(p.name) == target_norm]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            opts = ", ".join(p.name for p in matches)
            raise FileNotFoundError(
                f"Model '{model_dir}' is ambiguous. Candidates: {opts}"
            )
        available_preview = ", ".join(p.name for p in runs[:10])
        raise FileNotFoundError(
            f"Run directory not found for '{model_dir}'. "
            f"Latest available: {available_preview}"
        )
    return run_dir


def resolve_device(preferred: str = "auto") -> torch.device:
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


def load_model_from_run(
    *,
    project_root: Path,
    model_dir: str = "latest",
    checkpoint_name: str = "model_best.pt",
    device: str = "auto",
) -> tuple[ANN, torch.device, Path, dict]:
    """
    Load trained ANN_pricer model from outputs/runs/<run_id>.

    Returns
    - model (torch.nn.Module)
    - torch.device
    - run_dir (Path)
    - model_cfg (dict)
    """

    run_dir = resolve_run_dir(project_root=project_root, model_dir=model_dir)
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

    model_device = resolve_device(device)
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=model_device)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state': {ckpt_path}")

    model.load_state_dict(ckpt["model_state"])
    model.to(model_device)
    model.eval()
    return model, model_device, run_dir, model_cfg


def build_features_from_theta(
    theta: Sequence[float] | np.ndarray,
    *,
    moneyness: Sequence[float] | np.ndarray,
    tau: Sequence[float] | np.ndarray,
    r: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Build feature matrix expected by ANN_pricer.

    Inputs
    - theta: Heston parameters [rho, kappa, gamma, bar_v, v0], shape (5,)
    - moneyness, tau, r: quote vectors, shape (N,)

    Output
    - Features with shape (N, 8) in order:
      [rho, kappa, gamma, bar_v, v0, moneyness, tau, r]
    """

    theta_np = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_np.size != HESTON_PARAM_COUNT:
        raise ValueError(
            f"theta must have {HESTON_PARAM_COUNT} values; got {theta_np.size}"
        )

    m_np = np.asarray(moneyness, dtype=np.float64).reshape(-1)
    tau_np = np.asarray(tau, dtype=np.float64).reshape(-1)
    r_np = np.asarray(r, dtype=np.float64).reshape(-1)

    n = m_np.size
    if tau_np.size != n or r_np.size != n:
        raise ValueError(
            "moneyness, tau and r must share the same length. "
            f"Received {n}, {tau_np.size}, {r_np.size}"
        )
    if n == 0:
        raise ValueError("No quotes provided to build features")
    if np.any(tau_np <= 0.0):
        raise ValueError("tau must be strictly positive")

    theta_block = np.repeat(theta_np.reshape(1, -1), repeats=n, axis=0)
    features = np.column_stack([theta_block, m_np, tau_np, r_np]).astype(
        np.float32, copy=False
    )
    return features


def predict_iv(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device | str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Predict implied volatilities from ANN_pricer.

    Inputs
    - model: trained ANN
    - features: np.ndarray shape (N, 8)
    - device: optional override device
    - batch_size: optional chunk size for memory control

    Output
    - np.ndarray shape (N,)
    """

    x_np = np.asarray(features, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"features must be 2D; got ndim={x_np.ndim}")
    if x_np.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"features must have {len(FEATURE_ORDER)} columns; got {x_np.shape[1]}"
        )

    if device is None:
        model_device = next(model.parameters()).device
    else:
        model_device = torch.device(device)

    if batch_size is None or batch_size <= 0:
        batch_size = int(x_np.shape[0])

    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, x_np.shape[0], batch_size):
            stop = min(start + batch_size, x_np.shape[0])
            x_chunk = torch.from_numpy(x_np[start:stop]).to(model_device)
            y_chunk = model(x_chunk).detach().cpu().numpy().reshape(-1)
            outputs.append(y_chunk.astype(np.float64, copy=False))

    return np.concatenate(outputs, axis=0)
