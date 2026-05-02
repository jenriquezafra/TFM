from __future__ import annotations

from pathlib import Path
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.pinn.losses import (
    PINNLossTerms,
    compute_heston_pde_residual,
    compute_weighted_pinn_loss,
)
from src.pinn.model import build_pinn_model
from src.utils.callbacks import build_step_lr


def _resolve_device(device_pref: str) -> torch.device:
    pref = str(device_pref).lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref in {"cpu", "cuda", "mps"}:
        return torch.device(pref)
    raise ValueError(f"Unsupported device '{device_pref}'")


def _normalize_training_mode(name: str) -> str:
    raw = str(name).strip().lower()
    if raw in {"l-bfgs", "lbfgs"}:
        return "lbfgs"
    if raw in {"mix_half", "mix-half", "mix_once"}:
        return "mix_half"
    return raw


def _normalize_optimizer_name(name: str) -> str:
    raw = str(name).strip().lower()
    if raw in {"adam", "sgd"}:
        return raw
    if raw in {"l-bfgs", "lbfgs"}:
        return "lbfgs"
    raise ValueError(
        f"Unsupported optimizer '{name}'. Use one of: adam, sgd, l-bfgs"
    )


def _get_optimizer_cfg(training_config: dict, optimizer_name: str) -> dict:
    optimizers = training_config.get("optimizers", [])
    for item in optimizers:
        item_name = item.get("name", "")
        try:
            normalized = _normalize_optimizer_name(item_name)
        except ValueError:
            continue
        if normalized == optimizer_name:
            return dict(item)
    raise ValueError(
        f"Optimizer config for '{optimizer_name}' not found in training_config['optimizers']"
    )


def _build_optimizer(
    *,
    model: nn.Module,
    optimizer_name: str,
    optimizer_cfg: dict,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(optimizer_cfg.get("learn_rate", 1.0e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(optimizer_cfg.get("learn_rate", 1.0e-3)),
            momentum=float(optimizer_cfg.get("momentum", 0.0)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )

    if optimizer_name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=float(optimizer_cfg.get("learn_rate", 1.0)),
            max_iter=int(optimizer_cfg.get("max_iter", 20)),
            line_search_fn=optimizer_cfg.get("line_search_fn", "strong_wolfe"),
            history_size=int(
                optimizer_cfg.get("historic_size", optimizer_cfg.get("history_size", 100))
            ),
        )

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def _build_scheduler(
    *,
    training_config: dict,
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    cb_cfg = training_config.get("callbacks", {})
    sched_cfg = cb_cfg.get("lr_scheduler", {})
    enabled = bool(sched_cfg.get("enabled", False))
    if not enabled:
        return None

    # Keep scheduler only for Adam in mixed mode.
    if optimizer_name != "adam":
        return None

    sched_name = str(sched_cfg.get("name", "step")).lower()
    if sched_name == "step":
        return build_step_lr(
            optimizer=optimizer,
            step_size=int(sched_cfg.get("step_size", 500)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )

    raise ValueError(f"Unsupported lr scheduler '{sched_name}'. Use 'step'.")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _load_parquet_matrix(*, path: Path, feature_order: list[str]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Collocation file not found: {path}")
    df = pd.read_parquet(path)
    missing = [col for col in feature_order if col not in df.columns]
    if missing:
        raise KeyError(
            f"Collocation file {path} missing columns {missing}. "
            f"Available: {list(df.columns)}"
        )
    x = df.loc[:, feature_order].to_numpy(dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {path}, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"Collocation matrix contains non-finite values: {path}")
    return x


def _split_array(
    x: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Cannot split empty dataset.")
    if val_fraction <= 0.0:
        return x, x
    if val_fraction >= 1.0:
        raise ValueError(f"val_fraction must be < 1.0. Got {val_fraction}.")
    if x.shape[0] == 1:
        return x, x

    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(x.shape[0] * val_fraction))
    n_val = max(1, min(n_val, x.shape[0] - 1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], x[val_idx]


def _build_input_affine_from_train(
    *,
    x_train: dict[str, np.ndarray],
    feature_order: list[str],
    scaling_cfg: dict,
) -> dict | None:
    enabled = bool(scaling_cfg.get("enabled", False))
    if not enabled:
        return None

    method = str(scaling_cfg.get("method", "standardize_train")).strip().lower()
    if method not in {"standardize_train"}:
        raise ValueError(
            f"Unsupported data.input_scaling.method='{method}'. "
            "Use 'standardize_train'."
        )

    eps = float(scaling_cfg.get("eps", 1.0e-8))
    if eps <= 0.0:
        raise ValueError(f"data.input_scaling.eps must be > 0. Got {eps}.")

    blocks = [x_train["interior"], x_train["terminal"], x_train["lower"]]
    x_cat = np.concatenate(blocks, axis=0).astype(np.float64, copy=False)
    if x_cat.ndim != 2:
        raise ValueError(f"Expected 2D collocation matrix, got shape {x_cat.shape}.")
    if x_cat.shape[1] != len(feature_order):
        raise ValueError(
            f"Feature count mismatch: matrix has {x_cat.shape[1]}, "
            f"feature_order has {len(feature_order)}."
        )

    mean = x_cat.mean(axis=0)
    std = x_cat.std(axis=0)
    safe = std > eps

    # Affine transform used by the paper family: x_scaled = a + b * x.
    # For stable columns with non-negligible std: a = -mean/std, b = 1/std.
    # For nearly constant columns: keep identity (a=0, b=1).
    a = np.zeros_like(mean, dtype=np.float64)
    b = np.ones_like(std, dtype=np.float64)
    a[safe] = -mean[safe] / std[safe]
    b[safe] = 1.0 / std[safe]

    frozen_names = [name for name, is_safe in zip(feature_order, safe) if not bool(is_safe)]
    return {
        "enabled": True,
        "method": method,
        "eps": eps,
        "feature_order": list(feature_order),
        "a": a.astype(np.float32).tolist(),
        "b": b.astype(np.float32).tolist(),
        "mean_train": mean.astype(np.float32).tolist(),
        "std_train": std.astype(np.float32).tolist(),
        "frozen_features": frozen_names,
    }


def _resolve_config_path(path_raw: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_raw)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _load_input_affine_from_summary(
    *,
    summary_path: Path,
    feature_order: list[str],
) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Input-scaling summary file not found: {summary_path}")
    summary = _load_yaml(summary_path)
    input_affine = summary.get("input_scaling", {})
    if not isinstance(input_affine, dict):
        raise ValueError(f"train_summary input_scaling must be a dictionary: {summary_path}")
    if not bool(input_affine.get("enabled", False)):
        return {
            "enabled": False,
            "method": "none",
        }

    stats_feature_order = list(input_affine.get("feature_order", []))
    if not stats_feature_order:
        raise ValueError(f"input_scaling.feature_order missing in {summary_path}")
    missing = [name for name in feature_order if name not in stats_feature_order]
    if missing:
        raise KeyError(
            f"Input-scaling summary {summary_path} missing features {missing}. "
            f"Available: {stats_feature_order}"
        )

    a_all = np.asarray(input_affine.get("a", []), dtype=np.float32).reshape(-1)
    b_all = np.asarray(input_affine.get("b", []), dtype=np.float32).reshape(-1)
    if a_all.size != len(stats_feature_order) or b_all.size != len(stats_feature_order):
        raise ValueError(
            f"input_scaling dimension mismatch in {summary_path}: "
            f"feature_order={len(stats_feature_order)} a={a_all.size} b={b_all.size}"
        )
    lookup = {name: idx for idx, name in enumerate(stats_feature_order)}
    order_idx = [lookup[name] for name in feature_order]

    out = dict(input_affine)
    out["enabled"] = True
    out["method"] = "from_train_summary"
    out["source_summary_file"] = str(summary_path)
    out["feature_order"] = list(feature_order)
    out["a"] = a_all[order_idx].astype(np.float32).tolist()
    out["b"] = b_all[order_idx].astype(np.float32).tolist()
    for key in ("mean_train", "std_train"):
        if key in input_affine:
            values = np.asarray(input_affine.get(key, []), dtype=np.float32).reshape(-1)
            if values.size == len(stats_feature_order):
                out[key] = values[order_idx].astype(np.float32).tolist()
    return out


def _build_input_affine(
    *,
    x_train: dict[str, np.ndarray],
    feature_order: list[str],
    scaling_cfg: dict,
    output_dir: Path,
) -> dict | None:
    enabled = bool(scaling_cfg.get("enabled", False))
    if not enabled:
        return None

    method = str(scaling_cfg.get("method", "standardize_train")).strip().lower()
    if method in {"standardize_train"}:
        return _build_input_affine_from_train(
            x_train=x_train,
            feature_order=feature_order,
            scaling_cfg=scaling_cfg,
        )
    if method in {"from_train_summary", "frozen_train_summary"}:
        summary_raw = (
            scaling_cfg.get("summary_file")
            or scaling_cfg.get("train_summary_file")
            or scaling_cfg.get("source_summary_file")
        )
        if summary_raw is None:
            raise KeyError(
                "data.input_scaling.method='from_train_summary' requires "
                "'summary_file' or 'train_summary_file'."
            )
        loaded = _load_input_affine_from_summary(
            summary_path=_resolve_config_path(summary_raw, base_dir=output_dir),
            feature_order=feature_order,
        )
        return loaded if bool(loaded.get("enabled", False)) else None

    raise ValueError(
        f"Unsupported data.input_scaling.method='{method}'. "
        "Use 'standardize_train' or 'from_train_summary'."
    )


def _to_torch_input_affine(
    *,
    input_affine: dict | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if input_affine is None:
        return None
    return {
        "a": torch.tensor(input_affine["a"], dtype=torch.float32, device=device),
        "b": torch.tensor(input_affine["b"], dtype=torch.float32, device=device),
    }


def _load_checkpoint_state(path: Path, *, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Initial checkpoint file not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint payload type {type(payload)!r}: {path}")
    return payload


def _load_state_dict_flexible(
    *,
    model: nn.Module,
    state: dict,
) -> dict:
    """
    Load a checkpoint into a changed architecture where possible.

    Exact-shape tensors are copied normally. If the target tensor has a larger
    first dimension and all remaining dimensions match, the source tensor is
    copied into the leading slice. This supports initializing a multi-output
    head from a scalar-price checkpoint by copying the old price output into
    head 0 while leaving the new Greek heads randomly initialized.
    """

    current = model.state_dict()
    merged = dict(current)
    copied: list[str] = []
    partial: list[str] = []
    skipped: list[str] = []
    for key, source_value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        target_value = current[key]
        source_tensor = torch.as_tensor(source_value, dtype=target_value.dtype, device=target_value.device)
        if source_tensor.shape == target_value.shape:
            merged[key] = source_tensor
            copied.append(key)
            continue
        if (
            source_tensor.ndim == target_value.ndim
            and source_tensor.ndim >= 1
            and source_tensor.shape[0] <= target_value.shape[0]
            and source_tensor.shape[1:] == target_value.shape[1:]
        ):
            updated = target_value.clone()
            updated[: source_tensor.shape[0]] = source_tensor
            merged[key] = updated
            partial.append(key)
            continue
        skipped.append(key)
    model.load_state_dict(merged, strict=True)
    return {
        "copied": copied,
        "partial": partial,
        "skipped": skipped,
    }


def _build_loader(x: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0. Got {batch_size}.")
    ds = TensorDataset(torch.from_numpy(x))
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)


def _cycle_loader(loader: DataLoader):
    while True:
        for (xb,) in loader:
            yield xb


def _format_seconds(sec: float) -> str:
    sec_i = int(max(0.0, sec))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_torch_save(*, obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _evaluate_epoch_loss(
    *,
    model: nn.Module,
    loss_config: dict,
    device: torch.device,
    x_val: dict[str, np.ndarray],
    input_affine: dict[str, torch.Tensor] | None,
) -> tuple[float, PINNLossTerms]:
    model.eval()
    with torch.enable_grad():
        batch_payload = {
            "interior": torch.from_numpy(x_val["interior"]).to(device),
            "terminal": torch.from_numpy(x_val["terminal"]).to(device),
            "lower": torch.from_numpy(x_val["lower"]).to(device),
        }
        total, terms = compute_weighted_pinn_loss(
            model=model,
            loss_config=loss_config,
            batch_payload=batch_payload,
            input_affine=input_affine,
        )
    return float(total.detach().item()), terms


def _save_loss_curves(*, history_df: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if history_df.empty:
        return outputs

    epochs = history_df["epoch"].to_numpy()

    fig_total = figures_dir / "loss_curve.png"
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(epochs, history_df["train_total"].to_numpy(), label="train_total")
    plt.plot(epochs, history_df["val_total"].to_numpy(), label="val_total")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("weighted loss (log)")
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_total, dpi=300)
    plt.close()
    outputs["loss_curve"] = str(fig_total)

    fig_components = figures_dir / "loss_components_curve.png"
    component_specs = [
        ("pde", "PDE"),
        ("dpde", "Derivative PDE"),
        ("greek_delta", "Delta consistency"),
        ("greek_gamma", "Gamma consistency"),
        ("greek_vega", "Vega consistency"),
        ("term", "Terminal"),
        ("low", "Lower boundary"),
    ]
    component_specs = [
        (key, label)
        for key, label in component_specs
        if f"train_{key}" in history_df.columns and f"val_{key}" in history_df.columns
        and (
            np.nanmax(
                np.concatenate(
                    [
                        history_df[f"train_{key}"].to_numpy(dtype=np.float64),
                        history_df[f"val_{key}"].to_numpy(dtype=np.float64),
                    ]
                )
            )
            > 0.0
        )
    ]
    _, axes = plt.subplots(1, len(component_specs), figsize=(4.8 * len(component_specs), 4.3), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, component_specs):
        ax.plot(epochs, history_df[f"train_{key}"].to_numpy(), label=f"train_{key}")
        ax.plot(epochs, history_df[f"val_{key}"].to_numpy(), label=f"val_{key}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{label} loss (log)")
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(fig_components, dpi=300)
    plt.close()
    outputs["loss_components_curve"] = str(fig_components)

    fig_lr = figures_dir / "learning_rate_curve.png"
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(epochs, history_df["lr"].to_numpy())
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_lr, dpi=300)
    plt.close()
    outputs["learning_rate_curve"] = str(fig_lr)

    return outputs


def _save_pde_residual_zone_map(
    *,
    model: nn.Module,
    device: torch.device,
    x_val_interior: np.ndarray,
    figures_dir: Path,
    input_affine: dict[str, torch.Tensor] | None,
    n_bins_m: int = 24,
    n_bins_tau: int = 24,
) -> dict[str, str]:
    if x_val_interior.shape[0] == 0:
        raise ValueError("Cannot build PDE residual map with empty validation interior set.")

    model.eval()
    with torch.enable_grad():
        x_t = torch.from_numpy(x_val_interior).to(device)
        residual = compute_heston_pde_residual(
            model=model,
            x_interior=x_t,
            input_affine=input_affine,
        )
        abs_residual = residual.detach().abs().cpu().numpy().reshape(-1)

    tau = x_val_interior[:, 0]
    moneyness = x_val_interior[:, 1]
    tau_edges = np.linspace(float(tau.min()), float(tau.max()), int(n_bins_tau) + 1)
    m_edges = np.linspace(float(moneyness.min()), float(moneyness.max()), int(n_bins_m) + 1)

    tau_idx = np.clip(np.digitize(tau, bins=tau_edges, right=False) - 1, 0, n_bins_tau - 1)
    m_idx = np.clip(np.digitize(moneyness, bins=m_edges, right=False) - 1, 0, n_bins_m - 1)

    sums = np.zeros((n_bins_tau, n_bins_m), dtype=np.float64)
    counts = np.zeros((n_bins_tau, n_bins_m), dtype=np.int64)
    np.add.at(sums, (tau_idx, m_idx), abs_residual)
    np.add.at(counts, (tau_idx, m_idx), 1)

    heat = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )

    outputs: dict[str, str] = {}

    fig_path = figures_dir / "pde_residual_map_m_tau.png"
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[m_edges[0], m_edges[-1], tau_edges[0], tau_edges[-1]],
        cmap="magma",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("mean |PDE residual|")
    ax.set_xlabel("moneyness")
    ax.set_ylabel("tau")
    ax.set_title("Validation PDE Residual Map by Zone")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    outputs["pde_residual_map_m_tau"] = str(fig_path)

    valid = np.isfinite(heat) & (heat > 0.0)
    if np.any(valid):
        flat = heat[valid]
        vmin = float(np.nanpercentile(flat, 5.0))
        vmax = float(np.nanpercentile(flat, 95.0))
        vmin = max(vmin, float(np.finfo(np.float64).tiny))
        if vmax <= vmin:
            vmax = vmin * 10.0
        heat_log = np.where(np.isfinite(heat), np.maximum(heat, vmin), np.nan)

        fig_log_path = figures_dir / "pde_residual_map_m_tau_log.png"
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        im = ax.imshow(
            heat_log,
            origin="lower",
            aspect="auto",
            extent=[m_edges[0], m_edges[-1], tau_edges[0], tau_edges[-1]],
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("mean |PDE residual| (log scale)")
        ax.set_xlabel("moneyness")
        ax.set_ylabel("tau")
        ax.set_title("Validation PDE Residual Map by Zone (Log Scale)")
        plt.tight_layout()
        plt.savefig(fig_log_path, dpi=300)
        plt.close(fig)
        outputs["pde_residual_map_m_tau_log"] = str(fig_log_path)

    return outputs


class PINNTrainer:
    """
    PINN trainer (unsupervised):
    - consumes LHS collocation sets: interior / terminal / lower
    - supports: adam / sgd / lbfgs / mix_half
    - mix_half: Adam (with LR scheduler) then LBFGS at half training.
    """

    def __init__(self, *, output_dir: Path, training_config: dict):
        self.output_dir = Path(output_dir)
        self.training_config = dict(training_config)

    def train(self, *, model_config: dict, dataset_manifest: dict) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        collocation_manifest_file = dataset_manifest.get("collocation_manifest_file")
        if collocation_manifest_file is None:
            raise KeyError("dataset_manifest must include key 'collocation_manifest_file'")

        collocation_manifest_path = Path(collocation_manifest_file)
        if not collocation_manifest_path.exists():
            raise FileNotFoundError(
                f"Collocation manifest file not found: {collocation_manifest_path}"
            )

        collocation_manifest = _load_yaml(collocation_manifest_path)
        datasets_cfg = collocation_manifest.get("datasets", {})
        required_keys = ("interior", "terminal", "lower")
        missing = [key for key in required_keys if key not in datasets_cfg]
        if missing:
            raise KeyError(
                f"Collocation manifest missing datasets keys {missing}. "
                f"Available: {list(datasets_cfg.keys())}"
            )

        feature_order = collocation_manifest.get("feature_order")
        if not isinstance(feature_order, list) or not feature_order:
            raise ValueError(
                "Collocation manifest must include a non-empty 'feature_order' list."
            )

        x_interior = _load_parquet_matrix(
            path=Path(datasets_cfg["interior"]),
            feature_order=feature_order,
        )
        x_terminal = _load_parquet_matrix(
            path=Path(datasets_cfg["terminal"]),
            feature_order=feature_order,
        )
        x_lower = _load_parquet_matrix(
            path=Path(datasets_cfg["lower"]),
            feature_order=feature_order,
        )

        meta_cfg = self.training_config.get("meta", {})
        loop_cfg = self.training_config.get("loop", {})
        data_cfg = self.training_config.get("data", {})
        loss_cfg = self.training_config.get("loss", {})
        cb_ckpt_cfg = self.training_config.get("callbacks", {}).get("checkpoint", {})

        seed = int(meta_cfg.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = _resolve_device(meta_cfg.get("device", "auto"))
        epochs = int(loop_cfg.get("epochs", 200))
        batch_size_collocation = int(loop_cfg.get("batch_size_collocation", 2048))
        batch_size_boundary = int(loop_cfg.get("batch_size_boundary", 512))
        val_fraction = float(data_cfg.get("val_fraction", 0.2))
        input_scaling_cfg = data_cfg.get("input_scaling", {})
        if not isinstance(input_scaling_cfg, dict):
            raise ValueError("data.input_scaling must be a dictionary when provided.")
        log_every = int(loop_cfg.get("log_every", 50))
        if log_every <= 0:
            log_every = 1

        mode = _normalize_training_mode(meta_cfg.get("optimizer", "adam"))
        supported_modes = {"adam", "sgd", "lbfgs", "mix_half"}
        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported meta.optimizer '{mode}'. "
                f"Use one of: {sorted(supported_modes)}"
            )

        x_interior_train, x_interior_val = _split_array(
            x_interior, val_fraction=val_fraction, seed=seed
        )
        x_terminal_train, x_terminal_val = _split_array(
            x_terminal, val_fraction=val_fraction, seed=seed + 1
        )
        x_lower_train, x_lower_val = _split_array(
            x_lower, val_fraction=val_fraction, seed=seed + 2
        )

        x_train = {
            "interior": x_interior_train,
            "terminal": x_terminal_train,
            "lower": x_lower_train,
        }
        x_val = {
            "interior": x_interior_val,
            "terminal": x_terminal_val,
            "lower": x_lower_val,
        }
        input_affine_np = _build_input_affine(
            x_train=x_train,
            feature_order=feature_order,
            scaling_cfg=input_scaling_cfg,
            output_dir=self.output_dir,
        )

        model = build_pinn_model(model_config).to(device)
        configure_input_affine = getattr(model, "configure_input_affine", None)
        if callable(configure_input_affine):
            configure_input_affine(input_affine_np)
        input_affine = _to_torch_input_affine(
            input_affine=input_affine_np,
            device=device,
        )

        initial_checkpoint_raw = (
            meta_cfg.get("initial_checkpoint")
            or meta_cfg.get("warm_start_checkpoint")
            or self.training_config.get("initial_checkpoint")
        )
        initial_checkpoint_path: Path | None = None
        initial_checkpoint_report: dict | None = None
        if initial_checkpoint_raw:
            initial_checkpoint_path = _resolve_config_path(
                initial_checkpoint_raw,
                base_dir=self.output_dir,
            )
            initial_state = _load_checkpoint_state(initial_checkpoint_path, device=device)
            initial_checkpoint_strict = bool(meta_cfg.get("initial_checkpoint_strict", True))
            if initial_checkpoint_strict:
                model.load_state_dict(initial_state, strict=True)
                initial_checkpoint_report = {
                    "strict": True,
                    "copied": sorted(initial_state.keys()),
                    "partial": [],
                    "skipped": [],
                }
            else:
                initial_checkpoint_report = {
                    "strict": False,
                    **_load_state_dict_flexible(model=model, state=initial_state),
                }
            if callable(configure_input_affine):
                configure_input_affine(input_affine_np)
            print(f"[PINN] warm-start checkpoint loaded: {initial_checkpoint_path}")
            if initial_checkpoint_report is not None and not initial_checkpoint_report["strict"]:
                print(
                    "[PINN] flexible warm-start: "
                    f"copied={len(initial_checkpoint_report['copied'])} "
                    f"partial={len(initial_checkpoint_report['partial'])} "
                    f"skipped={len(initial_checkpoint_report['skipped'])}"
                )

        optimizers_by_name: dict[str, torch.optim.Optimizer] = {}
        schedulers_by_name: dict[str, torch.optim.lr_scheduler.LRScheduler | None] = {}
        train_loaders_by_name: dict[str, tuple[DataLoader, DataLoader, DataLoader]] = {}

        mix_half_switch_epoch: int | None = None
        if mode == "mix_half":
            adam_cfg = _get_optimizer_cfg(self.training_config, "adam")
            lbfgs_cfg = _get_optimizer_cfg(self.training_config, "lbfgs")

            optimizers_by_name["adam"] = _build_optimizer(
                model=model,
                optimizer_name="adam",
                optimizer_cfg=adam_cfg,
            )
            optimizers_by_name["lbfgs"] = _build_optimizer(
                model=model,
                optimizer_name="lbfgs",
                optimizer_cfg=lbfgs_cfg,
            )
            schedulers_by_name["adam"] = _build_scheduler(
                training_config=self.training_config,
                optimizer_name="adam",
                optimizer=optimizers_by_name["adam"],
            )
            schedulers_by_name["lbfgs"] = None

            lbfgs_full_batch = bool(lbfgs_cfg.get("full_batch", True))
            lbfgs_batch_size = int(
                lbfgs_cfg.get(
                    "batch_size",
                    max(batch_size_collocation, min(8192, batch_size_collocation * 4)),
                )
            )
            train_loaders_by_name["adam"] = (
                _build_loader(x_train["interior"], batch_size=batch_size_collocation, shuffle=True),
                _build_loader(x_train["terminal"], batch_size=batch_size_boundary, shuffle=True),
                _build_loader(x_train["lower"], batch_size=batch_size_boundary, shuffle=True),
            )
            train_loaders_by_name["lbfgs"] = (
                _build_loader(
                    x_train["interior"],
                    batch_size=len(x_train["interior"]) if lbfgs_full_batch else lbfgs_batch_size,
                    shuffle=not lbfgs_full_batch,
                ),
                _build_loader(
                    x_train["terminal"],
                    batch_size=len(x_train["terminal"]) if lbfgs_full_batch else lbfgs_batch_size,
                    shuffle=not lbfgs_full_batch,
                ),
                _build_loader(
                    x_train["lower"],
                    batch_size=len(x_train["lower"]) if lbfgs_full_batch else lbfgs_batch_size,
                    shuffle=not lbfgs_full_batch,
                ),
            )

            default_switch = max(2, (epochs // 2) + 1)
            switch_raw = loop_cfg.get("mix_half_switch_epoch", default_switch)
            mix_half_switch_epoch = default_switch if switch_raw is None else int(switch_raw)
            if mix_half_switch_epoch < 2:
                mix_half_switch_epoch = 2
            if mix_half_switch_epoch > epochs:
                mix_half_switch_epoch = epochs + 1

            active_optimizer_name = "adam"
        else:
            opt_name = mode
            opt_cfg = _get_optimizer_cfg(self.training_config, opt_name)
            optimizers_by_name[opt_name] = _build_optimizer(
                model=model,
                optimizer_name=opt_name,
                optimizer_cfg=opt_cfg,
            )
            schedulers_by_name[opt_name] = _build_scheduler(
                training_config=self.training_config,
                optimizer_name=opt_name,
                optimizer=optimizers_by_name[opt_name],
            )

            if opt_name == "lbfgs":
                lbfgs_full_batch = bool(opt_cfg.get("full_batch", True))
                lbfgs_batch_size = int(
                    opt_cfg.get(
                        "batch_size",
                        max(batch_size_collocation, min(8192, batch_size_collocation * 4)),
                    )
                )
                train_loaders_by_name[opt_name] = (
                    _build_loader(
                        x_train["interior"],
                        batch_size=len(x_train["interior"]) if lbfgs_full_batch else lbfgs_batch_size,
                        shuffle=not lbfgs_full_batch,
                    ),
                    _build_loader(
                        x_train["terminal"],
                        batch_size=len(x_train["terminal"]) if lbfgs_full_batch else lbfgs_batch_size,
                        shuffle=not lbfgs_full_batch,
                    ),
                    _build_loader(
                        x_train["lower"],
                        batch_size=len(x_train["lower"]) if lbfgs_full_batch else lbfgs_batch_size,
                        shuffle=not lbfgs_full_batch,
                    ),
                )
            else:
                train_loaders_by_name[opt_name] = (
                    _build_loader(x_train["interior"], batch_size=batch_size_collocation, shuffle=True),
                    _build_loader(x_train["terminal"], batch_size=batch_size_boundary, shuffle=True),
                    _build_loader(x_train["lower"], batch_size=batch_size_boundary, shuffle=True),
                )

            active_optimizer_name = opt_name

        optimizer = optimizers_by_name[active_optimizer_name]
        lr_scheduler = schedulers_by_name[active_optimizer_name]

        ckpt_dir = self.output_dir / "checkpoints"
        metrics_dir = self.output_dir / "metrics"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        filename_best = str(cb_ckpt_cfg.get("filename_best", "model_best.pt"))
        filename_last = str(cb_ckpt_cfg.get("filename_last", "model_last.pt"))
        best_ckpt = ckpt_dir / filename_best
        last_ckpt = ckpt_dir / filename_last

        history_rows: list[dict] = []
        best_val_loss = float("inf")
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_t0 = time.perf_counter()
            if mode == "mix_half":
                desired = "adam" if epoch < mix_half_switch_epoch else "lbfgs"
                if desired != active_optimizer_name:
                    active_optimizer_name = desired
                    optimizer = optimizers_by_name[active_optimizer_name]
                    lr_scheduler = schedulers_by_name[active_optimizer_name]

            model.train()
            loader_interior, loader_terminal, loader_lower = train_loaders_by_name[active_optimizer_name]
            n_steps = max(len(loader_interior), len(loader_terminal), len(loader_lower))
            it_interior = _cycle_loader(loader_interior)
            it_terminal = _cycle_loader(loader_terminal)
            it_lower = _cycle_loader(loader_lower)

            train_loss_sum = 0.0
            train_pde_sum = 0.0
            train_dpde_sum = 0.0
            train_greek_delta_sum = 0.0
            train_greek_gamma_sum = 0.0
            train_greek_vega_sum = 0.0
            train_term_sum = 0.0
            train_low_sum = 0.0
            train_no_arb_sum = 0.0

            for _ in range(n_steps):
                xb_interior = next(it_interior).to(device)
                xb_terminal = next(it_terminal).to(device)
                xb_lower = next(it_lower).to(device)

                if active_optimizer_name == "lbfgs":
                    terms_holder: dict[str, PINNLossTerms] = {}

                    def closure() -> torch.Tensor:
                        optimizer.zero_grad()
                        total_local, terms_local = compute_weighted_pinn_loss(
                            model=model,
                            loss_config=loss_cfg,
                            batch_payload={
                                "interior": xb_interior,
                                "terminal": xb_terminal,
                                "lower": xb_lower,
                            },
                            input_affine=input_affine,
                        )
                        total_local.backward()
                        terms_holder["value"] = terms_local
                        return total_local

                    loss_tensor = optimizer.step(closure)
                    step_total = float(loss_tensor.item())
                    step_terms = terms_holder["value"]
                else:
                    optimizer.zero_grad()
                    loss_tensor, step_terms = compute_weighted_pinn_loss(
                        model=model,
                        loss_config=loss_cfg,
                        batch_payload={
                            "interior": xb_interior,
                            "terminal": xb_terminal,
                            "lower": xb_lower,
                        },
                        input_affine=input_affine,
                    )
                    loss_tensor.backward()
                    optimizer.step()
                    step_total = float(loss_tensor.detach().item())

                train_loss_sum += step_total
                train_pde_sum += step_terms.pde
                train_dpde_sum += step_terms.dpde
                train_greek_delta_sum += step_terms.greek_delta
                train_greek_gamma_sum += step_terms.greek_gamma
                train_greek_vega_sum += step_terms.greek_vega
                train_term_sum += step_terms.term
                train_low_sum += step_terms.low
                train_no_arb_sum += step_terms.no_arbitrage

            train_total = train_loss_sum / max(n_steps, 1)
            train_terms = PINNLossTerms(
                pde=train_pde_sum / max(n_steps, 1),
                dpde=train_dpde_sum / max(n_steps, 1),
                greek_delta=train_greek_delta_sum / max(n_steps, 1),
                greek_gamma=train_greek_gamma_sum / max(n_steps, 1),
                greek_vega=train_greek_vega_sum / max(n_steps, 1),
                term=train_term_sum / max(n_steps, 1),
                low=train_low_sum / max(n_steps, 1),
                no_arbitrage=train_no_arb_sum / max(n_steps, 1),
            )

            val_total, val_terms = _evaluate_epoch_loss(
                model=model,
                loss_config=loss_cfg,
                device=device,
                x_val=x_val,
                input_affine=input_affine,
            )

            if lr_scheduler is not None:
                lr_scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])

            history_rows.append(
                {
                    "epoch": epoch,
                    "optimizer": active_optimizer_name,
                    "lr": current_lr,
                    "train_total": float(train_total),
                    "train_pde": float(train_terms.pde),
                    "train_dpde": float(train_terms.dpde),
                    "train_greek_delta": float(train_terms.greek_delta),
                    "train_greek_gamma": float(train_terms.greek_gamma),
                    "train_greek_vega": float(train_terms.greek_vega),
                    "train_term": float(train_terms.term),
                    "train_low": float(train_terms.low),
                    "train_no_arbitrage": float(train_terms.no_arbitrage),
                    "val_total": float(val_total),
                    "val_pde": float(val_terms.pde),
                    "val_dpde": float(val_terms.dpde),
                    "val_greek_delta": float(val_terms.greek_delta),
                    "val_greek_gamma": float(val_terms.greek_gamma),
                    "val_greek_vega": float(val_terms.greek_vega),
                    "val_term": float(val_terms.term),
                    "val_low": float(val_terms.low),
                    "val_no_arbitrage": float(val_terms.no_arbitrage),
                }
            )

            if val_total < best_val_loss:
                best_val_loss = val_total
                _safe_torch_save(obj=model.state_dict(), path=best_ckpt)

            elapsed_total = time.perf_counter() - t0
            epoch_elapsed = time.perf_counter() - epoch_t0
            mean_epoch = elapsed_total / max(epoch, 1)
            eta = (epochs - epoch) * mean_epoch
            if epoch == 1 or (epoch % log_every == 0) or (epoch == epochs):
                print(
                    "[PINN] "
                    f"epoch {epoch:4d}/{epochs} | "
                    f"opt={active_optimizer_name:<6} | "
                    f"lr={current_lr:.3e} | "
                    f"train={train_total:.3e} | "
                    f"val={val_total:.3e} | "
                    f"train(pde/dpde/gD/gG/gV/term/low)=({train_terms.pde:.3e}, {train_terms.dpde:.3e}, {train_terms.greek_delta:.3e}, {train_terms.greek_gamma:.3e}, {train_terms.greek_vega:.3e}, {train_terms.term:.3e}, {train_terms.low:.3e}) | "
                    f"val(pde/dpde/gD/gG/gV/term/low)=({val_terms.pde:.3e}, {val_terms.dpde:.3e}, {val_terms.greek_delta:.3e}, {val_terms.greek_gamma:.3e}, {val_terms.greek_vega:.3e}, {val_terms.term:.3e}, {val_terms.low:.3e}) | "
                    f"best_val={best_val_loss:.3e} | "
                    f"epoch_t={_format_seconds(epoch_elapsed)} | "
                    f"elapsed={_format_seconds(elapsed_total)} | "
                    f"eta={_format_seconds(eta)}"
                )

        _safe_torch_save(obj=model.state_dict(), path=last_ckpt)

        history_df = pd.DataFrame(history_rows)
        history_path = metrics_dir / "train_history.csv"
        history_df.to_csv(history_path, index=False)

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = _save_loss_curves(history_df=history_df, figures_dir=figures_dir)

        best_state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best_state)
        model.to(device)
        pde_map_paths = _save_pde_residual_zone_map(
            model=model,
            device=device,
            x_val_interior=x_val["interior"],
            figures_dir=figures_dir,
            input_affine=input_affine,
            n_bins_m=24,
            n_bins_tau=24,
        )
        figure_paths.update(pde_map_paths)

        summary = {
            "collocation_manifest_file": str(collocation_manifest_path),
            "device": str(device),
            "optimizer_mode": mode,
            "mix_half_switch_epoch": mix_half_switch_epoch,
            "epochs": epochs,
            "batch_size_collocation": batch_size_collocation,
            "batch_size_boundary": batch_size_boundary,
            "initial_checkpoint": (
                str(initial_checkpoint_path) if initial_checkpoint_path is not None else None
            ),
            "initial_checkpoint_report": initial_checkpoint_report,
            "val_fraction": val_fraction,
            "n_train_interior": int(len(x_train["interior"])),
            "n_train_terminal": int(len(x_train["terminal"])),
            "n_train_lower": int(len(x_train["lower"])),
            "n_val_interior": int(len(x_val["interior"])),
            "n_val_terminal": int(len(x_val["terminal"])),
            "n_val_lower": int(len(x_val["lower"])),
            "input_scaling": input_affine_np
            if input_affine_np is not None
            else {
                "enabled": False,
                "method": "none",
            },
            "best_val_total": float(best_val_loss),
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
            "history_file": str(history_path),
            "figures": figure_paths,
            "total_training_seconds": float(time.perf_counter() - t0),
        }
        summary_path = metrics_dir / "train_summary.yaml"
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        return best_ckpt
