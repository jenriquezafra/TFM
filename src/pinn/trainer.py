from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.pinn.model import build_pinn_model


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


def _split_indices(
    n_samples: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to create train/val split.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1). Got {val_fraction}.")

    idx = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = int(round(n_samples * val_fraction))
    n_val = max(1, min(n_val, n_samples - 1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def _build_optimizer(model: nn.Module, optim_cfg: dict) -> torch.optim.Optimizer:
    name = str(optim_cfg.get("name", "adam")).lower()
    lr = float(optim_cfg.get("learn_rate", 1.0e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.0))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(
        f"Unsupported optimizer '{optim_cfg.get('name')}'. "
        "Use 'adam' or 'sgd' for this minimal trainer."
    )


class PINNTrainer:
    """
    Minimal trainer: supervised only (MSE), no PDE terms yet.
    """

    def __init__(self, *, output_dir: Path, training_config: dict):
        self.output_dir = Path(output_dir)
        self.training_config = dict(training_config)

    def train(self, *, model_config: dict, dataset_manifest: dict) -> Path:
        dataset_file = dataset_manifest.get("dataset_file")
        if dataset_file is None:
            raise KeyError("dataset_manifest must include key 'dataset_file'")

        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        arrays = np.load(dataset_path)
        x = arrays["X"].astype(np.float32)
        y = arrays["y"].astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected X as 2D array, got shape {x.shape}")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(f"Expected y as shape [N,1], got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"X/y row mismatch: {x.shape[0]} vs {y.shape[0]}")

        meta_cfg = self.training_config.get("meta", {})
        loop_cfg = self.training_config.get("loop", {})
        data_cfg = self.training_config.get("data", {})
        cb_cfg = self.training_config.get("callbacks", {}).get("checkpoint", {})
        opt_list = self.training_config.get("optimizers", [])

        seed = int(meta_cfg.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = _resolve_device(meta_cfg.get("device", "auto"))
        epochs = int(loop_cfg.get("epochs", 200))
        batch_size = int(loop_cfg.get("batch_size_supervised", 256))
        val_fraction = float(data_cfg.get("val_fraction", 0.2))

        optim_cfg = opt_list[0] if opt_list else {"name": "adam", "learn_rate": 1.0e-3}

        train_idx, val_idx = _split_indices(
            x.shape[0],
            val_fraction=val_fraction,
            seed=seed,
        )

        x_train = torch.from_numpy(x[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        x_val = torch.from_numpy(x[val_idx]).to(device)
        y_val = torch.from_numpy(y[val_idx]).to(device)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        model = build_pinn_model(model_config).to(device)
        loss_fn = nn.MSELoss()
        optimizer = _build_optimizer(model, optim_cfg)

        ckpt_dir = self.output_dir / "checkpoints"
        metrics_dir = self.output_dir / "metrics"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        filename_best = str(cb_cfg.get("filename_best", "model_best.pt"))
        filename_last = str(cb_cfg.get("filename_last", "model_last.pt"))
        best_ckpt = ckpt_dir / filename_best
        last_ckpt = ckpt_dir / filename_last

        history_rows = []
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

                batch_size_eff = int(xb.shape[0])
                train_loss_sum += float(loss.item()) * batch_size_eff
                train_count += batch_size_eff

            train_loss = train_loss_sum / max(train_count, 1)

            model.eval()
            with torch.inference_mode():
                val_pred = model(x_val)
                val_loss = float(loss_fn(val_pred, y_val).item())

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_mse": float(train_loss),
                    "val_mse": float(val_loss),
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)

        torch.save(model.state_dict(), last_ckpt)

        history_df = pd.DataFrame(history_rows)
        history_path = metrics_dir / "train_history.csv"
        history_df.to_csv(history_path, index=False)

        split_indices_path = metrics_dir / "split_indices.npz"
        np.savez(
            split_indices_path,
            train_idx=train_idx.astype(np.int64),
            val_idx=val_idx.astype(np.int64),
        )

        summary = {
            "dataset_file": str(dataset_path),
            "device": str(device),
            "epochs": epochs,
            "batch_size_supervised": batch_size,
            "val_fraction": val_fraction,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "optimizer": {
                "name": str(optim_cfg.get("name", "adam")),
                "learn_rate": float(optim_cfg.get("learn_rate", 1.0e-3)),
                "weight_decay": float(optim_cfg.get("weight_decay", 0.0)),
            },
            "best_val_mse": float(best_val_loss),
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
            "history_file": str(history_path),
            "split_indices_file": str(split_indices_path),
        }
        summary_path = metrics_dir / "train_summary.yaml"
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        return best_ckpt
