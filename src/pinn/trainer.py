from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

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

    # Keep scheduler only for Adam in mixed mode (same behavior as ANN training flow).
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


def _evaluate_mse_in_batches(
    *,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:stop]).to(device)
            yb = torch.from_numpy(y[start:stop]).to(device)
            pred = model(xb)
            batch_loss = loss_fn(pred, yb)
            n_batch = int(stop - start)
            total_loss += float(batch_loss.item()) * n_batch
            total_count += n_batch
    return total_loss / max(total_count, 1)


class PINNTrainer:
    """
    Supervised trainer for PINN baseline:
    - supports: adam / sgd / lbfgs / mix_half
    - mix_half: Adam (with LR scheduler) then LBFGS at half training.
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
        cb_ckpt_cfg = self.training_config.get("callbacks", {}).get("checkpoint", {})

        seed = int(meta_cfg.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = _resolve_device(meta_cfg.get("device", "auto"))
        epochs = int(loop_cfg.get("epochs", 200))
        batch_size_train = int(loop_cfg.get("batch_size_supervised", 256))
        batch_size_val = int(loop_cfg.get("batch_size_val", 8192))
        val_fraction = float(data_cfg.get("val_fraction", 0.2))

        mode = _normalize_training_mode(meta_cfg.get("optimizer", "adam"))
        supported_modes = {"adam", "sgd", "lbfgs", "mix_half"}
        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported meta.optimizer '{mode}'. "
                f"Use one of: {sorted(supported_modes)}"
            )

        train_idx, val_idx = _split_indices(
            x.shape[0],
            val_fraction=val_fraction,
            seed=seed,
        )
        x_train_np = x[train_idx]
        y_train_np = y[train_idx]
        x_val_np = x[val_idx]
        y_val_np = y[val_idx]

        # Build loaders by optimizer flavor.
        train_ds = TensorDataset(
            torch.from_numpy(x_train_np),
            torch.from_numpy(y_train_np),
        )

        optimizers_by_name: dict[str, torch.optim.Optimizer] = {}
        schedulers_by_name: dict[str, torch.optim.lr_scheduler.LRScheduler | None] = {}
        train_loaders_by_name: dict[str, DataLoader] = {}

        model = build_pinn_model(model_config).to(device)
        loss_fn = nn.MSELoss()

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
            lbfgs_batch_size = len(train_ds) if lbfgs_full_batch else int(
                lbfgs_cfg.get("batch_size", max(batch_size_train, min(8192, batch_size_train * 4)))
            )

            train_loaders_by_name["adam"] = DataLoader(
                train_ds,
                batch_size=batch_size_train,
                shuffle=True,
            )
            train_loaders_by_name["lbfgs"] = DataLoader(
                train_ds,
                batch_size=lbfgs_batch_size,
                shuffle=not lbfgs_full_batch,
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
                lbfgs_batch_size = len(train_ds) if lbfgs_full_batch else int(
                    opt_cfg.get("batch_size", max(batch_size_train, min(8192, batch_size_train * 4)))
                )
                train_loaders_by_name[opt_name] = DataLoader(
                    train_ds,
                    batch_size=lbfgs_batch_size,
                    shuffle=not lbfgs_full_batch,
                )
            else:
                train_loaders_by_name[opt_name] = DataLoader(
                    train_ds,
                    batch_size=batch_size_train,
                    shuffle=True,
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

        for epoch in range(1, epochs + 1):
            # mix_half switching policy
            if mode == "mix_half":
                desired = "adam" if epoch < mix_half_switch_epoch else "lbfgs"
                if desired != active_optimizer_name:
                    active_optimizer_name = desired
                    optimizer = optimizers_by_name[active_optimizer_name]
                    lr_scheduler = schedulers_by_name[active_optimizer_name]

            model.train()
            train_loss_sum = 0.0
            train_count = 0

            train_loader = train_loaders_by_name[active_optimizer_name]
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                if active_optimizer_name == "lbfgs":
                    def closure() -> torch.Tensor:
                        optimizer.zero_grad()
                        pred_local = model(xb)
                        loss_local = loss_fn(pred_local, yb)
                        loss_local.backward()
                        return loss_local

                    loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()

                n_batch = int(xb.shape[0])
                train_loss_sum += float(loss.item()) * n_batch
                train_count += n_batch

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss = _evaluate_mse_in_batches(
                model=model,
                x=x_val_np,
                y=y_val_np,
                batch_size=batch_size_val,
                device=device,
                loss_fn=loss_fn,
            )

            if lr_scheduler is not None:
                lr_scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])

            history_rows.append(
                {
                    "epoch": epoch,
                    "optimizer": active_optimizer_name,
                    "lr": current_lr,
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
            "optimizer_mode": mode,
            "mix_half_switch_epoch": mix_half_switch_epoch,
            "epochs": epochs,
            "batch_size_supervised": batch_size_train,
            "batch_size_val": batch_size_val,
            "val_fraction": val_fraction,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
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
