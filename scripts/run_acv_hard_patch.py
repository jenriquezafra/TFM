from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar
from src.pinn.acv_hard_patch import (
    RAW_FEATURE_ORDER,
    baseline_distill_loss,
    build_acv_hard_patch_model,
    global_replay_loss,
    interface_loss,
    normalized_pde_loss,
    price_label_loss,
    stencil_price_loss,
    terminal_loss,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "acv_hard_patch_experimental.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _parse_dtype(raw: str) -> torch.dtype:
    key = str(raw).strip().lower()
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError("dtype must be one of {'float64', 'float32'}")


def _resolve_run_dir(cfg: dict) -> Path:
    outputs = cfg.get("outputs", {})
    root = Path(outputs.get("root_dir", "outputs/pinn"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_name = str(outputs.get("run_name", "acv_hard_patch_experimental"))
    return root / run_name


def _bounds(domain: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = domain.get(key, list(default))
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"domain.{key} must be [low, high]")
    low, high = float(raw[0]), float(raw[1])
    if high < low:
        raise ValueError(f"domain.{key} requires high >= low, got {raw}")
    if high == low:
        high = float(np.nextafter(low, np.inf))
    return low, high


def _uniform(rng: np.random.Generator, n: int, bounds: tuple[float, float]) -> np.ndarray:
    return rng.uniform(bounds[0], bounds[1], size=int(n)).astype(np.float64)


def _log_uniform(rng: np.random.Generator, n: int, bounds: tuple[float, float]) -> np.ndarray:
    low = max(float(bounds[0]), np.finfo(np.float64).tiny)
    high = max(float(bounds[1]), low * (1.0 + 1.0e-12))
    return np.exp(rng.uniform(np.log(low), np.log(high), size=int(n))).astype(np.float64)


def _domain_cfg(cfg: dict) -> dict:
    sampling = cfg.get("sampling", {})
    domain = sampling.get("domain", {})
    heston = domain.get("heston_params", {})
    return {
        "tau": _bounds(domain, "tau", (0.0, 3.0)),
        "moneyness": _bounds(domain, "moneyness", (0.01, 2.0)),
        "v": _bounds(domain, "v", (0.01, 0.5)),
        "r": _bounds(domain, "r", (0.0, 0.05)),
        "rho": _bounds(heston, "rho", (-0.9, 0.0)),
        "kappa": _bounds(heston, "kappa", (0.0, 3.0)),
        "gamma": _bounds(heston, "gamma", (0.01, 0.8)),
        "bar_v": _bounds(heston, "bar_v", (0.01, 0.5)),
    }


def _region_cfg(cfg: dict) -> dict:
    sampling = cfg.get("sampling", {})
    hard = sampling.get("hard_region", {})
    patch = sampling.get("patch_region", {})
    return {
        "tau_min": float(sampling.get("tau_min", 1.0e-4)),
        "hard_x": float(hard.get("x_abs", 0.03)),
        "hard_tau": float(hard.get("tau_max", 0.05)),
        "patch_x": float(patch.get("x_abs", 0.06)),
        "patch_tau": float(patch.get("tau_max", 0.08)),
        "buffer_x": float(patch.get("buffer_x_abs", 0.08)),
        "buffer_tau": float(patch.get("buffer_tau_max", 0.10)),
        "z_min": float(sampling.get("z_sampling", {}).get("z_min", -4.0)),
        "z_max": float(sampling.get("z_sampling", {}).get("z_max", 4.0)),
    }


def _sample_params(
    *,
    rng: np.random.Generator,
    n: int,
    domain: dict,
) -> dict[str, np.ndarray]:
    return {
        "v": _uniform(rng, n, domain["v"]),
        "rho": _uniform(rng, n, domain["rho"]),
        "kappa": _uniform(rng, n, domain["kappa"]),
        "gamma": _uniform(rng, n, domain["gamma"]),
        "bar_v": _uniform(rng, n, domain["bar_v"]),
        "r": _uniform(rng, n, domain["r"]),
    }


def _raw_from_tau_x_params(tau: np.ndarray, x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.column_stack(
        [
            tau,
            np.exp(x),
            params["v"],
            params["rho"],
            params["kappa"],
            params["gamma"],
            params["bar_v"],
            params["r"],
        ]
    )
    return raw.astype(np.float64, copy=False)


def _sample_raw(
    *,
    rng: np.random.Generator,
    n: int,
    cfg: dict,
    block: str,
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.empty((0, len(RAW_FEATURE_ORDER)), dtype=np.float64)

    domain = _domain_cfg(cfg)
    region = _region_cfg(cfg)
    params = _sample_params(rng=rng, n=n, domain=domain)
    block_key = str(block).strip().lower()

    if block_key == "hard_core":
        tau = _log_uniform(rng, n, (region["tau_min"], region["hard_tau"]))
        x = rng.uniform(-region["hard_x"], region["hard_x"], size=n)
        return _raw_from_tau_x_params(tau, x, params)

    if block_key == "patch":
        tau = _log_uniform(rng, n, (region["tau_min"], region["patch_tau"]))
        x = rng.uniform(-region["patch_x"], region["patch_x"], size=n)
        return _raw_from_tau_x_params(tau, x, params)

    if block_key == "z_layer":
        tau = _log_uniform(rng, n, (region["tau_min"], region["patch_tau"]))
        z = rng.uniform(region["z_min"], region["z_max"], size=n)
        x = z * np.sqrt(np.maximum(params["v"] * tau, 0.0))
        x = np.clip(x, -region["patch_x"], region["patch_x"])
        return _raw_from_tau_x_params(tau, x, params)

    if block_key == "buffer":
        tau = _log_uniform(rng, n, (region["tau_min"], region["buffer_tau"]))
        x = rng.uniform(-region["buffer_x"], region["buffer_x"], size=n)
        shell = rng.random(n) < 0.5
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        x_shell = signs * rng.uniform(region["hard_x"], region["buffer_x"], size=n)
        tau_shell = rng.uniform(region["hard_tau"], region["buffer_tau"], size=n)
        x = np.where(shell, x_shell, x)
        tau = np.where(shell, tau, tau_shell)
        return _raw_from_tau_x_params(tau, x, params)

    if block_key == "terminal":
        tau = np.zeros(n, dtype=np.float64)
        x = rng.uniform(-region["buffer_x"], region["buffer_x"], size=n)
        return _raw_from_tau_x_params(tau, x, params)

    if block_key == "interface":
        half = n // 2
        tau = _log_uniform(rng, n, (region["tau_min"], region["buffer_tau"]))
        x = rng.uniform(-region["patch_x"], region["patch_x"], size=n)
        signs = rng.choice(np.array([-1.0, 1.0]), size=half)
        x[:half] = signs * region["patch_x"]
        tau[half:] = region["patch_tau"]
        return _raw_from_tau_x_params(tau, x, params)

    if block_key == "global":
        m_min, m_max = domain["moneyness"]
        m_min = max(m_min, 1.0e-6)
        tau = _uniform(rng, n, domain["tau"])
        m = _uniform(rng, n, (m_min, m_max))
        x = np.log(m)
        return _raw_from_tau_x_params(tau, x, params)

    raise ValueError(
        "block must be one of {'hard_core', 'patch', 'z_layer', 'buffer', "
        "'terminal', 'interface', 'global'}"
    )


def _sample_mixed_pde(*, rng: np.random.Generator, n: int, cfg: dict) -> np.ndarray:
    mix = cfg.get("sampling", {}).get("pde_mix", {})
    weights = {
        "hard_core": float(mix.get("hard_core", 0.35)),
        "z_layer": float(mix.get("z_layer", 0.20)),
        "buffer": float(mix.get("buffer", 0.30)),
        "global": float(mix.get("global", 0.15)),
    }
    total_w = sum(max(0.0, x) for x in weights.values())
    if total_w <= 0.0:
        raise ValueError("sampling.pde_mix weights must sum to a positive value")
    counts = {name: int(round(n * max(0.0, w) / total_w)) for name, w in weights.items()}
    delta = n - sum(counts.values())
    counts["hard_core"] += delta
    parts = [
        _sample_raw(rng=rng, n=count, cfg=cfg, block=name)
        for name, count in counts.items()
        if count > 0
    ]
    out = np.concatenate(parts, axis=0)
    rng.shuffle(out)
    return out[:n]


def _to_tensor(raw: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(raw, dtype=dtype, device=device)


def _cf_settings(cfg: dict) -> HestonCFGreeksSettings:
    cf = cfg.get("cf_integration", {})
    return HestonCFGreeksSettings(
        u_min=float(cf.get("u_min", 1.0e-6)),
        u_max=float(cf.get("u_max", 200.0)),
        n_u=int(cf.get("n_u", 1200)),
    )


def _cf_put_price(row: np.ndarray, settings: HestonCFGreeksSettings) -> float:
    return heston_cf_greeks_scalar(
        option_type="put",
        S0=float(row[1]),
        K=1.0,
        tau=float(row[0]),
        r=float(row[7]),
        rho=float(row[3]),
        kappa=float(row[4]),
        gamma=float(row[5]),
        bar_v=float(row[6]),
        v0=float(row[2]),
        settings=settings,
    )["price"]


def _build_price_labels(
    *,
    cfg: dict,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    label_cfg = cfg.get("price_labels", {})
    if not bool(label_cfg.get("enabled", False)):
        return None
    n = int(label_cfg.get("n_points", 1024))
    if n <= 0:
        return None
    settings = _cf_settings(label_cfg)
    raw = _sample_raw(rng=rng, n=n, cfg=cfg, block=str(label_cfg.get("sampling_block", "patch")))
    prices = np.array([_cf_put_price(row, settings) for row in raw], dtype=np.float64).reshape(-1, 1)
    return (
        torch.as_tensor(raw, dtype=dtype, device=device),
        torch.as_tensor(prices, dtype=dtype, device=device),
    )


def _build_stencil_labels(
    *,
    cfg: dict,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    stencil_cfg = cfg.get("stencil_labels", {})
    if not bool(stencil_cfg.get("enabled", False)):
        return None
    n = int(stencil_cfg.get("n_points", 512))
    if n <= 0:
        return None
    settings = _cf_settings(stencil_cfg)
    centers = _sample_raw(rng=rng, n=n, cfg=cfg, block=str(stencil_cfg.get("sampling_block", "hard_core")))
    tau = np.maximum(centers[:, 0], 0.0)
    v = np.maximum(centers[:, 2], 0.0)
    h_raw = 0.2 * np.sqrt(np.maximum(v * tau, 0.0))
    h = np.clip(
        h_raw,
        float(stencil_cfg.get("h_min", 5.0e-4)),
        float(stencil_cfg.get("h_max", 5.0e-3)),
    )
    x0 = np.log(np.maximum(centers[:, 1], 1.0e-12))
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    raw_stencil = np.repeat(centers[:, None, :], repeats=5, axis=1)
    for j, offset in enumerate(offsets):
        raw_stencil[:, j, 1] = np.exp(x0 + offset * h)
    flat = raw_stencil.reshape(n * 5, centers.shape[1])
    prices = np.array([_cf_put_price(row, settings) for row in flat], dtype=np.float64).reshape(n, 5)
    return (
        torch.as_tensor(raw_stencil, dtype=dtype, device=device),
        torch.as_tensor(prices, dtype=dtype, device=device),
        torch.as_tensor(h, dtype=dtype, device=device),
    )


def _random_batch_pair(
    data: tuple[torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, ref = data
    idx = torch.randint(raw.shape[0], (min(batch_size, raw.shape[0]),), device=raw.device)
    return raw.index_select(0, idx), ref.index_select(0, idx)


def _random_stencil_batch(
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw, ref, h = data
    idx = torch.randint(raw.shape[0], (min(batch_size, raw.shape[0]),), device=raw.device)
    return raw.index_select(0, idx), ref.index_select(0, idx), h.index_select(0, idx)


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    cfg: dict,
    stage_name: str,
    step: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "stage": stage_name,
            "step": int(step),
            "loss": float(loss),
        },
        path,
    )


def _resolve_optional_checkpoint(raw: str | Path | None, *, run_dir: Path) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    candidate = run_dir / "checkpoints" / path
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / path).resolve()


def _load_model_checkpoint(
    *,
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    strict: bool = True,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Initial ACV checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state") if isinstance(payload, dict) and "model_state" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(payload)!r}")
    missing, unexpected = model.load_state_dict(state, strict=bool(strict))
    return {
        "path": str(path),
        "strict": bool(strict),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def _safe_stage_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name).strip())
    return out or "stage"


def _format_seconds(sec: float) -> str:
    sec_i = int(max(0.0, sec))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_acv_hard_patch(
    *,
    config_path: Path,
    max_steps_per_stage: int | None = None,
    dry_run: bool = False,
) -> Path:
    cfg = _load_yaml(config_path)
    meta = cfg.get("meta", {})
    seed = int(meta.get("seed", 42))
    dtype = _parse_dtype(str(meta.get("dtype", "float64")))
    device_pref = str(meta.get("device", "auto"))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = _resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    loaded = build_acv_hard_patch_model(
        project_root=PROJECT_ROOT,
        config=cfg,
        device=device_pref,
        dtype=dtype,
    )
    model = loaded.model
    device = loaded.device
    print(f"[ACV] device={device} dtype={dtype} run_dir={run_dir}")
    print(f"[ACV] baseline={loaded.baseline.run_dir} checkpoint={loaded.baseline.checkpoint_path}")

    smoke = _to_tensor(_sample_raw(rng=rng, n=4, cfg=cfg, block="patch"), device=device, dtype=dtype)
    with torch.no_grad():
        smoke_price = model(smoke)
    if not torch.isfinite(smoke_price).all():
        raise RuntimeError("ACV smoke forward produced non-finite prices")

    train_cfg = cfg.get("training", {})
    initial_checkpoint = _resolve_optional_checkpoint(
        train_cfg.get("initial_checkpoint") or meta.get("initial_checkpoint"),
        run_dir=run_dir,
    )
    initial_checkpoint_report = None
    if initial_checkpoint is not None:
        initial_checkpoint_report = _load_model_checkpoint(
            model=model,
            path=initial_checkpoint,
            device=device,
            strict=bool(train_cfg.get("initial_checkpoint_strict", True)),
        )
        print(f"[ACV] warm-start checkpoint loaded: {initial_checkpoint}")

    if bool(train_cfg.get("save_initial_checkpoint_only", False)):
        _save_checkpoint(
            path=run_dir / "checkpoints" / "model_best.pt",
            model=model,
            cfg=cfg,
            stage_name="initial_control_variate",
            step=0,
            loss=0.0,
        )
        _save_checkpoint(
            path=run_dir / "checkpoints" / "model_last.pt",
            model=model,
            cfg=cfg,
            stage_name="initial_control_variate",
            step=0,
            loss=0.0,
        )
        summary = {
            "experiment_id": str(meta.get("experiment_id", "acv_hard_patch_control_variate")),
            "status": "completed",
            "mode": "save_initial_checkpoint_only",
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "device": str(device),
            "dtype": str(dtype),
            "best_loss": 0.0,
            "best_stage": "initial_control_variate",
            "n_steps": 0,
            "best_checkpoint": str(run_dir / "checkpoints" / "model_best.pt"),
            "last_checkpoint": str(run_dir / "checkpoints" / "model_last.pt"),
            "baseline_run_dir": str(loaded.baseline.run_dir),
            "baseline_checkpoint": str(loaded.baseline.checkpoint_path),
        }
        if initial_checkpoint_report is not None:
            summary["initial_checkpoint"] = initial_checkpoint_report
        with open(run_dir / "metrics" / "train_summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        pd.DataFrame([], columns=["global_step", "stage", "local_step", "total", "lr"]).to_csv(
            run_dir / "metrics" / "train_history.csv",
            index=False,
        )
        print(f"[ACV] saved initial checkpoint only: {run_dir}")
        return run_dir

    if dry_run:
        print("[ACV] dry-run completed; no training executed.")
        return run_dir

    price_data = _build_price_labels(cfg=cfg, rng=rng, device=device, dtype=dtype)
    stencil_data = _build_stencil_labels(cfg=cfg, rng=rng, device=device, dtype=dtype)

    stages = train_cfg.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("training.stages must be a non-empty list")
    default_batch = int(train_cfg.get("batch_size", 256))
    default_lr = float(train_cfg.get("learn_rate", 1.0e-3))
    default_wd = float(train_cfg.get("weight_decay", 0.0))
    log_every = int(train_cfg.get("log_every", 25))
    grad_clip = train_cfg.get("grad_clip_norm", None)
    grad_clip = None if grad_clip is None else float(grad_clip)
    best_loss = float("inf")
    best_stage_name = "none"
    history: list[dict] = []
    global_step = 0
    t0 = time.perf_counter()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("ACV model has no trainable parameters")

    for stage_idx, stage_cfg in enumerate(stages, start=1):
        if not bool(stage_cfg.get("enabled", True)):
            continue
        stage_name = str(stage_cfg.get("name", f"stage_{stage_idx}"))
        steps = int(stage_cfg.get("steps", 0))
        if max_steps_per_stage is not None:
            steps = min(steps, int(max_steps_per_stage))
        if steps <= 0:
            print(f"[ACV] skip stage={stage_name}: steps={steps}")
            continue

        batch_size = int(stage_cfg.get("batch_size", default_batch))
        optimizer = torch.optim.Adam(
            trainable,
            lr=float(stage_cfg.get("learn_rate", default_lr)),
            weight_decay=float(stage_cfg.get("weight_decay", default_wd)),
        )
        weights = stage_cfg.get("weights", {})
        pde_cfg = stage_cfg.get("pde", {})
        interface_cfg = stage_cfg.get("interface", {})
        price_cfg = stage_cfg.get("price_labels", {})
        stencil_cfg = stage_cfg.get("stencil_labels", {})

        print(f"[ACV] stage={stage_name} steps={steps} batch={batch_size}")
        stage_best_loss = float("inf")
        safe_stage = _safe_stage_name(stage_name)
        for local_step in range(1, steps + 1):
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            components: dict[str, torch.Tensor] = {}
            total = torch.zeros((), dtype=dtype, device=device)

            def add_component(name: str, value: torch.Tensor) -> None:
                nonlocal total
                weight = float(weights.get(name, 0.0))
                if weight <= 0.0:
                    return
                weighted = weight * value
                components[name] = weighted
                total = total + weighted

            if float(weights.get("distill", 0.0)) > 0.0:
                raw = _to_tensor(
                    _sample_raw(rng=rng, n=batch_size, cfg=cfg, block="patch"),
                    device=device,
                    dtype=dtype,
                )
                add_component("distill", baseline_distill_loss(model=model, raw_patch=raw))

            if float(weights.get("pde", 0.0)) > 0.0:
                raw = _to_tensor(_sample_mixed_pde(rng=rng, n=batch_size, cfg=cfg), device=device, dtype=dtype)
                loss, residual = normalized_pde_loss(
                    model=model,
                    raw=raw,
                    target=str(pde_cfg.get("target", "final")),
                    scale_epsilon=float(pde_cfg.get("scale_epsilon", 1.0e-8)),
                    huber_beta=float(pde_cfg.get("huber_beta", 1.0)),
                )
                add_component("pde", loss)
                components["pde_residual_rmse_unweighted"] = torch.sqrt(torch.mean(residual.detach() ** 2))

            if float(weights.get("terminal", 0.0)) > 0.0:
                raw = _to_tensor(
                    _sample_raw(rng=rng, n=max(8, batch_size // 2), cfg=cfg, block="terminal"),
                    device=device,
                    dtype=dtype,
                )
                add_component("terminal", terminal_loss(model=model, raw_terminal=raw))

            if float(weights.get("interface", 0.0)) > 0.0:
                raw = _to_tensor(
                    _sample_raw(rng=rng, n=max(8, batch_size // 2), cfg=cfg, block="interface"),
                    device=device,
                    dtype=dtype,
                )
                add_component(
                    "interface",
                    interface_loss(
                        model=model,
                        raw_interface=raw,
                        lambda_x=float(interface_cfg.get("lambda_x", 0.1)),
                        lambda_v=float(interface_cfg.get("lambda_v", 0.1)),
                    ),
                )

            if float(weights.get("global_replay", 0.0)) > 0.0:
                raw = _to_tensor(
                    _sample_raw(rng=rng, n=max(8, batch_size // 2), cfg=cfg, block="global"),
                    device=device,
                    dtype=dtype,
                )
                add_component("global_replay", global_replay_loss(model=model, raw_replay=raw))

            if float(weights.get("price", 0.0)) > 0.0:
                if price_data is None:
                    raise ValueError("Stage uses weights.price > 0 but price_labels.enabled=false")
                raw, ref = _random_batch_pair(price_data, batch_size=int(price_cfg.get("batch_size", batch_size)))
                add_component(
                    "price",
                    price_label_loss(
                        model=model,
                        raw=raw,
                        ref_price=ref,
                        alpha=float(price_cfg.get("alpha", 0.5)),
                        floor=float(price_cfg.get("floor", 1.0e-4)),
                    ),
                )

            if float(weights.get("stencil", 0.0)) > 0.0:
                if stencil_data is None:
                    raise ValueError("Stage uses weights.stencil > 0 but stencil_labels.enabled=false")
                raw_st, ref_st, h_st = _random_stencil_batch(
                    stencil_data,
                    batch_size=int(stencil_cfg.get("batch_size", max(8, batch_size // 4))),
                )
                add_component(
                    "stencil",
                    stencil_price_loss(
                        model=model,
                        raw_stencil=raw_st,
                        ref_price=ref_st,
                        h_x=h_st,
                        mode=str(stencil_cfg.get("mode", "price")),
                        epsilon=float(stencil_cfg.get("epsilon", 1.0e-12)),
                    ),
                )

            if len([k for k in components if not k.endswith("_unweighted")]) == 0:
                raise ValueError(f"Stage {stage_name} has no active positive loss weights")
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite ACV loss at stage={stage_name} step={local_step}")

            total.backward()
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
            optimizer.step()

            loss_value = float(total.detach().item())
            row = {
                "global_step": global_step,
                "stage": stage_name,
                "local_step": local_step,
                "total": loss_value,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for key, val in components.items():
                row[key] = float(val.detach().item())
            history.append(row)

            if loss_value < stage_best_loss:
                stage_best_loss = loss_value
                best_loss = loss_value
                best_stage_name = stage_name
                _save_checkpoint(
                    path=run_dir / "checkpoints" / f"model_best_{safe_stage}.pt",
                    model=model,
                    cfg=cfg,
                    stage_name=stage_name,
                    step=global_step,
                    loss=loss_value,
                )
                _save_checkpoint(
                    path=run_dir / "checkpoints" / "model_best.pt",
                    model=model,
                    cfg=cfg,
                    stage_name=stage_name,
                    step=global_step,
                    loss=loss_value,
                )

            if local_step == 1 or local_step % log_every == 0 or local_step == steps:
                elapsed = _format_seconds(time.perf_counter() - t0)
                bits = [
                    f"[ACV] {stage_name} step={local_step}/{steps}",
                    f"global={global_step}",
                    f"loss={loss_value:.6e}",
                    f"stage_best={stage_best_loss:.6e}",
                    f"elapsed={elapsed}",
                ]
                for key in ("distill", "pde", "terminal", "interface", "global_replay", "price", "stencil"):
                    if key in row:
                        bits.append(f"{key}={row[key]:.3e}")
                print(" ".join(bits))

    _save_checkpoint(
        path=run_dir / "checkpoints" / "model_last.pt",
        model=model,
        cfg=cfg,
        stage_name=str(history[-1]["stage"]) if history else "none",
        step=int(history[-1]["global_step"]) if history else 0,
        loss=float(history[-1]["total"]) if history else float("nan"),
    )
    history_df = pd.DataFrame(history)
    history_path = run_dir / "metrics" / "train_history.csv"
    history_df.to_csv(history_path, index=False)
    summary = {
        "experiment_id": str(meta.get("experiment_id", "acv_hard_patch_experimental")),
        "status": "completed",
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "dtype": str(dtype),
        "best_loss": float(best_loss),
        "best_stage": str(best_stage_name),
        "n_steps": int(len(history)),
        "history_file": str(history_path),
        "best_checkpoint": str(run_dir / "checkpoints" / "model_best.pt"),
        "last_checkpoint": str(run_dir / "checkpoints" / "model_last.pt"),
        "baseline_run_dir": str(loaded.baseline.run_dir),
        "baseline_checkpoint": str(loaded.baseline.checkpoint_path),
    }
    if initial_checkpoint_report is not None:
        summary["initial_checkpoint"] = initial_checkpoint_report
    with open(run_dir / "metrics" / "train_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"[ACV] completed run_dir={run_dir}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train experimental ACV-HardPatch model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to ACV-HardPatch YAML config.",
    )
    parser.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Full config is unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load baseline, build patch, and run a small forward pass without training.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train_acv_hard_patch(
        config_path=args.config,
        max_steps_per_stage=args.max_steps_per_stage,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
