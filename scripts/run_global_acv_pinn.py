from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.pinn.global_acv_pinn import LOG_FEATURE_ORDER
from src.pinn.trainer import PINNTrainer


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "global_acv_pinn_experimental.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}.")
    return payload


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _bounds(domain: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = domain.get(key, list(default))
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"domain.{key} must be [low, high].")
    low = float(raw[0])
    high = float(raw[1])
    if high < low:
        raise ValueError(f"domain.{key} requires high >= low, got {raw}.")
    if high == low:
        high = float(np.nextafter(low, np.inf))
    return low, high


def _uniform(rng: np.random.Generator, n: int, bounds: tuple[float, float]) -> np.ndarray:
    return rng.uniform(float(bounds[0]), float(bounds[1]), size=int(n)).astype(np.float64)


def _log_uniform(rng: np.random.Generator, n: int, bounds: tuple[float, float]) -> np.ndarray:
    low = max(float(bounds[0]), np.finfo(np.float64).tiny)
    high = max(float(bounds[1]), low * (1.0 + 1.0e-12))
    return np.exp(rng.uniform(np.log(low), np.log(high), size=int(n))).astype(np.float64)


def _domain_cfg(sampling_cfg: dict) -> dict:
    domain = sampling_cfg.get("domain", {})
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


def _region_cfg(sampling_cfg: dict) -> dict:
    hard = sampling_cfg.get("hard_region", {})
    z_layer = sampling_cfg.get("z_layer", {})
    buffer = sampling_cfg.get("buffer_region", sampling_cfg.get("patch_region", {}))
    return {
        "tau_min": float(sampling_cfg.get("tau_min", 1.0e-4)),
        "hard_x": float(hard.get("x_abs", 0.03)),
        "hard_tau": float(hard.get("tau_max", 0.05)),
        "z_min": float(z_layer.get("z_min", -4.0)),
        "z_max": float(z_layer.get("z_max", 4.0)),
        "z_tau": float(z_layer.get("tau_max", 0.10)),
        "buffer_x": float(buffer.get("x_abs", 0.08)),
        "buffer_tau": float(buffer.get("tau_max", 0.10)),
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


def _from_tau_x_params(tau: np.ndarray, x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            tau,
            x,
            params["v"],
            params["rho"],
            params["kappa"],
            params["gamma"],
            params["bar_v"],
            params["r"],
        ]
    ).astype(np.float32, copy=False)


def _sample_block(
    *,
    rng: np.random.Generator,
    n: int,
    sampling_cfg: dict,
    block: str,
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.empty((0, len(LOG_FEATURE_ORDER)), dtype=np.float32)

    domain = _domain_cfg(sampling_cfg)
    region = _region_cfg(sampling_cfg)
    params = _sample_params(rng=rng, n=n, domain=domain)
    block_key = str(block).strip().lower()

    if block_key == "global":
        m_low = max(float(domain["moneyness"][0]), np.finfo(np.float64).tiny)
        m = _uniform(rng, n, (m_low, float(domain["moneyness"][1])))
        tau = _uniform(rng, n, domain["tau"])
        return _from_tau_x_params(tau, np.log(m), params)

    if block_key == "hard_core":
        tau = _log_uniform(rng, n, (region["tau_min"], region["hard_tau"]))
        x = rng.uniform(-region["hard_x"], region["hard_x"], size=n)
        return _from_tau_x_params(tau, x, params)

    if block_key == "z_layer":
        tau = _log_uniform(rng, n, (region["tau_min"], region["z_tau"]))
        z = rng.uniform(region["z_min"], region["z_max"], size=n)
        x = z * np.sqrt(np.maximum(params["v"] * tau, 0.0))
        x = np.clip(x, -region["buffer_x"], region["buffer_x"])
        return _from_tau_x_params(tau, x, params)

    if block_key == "buffer":
        tau = _log_uniform(rng, n, (region["tau_min"], region["buffer_tau"]))
        x = rng.uniform(-region["buffer_x"], region["buffer_x"], size=n)
        shell = rng.random(n) < 0.5
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        x_shell = signs * rng.uniform(region["hard_x"], region["buffer_x"], size=n)
        tau_shell = rng.uniform(region["hard_tau"], region["buffer_tau"], size=n)
        x = np.where(shell, x_shell, x)
        tau = np.where(shell, tau, tau_shell)
        return _from_tau_x_params(tau, x, params)

    if block_key == "terminal":
        tau = np.zeros(n, dtype=np.float64)
        m_low = max(float(domain["moneyness"][0]), np.finfo(np.float64).tiny)
        m = _uniform(rng, n, (m_low, float(domain["moneyness"][1])))
        return _from_tau_x_params(tau, np.log(m), params)

    if block_key == "lower":
        boundary_cfg = sampling_cfg.get("boundaries", {})
        boundary_cfg = boundary_cfg if isinstance(boundary_cfg, dict) else {}
        lower_m = float(boundary_cfg.get("lower_moneyness", domain["moneyness"][0]))
        if lower_m <= 0.0:
            raise ValueError("boundaries.lower_moneyness must be > 0 for log-moneyness sampling.")
        tau = _uniform(rng, n, domain["tau"])
        x = np.full(n, np.log(lower_m), dtype=np.float64)
        return _from_tau_x_params(tau, x, params)

    raise ValueError("block must be one of {'global', 'hard_core', 'z_layer', 'buffer', 'terminal', 'lower'}.")


def _sample_mixed_interior(
    *,
    rng: np.random.Generator,
    n: int,
    sampling_cfg: dict,
) -> np.ndarray:
    mix = sampling_cfg.get("mix", {})
    weights = {
        "global": float(mix.get("global", 0.35)),
        "hard_core": float(mix.get("hard_core", 0.20)),
        "z_layer": float(mix.get("z_layer", 0.20)),
        "buffer": float(mix.get("buffer", 0.25)),
    }
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("sampling.mix weights must sum to a positive value.")
    counts = {name: int(round(int(n) * max(0.0, weight) / total)) for name, weight in weights.items()}
    counts["global"] += int(n) - sum(counts.values())
    parts = [
        _sample_block(rng=rng, n=count, sampling_cfg=sampling_cfg, block=name)
        for name, count in counts.items()
        if count > 0
    ]
    out = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    rng.shuffle(out)
    return out[: int(n)]


def _validate_matrix(name: str, data: np.ndarray) -> None:
    if data.ndim != 2 or data.shape[1] != len(LOG_FEATURE_ORDER):
        raise ValueError(f"{name} must have shape [N,{len(LOG_FEATURE_ORDER)}], got {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values.")


def _build_collocation_manifest(
    *,
    sampling_cfg: dict,
    output_dir: Path,
) -> Path:
    seed = int(sampling_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    sizes = sampling_cfg.get("sizes", {})
    n_interior = int(sizes.get("n_interior", 40_000))
    n_terminal = int(sizes.get("n_terminal", 10_000))
    n_lower = int(sizes.get("n_lower", 10_000))
    if min(n_interior, n_terminal, n_lower) <= 0:
        raise ValueError("sampling.sizes n_interior, n_terminal and n_lower must be > 0.")

    interior = _sample_mixed_interior(rng=rng, n=n_interior, sampling_cfg=sampling_cfg)
    terminal = _sample_block(rng=rng, n=n_terminal, sampling_cfg=sampling_cfg, block="terminal")
    lower = _sample_block(rng=rng, n=n_lower, sampling_cfg=sampling_cfg, block="lower")
    _validate_matrix("interior", interior)
    _validate_matrix("terminal", terminal)
    _validate_matrix("lower", lower)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "interior": output_dir / "interior.parquet",
        "terminal": output_dir / "terminal.parquet",
        "lower": output_dir / "lower.parquet",
    }
    pd.DataFrame(interior, columns=LOG_FEATURE_ORDER).to_parquet(paths["interior"], index=False)
    pd.DataFrame(terminal, columns=LOG_FEATURE_ORDER).to_parquet(paths["terminal"], index=False)
    pd.DataFrame(lower, columns=LOG_FEATURE_ORDER).to_parquet(paths["lower"], index=False)

    domain = _domain_cfg(sampling_cfg)
    manifest = {
        "dataset_format": "parquet",
        "datasets": {key: str(path) for key, path in paths.items()},
        "feature_order": list(LOG_FEATURE_ORDER),
        "coordinate_space": "log_moneyness",
        "sampling_strategy": "global_acv_mixed",
        "seed": seed,
        "sizes": {
            "n_interior": int(interior.shape[0]),
            "n_terminal": int(terminal.shape[0]),
            "n_lower": int(lower.shape[0]),
        },
        "domain": {
            "tau": [float(domain["tau"][0]), float(domain["tau"][1])],
            "moneyness": [float(domain["moneyness"][0]), float(domain["moneyness"][1])],
            "log_moneyness": [
                float(np.log(domain["moneyness"][0])),
                float(np.log(domain["moneyness"][1])),
            ],
            "v": [float(domain["v"][0]), float(domain["v"][1])],
            "r": [float(domain["r"][0]), float(domain["r"][1])],
        },
    }
    manifest_path = output_dir / "collocation_sets_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    return manifest_path


def _resolve_run_dir(cfg: dict) -> Path:
    outputs = cfg.get("outputs", {})
    root = _resolve_path(outputs.get("root_dir", "outputs/pinn"))
    run_name = str(outputs.get("run_name", "global_acv_pinn_experimental"))
    return root / run_name


def _apply_smoke_overrides(cfg: dict) -> dict:
    smoke = copy.deepcopy(cfg)
    smoke.setdefault("outputs", {})
    smoke["outputs"]["run_name"] = str(smoke["outputs"].get("run_name", "global_acv_pinn_experimental")) + "_smoke"

    sampling = smoke.setdefault("sampling", {})
    sampling["output_dir"] = "data/synth/global_acv_pinn_smoke"
    sampling["sizes"] = {"n_interior": 96, "n_terminal": 32, "n_lower": 32}

    architecture = smoke.setdefault("architecture", {})
    architecture.setdefault("hidden", {})
    architecture["hidden"]["dims"] = [16, 16]
    architecture.setdefault("global_acv", {})
    architecture["global_acv"].setdefault("fourier", {})
    architecture["global_acv"]["fourier"]["frequencies"] = 2

    training = smoke.setdefault("training", {})
    training.setdefault("meta", {})
    training["meta"]["device"] = "cpu"
    training["meta"]["optimizer"] = "adam"
    training.setdefault("loop", {})
    training["loop"]["epochs"] = 2
    training["loop"]["batch_size_collocation"] = 16
    training["loop"]["batch_size_boundary"] = 16
    training["loop"]["log_every"] = 1
    training.setdefault("data", {})
    training["data"]["val_fraction"] = 0.25
    training.setdefault("callbacks", {})
    training["callbacks"].setdefault("lr_scheduler", {})
    training["callbacks"]["lr_scheduler"]["enabled"] = False
    return smoke


def run_global_acv_pinn(
    *,
    config_path: Path,
    smoke: bool = False,
    dry_run: bool = False,
) -> Path:
    cfg = _load_yaml(config_path)
    if smoke:
        cfg = _apply_smoke_overrides(cfg)

    architecture_cfg = cfg.get("architecture", {})
    training_cfg = cfg.get("training", {})
    sampling_cfg = cfg.get("sampling", {})
    if not isinstance(architecture_cfg, dict) or not isinstance(training_cfg, dict) or not isinstance(sampling_cfg, dict):
        raise ValueError("Config must contain dictionary sections: architecture, training and sampling.")

    run_dir = _resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = run_dir / "configs"
    _write_yaml(run_dir / "run_config.yaml", cfg)
    _write_yaml(cfg_dir / "architecture.yaml", architecture_cfg)
    _write_yaml(cfg_dir / "training.yaml", training_cfg)
    _write_yaml(cfg_dir / "sampling.yaml", sampling_cfg)

    sampling_output_raw = sampling_cfg.get("output_dir", f"data/synth/{run_dir.name}")
    sampling_output_dir = _resolve_path(sampling_output_raw)
    collocation_manifest = _build_collocation_manifest(
        sampling_cfg=sampling_cfg,
        output_dir=sampling_output_dir,
    )

    execution = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "smoke": bool(smoke),
        "dry_run": bool(dry_run),
        "architecture_config": str(cfg_dir / "architecture.yaml"),
        "training_config": str(cfg_dir / "training.yaml"),
        "collocation_manifest_file": str(collocation_manifest),
    }
    _write_yaml(run_dir / "global_acv_execution.yaml", execution)

    if dry_run:
        print(f"[GlobalACV] dry-run completed. Run dir: {run_dir}")
        return run_dir

    trainer = PINNTrainer(output_dir=run_dir / "train", training_config=training_cfg)
    best_ckpt = trainer.train(
        model_config=architecture_cfg,
        dataset_manifest={"collocation_manifest_file": str(collocation_manifest)},
    )
    execution["best_checkpoint"] = str(best_ckpt)
    execution["status"] = "completed"
    _write_yaml(run_dir / "global_acv_execution.yaml", execution)
    print(f"[GlobalACV] training completed. Run dir: {run_dir}")
    print(f"[GlobalACV] best checkpoint: {best_ckpt}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experimental global ACV residual PINN.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to experiment YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CPU training job for validation.")
    parser.add_argument("--dry-run", action="store_true", help="Build configs and collocation data without training.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    run_global_acv_pinn(config_path=config_path, smoke=bool(args.smoke), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
