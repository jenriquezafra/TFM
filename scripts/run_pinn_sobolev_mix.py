from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sobolev_prototype import build_arg_parser, run


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pinn_sobolev_mix.yaml"

SECTION_FIELDS: dict[str, set[str]] = {
    "global": {
        "run_dir",
        "checkpoint_name",
        "architecture_config",
        "device",
        "dtype",
        "output_root",
    },
    "training": {
        "epochs",
        "switch_epoch",
        "seed",
        "adam_lr",
        "weight_decay",
        "step_size",
        "gamma_lr",
        "log_every",
        "batch_size_collocation",
        "batch_size_boundary",
        "val_fraction",
    },
    "sobolev": {
        "anchor_points",
        "anchor_batch_size",
        "lambda_g1",
        "lambda_g2",
        "warmup_epochs",
        "scale_floor",
        "lbfgs_lr",
        "lbfgs_max_iter",
        "lbfgs_history_size",
        "lbfgs_tolerance_grad",
        "lbfgs_tolerance_change",
        "lbfgs_line_search_fn",
        "lbfgs_lr_decay_on_fail",
        "lbfgs_min_lr",
        "lbfgs_max_failures",
    },
    "greeks": {
        "spot_feature",
        "vol_feature",
        "tau_feature",
        "rate_feature",
        "theta_sign",
        "strike",
        "option_type",
    },
    "benchmark": {
        "benchmark_config",
        "benchmark_mape_floor",
        "chunk_size_values",
        "chunk_size_jac",
        "chunk_size_hess",
    },
}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Sobolev mix fine-tuning from YAML config.")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Sobolev mix YAML config.",
    )
    return p


def _apply_section(*, args_ns: argparse.Namespace, cfg: dict, section: str) -> None:
    raw = cfg.get(section, {})
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError(f"config.{section} must be a dictionary.")

    allowed = SECTION_FIELDS[section]
    unknown = [k for k in raw.keys() if k not in allowed]
    if unknown:
        raise KeyError(f"Unknown keys in config.{section}: {unknown}")

    for key, val in raw.items():
        setattr(args_ns, key, val)


def _build_args_from_config(config_path: Path) -> argparse.Namespace:
    cfg = _load_yaml(config_path)

    default_parser = build_arg_parser()
    args_ns = default_parser.parse_args([])

    for section in ("global", "training", "sobolev", "greeks", "benchmark"):
        _apply_section(args_ns=args_ns, cfg=cfg, section=section)

    return args_ns


def main() -> None:
    cli = _build_parser().parse_args()
    config_path = cli.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    args = _build_args_from_config(config_path)
    args.config_source = str(config_path)

    print(f"Config: {config_path}")
    print(f"Run dir: {args.run_dir}")
    print(f"Checkpoint: {args.checkpoint_name}")
    print(f"Output root: {args.output_root}")
    print(f"Epochs: {args.epochs} (switch_epoch={args.switch_epoch})")
    run(args)


if __name__ == "__main__":
    main()
