from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_pinn_baseline_diagnostics


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "global_acv_pinn_diagnostics.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run diagnostics for the experimental global ACV residual PINN."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to diagnostics YAML config.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    old_argv = list(sys.argv)
    try:
        sys.argv = [
            "run_pinn_baseline_diagnostics.py",
            "--config",
            str(config_path),
        ]
        run_pinn_baseline_diagnostics.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
