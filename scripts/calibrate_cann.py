import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.model_inference import list_available_run_dirs
from src.calibration.pipeline import run_calibration_from_config


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "calibration.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate Heston parameters with CaNN (frozen forward NN + DE)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to calibration config YAML.",
    )
    parser.add_argument(
        "--quotes",
        type=Path,
        default=None,
        help="Optional override for market quotes file path.",
    )
    parser.add_argument(
        "--model-dir",
        "--model",
        dest="model_dir",
        type=str,
        default=None,
        help="Run directory under outputs/runs or 'latest'. Overrides config.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available trained model run directories and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output root directory.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional metadata tag stored in summary files.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Optional DE maxiter override.",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help="Optional DE popsize override (SciPy multiplier).",
    )
    parser.add_argument(
        "--theta-true",
        type=float,
        nargs=5,
        metavar=("RHO", "KAPPA", "GAMMA", "BAR_V", "V0"),
        default=None,
        help=(
            "Optional synthetic ground-truth parameters in order "
            "[rho, kappa, gamma, bar_v, v0]. Enables parameter error artifacts."
        ),
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        default=None,
        help=(
            "Optional YAML/JSON path with synthetic ground-truth parameters "
            "(expects key 'theta_true'). Enables parameter error artifacts."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_models:
        runs = list_available_run_dirs(PROJECT_ROOT)
        if not runs:
            print("No trained model run directories found under outputs/runs.")
            return
        print("Available model runs (newest first):")
        for run in runs:
            print(f" - {run.name}")
        return

    run_calibration_from_config(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        quotes_override=args.quotes,
        model_dir_override=args.model_dir,
        output_dir_override=args.output_dir,
        tag_override=args.tag,
        maxiter_override=args.maxiter,
        popsize_override=args.popsize,
        theta_true_override=args.theta_true,
        truth_file_override=args.truth_file,
    )


if __name__ == "__main__":
    main()
