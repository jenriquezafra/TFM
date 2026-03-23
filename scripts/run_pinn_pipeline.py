import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pinn.pipeline import run_pinn_pipeline_from_config


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pinn_pipeline.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PINN pipeline (prepare_dataset + train + evaluate)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to PINN pipeline YAML config.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "prepare_dataset", "train", "evaluate"],
        default="all",
        help="Optional stage filter. Default runs all enabled stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate wiring and optionally dump plan, without executing stages.",
    )
    parser.add_argument(
        "--dump-plan",
        action="store_true",
        help="Dump resolved pipeline plan to outputs directory.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    selected_stage = None if args.stage == "all" else args.stage
    plan = run_pinn_pipeline_from_config(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        stage=selected_stage,
        dry_run=bool(args.dry_run),
        dump_plan=bool(args.dump_plan),
    )

    print("PINN pipeline scaffold")
    print(f"Config: {plan.config_path}")
    print(f"Experiment: {plan.experiment_name}")
    print(f"Description: {plan.description}")
    print(f"CaNN calibration dir: {plan.cann.calibration_dir}")
    print(f"CaNN summary: {plan.cann.summary_file}")
    print(f"CaNN quotes: {plan.cann.quotes_file}")
    print(f"CaNN parameter key: {plan.cann.parameter_key}")
    print("Stages:")
    for stage_obj in plan.stages:
        status = "enabled" if stage_obj.enabled else "disabled"
        print(
            f" - {stage_obj.name:<16} [{status}]"
            f" -> {stage_obj.owner_module}"
        )
    if plan.plan_path is not None:
        print(f"Plan file: {plan.plan_path}")
    if args.dry_run:
        print("Dry-run completed. No training/evaluation executed.")
    else:
        run_dir = plan.outputs_root / plan.run_name
        print(f"Execution completed. Run dir: {run_dir}")
        print(f"Execution summary: {run_dir / 'pipeline_execution.yaml'}")


if __name__ == "__main__":
    main()
