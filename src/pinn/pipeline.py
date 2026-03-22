from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from src.pinn.cann_bridge import load_cann_inputs, validate_cann_artifacts
from src.pinn.config import build_pipeline_plan_from_config, load_yaml
from src.pinn.contracts import PINNPipelinePlan
from src.pinn.data_builder import build_supervised_dataset
from src.pinn.trainer import PINNTrainer


SUPPORTED_STAGES = {"prepare_dataset", "train", "evaluate"}


def _assert_required_files(plan: PINNPipelinePlan) -> None:
    required_files = [
        plan.architecture_config,
        plan.training_config,
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing PINN config files: {missing_str}")


def dump_plan_yaml(plan: PINNPipelinePlan) -> Path:
    run_dir = plan.outputs_root / plan.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / plan.plan_filename

    with open(plan_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plan.to_dict(), f, sort_keys=False)

    return plan_path


def _is_stage_enabled(plan: PINNPipelinePlan, stage_name: str) -> bool:
    return any(stage.name == stage_name and stage.enabled for stage in plan.stages)


def _build_execution_summary_path(plan: PINNPipelinePlan) -> Path:
    run_dir = plan.outputs_root / plan.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "pipeline_execution.yaml"


def run_pinn_pipeline_from_config(
    *,
    project_root: Path,
    config_path: Path,
    stage: str | None = None,
    dry_run: bool = False,
    dump_plan: bool = True,
) -> PINNPipelinePlan:
    """
    Build and validate a PINN pipeline plan.
    """
    if stage is not None and stage not in SUPPORTED_STAGES:
        raise ValueError(
            f"Unsupported stage '{stage}'. Valid values: {sorted(SUPPORTED_STAGES)}"
        )

    plan = build_pipeline_plan_from_config(
        project_root=project_root,
        config_path=config_path,
    )

    _assert_required_files(plan)
    validate_cann_artifacts(plan.cann)

    if stage is not None:
        for item in plan.stages:
            item.enabled = bool(item.enabled and item.name == stage)

    if dump_plan:
        plan.plan_path = dump_plan_yaml(plan)

    if dry_run:
        return plan

    cfg = load_yaml(plan.config_path)
    inputs_cfg = cfg.get("inputs", {})
    cann_cfg = inputs_cfg.get("cann", {})
    pricing_cfg = inputs_cfg.get("pricing_context", {})

    cann_columns = cann_cfg.get("columns", {})
    col_m = str(cann_columns.get("moneyness", "moneyness"))
    col_tau = str(cann_columns.get("tau", "tau"))
    col_r = str(cann_columns.get("r", "r"))
    fallback_target = str(cann_columns.get("iv_market", "iv_market"))

    target_column = str(pricing_cfg.get("target_name", "price_market"))
    feature_columns = (col_m, col_tau, col_r)

    run_dir = plan.outputs_root / plan.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    execution = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "config_path": str(plan.config_path),
        "run_dir": str(run_dir),
        "stages": {},
    }

    dataset_path: Path | None = None
    if _is_stage_enabled(plan, "prepare_dataset"):
        theta_star, _, _ = load_cann_inputs(
            plan.cann,
            required_quote_columns=feature_columns,
        )
        dataset_path = build_supervised_dataset(
            cann_quotes_path=plan.cann.quotes_file,
            theta_star=theta_star,
            output_dir=run_dir / "data",
            feature_columns=feature_columns,
            target_column=target_column,
            fallback_target_columns=(fallback_target,),
        )
        execution["stages"]["prepare_dataset"] = {
            "status": "completed",
            "dataset_file": str(dataset_path),
            "target_column_requested": target_column,
            "feature_columns": list(feature_columns),
        }
    else:
        execution["stages"]["prepare_dataset"] = {"status": "skipped"}

    if _is_stage_enabled(plan, "train"):
        if dataset_path is None:
            dataset_path = run_dir / "data" / "supervised_dataset.npz"
        if not dataset_path.exists():
            theta_star, _, _ = load_cann_inputs(
                plan.cann,
                required_quote_columns=feature_columns,
            )
            dataset_path = build_supervised_dataset(
                cann_quotes_path=plan.cann.quotes_file,
                theta_star=theta_star,
                output_dir=run_dir / "data",
                feature_columns=feature_columns,
                target_column=target_column,
                fallback_target_columns=(fallback_target,),
            )
            if execution["stages"]["prepare_dataset"].get("status") == "skipped":
                execution["stages"]["prepare_dataset"] = {
                    "status": "completed",
                    "mode": "auto_from_train",
                    "dataset_file": str(dataset_path),
                    "target_column_requested": target_column,
                    "feature_columns": list(feature_columns),
                }

        model_cfg = load_yaml(plan.architecture_config)
        training_cfg = load_yaml(plan.training_config)
        trainer = PINNTrainer(
            output_dir=run_dir / "train",
            training_config=training_cfg,
        )
        best_ckpt = trainer.train(
            model_config=model_cfg,
            dataset_manifest={"dataset_file": str(dataset_path)},
        )
        execution["stages"]["train"] = {
            "status": "completed",
            "best_checkpoint": str(best_ckpt),
        }
    else:
        execution["stages"]["train"] = {"status": "skipped"}

    if _is_stage_enabled(plan, "evaluate"):
        execution["stages"]["evaluate"] = {
            "status": "pending",
            "message": "Evaluate stage is not implemented yet.",
        }
    else:
        execution["stages"]["evaluate"] = {"status": "skipped"}

    summary_path = _build_execution_summary_path(plan)
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(execution, f, sort_keys=False)

    return plan
