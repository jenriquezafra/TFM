from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from src.pinn.cann_bridge import load_cann_inputs, validate_cann_artifacts
from src.pinn.config import build_pipeline_plan_from_config, load_yaml
from src.pinn.contracts import PINNPipelinePlan
from src.pinn.data_builder import build_collocation_dataset, build_supervised_dataset
from src.pinn.evaluator import evaluate_pinn_run
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
    model_cfg = load_yaml(plan.architecture_config)
    training_cfg = load_yaml(plan.training_config)

    dataset_path: Path | None = None
    collocation_dataset_path: Path | None = None
    if _is_stage_enabled(plan, "prepare_dataset"):
        theta_star, parameter_order, quotes_df = load_cann_inputs(
            plan.cann,
            required_quote_columns=feature_columns,
        )
        if target_column in quotes_df.columns:
            dataset_path = build_supervised_dataset(
                cann_quotes_path=plan.cann.quotes_file,
                theta_star=theta_star,
                output_dir=run_dir / "data",
                feature_columns=feature_columns,
                target_column=target_column,
            )
        collocation_dataset_path = build_collocation_dataset(
            sampling_config=training_cfg.get("sampling", {}),
            output_dir=run_dir / "data",
            theta_star=theta_star,
            parameter_order=parameter_order,
        )
        stage_summary = {
            "status": "completed",
            "collocation_dataset_file": str(collocation_dataset_path),
            "target_column_requested": target_column,
            "feature_columns": list(feature_columns),
        }
        if dataset_path is not None:
            stage_summary["dataset_file"] = str(dataset_path)
        else:
            stage_summary["dataset_file"] = None
            stage_summary["dataset_status"] = "skipped_target_missing"
            stage_summary["dataset_skip_reason"] = (
                f"Target column '{target_column}' not present in CaNN quotes."
            )
        execution["stages"]["prepare_dataset"] = {
            **stage_summary,
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
            )
            if execution["stages"]["prepare_dataset"].get("status") == "skipped":
                execution["stages"]["prepare_dataset"] = {
                    "status": "completed",
                    "mode": "auto_from_train",
                    "dataset_file": str(dataset_path),
                    "target_column_requested": target_column,
                    "feature_columns": list(feature_columns),
                }

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
        eval_cfg = training_cfg.get("evaluation", {})
        eval_result = evaluate_pinn_run(
            run_dir=run_dir,
            model_config=model_cfg,
            training_config=training_cfg,
            evaluation_config=eval_cfg,
            dataset_file=dataset_path,
            checkpoint_file=(
                execution["stages"]["train"].get("best_checkpoint")
                if execution["stages"]["train"].get("status") == "completed"
                else None
            ),
            split_indices_file=run_dir / "train" / "metrics" / "split_indices.npz",
        )
        execution["stages"]["evaluate"] = {
            "status": "completed",
            "metrics_yaml": eval_result["metrics_yaml"],
            "metrics_csv": eval_result["metrics_csv"],
            "metrics_all": eval_result["metrics_all"],
            "metrics_val": eval_result["metrics_val"],
        }
    else:
        execution["stages"]["evaluate"] = {"status": "skipped"}

    summary_path = _build_execution_summary_path(plan)
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(execution, f, sort_keys=False)

    return plan
