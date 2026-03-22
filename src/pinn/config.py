from __future__ import annotations

from pathlib import Path

import yaml

from src.pinn.contracts import CaNNArtifactsSpec, PINNPipelinePlan, PipelineStage


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def build_pipeline_plan_from_config(
    *,
    project_root: Path,
    config_path: Path,
) -> PINNPipelinePlan:
    cfg_path = resolve_path(config_path, base_dir=project_root)
    cfg = load_yaml(cfg_path)

    meta = cfg.get("meta", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    outputs = cfg.get("outputs", {})
    inputs = cfg.get("inputs", {})
    cann = inputs.get("cann", {})
    pipe_cfg = cfg.get("pipeline", {})
    stage_flags = pipe_cfg.get("stages", {})

    calibration_dir = resolve_path(
        cann.get("calibration_dir", "outputs/calibration"),
        base_dir=project_root,
    )
    summary_rel = cann.get("summary_file", "summary.yaml")
    quotes_rel = cann.get("quotes_file", "quotes_comparison.parquet")

    stage_table = [
        PipelineStage(
            name="prepare_dataset",
            enabled=bool(stage_flags.get("prepare_dataset", True)),
            description=(
                "Read calibrated parameters from CaNN and build supervised + physics datasets."
            ),
            owner_module="src.pinn.data_builder",
        ),
        PipelineStage(
            name="train",
            enabled=bool(stage_flags.get("train", True)),
            description="Train the PINN with data loss and PDE residual losses.",
            owner_module="src.pinn.trainer",
        ),
        PipelineStage(
            name="evaluate",
            enabled=bool(stage_flags.get("evaluate", True)),
            description="Evaluate PINN pricing quality and physics consistency metrics.",
            owner_module="src.pinn.evaluator",
        ),
    ]

    return PINNPipelinePlan(
        project_root=project_root,
        config_path=cfg_path,
        experiment_name=str(meta.get("experiment_name", "PINN_scaffold")),
        description=str(meta.get("description", "PINN scaffold pipeline")),
        architecture_config=resolve_path(
            model.get("architecture_config", "configs/pinn_model_architecture.yaml"),
            base_dir=project_root,
        ),
        training_config=resolve_path(
            training.get("training_config", "configs/pinn_training.yaml"),
            base_dir=project_root,
        ),
        outputs_root=resolve_path(outputs.get("root_dir", "outputs/pinn"), base_dir=project_root),
        run_name=str(outputs.get("run_name", "PINN_v01")),
        plan_filename=str(outputs.get("dump_plan_filename", "pipeline_plan.yaml")),
        cann=CaNNArtifactsSpec(
            calibration_dir=calibration_dir,
            summary_file=calibration_dir / summary_rel,
            quotes_file=calibration_dir / quotes_rel,
            parameter_key=str(cann.get("parameter_key", "theta_star")),
        ),
        stages=stage_table,
    )

