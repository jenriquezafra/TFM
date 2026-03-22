from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineStage:
    """
    Declarative stage for the PINN pipeline.
    """

    name: str
    enabled: bool
    description: str
    owner_module: str


@dataclass
class CaNNArtifactsSpec:
    """
    Contract for the CaNN calibration artifacts consumed by PINN.
    """

    calibration_dir: Path
    summary_file: Path
    quotes_file: Path
    parameter_key: str


@dataclass
class PINNPipelinePlan:
    """
    Fully resolved, execution-ready scaffold plan.
    """

    project_root: Path
    config_path: Path
    experiment_name: str
    description: str
    architecture_config: Path
    training_config: Path
    outputs_root: Path
    run_name: str
    plan_filename: str
    cann: CaNNArtifactsSpec
    stages: list[PipelineStage]
    plan_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "experiment_name": self.experiment_name,
            "description": self.description,
            "architecture_config": str(self.architecture_config),
            "training_config": str(self.training_config),
            "outputs_root": str(self.outputs_root),
            "run_name": self.run_name,
            "plan_filename": self.plan_filename,
            "plan_path": str(self.plan_path) if self.plan_path is not None else None,
            "cann_inputs": {
                "calibration_dir": str(self.cann.calibration_dir),
                "summary_file": str(self.cann.summary_file),
                "quotes_file": str(self.cann.quotes_file),
                "parameter_key": self.cann.parameter_key,
            },
            "pipeline": {
                "stages": [
                    {
                        "name": stage.name,
                        "enabled": bool(stage.enabled),
                        "description": stage.description,
                        "owner_module": stage.owner_module,
                    }
                    for stage in self.stages
                ]
            },
        }

