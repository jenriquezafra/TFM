"""Train a PINN from a resolved pipeline configuration.

This helper intentionally avoids importing the full pipeline runner. On macOS
MPS, importing plotting/data tooling before Torch has occasionally made the MPS
backend unavailable in this project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _resolve(path_like: str | Path, *, project_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_config", type=Path)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    sys.path.insert(0, str(project_root))

    from src.pinn.trainer import PINNTrainer

    pipeline_path = _resolve(args.pipeline_config, project_root=project_root)
    pipeline_cfg = _load_yaml(pipeline_path)

    model_cfg_path = _resolve(
        pipeline_cfg.get("model", {}).get(
            "architecture_config",
            "configs/pinn_model_architecture.yaml",
        ),
        project_root=project_root,
    )
    training_cfg_path = _resolve(
        pipeline_cfg.get("training", {}).get(
            "training_config",
            "configs/pinn_training.yaml",
        ),
        project_root=project_root,
    )
    model_cfg = _load_yaml(model_cfg_path)
    training_cfg = _load_yaml(training_cfg_path)

    outputs_cfg = pipeline_cfg.get("outputs", {})
    root_dir = _resolve(outputs_cfg.get("root_dir", "outputs/pinn"), project_root=project_root)
    run_name = str(outputs_cfg.get("run_name", "PINN_v01"))
    run_dir = root_dir / run_name

    sampling_cfg = training_cfg.get("sampling", {})
    sampling_output = sampling_cfg.get("output_dir")
    if sampling_output:
        sampling_dir = _resolve(sampling_output, project_root=project_root)
    else:
        sampling_dir = project_root / "data" / "synth" / run_name
    collocation_manifest = sampling_dir / "collocation_sets_manifest.yaml"
    if not collocation_manifest.exists():
        raise FileNotFoundError(f"Collocation manifest not found: {collocation_manifest}")

    print(f"[PINN] pipeline: {pipeline_path}")
    print(f"[PINN] architecture: {model_cfg_path}")
    print(f"[PINN] training: {training_cfg_path}")
    print(f"[PINN] output_dir: {run_dir / 'train'}")
    print(f"[PINN] collocation: {collocation_manifest}")

    trainer = PINNTrainer(output_dir=run_dir / "train", training_config=training_cfg)
    trainer.train(
        model_config=model_cfg,
        dataset_manifest={"collocation_manifest_file": str(collocation_manifest)},
    )


if __name__ == "__main__":
    main()
