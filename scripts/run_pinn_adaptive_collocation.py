import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pinn.adaptive_collocation import build_adaptive_collocation_dataset
from src.pinn.cann_bridge import load_cann_inputs, validate_cann_artifacts
from src.pinn.config import load_yaml
from src.pinn.contracts import CaNNArtifactsSpec
from src.pinn.trainer import PINNTrainer


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pinn_adaptive_collocation.yaml"


def _resolve_path(path_raw: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_raw)
    return path if path.is_absolute() else project_root / path


def _load_base_paths(config: dict) -> dict[str, Path]:
    base_cfg = config.setdefault("base", {})
    run_dir_raw = base_cfg.get("run_dir")
    run_dir = _resolve_path(run_dir_raw) if run_dir_raw else None

    if not base_cfg.get("checkpoint"):
        if run_dir is None:
            raise KeyError("base.checkpoint is required when base.run_dir is not set.")
        base_cfg["checkpoint"] = str(run_dir / "train" / "checkpoints" / "model_best.pt")
    if not base_cfg.get("train_summary"):
        if run_dir is None:
            raise KeyError("base.train_summary is required when base.run_dir is not set.")
        base_cfg["train_summary"] = str(run_dir / "train" / "metrics" / "train_summary.yaml")

    train_summary = _resolve_path(base_cfg["train_summary"])
    summary = load_yaml(train_summary)
    if not base_cfg.get("collocation_manifest"):
        manifest_raw = summary.get("collocation_manifest_file")
        if not manifest_raw:
            raise KeyError(
                "base.collocation_manifest is required because the base train summary "
                "does not include collocation_manifest_file."
            )
        base_cfg["collocation_manifest"] = str(_resolve_path(manifest_raw))

    return {
        "checkpoint": _resolve_path(base_cfg["checkpoint"]),
        "train_summary": train_summary,
        "collocation_manifest": _resolve_path(base_cfg["collocation_manifest"]),
    }


def _load_theta_context(config: dict) -> tuple[list[float] | None, list[str] | None]:
    sampling_cfg = config.get("adaptive", {}).get("sampling", {})
    mode = str(sampling_cfg.get("mode", "parametric_theta")).strip().lower()
    if mode != "fixed_theta":
        return None, None

    inputs_cfg = config.get("inputs", {})
    cann_cfg = inputs_cfg.get("cann", {})
    if not cann_cfg:
        raise KeyError("inputs.cann is required when adaptive.sampling.mode='fixed_theta'.")

    calibration_dir = _resolve_path(cann_cfg["calibration_dir"])
    artifacts = CaNNArtifactsSpec(
        calibration_dir=calibration_dir,
        summary_file=calibration_dir / cann_cfg.get("summary_file", "summary.yaml"),
        quotes_file=calibration_dir / cann_cfg.get("quotes_file", "quotes_comparison.parquet"),
        parameter_key=str(cann_cfg.get("parameter_key", "theta_star")),
    )
    validate_cann_artifacts(artifacts)
    theta_star, parameter_order, _ = load_cann_inputs(artifacts)
    return list(theta_star), list(parameter_order)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build residual-adaptive PINN collocation points and fine-tune a base PINN."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to adaptive collocation YAML config.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "build_dataset", "train"],
        default="all",
        help="Run only one stage or the full adaptive collocation workflow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/config and print the resolved plan without generating data or training.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = load_yaml(config_path)
    paths = _load_base_paths(config)

    model_cfg_path = _resolve_path(config["model"]["architecture_config"])
    model_cfg = load_yaml(model_cfg_path)
    training_cfg_path = _resolve_path(config["training"]["training_config"])
    training_cfg = load_yaml(training_cfg_path)

    train_output_dir = _resolve_path(
        config.get("training", {}).get(
            "output_dir",
            "outputs/pinn/adaptive_collocation/train",
        )
    )
    adaptive_output_dir = _resolve_path(
        config.get("adaptive", {}).get(
            "output_dir",
            "data/synth/PINN_param_2x_v01_adaptive_collocation",
        )
    )

    print("Adaptive collocation plan")
    print(f"Config: {config_path}")
    print(f"Base checkpoint: {paths['checkpoint']}")
    print(f"Base train summary: {paths['train_summary']}")
    print(f"Base collocation manifest: {paths['collocation_manifest']}")
    print(f"Architecture config: {model_cfg_path}")
    print(f"Adaptive output dir: {adaptive_output_dir}")
    print(f"Train output dir: {train_output_dir}")
    if args.dry_run:
        print("Dry-run completed. No adaptive data or training executed.")
        return

    result = None
    if args.stage in {"all", "build_dataset"}:
        theta_star, parameter_order = _load_theta_context(config)
        result = build_adaptive_collocation_dataset(
            project_root=PROJECT_ROOT,
            config=config,
            architecture_config=model_cfg,
            theta_star=theta_star,
            parameter_order=parameter_order,
        )
        print(f"Adaptive manifest: {result.manifest_path}")
        print(f"Adaptive summary: {result.summary_path}")

    if args.stage in {"all", "train"}:
        manifest_path = (
            result.manifest_path
            if result is not None
            else adaptive_output_dir / "collocation_sets_manifest.yaml"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Adaptive manifest not found: {manifest_path}. "
                "Run --stage build_dataset first."
            )

        training_cfg.setdefault("meta", {})
        training_cfg["meta"]["initial_checkpoint"] = str(paths["checkpoint"])
        training_cfg.setdefault("data", {})
        training_cfg["data"]["input_scaling"] = {
            "enabled": True,
            "method": "from_train_summary",
            "summary_file": str(paths["train_summary"]),
        }

        train_output_dir.mkdir(parents=True, exist_ok=True)
        with open(train_output_dir / "adaptive_training_config.resolved.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(training_cfg, f, sort_keys=False)

        trainer = PINNTrainer(output_dir=train_output_dir, training_config=training_cfg)
        best_ckpt = trainer.train(
            model_config=model_cfg,
            dataset_manifest={"collocation_manifest_file": str(manifest_path)},
        )
        print(f"Adaptive fine-tune completed. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
