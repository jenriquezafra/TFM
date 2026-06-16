from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.pinn.config import load_yaml


DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "pinn_greek_ablation_suite.yaml"
VALID_STEPS = {"prepare", "train", "diagnose", "compare"}


def _resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = deepcopy(base)
        for key, value in override.items():
            out[key] = _deep_merge(out.get(key), value)
        return out
    return deepcopy(override)


def _suite_id(manifest: dict) -> str:
    return str(manifest.get("suite_id", "pinn_greek_ablation_suite"))


def _enabled_variants(manifest: dict, selected: set[str] | None) -> list[dict]:
    variants = manifest.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("Ablation manifest must include a non-empty variants list.")
    out = []
    for item in variants:
        variant = dict(item)
        key = str(variant.get("key", ""))
        if not key:
            raise KeyError(f"Variant without key: {variant}")
        if not bool(variant.get("enabled", True)):
            continue
        if selected is not None and key not in selected:
            continue
        out.append(variant)
    if not out:
        raise ValueError("No enabled variants selected.")
    return out


def _mode_overrides(manifest: dict, mode: str) -> dict:
    modes = manifest.get("modes", {})
    if not isinstance(modes, dict):
        return {}
    value = modes.get(mode, {})
    return value if isinstance(value, dict) else {}


def _run_name_for_key(manifest: dict, key: str, mode: str) -> str:
    suite_id = _suite_id(manifest)
    if mode == "full":
        return f"{suite_id}/{key}"
    return f"{suite_id}_{mode}/{key}"


def _run_name(manifest: dict, variant: dict, mode: str) -> str:
    return _run_name_for_key(manifest, str(variant["key"]), mode)


def _resolved_root(manifest: dict, mode: str) -> Path:
    raw = manifest.get("resolved_config_root")
    if raw:
        return _resolve_path(raw) / mode
    return PROJECT_ROOT / "outputs" / "pinn" / "ablation_suites" / _suite_id(manifest) / "resolved_configs" / mode


def _prepare_variant_configs(
    *,
    manifest: dict,
    variant: dict,
    mode: str,
) -> dict[str, Path]:
    base_cfg = manifest.get("base_configs", {})
    if not isinstance(base_cfg, dict):
        raise ValueError("manifest.base_configs must be a dictionary.")

    architecture = load_yaml(_resolve_path(base_cfg.get("architecture", "configs/pinn_model_architecture.yaml")))
    training = load_yaml(_resolve_path(base_cfg.get("training", "configs/pinn_training_fixed_theta.yaml")))
    pipeline = load_yaml(_resolve_path(base_cfg.get("pipeline", "configs/pinn_pipeline_fixed_theta.yaml")))
    diagnostics = load_yaml(_resolve_path(base_cfg.get("diagnostics", "configs/pinn_baseline_diagnostics.yaml")))
    adaptive = load_yaml(_resolve_path(base_cfg.get("adaptive", "configs/pinn_adaptive_collocation.yaml")))

    mode_overrides = _mode_overrides(manifest, mode)
    architecture = _deep_merge(architecture, variant.get("architecture_overrides", {}))
    training = _deep_merge(training, variant.get("training_overrides", {}))
    pipeline = _deep_merge(pipeline, variant.get("pipeline_overrides", {}))
    diagnostics = _deep_merge(diagnostics, variant.get("diagnostics_overrides", {}))

    architecture = _deep_merge(architecture, mode_overrides.get("architecture_overrides", {}))
    training = _deep_merge(training, mode_overrides.get("training_overrides", {}))
    pipeline = _deep_merge(pipeline, mode_overrides.get("pipeline_overrides", {}))
    diagnostics = _deep_merge(diagnostics, mode_overrides.get("diagnostics_overrides", {}))

    key = str(variant["key"])
    run_name = _run_name(manifest, variant, mode)
    resolved_root = _resolved_root(manifest, mode) / key
    architecture_path = resolved_root / "architecture.yaml"
    training_path = resolved_root / "training.yaml"
    pipeline_path = resolved_root / "pipeline.yaml"
    diagnostics_path = resolved_root / "diagnostics.yaml"
    adaptive_path = resolved_root / "adaptive.yaml"

    training.setdefault("meta", {})
    training["meta"]["seed"] = int(manifest.get("seed", training["meta"].get("seed", 42)))
    training.setdefault("sampling", {})
    training["sampling"]["seed"] = int(manifest.get("seed", training["sampling"].get("seed", 42)))
    training["sampling"]["output_dir"] = str(PROJECT_ROOT / "data" / "synth" / _suite_id(manifest) / key / mode)

    pipeline.setdefault("meta", {})
    pipeline["meta"]["experiment_name"] = key
    pipeline["meta"]["description"] = str(variant.get("description", variant.get("label", key)))
    pipeline.setdefault("model", {})
    pipeline["model"]["architecture_config"] = str(architecture_path)
    pipeline.setdefault("training", {})
    pipeline["training"]["training_config"] = str(training_path)
    pipeline.setdefault("pipeline", {}).setdefault("stages", {})
    pipeline["pipeline"]["stages"]["evaluate"] = False
    pipeline.setdefault("outputs", {})
    pipeline["outputs"]["root_dir"] = str(PROJECT_ROOT / "outputs" / "pinn")
    pipeline["outputs"]["run_name"] = run_name

    diagnostics.setdefault("global", {})
    diagnostics["global"]["experiment_id"] = key
    diagnostics["global"]["run_dir"] = run_name
    diagnostics["global"]["architecture_config"] = str(architecture_path)
    diagnostics.setdefault("diagnostics", {})
    diagnostics["diagnostics"]["output_subdir"] = str(
        variant.get(
            "diagnostics_subdir",
            manifest.get("diagnostics", {}).get("output_subdir", "baseline_diagnostics"),
        )
    )

    _write_yaml(architecture_path, architecture)
    _write_yaml(training_path, training)
    _write_yaml(pipeline_path, pipeline)
    _write_yaml(diagnostics_path, diagnostics)

    if str(variant.get("workflow", "pipeline")).strip().lower() == "adaptive_collocation":
        base_variant = str(variant.get("base_variant", manifest.get("baseline_key", "abl00_baseline")))
        adaptive = _deep_merge(adaptive, variant.get("adaptive_overrides", {}))
        adaptive = _deep_merge(adaptive, mode_overrides.get("adaptive_overrides", {}))
        adaptive.setdefault("base", {})
        adaptive["base"]["run_dir"] = str(PROJECT_ROOT / "outputs" / "pinn" / _run_name_for_key(manifest, base_variant, mode))
        adaptive["base"]["checkpoint"] = None
        adaptive["base"]["train_summary"] = None
        adaptive["base"]["collocation_manifest"] = None
        adaptive.setdefault("model", {})
        adaptive["model"]["architecture_config"] = str(architecture_path)
        adaptive.setdefault("training", {})
        adaptive["training"]["training_config"] = str(training_path)
        adaptive["training"]["output_dir"] = str(PROJECT_ROOT / "outputs" / "pinn" / _run_name_for_key(manifest, key, mode) / "train")
        adaptive.setdefault("adaptive", {})
        adaptive["adaptive"]["output_dir"] = str(PROJECT_ROOT / "data" / "synth" / _suite_id(manifest) / key / f"{mode}_adaptive")
        adaptive["adaptive"]["seed"] = int(manifest.get("seed", adaptive["adaptive"].get("seed", 42)))
        _write_yaml(adaptive_path, adaptive)

    return {
        "architecture": architecture_path,
        "training": training_path,
        "pipeline": pipeline_path,
        "diagnostics": diagnostics_path,
        "adaptive": adaptive_path,
    }


def _write_resolved_manifest(
    *,
    manifest: dict,
    variants: list[dict],
    mode: str,
    config_paths: dict[str, dict[str, Path]],
) -> Path:
    resolved = deepcopy(manifest)
    resolved["mode"] = mode
    if mode != "full":
        resolved.setdefault("comparison", {})
        if isinstance(resolved["comparison"], dict):
            resolved["comparison"]["output_dir"] = str(
                PROJECT_ROOT
                / "outputs"
                / "pinn"
                / "ablation_suites"
                / _suite_id(manifest)
                / mode
            )
    resolved["variants"] = []
    for variant in variants:
        item = deepcopy(variant)
        key = str(item["key"])
        item["run_dir"] = str(PROJECT_ROOT / "outputs" / "pinn" / _run_name(manifest, item, mode))
        item["diagnostics_subdir"] = str(
            item.get("diagnostics_subdir", manifest.get("diagnostics", {}).get("output_subdir", "baseline_diagnostics"))
        )
        item["resolved_configs"] = {name: str(path) for name, path in config_paths[key].items()}
        resolved["variants"].append(item)
    path = _resolved_root(manifest, mode) / "resolved_manifest.yaml"
    _write_yaml(path, resolved)
    return path


def _run_diagnostics(config_path: Path) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pinn_baseline_diagnostics.py"), "--config", str(config_path)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _run_pipeline_stage(config_path: Path, stage: str) -> None:
    child_env = dict(os.environ)
    child_env.pop("PYTHONUNBUFFERED", None)
    if stage == "train":
        code = (
            "import sys, yaml; "
            "from pathlib import Path; "
            "from src.pinn.trainer import PINNTrainer; "
            "project=Path('.').resolve(); "
            "cfg_path=Path(sys.argv[1]); "
            "cfg=yaml.safe_load(cfg_path.read_text()) or {}; "
            "resolve=lambda raw: (Path(raw) if Path(raw).is_absolute() else project / Path(raw)); "
            "arch_path=resolve(cfg.get('model', {}).get('architecture_config', 'configs/pinn_model_architecture.yaml')); "
            "train_path=resolve(cfg.get('training', {}).get('training_config', 'configs/pinn_training.yaml')); "
            "model_cfg=yaml.safe_load(arch_path.read_text()) or {}; "
            "train_cfg=yaml.safe_load(train_path.read_text()) or {}; "
            "outputs=cfg.get('outputs', {}); "
            "run_dir=resolve(outputs.get('root_dir', 'outputs/pinn')) / str(outputs.get('run_name', 'PINN_v01')); "
            "sampling=train_cfg.get('sampling', {}); "
            "sampling_raw=sampling.get('output_dir'); "
            "sampling_dir=resolve(sampling_raw) if sampling_raw else project / 'data' / 'synth' / str(outputs.get('run_name', 'PINN_v01')); "
            "manifest=sampling_dir / 'collocation_sets_manifest.yaml'; "
            "trainer=PINNTrainer(output_dir=run_dir / 'train', training_config=train_cfg); "
            "trainer.train(model_config=model_cfg, dataset_manifest={'collocation_manifest_file': str(manifest)})"
        )
        cmd = [sys.executable, "-u", "-c", code, str(config_path)]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=child_env)
        return

    code = (
        "import sys; "
        "from pathlib import Path; "
        "from src.pinn.pipeline import run_pinn_pipeline_from_config; "
        "run_pinn_pipeline_from_config("
        "project_root=Path('.').resolve(), "
        "config_path=Path(sys.argv[1]), "
        "stage=sys.argv[2], "
        "dry_run=False, "
        "dump_plan=True)"
    )
    cmd = [
        sys.executable,
        "-u",
        "-c",
        code,
        str(config_path),
        stage,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=child_env)


def _run_adaptive(config_path: Path, stage: str) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pinn_adaptive_collocation.py"),
        "--config",
        str(config_path),
        "--stage",
        stage,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _parse_steps(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(VALID_STEPS)
    steps = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = steps - VALID_STEPS
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}. Valid steps: {sorted(VALID_STEPS)}")
    return steps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a manifest-driven PINN Greek ablation suite.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--steps", default="all", help="Comma-separated subset of prepare,train,diagnose,compare or 'all'.")
    parser.add_argument("--variants", default=None, help="Comma-separated variant keys. Default runs all enabled variants.")
    parser.add_argument("--dry-run", action="store_true", help="Write resolved configs but do not execute stages.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    manifest = load_yaml(manifest_path)
    selected = None if args.variants is None else {v.strip() for v in args.variants.split(",") if v.strip()}
    variants = _enabled_variants(manifest, selected)
    steps = _parse_steps(args.steps)

    config_paths: dict[str, dict[str, Path]] = {}
    for variant in variants:
        key = str(variant["key"])
        config_paths[key] = _prepare_variant_configs(manifest=manifest, variant=variant, mode=args.mode)
    resolved_manifest = _write_resolved_manifest(
        manifest=manifest,
        variants=variants,
        mode=args.mode,
        config_paths=config_paths,
    )

    print(f"Ablation suite: {_suite_id(manifest)}")
    print(f"Mode: {args.mode}")
    print(f"Resolved manifest: {resolved_manifest}")
    for variant in variants:
        key = str(variant["key"])
        print(f" - {key}: {config_paths[key]['pipeline']}")

    if args.dry_run:
        print("Dry-run completed. No training, diagnostics, or comparison executed.")
        return

    for variant in variants:
        key = str(variant["key"])
        workflow = str(variant.get("workflow", "pipeline")).strip().lower()
        if "prepare" in steps:
            if workflow == "adaptive_collocation":
                _run_adaptive(config_paths[key]["adaptive"], "build_dataset")
            else:
                _run_pipeline_stage(config_paths[key]["pipeline"], "prepare_dataset")
        if "train" in steps:
            if workflow == "adaptive_collocation":
                _run_adaptive(config_paths[key]["adaptive"], "train")
            else:
                _run_pipeline_stage(config_paths[key]["pipeline"], "train")
        if "diagnose" in steps:
            _run_diagnostics(config_paths[key]["diagnostics"])

    if "compare" in steps:
        from scripts.build_pinn_ablation_study import build_ablation_study

        outputs = build_ablation_study(manifest_path=resolved_manifest)
        print(f"Scores CSV: {outputs['scores_csv']}")
        print(f"Pairwise figures: {outputs['pairwise_dir']}")


if __name__ == "__main__":
    main()
