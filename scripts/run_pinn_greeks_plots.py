from __future__ import annotations

import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pinn_greeks_surface import run_surface


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pinn_greeks_plots.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _fixed_values_to_cli_text(raw: dict | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(
            "Each plot.fixed_values must be either a dictionary or a string "
            "in format 'name=value,name=value'."
        )
    items: list[str] = []
    for key, val in raw.items():
        items.append(f"{key}={float(val)}")
    return ",".join(items)


def _build_namespace(*, global_cfg: dict, plot_cfg: dict) -> Namespace:
    required = [
        "x_feature",
        "x_min",
        "x_max",
        "y_feature",
        "y_min",
        "y_max",
    ]
    missing = [k for k in required if k not in plot_cfg]
    if missing:
        raise KeyError(f"Plot entry missing required keys: {missing}")

    return Namespace(
        run_dir=str(global_cfg.get("run_dir", "latest")),
        checkpoint_name=str(global_cfg.get("checkpoint_name", "model_best.pt")),
        architecture_config=global_cfg.get("architecture_config"),
        device=str(global_cfg.get("device", "auto")),
        dtype=str(global_cfg.get("dtype", "float64")),
        feature_order=global_cfg.get("feature_order"),
        x_feature=str(plot_cfg["x_feature"]),
        x_min=float(plot_cfg["x_min"]),
        x_max=float(plot_cfg["x_max"]),
        x_points=int(plot_cfg.get("x_points", global_cfg.get("x_points", 81))),
        y_feature=str(plot_cfg["y_feature"]),
        y_min=float(plot_cfg["y_min"]),
        y_max=float(plot_cfg["y_max"]),
        y_points=int(plot_cfg.get("y_points", global_cfg.get("y_points", 81))),
        fixed_values=_fixed_values_to_cli_text(plot_cfg.get("fixed_values", global_cfg.get("fixed_values"))),
        spot_feature=str(plot_cfg.get("spot_feature", global_cfg.get("spot_feature", "moneyness"))),
        vol_feature=plot_cfg.get("vol_feature", global_cfg.get("vol_feature", "v")),
        tau_feature=plot_cfg.get("tau_feature", global_cfg.get("tau_feature", "tau")),
        rate_feature=plot_cfg.get("rate_feature", global_cfg.get("rate_feature", "r")),
        theta_sign=str(plot_cfg.get("theta_sign", global_cfg.get("theta_sign", "minus_dv_dtau"))),
        strike=plot_cfg.get("strike", global_cfg.get("strike")),
        chunk_size_values=int(plot_cfg.get("chunk_size_values", global_cfg.get("chunk_size_values", 4096))),
        chunk_size_jac=int(plot_cfg.get("chunk_size_jac", global_cfg.get("chunk_size_jac", 512))),
        chunk_size_hess=int(plot_cfg.get("chunk_size_hess", global_cfg.get("chunk_size_hess", 64))),
        metric=str(plot_cfg.get("metric", global_cfg.get("metric", "value"))),
        all_greeks=bool(plot_cfg.get("all_greeks", global_cfg.get("all_greeks", True))),
        no_plot=bool(plot_cfg.get("no_plot", global_cfg.get("no_plot", False))),
        output_csv=plot_cfg.get("output_csv"),
        plot_path=plot_cfg.get("plot_path"),
        plots_dir=plot_cfg.get("plots_dir"),
    )


def main() -> None:
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {DEFAULT_CONFIG_PATH}. "
            "Create it first or restore the template."
        )

    cfg = _load_yaml(DEFAULT_CONFIG_PATH)
    global_cfg = cfg.get("global", {})
    if not isinstance(global_cfg, dict):
        raise ValueError("config.global must be a dictionary.")

    plots = cfg.get("plots", [])
    if not isinstance(plots, list) or not plots:
        raise ValueError("config.plots must be a non-empty list.")

    print(f"Config: {DEFAULT_CONFIG_PATH}")
    print(f"Total plot jobs: {len(plots)}")
    execution_rows: list[dict] = []
    for i, plot_cfg in enumerate(plots, start=1):
        if not isinstance(plot_cfg, dict):
            raise ValueError(f"plots[{i-1}] must be a dictionary.")
        job_name = str(plot_cfg.get("name", f"job_{i}"))
        print(f"[{i}/{len(plots)}] Running {job_name} ...")
        args = _build_namespace(global_cfg=global_cfg, plot_cfg=plot_cfg)
        result = run_surface(args)
        execution_rows.append(
            {
                "job_index": i,
                "job_name": job_name,
                "job_config": dict(plot_cfg),
                "resolved_args": vars(args),
                "result": result,
            }
        )

    run_dirs = sorted({row["result"]["run_dir"] for row in execution_rows})
    for run_dir_raw in run_dirs:
        run_dir = Path(run_dir_raw)
        greeks_dir = run_dir / "greeks"
        greeks_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = greeks_dir / "plots_config_execution.yaml"
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "config_source": str(DEFAULT_CONFIG_PATH),
            "global_config": global_cfg,
            "executions": [row for row in execution_rows if row["result"]["run_dir"] == run_dir_raw],
        }
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"Saved config snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
