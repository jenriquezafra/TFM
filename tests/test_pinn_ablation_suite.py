from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.build_pinn_ablation_study import build_ablation_study
from scripts.run_pinn_baseline_diagnostics import _build_surface_grid
from src.greeks.chain_rule import log_moneyness_delta_gamma_to_moneyness
from src.pinn.data_builder import build_lhs_pinn_sets
from src.pinn.losses import compute_heston_pde_residual


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _metrics_frame(scale: float) -> pd.DataFrame:
    rows = []
    variables = ["price", "delta", "gamma", "vega", "theta", "rho", "pde_residual"]
    for region in ["full", "hard"]:
        for variable in variables:
            base = 1.0 if variable != "price" else 0.1
            if variable == "pde_residual":
                base = 0.2
            rows.append(
                {
                    "region": region,
                    "variable": variable,
                    "rmse": base * scale,
                    "p99_abs_error": 2.0 * base * scale,
                    "mae": 0.8 * base * scale,
                    "mse": (base * scale) ** 2,
                }
            )
    return pd.DataFrame(rows)


def _points_frame(scale: float) -> pd.DataFrame:
    tau = np.array([0.01, 0.05, 0.1])
    m = np.array([0.97, 1.0, 1.03])
    rows = []
    for tau_i in tau:
        for m_i in m:
            row = {"tau": tau_i, "moneyness": m_i}
            for greek in ["delta", "gamma", "vega", "theta", "rho"]:
                row[f"abs_error_{greek}"] = scale * (abs(m_i - 1.0) + tau_i + 1.0e-3)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_run(root: Path, key: str, scale: float) -> Path:
    run_dir = root / key
    diag_dir = run_dir / "greeks" / "baseline_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    _metrics_frame(scale).to_csv(diag_dir / "metrics_by_region.csv", index=False)
    _points_frame(scale).to_csv(diag_dir / "points_baseline_diagnostics.csv", index=False)
    _write_yaml(run_dir / "train" / "metrics" / "train_summary.yaml", {"total_training_seconds": 1.0})
    return run_dir


def test_ablation_study_scores_and_pairwise_figure(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "abl00_baseline", 1.0)
    variant = _write_run(tmp_path, "abl01_variant", 0.5)
    manifest = {
        "suite_id": "fixture_suite",
        "baseline_key": "abl00_baseline",
        "diagnostics": {
            "output_subdir": "baseline_diagnostics",
            "hard_region": {"epsilon_m": 0.03, "epsilon_tau": 0.05},
        },
        "variants": [
            {"key": "abl00_baseline", "label": "Baseline", "role": "baseline", "run_dir": str(baseline)},
            {"key": "abl01_variant", "label": "Variant", "run_dir": str(variant)},
        ],
    }
    manifest_path = tmp_path / "suite.yaml"
    _write_yaml(manifest_path, manifest)

    outputs = build_ablation_study(manifest_path=manifest_path, output_dir=tmp_path / "out")

    scores = pd.read_csv(outputs["scores_csv"])
    row = scores[scores["key"] == "abl01_variant"].iloc[0]
    assert row["status"] == "global_improvement"
    assert row["global_score"] < 1.0
    assert row["hard_score"] < 1.0
    assert (outputs["pairwise_dir"] / "abl01_variant_greek_error_maps_vs_baseline.png").exists()


def test_log_moneyness_delta_gamma_chain_rule() -> None:
    x = torch.tensor([-0.2, 0.0, 0.3], dtype=torch.float64)
    m = torch.exp(x)
    u_x = 3.0 * torch.exp(3.0 * x)
    u_xx = 9.0 * torch.exp(3.0 * x)

    delta, gamma = log_moneyness_delta_gamma_to_moneyness(u_x=u_x, u_xx=u_xx, x=x)

    assert torch.allclose(delta, 3.0 * m**2)
    assert torch.allclose(gamma, 6.0 * m)


def test_log_moneyness_collocation_manifest_and_columns(tmp_path: Path) -> None:
    manifest_path = build_lhs_pinn_sets(
        sampling_config={
            "strategy": "lhs_static",
            "mode": "fixed_theta",
            "seed": 7,
            "coordinate_space": "log_moneyness",
            "sizes": {"n_interior": 8, "n_terminal": 4, "n_lower": 4},
            "domain": {
                "tau": [0.0, 0.25],
                "moneyness": [0.05, 2.0],
                "v": [0.02, 0.10],
                "r": [0.0, 0.02],
            },
            "boundaries": {"lower_moneyness": 0.05},
        },
        output_dir=tmp_path,
        theta_star=[-0.7, 2.0, 0.3, 0.04],
        parameter_order=["rho", "kappa", "gamma", "bar_v"],
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["coordinate_space"] == "log_moneyness"
    assert manifest["feature_order"][1] == "log_moneyness"

    interior = pd.read_parquet(manifest["datasets"]["interior"])
    assert "log_moneyness" in interior.columns
    m = np.exp(interior["log_moneyness"].to_numpy(dtype=np.float64))
    assert float(m.min()) >= 0.05
    assert float(m.max()) <= 2.0


def test_log_moneyness_pde_matches_moneyness_pde_for_same_surface() -> None:
    class MoneynessSurface(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, 1:2] ** 2

    class LogMoneynessSurface(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.exp(2.0 * x[:, 1:2])

    x_m = torch.tensor(
        [[0.4, 1.15, 0.07, -0.6, 1.8, 0.25, 0.05, 0.015]],
        dtype=torch.float64,
    )
    x_log = x_m.clone()
    x_log[:, 1] = torch.log(x_log[:, 1])

    residual_m = compute_heston_pde_residual(
        model=MoneynessSurface(),
        x_interior=x_m,
        coordinate="moneyness",
    )
    residual_log = compute_heston_pde_residual(
        model=LogMoneynessSurface(),
        x_interior=x_log,
        coordinate="log_moneyness",
    )

    assert torch.allclose(residual_m, residual_log, atol=1.0e-12, rtol=1.0e-12)


def test_diagnostics_surface_grid_maps_moneyness_to_log_coordinate() -> None:
    feature_order = ["tau", "log_moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
    grid = _build_surface_grid(
        feature_order=feature_order,
        diagnostics_cfg={
            "surface_grid": {
                "x_feature": "moneyness",
                "x_min": 0.8,
                "x_max": 1.2,
                "x_points": 3,
                "y_feature": "tau",
                "y_min": 0.1,
                "y_max": 0.2,
                "y_points": 2,
            },
            "fixed_values": {"moneyness": 1.0, "v": 0.04},
        },
    )

    assert grid.shape == (6, 8)
    assert np.allclose(np.exp(grid[:, 1]).reshape(3, 2)[:, 0], [0.8, 1.0, 1.2])
