from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.build_pinn_ablation_study import build_ablation_study
from scripts.run_pinn_baseline_diagnostics import _build_surface_grid, _region_masks
from src.greeks.chain_rule import log_moneyness_delta_gamma_to_moneyness
from src.pinn.adaptive_collocation import _evaluate_candidate_scores
from src.pinn.data_builder import build_lhs_pinn_sets
from src.pinn.dynamic_collocation import refresh_dynamic_collocation_interior
from src.pinn.global_acv_pinn import GlobalACVResidualPINN, head_greeks_x_to_financial
from src.pinn.losses import compute_heston_pde_residual, compute_weighted_pinn_loss
from src.pinn.model import build_pinn_model
from src.pinn.trainer import _build_optimizer


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _metrics_frame(scale: float, *, n_points: int = 9) -> pd.DataFrame:
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
                    "n_points": n_points,
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


def _write_run(root: Path, key: str, scale: float, *, n_points: int = 9) -> Path:
    run_dir = root / key
    diag_dir = run_dir / "greeks" / "baseline_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    _metrics_frame(scale, n_points=n_points).to_csv(diag_dir / "metrics_by_region.csv", index=False)
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


def test_pricing_focused_ablation_status_uses_price_score(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "abl00_baseline", 1.0)
    variant = _write_run(tmp_path, "abl12_price_only", 0.5)
    manifest = {
        "suite_id": "fixture_suite",
        "baseline_key": "abl00_baseline",
        "diagnostics": {
            "output_subdir": "baseline_diagnostics",
            "hard_region": {"epsilon_m": 0.03, "epsilon_tau": 0.05},
        },
        "variants": [
            {"key": "abl00_baseline", "label": "Baseline", "role": "baseline", "run_dir": str(baseline)},
            {
                "key": "abl12_price_only",
                "label": "Price only",
                "metric_focus": "pricing",
                "run_dir": str(variant),
            },
        ],
    }
    manifest_path = tmp_path / "suite.yaml"
    _write_yaml(manifest_path, manifest)

    outputs = build_ablation_study(manifest_path=manifest_path, output_dir=tmp_path / "out")

    scores = pd.read_csv(outputs["scores_csv"])
    row = scores[scores["key"] == "abl12_price_only"].iloc[0]
    assert row["status"] == "pricing_improvement"
    assert row["price_score"] < 1.0


def test_ablation_study_marks_incompatible_metric_grids(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "abl00_baseline", 1.0, n_points=9)
    variant = _write_run(tmp_path, "abl13_price_grid_mismatch", 0.5, n_points=25)
    manifest = {
        "suite_id": "fixture_suite",
        "baseline_key": "abl00_baseline",
        "diagnostics": {
            "output_subdir": "baseline_diagnostics",
            "hard_region": {"epsilon_m": 0.03, "epsilon_tau": 0.05},
        },
        "variants": [
            {"key": "abl00_baseline", "label": "Baseline", "role": "baseline", "run_dir": str(baseline)},
            {
                "key": "abl13_price_grid_mismatch",
                "label": "Price grid mismatch",
                "metric_focus": "pricing",
                "run_dir": str(variant),
            },
        ],
    }
    manifest_path = tmp_path / "suite.yaml"
    _write_yaml(manifest_path, manifest)

    outputs = build_ablation_study(manifest_path=manifest_path, output_dir=tmp_path / "out")

    scores = pd.read_csv(outputs["scores_csv"])
    row = scores[scores["key"] == "abl13_price_grid_mismatch"].iloc[0]
    assert row["status"] == "incompatible_grid"
    assert not bool(row["comparison_grid_ok"])
    assert pd.isna(row["price_score"])


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


def test_kink_bulk_pde_weights_are_applied_separately() -> None:
    class SmoothLogSurface(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tau = x[:, 0:1]
            log_m = x[:, 1:2]
            v = x[:, 2:3]
            return tau**2 + torch.exp(2.0 * log_m) + v**3

    interior = torch.tensor(
        [
            [0.01, 0.0, 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [1.00, np.log(1.5), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
        ],
        dtype=torch.float64,
    )
    terminal = interior.clone()
    terminal[:, 0] = 0.0
    lower = interior.clone()
    lower[:, 1] = np.log(0.05)
    payload = {"interior": interior, "terminal": terminal, "lower": lower}

    config = {
        "pde": {"coordinate": "log_moneyness"},
        "kink_bulk": {
            "enabled": True,
            "c": 2.0,
            "tau_c": 0.15,
            "delta_x": 0.05,
            "delta_tau": 0.02,
            "ell_epsilon": 1.0e-8,
        },
        "weights": {
            "pde_bulk": 1.0,
            "pde_kink": 1.0,
            "term": 0.0,
            "low": 0.0,
            "no_arbitrage": 0.0,
        },
    }
    _, terms_equal = compute_weighted_pinn_loss(
        model=SmoothLogSurface(),
        loss_config=config,
        batch_payload=payload,
    )

    tilted = dict(config)
    tilted["weights"] = dict(config["weights"])
    tilted["weights"]["pde_kink"] = 5.0
    _, terms_tilted = compute_weighted_pinn_loss(
        model=SmoothLogSurface(),
        loss_config=tilted,
        batch_payload=payload,
    )

    assert terms_tilted.pde > terms_equal.pde


def test_normalized_pde_residual_scales_local_terms() -> None:
    class QuadraticSurface(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tau = x[:, 0:1]
            m = x[:, 1:2]
            v = x[:, 2:3]
            return tau + m**2 + v**2

    interior = torch.tensor(
        [
            [0.50, 1.0, 0.20, -0.6, 1.8, 0.25, 0.05, 0.015],
            [1.00, 1.2, 0.30, -0.6, 1.8, 0.25, 0.05, 0.015],
        ],
        dtype=torch.float64,
    )
    terminal = interior.clone()
    terminal[:, 0] = 0.0
    lower = interior.clone()
    lower[:, 1] = 0.0
    payload = {"interior": interior, "terminal": terminal, "lower": lower}
    base_config = {
        "weights": {"pde": 1.0, "term": 0.0, "low": 0.0, "no_arbitrage": 0.0}
    }
    normalized_config = {
        "pde": {"normalization": {"enabled": True, "floor": 1.0e-6}},
        "weights": {"pde": 1.0, "term": 0.0, "low": 0.0, "no_arbitrage": 0.0},
    }

    _, base_terms = compute_weighted_pinn_loss(
        model=QuadraticSurface(),
        loss_config=base_config,
        batch_payload=payload,
    )
    _, normalized_terms = compute_weighted_pinn_loss(
        model=QuadraticSurface(),
        loss_config=normalized_config,
        batch_payload=payload,
    )

    assert normalized_terms.pde < base_terms.pde


def test_adaptive_log_score_includes_financial_curvature() -> None:
    class SmoothLogSurface(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tau = x[:, 0:1]
            log_m = x[:, 1:2]
            v = x[:, 2:3]
            return tau**2 + torch.exp(2.0 * log_m) + v**3

    candidates = np.array(
        [
            [0.01, 0.0, 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.10, np.log(0.9), 0.09, -0.6, 1.8, 0.25, 0.05, 0.015],
            [1.00, np.log(1.5), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
        ],
        dtype=np.float32,
    )

    scores = _evaluate_candidate_scores(
        model=SmoothLogSurface(),
        x=candidates,
        input_affine=None,
        coordinate_space="log_moneyness",
        score_config={"residual_weight": 1.0, "curvature_weight": 0.5},
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert set(scores) == {"adaptive_score", "abs_pde_residual", "abs_financial_curvature"}
    assert scores["adaptive_score"].shape == (3,)
    assert np.all(np.isfinite(scores["adaptive_score"]))
    assert np.all(scores["abs_financial_curvature"] > 0.0)
    assert np.all(scores["adaptive_score"] >= scores["abs_pde_residual"])


def test_dynamic_collocation_refresh_mixes_kink_and_adaptive_points() -> None:
    class SmoothLogSurface(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tau = x[:, 0:1]
            log_m = x[:, 1:2]
            v = x[:, 2:3]
            return tau**2 + torch.exp(2.0 * log_m) + v**3

    current = np.array(
        [
            [0.01, 0.0, 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.10, np.log(0.9), 0.09, -0.6, 1.8, 0.25, 0.05, 0.015],
            [1.00, np.log(1.5), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.50, np.log(1.1), 0.12, -0.6, 1.8, 0.25, 0.05, 0.015],
        ],
        dtype=np.float32,
    )

    refresh = refresh_dynamic_collocation_interior(
        model=SmoothLogSurface(),
        input_affine=None,
        device=torch.device("cpu"),
        current_interior=current,
        reference_blocks=[current],
        sampling_config={
            "coordinate_space": "log_moneyness",
            "domain": {
                "tau": [0.0, 0.2],
                "moneyness": [0.8, 1.2],
                "v": [0.02, 0.12],
                "r": [0.0, 0.02],
            },
        },
        manifest_domain={},
        coordinate_space="log_moneyness",
        dynamic_config={
            "n_interior": 12,
            "fractions": {"global": 0.25, "kink": 0.25, "adaptive": 0.50},
            "hard_region": {"epsilon_m": 0.03, "epsilon_tau": 0.05},
            "kink_band": {"c": 2.0, "tau_c": 0.10, "epsilon": 1.0e-8},
            "candidate_pool": {
                "n_candidates": 32,
                "batch_size": 8,
                "region_shares": {"global": 0.25, "hard": 0.25, "atm": 0.25, "kink": 0.25},
            },
            "score": {"residual_weight": 1.0, "curvature_weight": 0.1},
            "selection": {
                "method": "weighted_without_replacement",
                "floor": 0.05,
                "min_source_shares": {"hard": 0.25, "kink": 0.25},
            },
        },
        seed=7,
        epoch=3,
    )

    assert refresh.interior.shape == (12, 8)
    assert np.isfinite(refresh.interior).all()
    m = np.exp(refresh.interior[:, 1])
    assert float(m.min()) >= 0.8
    assert float(m.max()) <= 1.2
    assert refresh.report["source_counts"]["kink"] == 3
    assert refresh.report["source_counts"]["adaptive"] == 6
    assert refresh.report["selected_counts"]


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


def test_region_masks_include_non_hard_complement() -> None:
    points = pd.DataFrame(
        {
            "moneyness": [1.0, 1.1, 1.0, 0.9],
            "tau": [0.01, 0.01, 0.10, 0.10],
        }
    )

    masks = _region_masks(
        points=points,
        spot_feature="moneyness",
        tau_feature="tau",
        epsilon_m=0.03,
        epsilon_tau=0.05,
    )

    assert masks["hard"].tolist() == [True, False, False, False]
    assert masks["non_hard"].tolist() == [False, True, True, True]
    assert np.all(masks["full"] == (masks["hard"] | masks["non_hard"]))
    assert not np.any(masks["hard"] & masks["non_hard"])


def _bounded_price_arch() -> dict:
    return {
        "input": {
            "dim": 8,
            "features": ["tau", "log_moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"],
        },
        "hidden": {
            "dims": [16, 16],
            "activation": "tanh",
            "initialization": "xavier_uniform",
        },
        "output": {"dim": 1},
        "feature_map": {
            "enabled": True,
            "mode": "kink_adapted",
            "input_coordinate": "log_moneyness",
            "moneyness_floor": 1.0e-6,
            "tau_epsilon": 1.0e-6,
            "q_epsilon": 1.0e-8,
            "q_clip": 10.0,
        },
        "bounded_price": {
            "enabled": True,
            "option_type": "put",
            "strike": 1.0,
            "sigmoid_temperature": 2.0,
            "payoff_smoothing": 1.0e-5,
            "lower_smoothing": 1.0e-5,
            "time_gate": "rational_tau",
            "time_gate_tau": 2.0e-2,
        },
    }


def test_bounded_price_pinn_enforces_put_price_bounds() -> None:
    model = build_pinn_model(_bounded_price_arch()).double()
    x = torch.tensor(
        [
            [0.05, np.log(0.8), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.50, np.log(1.0), 0.09, -0.4, 2.1, 0.30, 0.06, 0.010],
            [1.00, np.log(1.3), 0.12, -0.7, 1.4, 0.35, 0.04, 0.020],
        ],
        dtype=torch.float64,
    )

    price = model(x)
    m = torch.exp(x[:, 1:2])
    discount = torch.exp(-x[:, 7:8] * x[:, 0:1])
    lower = torch.clamp(discount - m, min=0.0)
    upper = discount

    assert torch.all(price >= lower - 1.0e-5)
    assert torch.all(price <= upper + 1.0e-12)


def test_bounded_price_pinn_anchors_terminal_payoff() -> None:
    model = build_pinn_model(_bounded_price_arch()).double()
    x = torch.tensor(
        [
            [0.0, np.log(0.82), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.0, np.log(1.15), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
        ],
        dtype=torch.float64,
    )

    price = model(x)
    payoff = torch.clamp(1.0 - torch.exp(x[:, 1:2]), min=0.0)

    assert torch.allclose(price, payoff, atol=1.0e-5, rtol=1.0e-5)


def test_bounded_price_pinn_can_hard_constrain_right_put_boundary() -> None:
    arch = _bounded_price_arch()
    arch["bounded_price"]["payoff_smoothing"] = 0.0
    arch["bounded_price"]["lower_smoothing"] = 0.0
    arch["bounded_price"]["right_boundary"] = {
        "enabled": True,
        "moneyness": 3.0,
        "start_moneyness": 2.0,
        "power": 2.0,
    }
    model = build_pinn_model(arch).double()
    x = torch.tensor(
        [
            [0.20, np.log(3.0), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [1.00, np.log(3.0), 0.12, -0.7, 1.4, 0.35, 0.04, 0.020],
        ],
        dtype=torch.float64,
    )

    price = model(x)

    assert torch.allclose(price, torch.zeros_like(price), atol=1.0e-12, rtol=0.0)


def test_bounded_price_pinn_lower_boundary_is_exact_in_moneyness_coordinate() -> None:
    arch = _bounded_price_arch()
    arch["input"]["features"] = ["tau", "moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"]
    arch["feature_map"]["input_coordinate"] = "moneyness"
    arch["bounded_price"]["payoff_smoothing"] = 0.0
    arch["bounded_price"]["lower_smoothing"] = 0.0
    model = build_pinn_model(arch).double()
    x = torch.tensor(
        [
            [0.20, 0.0, 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [1.00, 0.0, 0.12, -0.7, 1.4, 0.35, 0.04, 0.020],
        ],
        dtype=torch.float64,
    )

    price = model(x)
    target = torch.exp(-x[:, 7:8] * x[:, 0:1])

    assert torch.allclose(price, target, atol=1.0e-12, rtol=1.0e-12)


def test_bounded_price_pinn_works_with_log_pde_residual() -> None:
    model = build_pinn_model(_bounded_price_arch()).double()
    x = torch.tensor(
        [
            [0.20, np.log(0.85), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.45, np.log(1.00), 0.09, -0.4, 2.1, 0.30, 0.06, 0.010],
        ],
        dtype=torch.float64,
    )

    residual = compute_heston_pde_residual(
        model=model,
        x_interior=x,
        coordinate="log_moneyness",
    )

    assert residual.shape == (2, 1)
    assert torch.isfinite(residual).all()


def _global_acv_arch() -> dict:
    return {
        "input": {
            "dim": 8,
            "features": ["tau", "log_moneyness", "v", "rho", "kappa", "gamma", "bar_v", "r"],
        },
        "hidden": {
            "dims": [16, 16],
            "activation": "tanh",
            "initialization": "xavier_uniform",
        },
        "output": {"dim": 4},
        "global_acv": {
            "enabled": True,
            "option_type": "put",
            "strike": 1.0,
            "q_epsilon": 1.0e-8,
            "bs_epsilon": 1.0e-12,
            "time_factor": "linear_tau",
            "final_init_scale": 0.0,
            "fourier": {"frequencies": 2},
        },
    }


def _global_acv_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.20, np.log(0.85), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015],
            [0.45, np.log(1.00), 0.09, -0.4, 2.1, 0.30, 0.06, 0.010],
            [1.00, np.log(1.20), 0.12, -0.7, 1.4, 0.35, 0.04, 0.020],
        ],
        dtype=torch.float64,
    )


def test_global_acv_terminal_payoff_overrides_nonzero_residual() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    with torch.no_grad():
        model.residual_net[-1].bias.fill_(2.0)

    x = torch.tensor(
        [[0.0, np.log(0.82), 0.04, -0.6, 1.8, 0.25, 0.05, 0.015]],
        dtype=torch.float64,
    )
    pred = model(x)
    payoff = torch.clamp(1.0 - torch.exp(x[:, 1:2]), min=0.0)

    assert torch.allclose(pred, payoff, atol=1.0e-12, rtol=1.0e-12)


def test_global_acv_zero_init_matches_local_bs() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    x = _global_acv_points()

    pred = model(x)
    expected = model.local_bs_put(x)

    assert torch.allclose(pred, expected, atol=1.0e-12, rtol=1.0e-12)


def test_global_acv_forward_shapes_are_finite() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    x = _global_acv_points()

    price = model(x)
    residual_heads = model.forward_residual_all(x)
    final_heads = model.forward_all(x)

    assert price.shape == (3, 1)
    assert residual_heads.shape == (3, 4)
    assert final_heads.shape == (3, 4)
    assert torch.isfinite(price).all()
    assert torch.isfinite(residual_heads).all()
    assert torch.isfinite(final_heads).all()


def test_global_acv_works_with_log_pde_and_greek_consistency_loss() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    interior = _global_acv_points()
    terminal = interior.clone()
    terminal[:, 0] = 0.0
    lower = interior.clone()
    lower[:, 1] = np.log(0.01)

    total, terms = compute_weighted_pinn_loss(
        model=model,
        loss_config={
            "pde": {"coordinate": "log_moneyness"},
            "weights": {
                "pde": 1.0,
                "term": 0.1,
                "low": 1.0,
                "no_arbitrage": 0.0,
                "greek_delta": 0.1,
                "greek_gamma": 0.01,
                "greek_vega": 0.1,
            },
        },
        batch_payload={"interior": interior, "terminal": terminal, "lower": lower},
    )

    assert torch.isfinite(total)
    assert np.isfinite(terms.total)


def test_learned_log_loss_weights_receive_gradients() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    interior = _global_acv_points()
    terminal = interior.clone()
    terminal[:, 0] = 0.0
    lower = interior.clone()
    lower[:, 1] = np.log(0.05)
    learned_log_vars = {
        "pde": torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
        "low": torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
    }

    total, terms = compute_weighted_pinn_loss(
        model=model,
        loss_config={
            "pde": {"coordinate": "log_moneyness"},
            "weights": {
                "pde": 1.0,
                "term": 0.1,
                "low": 1.0,
                "no_arbitrage": 0.0,
            },
            "learned_weights": {
                "enabled": True,
                "terms": ["pde", "low"],
                "min_log_var": -6.0,
                "max_log_var": 6.0,
            },
        },
        batch_payload={"interior": interior, "terminal": terminal, "lower": lower},
        learned_log_vars=learned_log_vars,
    )
    total.backward()

    assert torch.isfinite(total)
    assert np.isfinite(terms.total)
    assert learned_log_vars["pde"].grad is not None
    assert learned_log_vars["low"].grad is not None
    assert torch.isfinite(learned_log_vars["pde"].grad)
    assert torch.isfinite(learned_log_vars["low"].grad)


def test_optimizer_supports_separate_learned_weight_lr() -> None:
    model = torch.nn.Linear(2, 1)
    learned = [torch.nn.Parameter(torch.tensor(0.0))]

    optimizer = _build_optimizer(
        model=model,
        optimizer_name="adam",
        optimizer_cfg={"learn_rate": 1.0e-4, "weight_decay": 0.0},
        extra_parameters=learned,
        extra_learn_rate=1.0e-5,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert optimizer.param_groups[1]["lr"] == 1.0e-5


def test_learned_log_loss_weight_prior_pulls_back_outside_clamp() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    interior = _global_acv_points()
    terminal = interior.clone()
    terminal[:, 0] = 0.0
    lower = interior.clone()
    lower[:, 1] = np.log(0.05)
    learned_log_vars = {
        "pde": torch.nn.Parameter(torch.tensor(-10.0, dtype=torch.float64)),
    }

    total, _terms = compute_weighted_pinn_loss(
        model=model,
        loss_config={
            "pde": {"coordinate": "log_moneyness"},
            "weights": {
                "pde": 1.0,
                "term": 0.0,
                "low": 0.0,
                "no_arbitrage": 0.0,
            },
            "learned_weights": {
                "enabled": True,
                "terms": ["pde"],
                "min_log_var": -3.0,
                "max_log_var": 3.0,
                "prior": {
                    "strength": 0.2,
                    "target": 0.0,
                },
            },
        },
        batch_payload={"interior": interior, "terminal": terminal, "lower": lower},
        learned_log_vars=learned_log_vars,
    )
    total.backward()

    assert torch.isfinite(total)
    assert learned_log_vars["pde"].grad is not None
    assert learned_log_vars["pde"].grad.item() < 0.0


def test_global_acv_head_x_greeks_convert_like_autodiff() -> None:
    model = GlobalACVResidualPINN(_global_acv_arch()).double()
    x = _global_acv_points().detach().clone().requires_grad_(True)

    heads = model.forward_all(x)
    price = heads[:, 0:1]
    grad = torch.autograd.grad(
        outputs=price,
        inputs=x,
        grad_outputs=torch.ones_like(price),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad[:, 1:2]
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]

    head_delta, head_gamma = head_greeks_x_to_financial(
        u_x=heads[:, 1:2],
        u_xx=heads[:, 2:3],
        x=x[:, 1:2],
    )
    auto_delta, auto_gamma = head_greeks_x_to_financial(
        u_x=u_x,
        u_xx=u_xx,
        x=x[:, 1:2],
    )

    assert torch.allclose(head_delta, auto_delta, atol=1.0e-9, rtol=1.0e-7)
    assert torch.allclose(head_gamma, auto_gamma, atol=1.0e-7, rtol=1.0e-5)
