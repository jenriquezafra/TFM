from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.pinn.data_builder import (
    PINN_FEATURE_ORDER,
    _convert_spot_coordinate,
    _coordinate_space_key,
    _feature_order_for_coordinate,
    _moneyness_column,
    _sample_kink_band_points,
    _sample_lhs_points,
    _sampling_domain_and_ranges,
    _validate_lhs_set,
)
from src.pinn.losses import _price_derivatives, compute_heston_pde_residual
from src.pinn.model import build_pinn_model
from src.pinn.trainer import (
    _load_checkpoint_state,
    _load_input_affine_from_summary,
    _resolve_device,
    _to_torch_input_affine,
)


@dataclass(frozen=True)
class AdaptiveCollocationResult:
    manifest_path: Path
    summary_path: Path
    interior_path: Path
    selected_path: Path


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _resolve_path(path_raw: str | Path, *, project_root: Path) -> Path:
    path = Path(path_raw)
    return path if path.is_absolute() else project_root / path


def _read_collocation_matrix(path: Path, *, feature_order: Sequence[str]) -> np.ndarray:
    df = pd.read_parquet(path)
    missing = [name for name in feature_order if name not in df.columns]
    if missing:
        raise KeyError(f"Collocation file {path} missing columns {missing}")
    x = df.loc[:, list(feature_order)].to_numpy(dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(feature_order):
        raise ValueError(f"Invalid collocation matrix shape {x.shape}: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Collocation matrix contains non-finite values: {path}")
    return x


def _normalize_shares(raw: dict, *, default: dict[str, float]) -> dict[str, float]:
    source = raw if raw else default
    shares = {str(k): max(0.0, float(v)) for k, v in source.items()}
    total = float(sum(shares.values()))
    if total <= 0.0:
        raise ValueError(f"At least one positive share is required. Got: {source}")
    return {k: v / total for k, v in shares.items()}


def _allocate_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {k: 0 for k in shares}
    raw = {k: total * v for k, v in shares.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(shares, key=lambda k: raw[k] - counts[k], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _clip_bounds(bounds: np.ndarray, low: float, high: float) -> np.ndarray:
    out_low = max(float(bounds[0]), float(low))
    out_high = min(float(bounds[1]), float(high))
    if out_low >= out_high:
        out_high = float(np.nextafter(out_low, np.inf))
    return np.array([out_low, out_high], dtype=np.float64)


def _build_candidate_pool(
    *,
    sampling_config: dict,
    theta_star: Sequence[float] | None,
    parameter_order: Sequence[str] | None,
    n_candidates: int,
    seed: int,
    hard_region: dict,
    region_shares: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    domain, param_ranges = _sampling_domain_and_ranges(
        sampling_config=sampling_config,
        theta_star=theta_star,
        parameter_order=parameter_order,
    )
    tau_bounds = domain["tau"]
    moneyness_bounds = domain["moneyness"]
    r_bounds = domain["r"]

    eps_m = float(hard_region.get("epsilon_m", 0.03))
    eps_tau = float(hard_region.get("epsilon_tau", 0.05))
    if eps_m <= 0.0 or eps_tau <= 0.0:
        raise ValueError("hard_region epsilon_m and epsilon_tau must be > 0.")

    atm_bounds = _clip_bounds(moneyness_bounds, float(np.exp(-eps_m)), float(np.exp(eps_m)))
    short_tau_bounds = _clip_bounds(tau_bounds, float(tau_bounds[0]), eps_tau)
    counts = _allocate_counts(n_candidates, region_shares)

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for offset, (name, count) in enumerate(counts.items()):
        if count <= 0:
            continue
        local_tau = tau_bounds
        local_m = moneyness_bounds
        if name == "hard":
            local_tau = short_tau_bounds
            local_m = atm_bounds
        elif name == "short":
            local_tau = short_tau_bounds
        elif name == "atm":
            local_m = atm_bounds
        elif name != "uniform":
            raise ValueError(
                f"Unsupported candidate region '{name}'. "
                "Use one of: uniform, hard, short, atm."
            )

        block = _sample_lhs_points(
            n_samples=count,
            seed=seed + 101 * (offset + 1),
            param_ranges=param_ranges,
            tau_bounds=local_tau,
            moneyness_bounds=local_m,
            r_bounds=r_bounds,
        )
        blocks.append(block)
        labels.append(np.full(count, name, dtype=object))

    if not blocks:
        raise ValueError("Candidate pool is empty.")
    candidates_m = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    candidates = _convert_spot_coordinate(candidates_m, _coordinate_space_key(sampling_config))
    source_labels = np.concatenate(labels, axis=0)

    meta = {
        "domain": {
            "tau": [float(tau_bounds[0]), float(tau_bounds[1])],
            "moneyness": [float(moneyness_bounds[0]), float(moneyness_bounds[1])],
            "v": [float(domain["v"][0]), float(domain["v"][1])],
            "r": [float(r_bounds[0]), float(r_bounds[1])],
        },
        "hard_region": {
            "epsilon_m": eps_m,
            "epsilon_tau": eps_tau,
            "atm_moneyness_bounds": [float(atm_bounds[0]), float(atm_bounds[1])],
            "short_tau_bounds": [float(short_tau_bounds[0]), float(short_tau_bounds[1])],
        },
        "candidate_counts": {k: int(v) for k, v in counts.items()},
    }
    return candidates, source_labels, meta


def _evaluate_candidate_scores(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    input_affine: dict[str, torch.Tensor] | None,
    coordinate_space: str,
    score_config: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0. Got {batch_size}.")
    coordinate_key = _coordinate_space_key(coordinate_space)
    residual_weight = float(score_config.get("residual_weight", 1.0))
    curvature_weight = float(
        score_config.get(
            "curvature_weight",
            score_config.get("g_weight", score_config.get("curvature", 0.0)),
        )
    )
    residual_dx_weight = float(
        score_config.get("residual_dx_weight", score_config.get("dx_weight", 0.0))
    )
    residual_dv_weight = float(
        score_config.get("residual_dv_weight", score_config.get("dv_weight", 0.0))
    )
    model.eval()
    score_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    curvature_values: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        xb = torch.from_numpy(x[start:stop]).to(device)
        with torch.enable_grad():
            xb = xb.requires_grad_(True)
            residual = compute_heston_pde_residual(
                model=model,
                x_interior=xb,
                input_affine=input_affine,
                coordinate=coordinate_key,
            )
            score = residual_weight * torch.abs(residual)
            curvature_abs = torch.zeros_like(residual)
            if curvature_weight > 0.0:
                _, _u_s, _u_v, curvature = _price_derivatives(
                    model=model,
                    x=xb,
                    input_affine=input_affine,
                    coordinate=coordinate_key,
                )
                curvature_abs = torch.abs(curvature)
                score = score + curvature_weight * curvature_abs
            if residual_dx_weight > 0.0 or residual_dv_weight > 0.0:
                grad_residual = torch.autograd.grad(
                    outputs=residual,
                    inputs=xb,
                    grad_outputs=torch.ones_like(residual),
                    create_graph=False,
                    retain_graph=True,
                )[0]
                if residual_dx_weight > 0.0:
                    if coordinate_key == "log_moneyness":
                        residual_dx = grad_residual[:, 1:2]
                    else:
                        residual_dx = xb[:, 1:2] * grad_residual[:, 1:2]
                    score = score + residual_dx_weight * torch.abs(residual_dx)
                if residual_dv_weight > 0.0:
                    score = score + residual_dv_weight * torch.abs(grad_residual[:, 2:3])
        score_values.append(score.detach().cpu().numpy().reshape(-1))
        residual_values.append(residual.detach().abs().cpu().numpy().reshape(-1))
        curvature_values.append(curvature_abs.detach().cpu().numpy().reshape(-1))
    return {
        "adaptive_score": np.concatenate(score_values, axis=0).astype(np.float32, copy=False),
        "abs_pde_residual": np.concatenate(residual_values, axis=0).astype(np.float32, copy=False),
        "abs_financial_curvature": np.concatenate(curvature_values, axis=0).astype(np.float32, copy=False),
    }


def _select_adaptive_indices(
    *,
    abs_residual: np.ndarray,
    source_labels: np.ndarray,
    n_select: int,
    min_source_shares: dict[str, float],
) -> np.ndarray:
    if n_select <= 0:
        return np.empty(0, dtype=np.int64)
    if n_select > abs_residual.size:
        raise ValueError(
            f"Cannot select {n_select} adaptive points from {abs_residual.size} candidates."
        )

    selected = np.zeros(abs_residual.size, dtype=bool)
    min_counts = _allocate_counts(n_select, min_source_shares) if min_source_shares else {}
    for label, count in min_counts.items():
        if count <= 0:
            continue
        idx = np.flatnonzero(source_labels == label)
        if idx.size == 0:
            continue
        take = min(count, idx.size)
        best_local = idx[np.argsort(abs_residual[idx])[-take:]]
        selected[best_local] = True

    remaining = n_select - int(selected.sum())
    if remaining > 0:
        available = np.flatnonzero(~selected)
        best = available[np.argsort(abs_residual[available])[-remaining:]]
        selected[best] = True

    out = np.flatnonzero(selected)
    order = np.argsort(abs_residual[out])[::-1]
    return out[order]


def _save_residual_map(
    *,
    points: np.ndarray,
    abs_residual: np.ndarray,
    output_dir: Path,
    stem: str,
    n_bins_tau: int,
    n_bins_m: int,
    coordinate_space: str = "moneyness",
    cbar_label: str = "mean |PDE residual|",
) -> dict[str, str]:
    tau = points[:, 0]
    m = _moneyness_column(points, coordinate_space)
    tau_edges = np.linspace(float(tau.min()), float(tau.max()), int(n_bins_tau) + 1)
    m_edges = np.linspace(float(m.min()), float(m.max()), int(n_bins_m) + 1)
    tau_idx = np.clip(np.digitize(tau, tau_edges, right=False) - 1, 0, n_bins_tau - 1)
    m_idx = np.clip(np.digitize(m, m_edges, right=False) - 1, 0, n_bins_m - 1)

    sums = np.zeros((n_bins_tau, n_bins_m), dtype=np.float64)
    counts = np.zeros((n_bins_tau, n_bins_m), dtype=np.int64)
    np.add.at(sums, (tau_idx, m_idx), abs_residual.astype(np.float64, copy=False))
    np.add.at(counts, (tau_idx, m_idx), 1)
    heat = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    fig_path = output_dir / f"{stem}.png"
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[m_edges[0], m_edges[-1], tau_edges[0], tau_edges[-1]],
        cmap="magma",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel("moneyness")
    ax.set_ylabel("tau")
    ax.set_title("Candidate PDE Residual Map")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    outputs[stem] = str(fig_path)

    valid = np.isfinite(heat) & (heat > 0.0)
    if np.any(valid):
        flat = heat[valid]
        vmin = max(float(np.nanpercentile(flat, 5.0)), float(np.finfo(np.float64).tiny))
        vmax = float(np.nanpercentile(flat, 95.0))
        if vmax <= vmin:
            vmax = vmin * 10.0
        fig_log_path = output_dir / f"{stem}_log.png"
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        im = ax.imshow(
            np.where(np.isfinite(heat), np.maximum(heat, vmin), np.nan),
            origin="lower",
            aspect="auto",
            extent=[m_edges[0], m_edges[-1], tau_edges[0], tau_edges[-1]],
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{cbar_label} (log scale)")
        ax.set_xlabel("moneyness")
        ax.set_ylabel("tau")
        ax.set_title("Candidate PDE Residual Map (Log Scale)")
        plt.tight_layout()
        plt.savefig(fig_log_path, dpi=300)
        plt.close(fig)
        outputs[f"{stem}_log"] = str(fig_log_path)
    return outputs


def build_adaptive_collocation_dataset(
    *,
    project_root: Path,
    config: dict,
    architecture_config: dict,
    theta_star: Sequence[float] | None = None,
    parameter_order: Sequence[str] | None = None,
) -> AdaptiveCollocationResult:
    adaptive_cfg = config.get("adaptive", {})
    if not isinstance(adaptive_cfg, dict):
        raise ValueError("config.adaptive must be a dictionary.")

    base_cfg = config.get("base", {})
    if not isinstance(base_cfg, dict):
        raise ValueError("config.base must be a dictionary.")

    output_dir = _resolve_path(
        adaptive_cfg.get("output_dir", "data/synth/PINN_param_2x_v01_adaptive_collocation"),
        project_root=project_root,
    )
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _resolve_path(base_cfg["collocation_manifest"], project_root=project_root)
    manifest = _load_yaml(base_manifest)
    feature_order = list(manifest.get("feature_order", PINN_FEATURE_ORDER))
    coordinate_space = _coordinate_space_key(manifest.get("coordinate_space", "moneyness"))
    expected_order = list(_feature_order_for_coordinate(coordinate_space))
    if feature_order != expected_order:
        raise ValueError(
            "Adaptive collocation feature order mismatch for coordinate_space="
            f"{coordinate_space}: expected {expected_order}, got {feature_order}."
        )
    datasets = manifest.get("datasets", {})
    missing = [key for key in ("terminal", "lower") if key not in datasets]
    if missing:
        raise KeyError(f"Base collocation manifest missing datasets {missing}: {base_manifest}")

    terminal_path = _resolve_path(datasets["terminal"], project_root=project_root)
    lower_path = _resolve_path(datasets["lower"], project_root=project_root)
    terminal = _read_collocation_matrix(terminal_path, feature_order=feature_order)
    lower = _read_collocation_matrix(lower_path, feature_order=feature_order)
    optional_paths: dict[str, Path] = {}
    optional_sets: dict[str, np.ndarray] = {}
    for optional_key in ("right", "v_zero"):
        if optional_key not in datasets:
            continue
        optional_path = _resolve_path(datasets[optional_key], project_root=project_root)
        optional_paths[optional_key] = optional_path
        optional_sets[optional_key] = _read_collocation_matrix(
            optional_path,
            feature_order=feature_order,
        )

    sampling_cfg = adaptive_cfg.get("sampling", {})
    if not isinstance(sampling_cfg, dict):
        raise ValueError("adaptive.sampling must be a dictionary.")
    sampling_cfg = dict(sampling_cfg)
    sampling_cfg.setdefault("coordinate_space", coordinate_space)
    domain, _ = _sampling_domain_and_ranges(
        sampling_config=sampling_cfg,
        theta_star=theta_star,
        parameter_order=parameter_order,
    )

    seed = int(adaptive_cfg.get("seed", sampling_cfg.get("seed", 42)))
    sizes_cfg = adaptive_cfg.get("sizes", {})
    if not isinstance(sizes_cfg, dict):
        raise ValueError("adaptive.sizes must be a dictionary when provided.")
    total_interior = int(
        sizes_cfg.get("n_interior", sampling_cfg.get("sizes", {}).get("n_interior", 40000))
    )

    selection_cfg = adaptive_cfg.get("selection", {})
    if not isinstance(selection_cfg, dict):
        raise ValueError("adaptive.selection must be a dictionary when provided.")
    adaptive_ratio = float(selection_cfg.get("adaptive_ratio", 0.4))
    if not (0.0 < adaptive_ratio < 1.0):
        raise ValueError(f"adaptive.selection.adaptive_ratio must be in (0,1). Got {adaptive_ratio}.")
    n_adaptive = int(round(total_interior * adaptive_ratio))
    kink_cfg = sampling_cfg.get("kink_band", {})
    kink_enabled = isinstance(kink_cfg, dict) and bool(kink_cfg.get("enabled", False))
    kink_fraction = float(kink_cfg.get("fraction", 0.0)) if kink_enabled else 0.0
    if kink_fraction < 0.0 or kink_fraction >= 1.0:
        raise ValueError("adaptive.sampling.kink_band.fraction must be in [0, 1).")
    if adaptive_ratio + kink_fraction >= 1.0:
        raise ValueError("adaptive_ratio + kink_band.fraction must be < 1.")
    n_kink = int(round(total_interior * kink_fraction))
    n_uniform = total_interior - n_adaptive - n_kink

    pool_cfg = adaptive_cfg.get("candidate_pool", {})
    if not isinstance(pool_cfg, dict):
        raise ValueError("adaptive.candidate_pool must be a dictionary when provided.")
    n_candidates = int(pool_cfg.get("n_candidates", max(5 * n_adaptive, n_adaptive + 1)))
    if n_candidates < n_adaptive:
        raise ValueError("candidate_pool.n_candidates must be >= selected adaptive points.")
    region_shares = _normalize_shares(
        pool_cfg.get("region_shares", {}),
        default={"uniform": 0.50, "hard": 0.20, "short": 0.15, "atm": 0.15},
    )
    hard_region = adaptive_cfg.get("hard_region", {})
    if not isinstance(hard_region, dict):
        raise ValueError("adaptive.hard_region must be a dictionary when provided.")

    candidates, source_labels, candidate_meta = _build_candidate_pool(
        sampling_config=sampling_cfg,
        theta_star=theta_star,
        parameter_order=parameter_order,
        n_candidates=n_candidates,
        seed=seed + 10_000,
        hard_region=hard_region,
        region_shares=region_shares,
    )

    device = _resolve_device(base_cfg.get("device", "auto"))
    model = build_pinn_model(architecture_config).to(device)
    train_summary = _resolve_path(base_cfg["train_summary"], project_root=project_root)
    input_affine_loaded = _load_input_affine_from_summary(
        summary_path=train_summary,
        feature_order=feature_order,
    )
    input_affine_np = (
        input_affine_loaded if bool(input_affine_loaded.get("enabled", False)) else None
    )
    configure_input_affine = getattr(model, "configure_input_affine", None)
    if callable(configure_input_affine):
        configure_input_affine(input_affine_np)
    checkpoint_path = _resolve_path(base_cfg["checkpoint"], project_root=project_root)
    model.load_state_dict(_load_checkpoint_state(checkpoint_path, device=device), strict=True)
    if callable(configure_input_affine):
        configure_input_affine(input_affine_np)
    input_affine = _to_torch_input_affine(input_affine=input_affine_np, device=device)

    batch_size = int(pool_cfg.get("batch_size", 4096))
    score_cfg = selection_cfg.get("score", {})
    if not isinstance(score_cfg, dict):
        raise ValueError("adaptive.selection.score must be a dictionary when provided.")
    score_payload = _evaluate_candidate_scores(
        model=model,
        x=candidates,
        input_affine=input_affine,
        coordinate_space=coordinate_space,
        score_config=score_cfg,
        device=device,
        batch_size=batch_size,
    )
    adaptive_score = score_payload["adaptive_score"]
    abs_residual = score_payload["abs_pde_residual"]
    abs_curvature = score_payload["abs_financial_curvature"]

    min_source_shares = _normalize_shares(
        selection_cfg.get("min_source_shares", {}),
        default={},
    ) if selection_cfg.get("min_source_shares") else {}
    selected_idx = _select_adaptive_indices(
        abs_residual=adaptive_score,
        source_labels=source_labels,
        n_select=n_adaptive,
        min_source_shares=min_source_shares,
    )
    selected = candidates[selected_idx]
    selected_score = adaptive_score[selected_idx]
    selected_residual = abs_residual[selected_idx]
    selected_curvature = abs_curvature[selected_idx]
    selected_source = source_labels[selected_idx]

    domain_cfg = candidate_meta["domain"]
    uniform_m = _sample_lhs_points(
        n_samples=n_uniform,
        seed=seed + 20_000,
        param_ranges=_sampling_domain_and_ranges(
            sampling_config=sampling_cfg,
            theta_star=theta_star,
            parameter_order=parameter_order,
        )[1],
        tau_bounds=np.asarray(domain_cfg["tau"], dtype=np.float64),
        moneyness_bounds=np.asarray(domain_cfg["moneyness"], dtype=np.float64),
        r_bounds=np.asarray(domain_cfg["r"], dtype=np.float64),
    )
    uniform = _convert_spot_coordinate(uniform_m, coordinate_space)
    blocks = [uniform]
    if n_kink > 0:
        kink_m = _sample_kink_band_points(
            n_samples=n_kink,
            seed=seed + 30_000,
            param_ranges=_sampling_domain_and_ranges(
                sampling_config=sampling_cfg,
                theta_star=theta_star,
                parameter_order=parameter_order,
            )[1],
            tau_bounds=np.asarray(domain_cfg["tau"], dtype=np.float64),
            moneyness_bounds=np.asarray(domain_cfg["moneyness"], dtype=np.float64),
            r_bounds=np.asarray(domain_cfg["r"], dtype=np.float64),
            config=kink_cfg,
        )
        blocks.append(_convert_spot_coordinate(kink_m, coordinate_space))
    blocks.append(selected)
    interior = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)

    _validate_lhs_set(
        data=interior,
        name="interior",
        feature_order=feature_order,
        coordinate_space=coordinate_space,
        tau_bounds=np.asarray(domain_cfg["tau"], dtype=np.float64),
        moneyness_bounds=np.asarray(domain_cfg["moneyness"], dtype=np.float64),
        v_bounds=np.asarray(domain_cfg["v"], dtype=np.float64),
        r_bounds=np.asarray(domain_cfg["r"], dtype=np.float64),
    )
    _validate_lhs_set(
        data=terminal,
        name="terminal",
        feature_order=feature_order,
        coordinate_space=coordinate_space,
        tau_bounds=np.asarray(domain_cfg["tau"], dtype=np.float64),
        moneyness_bounds=np.asarray(domain_cfg["moneyness"], dtype=np.float64),
        v_bounds=np.asarray(domain_cfg["v"], dtype=np.float64),
        r_bounds=np.asarray(domain_cfg["r"], dtype=np.float64),
    )
    _validate_lhs_set(
        data=lower,
        name="lower",
        feature_order=feature_order,
        coordinate_space=coordinate_space,
        tau_bounds=np.asarray(domain_cfg["tau"], dtype=np.float64),
        moneyness_bounds=np.asarray(domain_cfg["moneyness"], dtype=np.float64),
        v_bounds=np.asarray(domain_cfg["v"], dtype=np.float64),
        r_bounds=np.asarray(domain_cfg["r"], dtype=np.float64),
    )
    for optional_key, optional_data in optional_sets.items():
        m_bounds = np.asarray(domain_cfg["moneyness"], dtype=np.float64)
        v_bounds = np.asarray(domain_cfg["v"], dtype=np.float64)
        if optional_key == "right":
            optional_m = _moneyness_column(optional_data, coordinate_space)
            m_bounds = np.asarray(
                [
                    min(float(m_bounds[0]), float(optional_m.min())),
                    max(float(m_bounds[1]), float(optional_m.max())),
                ],
                dtype=np.float64,
            )
        if optional_key == "v_zero":
            optional_v = optional_data[:, 2]
            v_bounds = np.asarray(
                [
                    min(float(v_bounds[0]), float(optional_v.min())),
                    max(float(v_bounds[1]), float(optional_v.max())),
                ],
                dtype=np.float64,
            )
        _validate_lhs_set(
            data=optional_data,
            name=optional_key,
            feature_order=feature_order,
            coordinate_space=coordinate_space,
            tau_bounds=np.asarray(domain_cfg["tau"], dtype=np.float64),
            moneyness_bounds=m_bounds,
            v_bounds=v_bounds,
            r_bounds=np.asarray(domain_cfg["r"], dtype=np.float64),
        )

    interior_path = output_dir / "interior.parquet"
    pd.DataFrame(interior, columns=feature_order).to_parquet(
        interior_path,
        engine="pyarrow",
        index=False,
    )

    selected_df = pd.DataFrame(selected, columns=feature_order)
    selected_df["adaptive_score"] = selected_score
    selected_df["abs_pde_residual"] = selected_residual
    selected_df["abs_financial_curvature"] = selected_curvature
    selected_df["candidate_source"] = selected_source
    selected_path = output_dir / "selected_adaptive_points.parquet"
    selected_df.to_parquet(selected_path, engine="pyarrow", index=False)

    residual_map_paths = _save_residual_map(
        points=candidates,
        abs_residual=adaptive_score,
        output_dir=figures_dir,
        stem="candidate_adaptive_score_map",
        n_bins_tau=int(pool_cfg.get("n_bins_tau", 36)),
        n_bins_m=int(pool_cfg.get("n_bins_m", 36)),
        coordinate_space=coordinate_space,
        cbar_label="mean adaptive score",
    )

    manifest_path = output_dir / "collocation_sets_manifest.yaml"
    manifest_datasets = {
        "interior": str(interior_path),
        "terminal": str(terminal_path),
        "lower": str(lower_path),
    }
    for optional_key, optional_path in optional_paths.items():
        manifest_datasets[optional_key] = str(optional_path)
    sizes = {
        "n_interior": int(interior.shape[0]),
        "n_terminal": int(terminal.shape[0]),
        "n_lower": int(lower.shape[0]),
        "n_interior_uniform": int(n_uniform),
        "n_interior_kink": int(n_kink),
        "n_interior_adaptive": int(n_adaptive),
    }
    for optional_key, optional_data in optional_sets.items():
        sizes[f"n_{optional_key}"] = int(optional_data.shape[0])
    adaptive_manifest = {
        "dataset_format": "parquet",
        "datasets": manifest_datasets,
        "feature_order": list(feature_order),
        "coordinate_space": coordinate_space,
        "sampling_strategy": "adaptive_residual",
        "sampling_mode": str(sampling_cfg.get("mode", "parametric_theta")),
        "seed": seed,
        "sizes": sizes,
        "domain": domain_cfg,
        "adaptive": {
            "base_collocation_manifest": str(base_manifest),
            "base_train_summary": str(train_summary),
            "base_checkpoint": str(checkpoint_path),
            "candidate_pool_size": int(candidates.shape[0]),
            "adaptive_ratio": adaptive_ratio,
            "kink_fraction": kink_fraction,
            "score": score_cfg,
            "hard_region": candidate_meta["hard_region"],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(adaptive_manifest, f, sort_keys=False)

    selected_counts = pd.Series(selected_source).value_counts().to_dict()
    summary = {
        "manifest_file": str(manifest_path),
        "interior_file": str(interior_path),
        "selected_adaptive_points_file": str(selected_path),
        "base_collocation_manifest": str(base_manifest),
        "base_train_summary": str(train_summary),
        "base_checkpoint": str(checkpoint_path),
        "device": str(device),
        "n_candidates": int(candidates.shape[0]),
        "n_uniform": int(n_uniform),
        "n_kink": int(n_kink),
        "n_adaptive": int(n_adaptive),
        "optional_boundary_sets": {
            key: {"path": str(optional_paths[key]), "n": int(value.shape[0])}
            for key, value in optional_sets.items()
        },
        "score": score_cfg,
        "candidate_counts": candidate_meta["candidate_counts"],
        "selected_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "abs_residual_summary": {
            "candidate_mean": float(np.mean(abs_residual)),
            "candidate_p50": float(np.percentile(abs_residual, 50.0)),
            "candidate_p90": float(np.percentile(abs_residual, 90.0)),
            "candidate_p99": float(np.percentile(abs_residual, 99.0)),
            "selected_min": float(np.min(selected_residual)),
            "selected_mean": float(np.mean(selected_residual)),
            "selected_max": float(np.max(selected_residual)),
        },
        "figures": residual_map_paths,
    }
    summary_path = output_dir / "adaptive_collocation_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    return AdaptiveCollocationResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        interior_path=interior_path,
        selected_path=selected_path,
    )
