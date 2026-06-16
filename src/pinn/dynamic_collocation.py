from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from src.pinn.data_builder import (
    PINN_FEATURE_ORDER,
    _convert_spot_coordinate,
    _coordinate_space_key,
    _moneyness_column,
    _sample_kink_band_points,
    _sample_lhs_points,
)
from src.pinn.losses import _price_derivatives, compute_heston_pde_residual


@dataclass(frozen=True)
class DynamicCollocationRefresh:
    interior: np.ndarray
    report: dict


def _bounds(raw: object, *, default: Sequence[float], name: str) -> np.ndarray:
    values = default if raw is None else raw
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} bounds must have two entries. Got {values!r}.")
    lo, hi = float(arr[0]), float(arr[1])
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError(f"Invalid {name} bounds: {values!r}.")
    return np.array([lo, hi], dtype=np.float64)


def _domain_from_config(sampling_config: dict, manifest_domain: dict) -> dict[str, np.ndarray]:
    domain_cfg = sampling_config.get("domain", {})
    domain_cfg = domain_cfg if isinstance(domain_cfg, dict) else {}
    manifest_domain = manifest_domain if isinstance(manifest_domain, dict) else {}
    return {
        "tau": _bounds(domain_cfg.get("tau", manifest_domain.get("tau")), default=[0.0, 3.0], name="tau"),
        "moneyness": _bounds(
            domain_cfg.get("moneyness", manifest_domain.get("moneyness")),
            default=[0.05, 2.0],
            name="moneyness",
        ),
        "v": _bounds(domain_cfg.get("v", manifest_domain.get("v")), default=[0.01, 0.5], name="v"),
        "r": _bounds(domain_cfg.get("r", manifest_domain.get("r")), default=[0.0, 0.05], name="r"),
    }


def _clip_bounds(bounds: np.ndarray, low: float, high: float) -> np.ndarray:
    out_low = max(float(bounds[0]), float(low))
    out_high = min(float(bounds[1]), float(high))
    if out_low >= out_high:
        out_high = float(np.nextafter(out_low, np.inf))
    return np.array([out_low, out_high], dtype=np.float64)


def _normalize_weights(raw: dict | None, *, default: dict[str, float]) -> dict[str, float]:
    source = default if not raw else raw
    weights = {str(k): max(0.0, float(v)) for k, v in source.items()}
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError(f"At least one positive weight is required. Got: {source!r}.")
    return {k: v / total for k, v in weights.items()}


def _allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * v for k, v in weights.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda k: raw[k] - counts[k], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _reference_param_ranges(reference_blocks: Sequence[np.ndarray], domain: dict[str, np.ndarray]) -> np.ndarray:
    if not reference_blocks:
        raise ValueError("reference_blocks cannot be empty.")
    ref = np.concatenate([x for x in reference_blocks if x.size], axis=0).astype(np.float64, copy=False)
    if ref.ndim != 2 or ref.shape[1] < len(PINN_FEATURE_ORDER):
        raise ValueError(f"Expected reference blocks with at least 8 columns, got {ref.shape}.")
    ranges = []
    for idx in (3, 4, 5, 6):
        value_min = float(np.nanmin(ref[:, idx]))
        value_max = float(np.nanmax(ref[:, idx]))
        if not np.isfinite([value_min, value_max]).all():
            raise ValueError("Reference collocation parameters contain non-finite values.")
        if value_min >= value_max:
            value_max = float(np.nextafter(value_min, np.inf))
        ranges.append([value_min, value_max])
    ranges.append([float(domain["v"][0]), float(domain["v"][1])])
    return np.asarray(ranges, dtype=np.float64)


def _sample_uniform(
    *,
    n_samples: int,
    seed: int,
    param_ranges: np.ndarray,
    domain: dict[str, np.ndarray],
    coordinate_space: str,
) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, len(PINN_FEATURE_ORDER)), dtype=np.float32)
    m_points = _sample_lhs_points(
        n_samples=n_samples,
        seed=seed,
        param_ranges=param_ranges,
        tau_bounds=domain["tau"],
        moneyness_bounds=domain["moneyness"],
        r_bounds=domain["r"],
    )
    return _convert_spot_coordinate(m_points, coordinate_space)


def _sample_kink(
    *,
    n_samples: int,
    seed: int,
    param_ranges: np.ndarray,
    domain: dict[str, np.ndarray],
    coordinate_space: str,
    kink_config: dict,
) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, len(PINN_FEATURE_ORDER)), dtype=np.float32)
    m_points = _sample_kink_band_points(
        n_samples=n_samples,
        seed=seed,
        param_ranges=param_ranges,
        tau_bounds=domain["tau"],
        moneyness_bounds=domain["moneyness"],
        r_bounds=domain["r"],
        config=kink_config,
    )
    return _convert_spot_coordinate(m_points, coordinate_space)


def _sample_candidate_region(
    *,
    region: str,
    n_samples: int,
    seed: int,
    param_ranges: np.ndarray,
    domain: dict[str, np.ndarray],
    coordinate_space: str,
    hard_region: dict,
    kink_config: dict,
) -> np.ndarray:
    if region == "kink":
        return _sample_kink(
            n_samples=n_samples,
            seed=seed,
            param_ranges=param_ranges,
            domain=domain,
            coordinate_space=coordinate_space,
            kink_config=kink_config,
        )

    eps_m = float(hard_region.get("epsilon_m", 0.03))
    eps_tau = float(hard_region.get("epsilon_tau", 0.05))
    tau_bounds = domain["tau"]
    m_bounds = domain["moneyness"]
    if region == "hard":
        tau_bounds = _clip_bounds(domain["tau"], float(domain["tau"][0]), eps_tau)
        m_bounds = _clip_bounds(domain["moneyness"], float(np.exp(-eps_m)), float(np.exp(eps_m)))
    elif region == "short":
        tau_bounds = _clip_bounds(domain["tau"], float(domain["tau"][0]), eps_tau)
    elif region == "atm":
        m_bounds = _clip_bounds(domain["moneyness"], float(np.exp(-eps_m)), float(np.exp(eps_m)))
    elif region not in {"uniform", "global"}:
        raise ValueError(f"Unsupported dynamic candidate region '{region}'.")

    local_domain = dict(domain)
    local_domain["tau"] = tau_bounds
    local_domain["moneyness"] = m_bounds
    return _sample_uniform(
        n_samples=n_samples,
        seed=seed,
        param_ranges=param_ranges,
        domain=local_domain,
        coordinate_space=coordinate_space,
    )


def _build_candidate_pool(
    *,
    n_candidates: int,
    seed: int,
    param_ranges: np.ndarray,
    domain: dict[str, np.ndarray],
    coordinate_space: str,
    hard_region: dict,
    kink_config: dict,
    region_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    counts = _allocate_counts(n_candidates, region_weights)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for offset, (region, count) in enumerate(counts.items()):
        if count <= 0:
            continue
        block = _sample_candidate_region(
            region=region,
            n_samples=count,
            seed=seed + 173 * (offset + 1),
            param_ranges=param_ranges,
            domain=domain,
            coordinate_space=coordinate_space,
            hard_region=hard_region,
            kink_config=kink_config,
        )
        blocks.append(block)
        labels.append(np.full(count, region, dtype=object))
    if not blocks:
        raise ValueError("Dynamic candidate pool is empty.")
    return (
        np.concatenate(blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(labels, axis=0),
        {k: int(v) for k, v in counts.items()},
    )


def _evaluate_scores(
    *,
    model: nn.Module,
    candidates: np.ndarray,
    input_affine: dict[str, torch.Tensor] | None,
    coordinate_space: str,
    score_config: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    residual_weight = float(score_config.get("residual_weight", 1.0))
    curvature_weight = float(score_config.get("curvature_weight", 0.0))
    coordinate_key = _coordinate_space_key(coordinate_space)
    score_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    curvature_parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, candidates.shape[0], batch_size):
        stop = min(start + batch_size, candidates.shape[0])
        xb = torch.from_numpy(candidates[start:stop]).to(device)
        with torch.enable_grad():
            xb = xb.requires_grad_(True)
            residual = compute_heston_pde_residual(
                model=model,
                x_interior=xb,
                input_affine=input_affine,
                coordinate=coordinate_key,
            )
            score = residual_weight * torch.abs(residual)
            curvature_abs = torch.zeros_like(score)
            if curvature_weight > 0.0:
                _, _, _, curvature = _price_derivatives(
                    model=model,
                    x=xb,
                    input_affine=input_affine,
                    coordinate=coordinate_key,
                )
                curvature_abs = torch.abs(curvature)
                score = score + curvature_weight * curvature_abs
        score_parts.append(score.detach().cpu().numpy().reshape(-1))
        residual_parts.append(residual.detach().abs().cpu().numpy().reshape(-1))
        curvature_parts.append(curvature_abs.detach().cpu().numpy().reshape(-1))
    return {
        "score": np.concatenate(score_parts, axis=0).astype(np.float64, copy=False),
        "abs_residual": np.concatenate(residual_parts, axis=0).astype(np.float64, copy=False),
        "abs_curvature": np.concatenate(curvature_parts, axis=0).astype(np.float64, copy=False),
    }


def _select_scored_indices(
    *,
    score: np.ndarray,
    labels: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
    selection_config: dict,
) -> np.ndarray:
    if n_select <= 0:
        return np.empty(0, dtype=np.int64)
    if n_select > score.size:
        raise ValueError(f"Cannot select {n_select} points from {score.size} candidates.")

    selected = np.zeros(score.size, dtype=bool)
    min_shares = _normalize_weights(selection_config.get("min_source_shares"), default={}) if selection_config.get("min_source_shares") else {}
    min_counts = _allocate_counts(n_select, min_shares) if min_shares else {}
    for label, count in min_counts.items():
        if count <= 0:
            continue
        pool = np.flatnonzero(labels == label)
        if pool.size == 0:
            continue
        take = min(count, pool.size)
        local = pool[np.argpartition(score[pool], -take)[-take:]]
        selected[local] = True

    remaining = n_select - int(selected.sum())
    if remaining <= 0:
        return np.flatnonzero(selected)[:n_select]

    pool = np.flatnonzero(~selected)
    method = str(selection_config.get("method", "weighted_without_replacement")).strip().lower()
    if method in {"topk", "rar", "largest"}:
        local = pool[np.argpartition(score[pool], -remaining)[-remaining:]]
    elif method in {"weighted_without_replacement", "rar-d", "rad", "importance"}:
        power = float(selection_config.get("power", 1.0))
        floor = float(selection_config.get("floor", 0.05))
        weights = np.maximum(score[pool], 0.0) ** power
        positive = weights[np.isfinite(weights) & (weights > 0.0)]
        scale = float(np.mean(positive)) if positive.size else 1.0
        weights = np.where(np.isfinite(weights), weights, 0.0) + floor * scale
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights, dtype=np.float64)
        probs = weights / float(weights.sum())
        local = rng.choice(pool, size=remaining, replace=False, p=probs)
    else:
        raise ValueError(
            "dynamic_collocation.selection.method must be one of "
            "{'weighted_without_replacement','topk'}."
        )
    selected[local] = True
    return np.flatnonzero(selected)


def refresh_dynamic_collocation_interior(
    *,
    model: nn.Module,
    input_affine: dict[str, torch.Tensor] | None,
    device: torch.device,
    current_interior: np.ndarray,
    reference_blocks: Sequence[np.ndarray],
    sampling_config: dict,
    manifest_domain: dict,
    coordinate_space: str,
    dynamic_config: dict,
    seed: int,
    epoch: int,
) -> DynamicCollocationRefresh:
    coordinate_key = _coordinate_space_key(coordinate_space)
    domain = _domain_from_config(sampling_config, manifest_domain)
    if coordinate_key == "log_moneyness" and domain["moneyness"][0] <= 0.0:
        raise ValueError("dynamic_collocation with log_moneyness requires moneyness lower bound > 0.")

    total_interior = int(dynamic_config.get("n_interior", current_interior.shape[0]))
    if total_interior <= 0:
        raise ValueError("dynamic_collocation.n_interior must be positive.")

    fractions = _normalize_weights(
        dynamic_config.get("fractions"),
        default={"global": 0.40, "kink": 0.35, "adaptive": 0.25},
    )
    counts = _allocate_counts(total_interior, fractions)
    n_global = int(counts.get("global", counts.get("uniform", 0)))
    n_kink = int(counts.get("kink", 0))
    n_adaptive = int(counts.get("adaptive", 0))
    unused = total_interior - n_global - n_kink - n_adaptive
    n_global += unused

    param_ranges = _reference_param_ranges(reference_blocks, domain=domain)
    kink_config = dict(dynamic_config.get("kink_band", {}))
    if not kink_config:
        kink_config = {"c": 2.0, "tau_c": 0.15, "epsilon": 1.0e-8}
    hard_region = dict(dynamic_config.get("hard_region", {"epsilon_m": 0.03, "epsilon_tau": 0.05}))
    rng = np.random.default_rng(seed + 10_009 * epoch)

    blocks: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    if n_global > 0:
        global_block = _sample_uniform(
            n_samples=n_global,
            seed=seed + 20_000 + epoch,
            param_ranges=param_ranges,
            domain=domain,
            coordinate_space=coordinate_key,
        )
        blocks.append(global_block)
        source_counts["global"] = int(global_block.shape[0])
    if n_kink > 0:
        kink_block = _sample_kink(
            n_samples=n_kink,
            seed=seed + 30_000 + epoch,
            param_ranges=param_ranges,
            domain=domain,
            coordinate_space=coordinate_key,
            kink_config=kink_config,
        )
        blocks.append(kink_block)
        source_counts["kink"] = int(kink_block.shape[0])

    score_summary: dict[str, float] = {}
    candidate_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    if n_adaptive > 0:
        candidate_cfg = dynamic_config.get("candidate_pool", {})
        candidate_cfg = candidate_cfg if isinstance(candidate_cfg, dict) else {}
        n_candidates = int(
            candidate_cfg.get(
                "n_candidates",
                dynamic_config.get(
                    "n_candidates",
                    max(n_adaptive + 1, int(float(candidate_cfg.get("multiplier", 6.0)) * n_adaptive)),
                ),
            )
        )
        n_candidates = max(n_candidates, n_adaptive)
        candidate_weights = _normalize_weights(
            candidate_cfg.get("region_shares"),
            default={"global": 0.35, "hard": 0.25, "short": 0.15, "atm": 0.10, "kink": 0.15},
        )
        candidates, labels, candidate_counts = _build_candidate_pool(
            n_candidates=n_candidates,
            seed=seed + 40_000 + epoch,
            param_ranges=param_ranges,
            domain=domain,
            coordinate_space=coordinate_key,
            hard_region=hard_region,
            kink_config=kink_config,
            region_weights=candidate_weights,
        )
        scores = _evaluate_scores(
            model=model,
            candidates=candidates,
            input_affine=input_affine,
            coordinate_space=coordinate_key,
            score_config=dict(dynamic_config.get("score", {})),
            device=device,
            batch_size=int(candidate_cfg.get("batch_size", 4096)),
        )
        selection_config = dict(dynamic_config.get("selection", {}))
        selected_idx = _select_scored_indices(
            score=scores["score"],
            labels=labels,
            n_select=n_adaptive,
            rng=rng,
            selection_config=selection_config,
        )
        adaptive_block = candidates[selected_idx]
        blocks.append(adaptive_block)
        source_counts["adaptive"] = int(adaptive_block.shape[0])
        selected_counts = {str(k): int(v) for k, v in zip(*np.unique(labels[selected_idx], return_counts=True))}
        selected_score = scores["score"][selected_idx]
        selected_residual = scores["abs_residual"][selected_idx]
        score_summary = {
            "candidate_score_mean": float(np.mean(scores["score"])),
            "candidate_score_p95": float(np.percentile(scores["score"], 95.0)),
            "candidate_score_p99": float(np.percentile(scores["score"], 99.0)),
            "selected_score_mean": float(np.mean(selected_score)),
            "selected_score_min": float(np.min(selected_score)),
            "selected_score_max": float(np.max(selected_score)),
            "selected_abs_residual_mean": float(np.mean(selected_residual)),
        }

    if not blocks:
        raise ValueError("Dynamic collocation refresh produced no interior points.")
    interior = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    interior = interior[rng.permutation(interior.shape[0])]

    m = _moneyness_column(interior, coordinate_key)
    report = {
        "epoch": int(epoch),
        "seed": int(seed),
        "n_interior": int(interior.shape[0]),
        "source_counts": source_counts,
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "score_summary": score_summary,
        "domain": {
            "tau": [float(domain["tau"][0]), float(domain["tau"][1])],
            "moneyness": [float(domain["moneyness"][0]), float(domain["moneyness"][1])],
            "v": [float(domain["v"][0]), float(domain["v"][1])],
            "r": [float(domain["r"][0]), float(domain["r"][1])],
        },
        "realized": {
            "tau_min": float(np.min(interior[:, 0])),
            "tau_max": float(np.max(interior[:, 0])),
            "moneyness_min": float(np.min(m)),
            "moneyness_max": float(np.max(m)),
        },
    }
    return DynamicCollocationRefresh(interior=interior, report=report)
