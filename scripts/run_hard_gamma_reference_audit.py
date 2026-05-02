from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.greeks.heston_cf_greeks import HestonCFGreeksSettings, heston_cf_greeks_scalar


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hard_gamma_reference_audit.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _bounds(raw: list[float] | tuple[float, float], default: tuple[float, float]) -> tuple[float, float]:
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("Bounds must be [low, high]")
    low, high = float(raw[0]), float(raw[1])
    if high < low:
        raise ValueError(f"Invalid bounds: {raw}")
    if high == low:
        high = float(np.nextafter(low, np.inf))
    return low, high


def _uniform(rng: np.random.Generator, n: int, bounds: tuple[float, float]) -> np.ndarray:
    return rng.uniform(bounds[0], bounds[1], size=int(n)).astype(np.float64)


def _log_uniform(rng: np.random.Generator, n: int, bounds: tuple[float, float]) -> np.ndarray:
    low = max(float(bounds[0]), np.finfo(np.float64).tiny)
    high = max(float(bounds[1]), low * (1.0 + 1.0e-12))
    return np.exp(rng.uniform(np.log(low), np.log(high), size=int(n))).astype(np.float64)


def _sample_hard_points(cfg: dict, *, n: int, rng: np.random.Generator) -> np.ndarray:
    sampling = cfg.get("sampling", {})
    domain = sampling.get("domain", {})
    heston = domain.get("heston_params", {})
    hard = sampling.get("hard_region", {})

    tau_min = float(sampling.get("tau_min", 1.0e-4))
    tau_max = float(hard.get("tau_max", 0.05))
    x_abs = float(hard.get("x_abs", 0.03))

    tau = _log_uniform(rng, n, (tau_min, tau_max))
    x = rng.uniform(-x_abs, x_abs, size=n)
    v = _uniform(rng, n, _bounds(domain.get("v", [0.01, 0.5]), (0.01, 0.5)))
    rho = _uniform(rng, n, _bounds(heston.get("rho", [-0.9, 0.0]), (-0.9, 0.0)))
    kappa = _uniform(rng, n, _bounds(heston.get("kappa", [0.0, 3.0]), (0.0, 3.0)))
    gamma = _uniform(rng, n, _bounds(heston.get("gamma", [0.01, 0.8]), (0.01, 0.8)))
    bar_v = _uniform(rng, n, _bounds(heston.get("bar_v", [0.01, 0.5]), (0.01, 0.5)))
    r = _uniform(rng, n, _bounds(domain.get("r", [0.0, 0.05]), (0.0, 0.05)))
    return np.column_stack([tau, np.exp(x), v, rho, kappa, gamma, bar_v, r]).astype(np.float64)


def _cf_settings(cfg: dict) -> HestonCFGreeksSettings:
    cf = cfg.get("cf_integration", {})
    return HestonCFGreeksSettings(
        u_min=float(cf.get("u_min", 1.0e-6)),
        u_max=float(cf.get("u_max", 200.0)),
        n_u=int(cf.get("n_u", 1200)),
    )


def _cf_put(row: np.ndarray, settings: HestonCFGreeksSettings) -> dict[str, float]:
    return heston_cf_greeks_scalar(
        option_type="put",
        S0=float(row[1]),
        K=1.0,
        tau=float(row[0]),
        r=float(row[7]),
        rho=float(row[3]),
        kappa=float(row[4]),
        gamma=float(row[5]),
        bar_v=float(row[6]),
        v0=float(row[2]),
        settings=settings,
    )


def _finite_difference_gamma_mm(
    *,
    row: np.ndarray,
    h_x: float,
    settings: HestonCFGreeksSettings,
) -> float:
    x0 = float(np.log(row[1]))
    row_m = row.copy()
    row_p = row.copy()
    row_m[1] = float(np.exp(x0 - h_x))
    row_p[1] = float(np.exp(x0 + h_x))
    p_m = _cf_put(row_m, settings)["price"]
    p_0 = _cf_put(row, settings)["price"]
    p_p = _cf_put(row_p, settings)["price"]
    v_x = (p_p - p_m) / (2.0 * h_x)
    v_xx = (p_p - 2.0 * p_0 + p_m) / (h_x * h_x)
    m = float(row[1])
    return float((v_xx - v_x) / (m * m))


def _error_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "p99_abs_error": float("nan"), "max_abs_error": float("nan")}
    abs_values = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(abs_values)),
        "p99_abs_error": float(np.percentile(abs_values, 99.0)),
        "max_abs_error": float(np.max(abs_values)),
    }


def run_reference_audit(
    *,
    config_path: Path,
    max_points: int | None = None,
) -> Path:
    cfg = _load_yaml(config_path)
    output_dir = Path(cfg.get("outputs", {}).get("output_dir", "outputs/pinn/acv_hard_patch_experimental/reference_audit"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("meta", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    n = int(cfg.get("sampling", {}).get("n_points", 128))
    if max_points is not None:
        n = min(n, int(max_points))
    h_values = [float(x) for x in cfg.get("finite_difference", {}).get("h_x_values", [5.0e-4, 1.0e-3, 2.0e-3])]
    if n <= 0:
        raise ValueError("n_points must be > 0")
    if not h_values:
        raise ValueError("finite_difference.h_x_values must not be empty")

    settings = _cf_settings(cfg)
    points = _sample_hard_points(cfg, n=n, rng=rng)
    rows: list[dict] = []
    for i, row in enumerate(points):
        ref = _cf_put(row, settings)
        base = {
            "idx": i,
            "tau": float(row[0]),
            "moneyness": float(row[1]),
            "log_moneyness": float(np.log(row[1])),
            "v": float(row[2]),
            "rho": float(row[3]),
            "kappa": float(row[4]),
            "gamma": float(row[5]),
            "bar_v": float(row[6]),
            "r": float(row[7]),
            "gamma_cf": float(ref["gamma"]),
        }
        for h_x in h_values:
            fd = _finite_difference_gamma_mm(row=row, h_x=h_x, settings=settings)
            rows.append({**base, "h_x": float(h_x), "gamma_fd": fd, "error": fd - float(ref["gamma"])})

    df = pd.DataFrame(rows)
    detail_path = output_dir / "hard_gamma_reference_audit.csv"
    df.to_csv(detail_path, index=False)
    summary = {
        "config_path": str(config_path),
        "n_points": int(n),
        "h_x_values": h_values,
        "detail_file": str(detail_path),
        "by_h_x": {},
    }
    for h_x, group in df.groupby("h_x"):
        summary["by_h_x"][float(h_x)] = _error_metrics(group["error"].to_numpy(dtype=np.float64))
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"[AUDIT] wrote {summary_path}")
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit hard-region Heston CF Gamma against price finite differences.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap for quick smoke runs.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_reference_audit(config_path=args.config, max_points=args.max_points)


if __name__ == "__main__":
    main()
