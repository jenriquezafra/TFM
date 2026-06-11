from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from src.datasets.make_synth import generate_all


PINN_FEATURE_ORDER = (
    "tau",
    "moneyness",
    "v",
    "rho",
    "kappa",
    "gamma",
    "bar_v",
    "r",
)

PINN_LOG_FEATURE_ORDER = (
    "tau",
    "log_moneyness",
    "v",
    "rho",
    "kappa",
    "gamma",
    "bar_v",
    "r",
)


def _coordinate_space_key(sampling_config: dict | str | None) -> str:
    if isinstance(sampling_config, dict):
        raw = sampling_config.get(
            "coordinate_space",
            sampling_config.get("coordinate", sampling_config.get("spot_coordinate", "moneyness")),
        )
    else:
        raw = sampling_config or "moneyness"
    key = str(raw).strip().lower()
    if key in {"moneyness", "m", "raw"}:
        return "moneyness"
    if key in {"log_moneyness", "log-moneyness", "x"}:
        return "log_moneyness"
    raise ValueError("sampling.coordinate_space must be 'moneyness' or 'log_moneyness'.")


def _feature_order_for_coordinate(coordinate_space: str) -> tuple[str, ...]:
    return PINN_LOG_FEATURE_ORDER if _coordinate_space_key(coordinate_space) == "log_moneyness" else PINN_FEATURE_ORDER


def _convert_spot_coordinate(data: np.ndarray, coordinate_space: str) -> np.ndarray:
    if _coordinate_space_key(coordinate_space) != "log_moneyness":
        return data.astype(np.float32, copy=False)
    out = data.astype(np.float32, copy=True)
    if (out[:, 1] <= 0.0).any():
        raise ValueError("log_moneyness collocation requires strictly positive moneyness before conversion.")
    out[:, 1] = np.log(out[:, 1].astype(np.float64)).astype(np.float32)
    return out


def _moneyness_column(data: np.ndarray, coordinate_space: str) -> np.ndarray:
    if _coordinate_space_key(coordinate_space) == "log_moneyness":
        return np.exp(data[:, 1].astype(np.float64))
    return data[:, 1].astype(np.float64)


def _outside_bounds(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    span = max(1.0, abs(float(bounds[0])), abs(float(bounds[1])))
    tol = 1.0e-6 * span
    return (values < float(bounds[0]) - tol) | (values > float(bounds[1]) + tol)


def _as_bounds(value: float | Sequence[float], *, name: str) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
        if low > high:
            raise ValueError(f"Invalid bounds for '{name}': [{low}, {high}]")
        if low == high:
            high = float(np.nextafter(low, np.inf))
        return np.array([low, high], dtype=np.float64)

    scalar = float(value)
    return np.array([scalar, float(np.nextafter(scalar, np.inf))], dtype=np.float64)


def _get_sampling_sizes(sampling_config: dict) -> tuple[int, int, int]:
    sizes_cfg = sampling_config.get("sizes", {})
    if sizes_cfg:
        n_interior = int(sizes_cfg.get("n_interior", 20_000))
        n_terminal = int(sizes_cfg.get("n_terminal", 5_000))
        n_lower = int(sizes_cfg.get("n_lower", 5_000))
    else:
        # Backward compatibility with previous scaffold schema.
        coll_cfg = sampling_config.get("collocation", {})
        n_interior = int(coll_cfg.get("n_interior", 20_000))
        n_terminal = int(coll_cfg.get("n_terminal", 5_000))
        n_lower = int(coll_cfg.get("n_boundary", 5_000))

    for name, value in (
        ("n_interior", n_interior),
        ("n_terminal", n_terminal),
        ("n_lower", n_lower),
    ):
        if value <= 0:
            raise ValueError(f"sampling.sizes.{name} must be > 0. Got {value}.")

    return n_interior, n_terminal, n_lower


def _get_optional_sampling_size(sampling_config: dict, key: str) -> int:
    sizes_cfg = sampling_config.get("sizes", {})
    value = int(sizes_cfg.get(key, 0)) if isinstance(sizes_cfg, dict) else 0
    if value < 0:
        raise ValueError(f"sampling.sizes.{key} must be >= 0. Got {value}.")
    return value


def _extract_fixed_theta(
    *,
    theta_star: Sequence[float] | None,
    parameter_order: Sequence[str] | None,
) -> dict[str, float]:
    if theta_star is None:
        raise ValueError("sampling.mode='fixed_theta' requires theta_star from calibration.")

    theta = np.asarray(theta_star, dtype=np.float64).reshape(-1)
    if theta.size < 4:
        raise ValueError(
            "sampling.mode='fixed_theta' requires at least 4 calibrated parameters: "
            "rho, kappa, gamma, bar_v."
        )

    default = {
        "rho": float(theta[0]),
        "kappa": float(theta[1]),
        "gamma": float(theta[2]),
        "bar_v": float(theta[3]),
    }

    if not parameter_order or len(parameter_order) != len(theta):
        return default

    lookup = {str(name).strip().lower(): float(value) for name, value in zip(parameter_order, theta)}
    aliases = {
        "rho": ("rho",),
        "kappa": ("kappa",),
        "gamma": ("gamma", "sigma", "sigma_v"),
        "bar_v": ("bar_v", "vbar", "v_bar", "theta"),
    }

    out = {}
    for key, names in aliases.items():
        found = None
        for name in names:
            if name in lookup:
                found = lookup[name]
                break
        out[key] = float(found if found is not None else default[key])
    return out


def _sampling_domain_and_ranges(
    *,
    sampling_config: dict,
    theta_star: Sequence[float] | None,
    parameter_order: Sequence[str] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    domain_cfg = sampling_config.get("domain", {})

    tau_bounds = _as_bounds(domain_cfg.get("tau", [0.0, 3.0]), name="tau")
    moneyness_bounds = _as_bounds(domain_cfg.get("moneyness", [0.0, 2.0]), name="moneyness")
    v_bounds = _as_bounds(domain_cfg.get("v", [0.01, 0.5]), name="v")
    r_bounds = _as_bounds(domain_cfg.get("r", [0.0, 0.05]), name="r")

    mode = str(sampling_config.get("mode", "fixed_theta")).strip().lower()
    if mode not in {"fixed_theta", "parametric_theta"}:
        raise ValueError(
            f"Unsupported sampling.mode='{mode}'. Use 'fixed_theta' or 'parametric_theta'."
        )

    if mode == "fixed_theta":
        fixed_theta = _extract_fixed_theta(theta_star=theta_star, parameter_order=parameter_order)
        param_ranges = np.vstack(
            [
                _as_bounds(fixed_theta["rho"], name="rho"),
                _as_bounds(fixed_theta["kappa"], name="kappa"),
                _as_bounds(fixed_theta["gamma"], name="gamma"),
                _as_bounds(fixed_theta["bar_v"], name="bar_v"),
                v_bounds,
            ]
        )
    else:
        heston_cfg = domain_cfg.get("heston_params", {})
        param_ranges = np.vstack(
            [
                _as_bounds(heston_cfg.get("rho", [-0.9, 0.0]), name="rho"),
                _as_bounds(heston_cfg.get("kappa", [0.0, 3.0]), name="kappa"),
                _as_bounds(heston_cfg.get("gamma", [0.01, 0.8]), name="gamma"),
                _as_bounds(heston_cfg.get("bar_v", [0.01, 0.5]), name="bar_v"),
                v_bounds,
            ]
        )

    return (
        {
            "tau": tau_bounds,
            "moneyness": moneyness_bounds,
            "v": v_bounds,
            "r": r_bounds,
        },
        param_ranges,
    )


def _sample_lhs_points(
    *,
    n_samples: int,
    seed: int,
    param_ranges: np.ndarray,
    tau_bounds: np.ndarray,
    moneyness_bounds: np.ndarray,
    r_bounds: np.ndarray,
) -> np.ndarray:
    grid_bounds = np.vstack([moneyness_bounds, tau_bounds])
    x_params, x_grid, x_r = generate_all(
        n_samples=n_samples,
        param_ranges=param_ranges,
        grid_bounds=grid_bounds,
        r_bounds=r_bounds,
        seed=seed,
    )

    # Output order for PINN residual assembly:
    # [tau, moneyness, v, rho, kappa, gamma, bar_v, r]
    x = np.column_stack(
        [
            x_grid[:, 1],   # tau
            x_grid[:, 0],   # moneyness
            x_params[:, 4], # v
            x_params[:, 0], # rho
            x_params[:, 1], # kappa
            x_params[:, 2], # gamma
            x_params[:, 3], # bar_v
            x_r[:, 0],      # r
        ]
    )
    return x.astype(np.float32, copy=False)


def _sample_kink_band_points(
    *,
    n_samples: int,
    seed: int,
    param_ranges: np.ndarray,
    tau_bounds: np.ndarray,
    moneyness_bounds: np.ndarray,
    r_bounds: np.ndarray,
    config: dict,
) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, len(PINN_FEATURE_ORDER)), dtype=np.float32)
    tau_c = float(config.get("tau_c", tau_bounds[1]))
    tau_high = min(float(tau_bounds[1]), max(float(tau_bounds[0]), tau_c))
    tau_band = np.array([float(tau_bounds[0]), float(np.nextafter(tau_high, np.inf))], dtype=np.float64)
    base = _sample_lhs_points(
        n_samples=n_samples,
        seed=seed,
        param_ranges=param_ranges,
        tau_bounds=tau_band,
        moneyness_bounds=moneyness_bounds,
        r_bounds=r_bounds,
    )
    rng = np.random.default_rng(seed + 7919)
    c = float(config.get("c", 2.0))
    eps = float(config.get("epsilon", 1.0e-8))
    tau = np.maximum(base[:, 0].astype(np.float64), 0.0)
    v = np.maximum(base[:, 2].astype(np.float64), 0.0)
    width = c * np.sqrt(v * tau + eps)
    x = rng.uniform(low=-width, high=width)
    m = np.exp(x)
    base[:, 1] = np.clip(m, float(moneyness_bounds[0]), float(moneyness_bounds[1])).astype(np.float32)
    return base.astype(np.float32, copy=False)


def _validate_lhs_set(
    *,
    data: np.ndarray,
    name: str,
    feature_order: Sequence[str],
    coordinate_space: str,
    tau_bounds: np.ndarray,
    moneyness_bounds: np.ndarray,
    v_bounds: np.ndarray,
    r_bounds: np.ndarray,
) -> None:
    if data.ndim != 2 or data.shape[1] != len(feature_order):
        raise ValueError(
            f"{name} must have shape [N,{len(feature_order)}]. Got {tuple(data.shape)}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values.")

    tau = data[:, 0]
    m = _moneyness_column(data, coordinate_space)
    v = data[:, 2]
    r = data[:, 7]

    if name != "terminal" and _outside_bounds(tau, tau_bounds).any():
        raise ValueError(f"{name}.tau outside configured bounds {tau_bounds.tolist()}.")
    if name != "lower" and _outside_bounds(m, moneyness_bounds).any():
        raise ValueError(f"{name}.moneyness outside configured bounds {moneyness_bounds.tolist()}.")
    if _outside_bounds(v, v_bounds).any():
        raise ValueError(f"{name}.v outside configured bounds {v_bounds.tolist()}.")
    if _outside_bounds(r, r_bounds).any():
        raise ValueError(f"{name}.r outside configured bounds {r_bounds.tolist()}.")

    if name == "terminal" and not np.allclose(tau, 0.0):
        raise ValueError("Terminal set must satisfy tau=0 for all points.")
    if name == "lower" and not np.allclose(m, m[0]):
        raise ValueError("Lower-boundary set must have constant moneyness.")
    if name == "right" and not np.allclose(m, m[0]):
        raise ValueError("Right-boundary set must have constant moneyness.")
    if name == "v_zero" and not np.allclose(v, v[0]):
        raise ValueError("Variance-zero boundary set must have constant v.")


def _write_collocation_manifest(
    *,
    manifest_path: Path,
    interior_path: Path,
    terminal_path: Path,
    lower_path: Path,
    right_path: Path | None,
    v_zero_path: Path | None,
    sampling_config: dict,
    feature_order: Sequence[str],
    interior: np.ndarray,
    terminal: np.ndarray,
    lower: np.ndarray,
    right: np.ndarray | None,
    v_zero: np.ndarray | None,
    tau_bounds: np.ndarray,
    moneyness_bounds: np.ndarray,
    v_bounds: np.ndarray,
    r_bounds: np.ndarray,
) -> None:
    mode = str(sampling_config.get("mode", "fixed_theta")).strip().lower()
    coordinate_space = _coordinate_space_key(sampling_config)
    datasets = {
        "interior": str(interior_path),
        "terminal": str(terminal_path),
        "lower": str(lower_path),
    }
    if right_path is not None and right is not None and right.shape[0] > 0:
        datasets["right"] = str(right_path)
    if v_zero_path is not None and v_zero is not None and v_zero.shape[0] > 0:
        datasets["v_zero"] = str(v_zero_path)

    sizes = {
        "n_interior": int(interior.shape[0]),
        "n_terminal": int(terminal.shape[0]),
        "n_lower": int(lower.shape[0]),
    }
    if right is not None:
        sizes["n_right"] = int(right.shape[0])
    if v_zero is not None:
        sizes["n_v_zero"] = int(v_zero.shape[0])

    manifest = {
        "dataset_format": "parquet",
        "datasets": datasets,
        "feature_order": list(feature_order),
        "coordinate_space": coordinate_space,
        "sampling_strategy": str(sampling_config.get("strategy", "lhs_static")),
        "sampling_mode": mode,
        "seed": int(sampling_config.get("seed", 42)),
        "sizes": sizes,
        "domain": {
            "tau": [float(tau_bounds[0]), float(tau_bounds[1])],
            "moneyness": [float(moneyness_bounds[0]), float(moneyness_bounds[1])],
            "v": [float(v_bounds[0]), float(v_bounds[1])],
            "r": [float(r_bounds[0]), float(r_bounds[1])],
        },
    }
    if coordinate_space == "log_moneyness":
        manifest["domain"]["log_moneyness"] = [
            float(np.log(moneyness_bounds[0])),
            float(np.log(moneyness_bounds[1])),
        ]
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset suffix '{path.suffix}'")
    if df.empty:
        raise ValueError(f"Dataset file is empty: {path}")
    return df


def _validate_target_column(
    quotes_df: pd.DataFrame,
    *,
    target_column: str,
) -> str:
    if target_column in quotes_df.columns:
        return target_column
    raise KeyError(
        f"Required target column '{target_column}' not found. "
        f"Available columns: {list(quotes_df.columns)}"
    )


def build_supervised_xy(
    *,
    theta_star: Sequence[float],
    quotes_df: pd.DataFrame,
    feature_columns: Sequence[str] = ("moneyness", "tau", "r"),
    target_column: str = "price_market",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Build X,y from CaNN quotes and calibrated theta.
    """
    missing = [col for col in feature_columns if col not in quotes_df.columns]
    if missing:
        raise KeyError(
            f"Missing feature columns: {missing}. "
            f"Available columns: {list(quotes_df.columns)}"
        )

    target_col_used = _validate_target_column(
        quotes_df,
        target_column=target_column,
    )

    theta = np.asarray(theta_star, dtype=np.float32).reshape(1, -1)
    if theta.size == 0:
        raise ValueError("theta_star cannot be empty")

    n_samples = int(len(quotes_df))
    theta_block = np.repeat(theta, repeats=n_samples, axis=0)
    market_block = quotes_df.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
    x = np.concatenate([theta_block, market_block], axis=1)
    y = quotes_df.loc[:, target_col_used].to_numpy(dtype=np.float32).reshape(-1, 1)

    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values")
    if not np.isfinite(y).all():
        raise ValueError("Target vector contains non-finite values")

    return x, y, target_col_used


def build_supervised_dataset(
    *,
    cann_quotes_path: Path,
    theta_star: Sequence[float],
    output_dir: Path,
    feature_columns: Sequence[str] = ("moneyness", "tau", "r"),
    target_column: str = "price_market",
) -> Path:
    """
    Build and persist supervised dataset artifacts from CaNN outputs.
    """
    quotes_df = _read_table(cann_quotes_path)
    x, y, target_col_used = build_supervised_xy(
        theta_star=theta_star,
        quotes_df=quotes_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "supervised_dataset.npz"
    np.savez(
        dataset_path,
        X=x,
        y=y,
        theta_star=np.asarray(theta_star, dtype=np.float32),
    )

    manifest = {
        "source_quotes_file": str(cann_quotes_path),
        "dataset_file": str(dataset_path),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "feature_columns": [f"theta_{i}" for i in range(len(theta_star))]
        + list(feature_columns),
        "target_column_requested": target_column,
        "target_column_used": target_col_used,
    }
    manifest_path = output_dir / "supervised_dataset_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return dataset_path


def build_collocation_dataset(
    *,
    sampling_config: dict,
    output_dir: Path,
    theta_star: Sequence[float] | None = None,
    parameter_order: Sequence[str] | None = None,
) -> Path:
    """
    Build and persist LHS datasets for interior / terminal / lower-boundary PINN terms.
    """
    return build_lhs_pinn_sets(
        sampling_config=sampling_config,
        output_dir=output_dir,
        theta_star=theta_star,
        parameter_order=parameter_order,
    )


def build_lhs_pinn_sets(
    *,
    sampling_config: dict,
    output_dir: Path,
    theta_star: Sequence[float] | None = None,
    parameter_order: Sequence[str] | None = None,
) -> Path:
    strategy = str(sampling_config.get("strategy", "lhs_static")).strip().lower()
    if strategy not in {"lhs_static", "lhs"}:
        raise ValueError(
            f"Unsupported sampling.strategy='{strategy}'. Use 'lhs_static' or 'lhs'."
        )

    n_interior, n_terminal, n_lower = _get_sampling_sizes(sampling_config)
    n_right = _get_optional_sampling_size(sampling_config, "n_right")
    n_v_zero = _get_optional_sampling_size(sampling_config, "n_v_zero")
    seed = int(sampling_config.get("seed", 42))
    coordinate_space = _coordinate_space_key(sampling_config)
    feature_order = _feature_order_for_coordinate(coordinate_space)

    domain, param_ranges = _sampling_domain_and_ranges(
        sampling_config=sampling_config,
        theta_star=theta_star,
        parameter_order=parameter_order,
    )
    tau_bounds = domain["tau"]
    moneyness_bounds = domain["moneyness"]
    v_bounds = domain["v"]
    r_bounds = domain["r"]
    if coordinate_space == "log_moneyness" and moneyness_bounds[0] <= 0.0:
        raise ValueError(
            "sampling.coordinate_space='log_moneyness' requires domain.moneyness[0] > 0."
        )

    kink_cfg = sampling_config.get("kink_band", {})
    kink_enabled = isinstance(kink_cfg, dict) and bool(kink_cfg.get("enabled", False))
    kink_fraction = float(kink_cfg.get("fraction", 0.0)) if kink_enabled else 0.0
    if kink_fraction < 0.0 or kink_fraction > 1.0:
        raise ValueError("sampling.kink_band.fraction must be in [0, 1].")
    n_kink = int(round(n_interior * kink_fraction))
    n_global = n_interior - n_kink

    interior_global = _sample_lhs_points(
        n_samples=n_global,
        seed=seed,
        param_ranges=param_ranges,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        r_bounds=r_bounds,
    )
    if n_kink > 0:
        interior_kink = _sample_kink_band_points(
            n_samples=n_kink,
            seed=seed + 101,
            param_ranges=param_ranges,
            tau_bounds=tau_bounds,
            moneyness_bounds=moneyness_bounds,
            r_bounds=r_bounds,
            config=kink_cfg,
        )
        interior = np.vstack([interior_global, interior_kink]).astype(np.float32, copy=False)
    else:
        interior = interior_global
    terminal = _sample_lhs_points(
        n_samples=n_terminal,
        seed=seed + 1,
        param_ranges=param_ranges,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        r_bounds=r_bounds,
    )
    terminal[:, 0] = 0.0

    lower = _sample_lhs_points(
        n_samples=n_lower,
        seed=seed + 2,
        param_ranges=param_ranges,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        r_bounds=r_bounds,
    )
    boundary_cfg = sampling_config.get("boundaries", {})
    boundary_cfg = boundary_cfg if isinstance(boundary_cfg, dict) else {}
    lower_moneyness = float(boundary_cfg.get("lower_moneyness", 0.0))
    right_moneyness = float(boundary_cfg.get("right_moneyness", moneyness_bounds[1]))
    v_zero_value = float(boundary_cfg.get("v_zero_value", 0.0))
    if coordinate_space == "log_moneyness":
        for boundary_name, boundary_value in (
            ("lower_moneyness", lower_moneyness),
            ("right_moneyness", right_moneyness),
        ):
            if boundary_value <= 0.0:
                raise ValueError(
                    "sampling.coordinate_space='log_moneyness' requires "
                    f"boundaries.{boundary_name} > 0."
                )
    lower[:, 1] = lower_moneyness

    right = None
    if n_right > 0:
        right = _sample_lhs_points(
            n_samples=n_right,
            seed=seed + 3,
            param_ranges=param_ranges,
            tau_bounds=tau_bounds,
            moneyness_bounds=moneyness_bounds,
            r_bounds=r_bounds,
        )
        right[:, 1] = right_moneyness

    v_zero = None
    if n_v_zero > 0:
        v_zero = _sample_lhs_points(
            n_samples=n_v_zero,
            seed=seed + 4,
            param_ranges=param_ranges,
            tau_bounds=tau_bounds,
            moneyness_bounds=moneyness_bounds,
            r_bounds=r_bounds,
        )
        v_zero[:, 2] = v_zero_value

    _validate_lhs_set(
        data=interior,
        name="interior",
        feature_order=PINN_FEATURE_ORDER,
        coordinate_space="moneyness",
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )
    _validate_lhs_set(
        data=terminal,
        name="terminal",
        feature_order=PINN_FEATURE_ORDER,
        coordinate_space="moneyness",
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )
    _validate_lhs_set(
        data=lower,
        name="lower",
        feature_order=PINN_FEATURE_ORDER,
        coordinate_space="moneyness",
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )
    if right is not None:
        _validate_lhs_set(
            data=right,
            name="right",
            feature_order=PINN_FEATURE_ORDER,
            coordinate_space="moneyness",
            tau_bounds=tau_bounds,
            moneyness_bounds=np.array(
                [min(float(moneyness_bounds[0]), right_moneyness), max(float(moneyness_bounds[1]), right_moneyness)],
                dtype=np.float64,
            ),
            v_bounds=v_bounds,
            r_bounds=r_bounds,
        )
    if v_zero is not None:
        _validate_lhs_set(
            data=v_zero,
            name="v_zero",
            feature_order=PINN_FEATURE_ORDER,
            coordinate_space="moneyness",
            tau_bounds=tau_bounds,
            moneyness_bounds=moneyness_bounds,
            v_bounds=np.array(
                [min(float(v_bounds[0]), v_zero_value), max(float(v_bounds[1]), v_zero_value)],
                dtype=np.float64,
            ),
            r_bounds=r_bounds,
        )

    interior_out = _convert_spot_coordinate(interior, coordinate_space)
    terminal_out = _convert_spot_coordinate(terminal, coordinate_space)
    lower_out = _convert_spot_coordinate(lower, coordinate_space)
    right_out = _convert_spot_coordinate(right, coordinate_space) if right is not None else None
    v_zero_out = _convert_spot_coordinate(v_zero, coordinate_space) if v_zero is not None else None

    _validate_lhs_set(
        data=interior_out,
        name="interior",
        feature_order=feature_order,
        coordinate_space=coordinate_space,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )
    _validate_lhs_set(
        data=terminal_out,
        name="terminal",
        feature_order=feature_order,
        coordinate_space=coordinate_space,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )
    _validate_lhs_set(
        data=lower_out,
        name="lower",
        feature_order=feature_order,
        coordinate_space=coordinate_space,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )
    if right_out is not None:
        _validate_lhs_set(
            data=right_out,
            name="right",
            feature_order=feature_order,
            coordinate_space=coordinate_space,
            tau_bounds=tau_bounds,
            moneyness_bounds=np.array(
                [min(float(moneyness_bounds[0]), right_moneyness), max(float(moneyness_bounds[1]), right_moneyness)],
                dtype=np.float64,
            ),
            v_bounds=v_bounds,
            r_bounds=r_bounds,
        )
    if v_zero_out is not None:
        _validate_lhs_set(
            data=v_zero_out,
            name="v_zero",
            feature_order=feature_order,
            coordinate_space=coordinate_space,
            tau_bounds=tau_bounds,
            moneyness_bounds=moneyness_bounds,
            v_bounds=np.array(
                [min(float(v_bounds[0]), v_zero_value), max(float(v_bounds[1]), v_zero_value)],
                dtype=np.float64,
            ),
            r_bounds=r_bounds,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    interior_path = output_dir / "interior.parquet"
    terminal_path = output_dir / "terminal.parquet"
    lower_path = output_dir / "lower.parquet"
    right_path = output_dir / "right.parquet" if right is not None else None
    v_zero_path = output_dir / "v_zero.parquet" if v_zero is not None else None
    pd.DataFrame(interior_out, columns=feature_order).to_parquet(
        interior_path,
        engine="pyarrow",
        index=False,
    )
    pd.DataFrame(terminal_out, columns=feature_order).to_parquet(
        terminal_path,
        engine="pyarrow",
        index=False,
    )
    pd.DataFrame(lower_out, columns=feature_order).to_parquet(
        lower_path,
        engine="pyarrow",
        index=False,
    )
    if right_out is not None and right_path is not None:
        pd.DataFrame(right_out, columns=feature_order).to_parquet(
            right_path,
            engine="pyarrow",
            index=False,
        )
    if v_zero_out is not None and v_zero_path is not None:
        pd.DataFrame(v_zero_out, columns=feature_order).to_parquet(
            v_zero_path,
            engine="pyarrow",
            index=False,
        )

    manifest_path = output_dir / "collocation_sets_manifest.yaml"
    _write_collocation_manifest(
        manifest_path=manifest_path,
        interior_path=interior_path,
        terminal_path=terminal_path,
        lower_path=lower_path,
        right_path=right_path,
        v_zero_path=v_zero_path,
        sampling_config=sampling_config,
        feature_order=feature_order,
        interior=interior_out,
        terminal=terminal_out,
        lower=lower_out,
        right=right_out,
        v_zero=v_zero_out,
        tau_bounds=tau_bounds,
        moneyness_bounds=moneyness_bounds,
        v_bounds=v_bounds,
        r_bounds=r_bounds,
    )

    return manifest_path


def build_boundary_dataset(*, boundary_config: dict, output_dir: Path) -> Path:
    """
    Build boundary/initial-condition samples used by PINN constraints.
    """
    raise NotImplementedError(
        "PINN scaffold only: boundary dataset builder not implemented yet."
    )
