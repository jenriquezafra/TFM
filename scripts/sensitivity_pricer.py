import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import PowerNorm, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ANN_pricer import ANN
from src.solvers.implied_vol import IV_Brent, IV_LM


PARAM_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]
HESTON_PARAM_COUNT = 5
STRICT_POSITIVE_PARAMS = {"kappa", "gamma", "bar_v", "v0", "tau", "moneyness"}


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _as_range(value) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    val = float(value)
    return val, val


def _expand_range(min_val: float, max_val: float, factor: float) -> Tuple[float, float]:
    span = max_val - min_val
    center = 0.5 * (min_val + max_val)
    expanded_half = 0.5 * span * factor
    return center - expanded_half, center + expanded_half


class SensitivityPricer2D:
    def __init__(self, config_path: Path, verbose: bool = True, model_dir_override: str | None = None) -> None:
        self.project_root = PROJECT_ROOT
        self.config_path = config_path
        self.config = _load_yaml(config_path)
        self.verbose = verbose

        model_dir = model_dir_override if model_dir_override is not None else self.config["model_dir"]
        self.run_dir = self._resolve_run_dir(model_dir)
        self.model_cfg = _load_yaml(self.run_dir / "model_architecture_copy.yaml")
        self.data_cfg = _load_yaml(self.run_dir / "synth_copy.yaml")

        self.fig_dir = self.run_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.param_ranges = self._build_param_ranges()
        self.range_overrides = self.config.get("parameters", {}).get("ranges", {})
        self.fixed_values = self.config["parameters"]["fixed_values"]
        self.pairs = [tuple(pair) for pair in self.config["parameters"]["pairs"]]
        if len(self.pairs) != 3:
            raise ValueError(
                f"sensitivity_config.yaml must define exactly 3 parameter pairs for a 3x2 grid; got {len(self.pairs)}"
            )
        self.grid_factor = float(self.config["grid"].get("expansion_factor", 1.2))
        self.extrapolation_ranges = self.config["grid"].get("extrapolation_ranges", {})
        self.n_points = int(self.config["grid"]["n_points"])
        self.error_metric = self.config.get("error_metric", "rmse").lower()

        self.device = self._get_device()
        self.model = self._load_model()

        self.cos_params, self.opt_type, self.K, self.cos_interval_rule = self._parse_market_config()
        self.root_method, self.root_params = self._parse_root_finder()

    def _resolve_run_dir(self, model_dir: str) -> Path:
        runs_dir = self.project_root / "outputs" / "runs"
        if model_dir == "latest":
            candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
            if not candidates:
                raise FileNotFoundError(f"No run directories found under {runs_dir}")
            return max(candidates, key=lambda p: p.stat().st_mtime)
        return runs_dir / model_dir

    def _build_param_ranges(self) -> dict:
        ranges = {}
        heston_params = self.data_cfg["data"]["heston_params"]
        grid_params = self.data_cfg["data"]["grid"]

        for source in (heston_params, grid_params):
            for name, value in source.items():
                if isinstance(value, dict):
                    continue
                ranges[name] = _as_range(value)

        market_cfg = self.data_cfg.get("market", {})
        if "r" in market_cfg:
            ranges["r"] = _as_range(market_cfg["r"])

        return ranges

    def _range_for_param(self, name: str) -> Tuple[float, float]:
        if name in self.range_overrides:
            return _as_range(self.range_overrides[name])
        if name in self.param_ranges:
            return self.param_ranges[name]
        if name in self.fixed_values:
            return _as_range(self.fixed_values[name])
        raise KeyError(f"Missing range for parameter '{name}'")

    @staticmethod
    def _normalize_bounds(min_val: float, max_val: float) -> Tuple[float, float]:
        if min_val <= max_val:
            return min_val, max_val
        return max_val, min_val

    def _clamp_param_bounds(self, name: str, min_val: float, max_val: float) -> Tuple[float, float]:
        min_val, max_val = self._normalize_bounds(float(min_val), float(max_val))

        if name in STRICT_POSITIVE_PARAMS:
            min_val = max(1.0e-12, min_val)
            max_val = max(1.0e-12, max_val)
        if name == "rho":
            min_val = max(-0.999, min_val)
            max_val = min(0.999, max_val)

        if min_val > max_val:
            raise ValueError(f"Invalid bounds for '{name}' after clamping: [{min_val}, {max_val}]")
        return min_val, max_val

    def _grid_ranges_for_param(self, name: str) -> Tuple[float, float, float, float]:
        interp_min, interp_max = self._clamp_param_bounds(name, *self._range_for_param(name))
        if name in self.extrapolation_ranges:
            grid_min, grid_max = _as_range(self.extrapolation_ranges[name])
        else:
            grid_min, grid_max = _expand_range(interp_min, interp_max, self.grid_factor)
        grid_min, grid_max = self._clamp_param_bounds(name, grid_min, grid_max)

        if grid_min > interp_min or grid_max < interp_max:
            raise ValueError(
                f"Grid range for '{name}' must include interpolation range. "
                f"Interpolation=[{interp_min}, {interp_max}], Grid=[{grid_min}, {grid_max}]"
            )
        return interp_min, interp_max, grid_min, grid_max

    def _parse_market_config(self) -> Tuple[np.ndarray, str, float, str]:
        cos_cfg = self.data_cfg["cos_solver"]
        cos_params = np.array([float(cos_cfg["N"]), float(cos_cfg["L"])], dtype=np.float64)
        cos_interval_rule = cos_cfg.get("interval_rule", "sqrt_t")
        market_cfg = self.data_cfg["market"]
        opt_type = market_cfg.get("option_type", "put")
        K = float(market_cfg.get("K", 1.0))
        return cos_params, opt_type, K, cos_interval_rule

    def _parse_root_finder(self) -> Tuple[str, dict]:
        root_cfg = self.data_cfg.get("root_finder", {})
        method = root_cfg.get("method", "LM")
        methods_cfg = root_cfg.get("methods", {})
        if method == "brent_iv":
            cfg = methods_cfg.get("brent_iv", {})
            return method, {
                "iv_bounds": cfg.get("iv_bounds", [1.0e-6, 5.0]),
                "tol": cfg.get("tol", 1.0e-8),
                "max_iter": cfg.get("max_iter", 100),
            }
        if method == "LM":
            cfg = methods_cfg.get("LM", {})
            return method, {
                "sigma0": cfg.get("sigma0", 0.2),
            }
        raise ValueError(f"Unsupported root-finder method: {method}")

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> ANN:
        model = ANN(
            input_dim=self.model_cfg["input"]["dim"],
            hidden_dims=self.model_cfg["hidden"]["dims"],
            output_dim=self.model_cfg["output"]["dim"],
            activation=self.model_cfg["hidden"]["activation"],
            dropout_rate=self.model_cfg["hidden"]["dropout_rate"],
            initialization=self.model_cfg["hidden"]["initialization"],
        ).to(self.device)

        ckpt_path = self.run_dir / "checkpoints" / "model_best.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        return model

    def _grid_for_pair(
        self, param1: str, param2: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
        p1_interp_min, p1_interp_max, p1_grid_min, p1_grid_max = self._grid_ranges_for_param(param1)
        p2_interp_min, p2_interp_max, p2_grid_min, p2_grid_max = self._grid_ranges_for_param(param2)

        p1_values = np.linspace(p1_grid_min, p1_grid_max, self.n_points, dtype=np.float64)
        p2_values = np.linspace(p2_grid_min, p2_grid_max, self.n_points, dtype=np.float64)

        P1, P2 = np.meshgrid(p1_values, p2_values, indexing="ij")
        interpolation_mask = (
            (P1 >= p1_interp_min)
            & (P1 <= p1_interp_max)
            & (P2 >= p2_interp_min)
            & (P2 <= p2_interp_max)
        )
        return P1, P2, interpolation_mask, (p1_interp_min, p1_interp_max), (p2_interp_min, p2_interp_max)

    def _build_feature_grid(
        self, param1: str, param2: str, P1: np.ndarray, P2: np.ndarray
    ) -> np.ndarray:
        grid = {}
        for param in PARAM_ORDER:
            if param == param1:
                grid[param] = P1
            elif param == param2:
                grid[param] = P2
            else:
                if param not in self.fixed_values:
                    raise KeyError(f"Missing fixed value for parameter '{param}'")
                grid[param] = np.full_like(P1, float(self.fixed_values[param]), dtype=np.float64)

        return np.stack([grid[param] for param in PARAM_ORDER], axis=-1)

    def _predict_nn(self, x_flat: np.ndarray) -> np.ndarray:
        x_tensor = torch.from_numpy(x_flat).float()
        self.model.eval()
        x = x_tensor.to(self.device)
        with torch.inference_mode():
            y_pred = self.model(x)
        return y_pred.cpu().numpy()

    def _compute_iv_surface(self, x_flat: np.ndarray) -> np.ndarray:
        total = x_flat.shape[0]
        iv = np.full(total, fill_value=np.nan, dtype=np.float64)
        failed = 0

        progress_step = max(1, total // 10)
        for idx in range(total):
            params_heston = x_flat[idx, :HESTON_PARAM_COUNT]
            S0 = x_flat[idx, 5] * self.K
            tau = x_flat[idx, 6]
            r = x_flat[idx, 7]
            try:
                iv[idx] = self._compute_single_iv(params_heston, S0, tau, r)
            except Exception:
                failed += 1
                iv[idx] = np.nan
            if self.verbose and (idx + 1) % progress_step == 0:
                print(f"Heston IV progress: {idx + 1}/{total}")

        if self.verbose and failed > 0:
            print(f"Heston IV: {failed}/{total} points failed and were set to NaN")
        return iv

    def _compute_single_iv(
        self, params_heston: np.ndarray, S0: float, tau: float, r: float
    ) -> float:
        if self.root_method == "brent_iv":
            return float(
                IV_Brent(
                    params_Heston=params_heston,
                    S0=S0,
                    K=self.K,
                    tau=tau,
                    r=r,
                    COS_params=self.cos_params,
                    cos_interval_rule=self.cos_interval_rule,
                    opt_type=self.opt_type,
                    iv_bounds=self.root_params["iv_bounds"],
                    tol=self.root_params["tol"],
                    max_iter=self.root_params["max_iter"],
                )
            )
        if self.root_method == "LM":
            return float(
                IV_LM(
                    params_Heston=params_heston,
                    S0=S0,
                    K=self.K,
                    tau=tau,
                    r=r,
                    COS_params=self.cos_params,
                    cos_interval_rule=self.cos_interval_rule,
                    opt_type=self.opt_type,
                    sigma0=self.root_params["sigma0"],
                )
            )
        raise ValueError(f"Unsupported root-finder method: {self.root_method}")

    def _compute_error_surface(self, y_nn: np.ndarray, y_heston: np.ndarray) -> np.ndarray:
        diff = y_nn - y_heston
        if self.error_metric == "rmse":
            return np.sqrt(diff**2)
        if self.error_metric == "mse":
            return diff**2
        if self.error_metric == "abs_diff":
            return np.abs(diff)
        if self.error_metric == "difference":
            return diff
        raise ValueError(f"Unsupported error metric: {self.error_metric}")

    @staticmethod
    def _safe_limits(values: List[np.ndarray], fallback: Tuple[float, float]) -> Tuple[float, float]:
        finite_parts = [v[np.isfinite(v)] for v in values if v.size > 0]
        finite_parts = [v for v in finite_parts if v.size > 0]
        if not finite_parts:
            return fallback
        finite = np.concatenate(finite_parts)
        if finite.size == 0:
            return fallback
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if np.isclose(lo, hi):
            eps = max(1.0e-8, abs(lo) * 1.0e-3)
            return lo - eps, hi + eps
        return lo, hi

    def _plot_grid_3x2(
        self,
        rows: List[
            Tuple[
                str,
                str,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                Tuple[float, float],
                Tuple[float, float],
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ],
    ) -> Path:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 16), constrained_layout=True)

        for (
            row_idx,
            (
                param1,
                param2,
                P1,
                P2,
                _interpolation_mask,
                p1_interp_bounds,
                p2_interp_bounds,
                y_nn,
                y_heston,
                error_surface,
            ),
        ) in enumerate(rows):
            ax_val = axes[row_idx, 0]
            ax_err = axes[row_idx, 1]

            val_lo, val_hi = self._safe_limits([y_nn, y_heston], fallback=(0.0, 1.0))
            err_lo, err_hi = self._safe_limits([error_surface], fallback=(0.0, 1.0))

            mesh_val = ax_val.pcolormesh(
                P1, P2, np.ma.masked_invalid(y_nn), shading="auto", cmap="viridis", vmin=val_lo, vmax=val_hi
            )
            try:
                ax_val.contour(
                    P1,
                    P2,
                    np.ma.masked_invalid(y_heston),
                    colors="white",
                    linewidths=0.6,
                    alpha=0.85,
                    levels=8,
                )
            except Exception:
                if self.verbose:
                    print(f"Warning: could not draw Heston contours for pair {param1}/{param2}")
            ax_val.set_xlabel(param1)
            ax_val.set_ylabel(param2)
            ax_val.set_title(f"{param1} vs {param2} | IV values (NN + Heston contours)")
            fig.colorbar(mesh_val, ax=ax_val, shrink=0.85)

            if self.error_metric == "difference":
                finite_abs = np.abs(error_surface[np.isfinite(error_surface)])
                if finite_abs.size > 0:
                    p90 = float(np.percentile(finite_abs, 90.0))
                    lim = float(np.percentile(finite_abs, 99.0))
                    if p90 > 0:
                        lim = min(lim, 5.0 * p90)
                    lim = lim if lim > 0 else float(finite_abs.max())
                else:
                    lim = 1.0
                lim = max(lim, 1.0e-12)
                err_norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
                err_cmap = "gray"
            else:
                finite = error_surface[np.isfinite(error_surface)]
                if finite.size > 0:
                    p90 = float(np.percentile(finite, 90.0))
                    vmax = float(np.percentile(finite, 99.0))
                    if p90 > 0:
                        vmax = min(vmax, 5.0 * p90)
                    vmax = vmax if vmax > 0 else float(finite.max())
                else:
                    vmax = 1.0
                vmax = max(vmax, 1.0e-12)
                err_norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)
                err_cmap = "gray"

            mesh_err = ax_err.pcolormesh(
                P1,
                P2,
                np.ma.masked_invalid(error_surface),
                shading="auto",
                cmap=err_cmap,
                norm=err_norm,
            )
            ax_err.set_xlabel(param1)
            ax_err.set_ylabel(param2)
            if self.error_metric == "abs_diff":
                ax_err.set_title(f"{param1} vs {param2} | |NN - Heston|")
            else:
                ax_err.set_title(f"{param1} vs {param2} | {self.error_metric.upper()} error")
            fig.colorbar(mesh_err, ax=ax_err, shrink=0.85)

            for ax in (ax_val, ax_err):
                ax.axvline(p1_interp_bounds[0], linestyle="--", linewidth=1.0, color="black")
                ax.axvline(p1_interp_bounds[1], linestyle="--", linewidth=1.0, color="black")
                ax.axhline(p2_interp_bounds[0], linestyle="--", linewidth=1.0, color="black")
                ax.axhline(p2_interp_bounds[1], linestyle="--", linewidth=1.0, color="black")
                ax.text(
                    0.02,
                    0.02,
                    "interpolation limits",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
                )

        fig_path = self.fig_dir / "sensitivity_grid_3x2.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path

    def run(self) -> None:
        plot_rows = []
        for param1, param2 in self.pairs:
            if self.verbose:
                print(f"Processing pair: {param1} vs {param2}")

            P1, P2, interpolation_mask, p1_interp_bounds, p2_interp_bounds = self._grid_for_pair(param1, param2)
            features = self._build_feature_grid(param1, param2, P1, P2)
            x_flat = features.reshape(-1, len(PARAM_ORDER))

            y_nn = self._predict_nn(x_flat).reshape(P1.shape)
            y_heston = self._compute_iv_surface(x_flat).reshape(P1.shape)
            error_surface = self._compute_error_surface(y_nn, y_heston)
            plot_rows.append(
                (
                    param1,
                    param2,
                    P1,
                    P2,
                    interpolation_mask,
                    p1_interp_bounds,
                    p2_interp_bounds,
                    y_nn,
                    y_heston,
                    error_surface,
                )
            )

        fig_path = self._plot_grid_3x2(plot_rows)
        if self.verbose:
            print(f"Saved figure: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D sensitivity error surfaces.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "sensitivity_config.yaml"),
        help="Path to sensitivity config.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional run dir name under outputs/runs. Overrides 'model_dir' from config.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args()

    pricer = SensitivityPricer2D(
        Path(args.config),
        verbose=not args.quiet,
        model_dir_override=args.model_dir,
    )
    pricer.run()


if __name__ == "__main__":
    main()
