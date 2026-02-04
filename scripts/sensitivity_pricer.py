import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ANN_pricer import ANN
from src.solvers.implied_vol import IV_Brent, IV_LM


PARAM_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]
HESTON_PARAM_COUNT = 5


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
    def __init__(self, config_path: Path, verbose: bool = True) -> None:
        self.project_root = PROJECT_ROOT
        self.config_path = config_path
        self.config = _load_yaml(config_path)
        self.verbose = verbose

        self.run_dir = self._resolve_run_dir(self.config["model_dir"])
        self.model_cfg = _load_yaml(self.run_dir / "model_architecture_copy.yaml")
        self.data_cfg = _load_yaml(self.run_dir / "synth_copy.yaml")

        self.fig_dir = self.run_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.param_ranges = self._build_param_ranges()
        self.range_overrides = self.config.get("parameters", {}).get("ranges", {})
        self.fixed_values = self.config["parameters"]["fixed_values"]
        self.pairs = [tuple(pair) for pair in self.config["parameters"]["pairs"]]
        self.grid_factor = float(self.config["grid"]["expansion_factor"])
        self.n_points = int(self.config["grid"]["n_points"])
        self.error_metric = self.config.get("error_metric", "rmse").lower()

        self.device = self._get_device()
        self.model = self._load_model()

        self.cos_params, self.opt_type, self.K = self._parse_market_config()
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

    def _parse_market_config(self) -> Tuple[np.ndarray, str, float]:
        cos_cfg = self.data_cfg["cos_solver"]
        cos_params = np.array([float(cos_cfg["N"]), float(cos_cfg["L"])], dtype=np.float64)
        market_cfg = self.data_cfg["market"]
        opt_type = market_cfg.get("option_type", "put")
        K = float(market_cfg.get("K", 1.0))
        return cos_params, opt_type, K

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

    def _grid_for_pair(self, param1: str, param2: str) -> Tuple[np.ndarray, np.ndarray]:
        p1_min, p1_max = _expand_range(*self._range_for_param(param1), self.grid_factor)
        p2_min, p2_max = _expand_range(*self._range_for_param(param2), self.grid_factor)
        if param1 == "tau":
            p1_min = max(0.0, p1_min)
            p1_max = max(0.0, p1_max)
        if param2 == "tau":
            p2_min = max(0.0, p2_min)
            p2_max = max(0.0, p2_max)

        p1_values = np.linspace(p1_min, p1_max, self.n_points, dtype=np.float64)
        p2_values = np.linspace(p2_min, p2_max, self.n_points, dtype=np.float64)

        P1, P2 = np.meshgrid(p1_values, p2_values, indexing="ij")
        return P1, P2

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
        iv = np.empty(total, dtype=np.float64)

        progress_step = max(1, total // 10)
        for idx in range(total):
            params_heston = x_flat[idx, :HESTON_PARAM_COUNT]
            S0 = x_flat[idx, 5] * self.K
            tau = x_flat[idx, 6]
            r = x_flat[idx, 7]
            iv[idx] = self._compute_single_iv(params_heston, S0, tau, r)
            if self.verbose and (idx + 1) % progress_step == 0:
                print(f"Heston IV progress: {idx + 1}/{total}")

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
        raise ValueError(f"Unsupported error metric: {self.error_metric}")

    def _plot_surface_2d(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        error_surface: np.ndarray,
        param1: str,
        param2: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(P1, P2, error_surface, shading="auto", cmap="viridis")
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f"{self.error_metric.upper()} between NN and Heston solver")
        fig.colorbar(mesh, ax=ax, shrink=0.85)
        fig.tight_layout()

        fig_path = self.fig_dir / f"error_surface_2d_{param1}_{param2}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path

    def run(self) -> None:
        for param1, param2 in self.pairs:
            if self.verbose:
                print(f"Processing pair: {param1} vs {param2}")

            P1, P2 = self._grid_for_pair(param1, param2)
            features = self._build_feature_grid(param1, param2, P1, P2)
            x_flat = features.reshape(-1, len(PARAM_ORDER))

            y_nn = self._predict_nn(x_flat).reshape(P1.shape)
            y_heston = self._compute_iv_surface(x_flat).reshape(P1.shape)
            error_surface = self._compute_error_surface(y_nn, y_heston)

            fig_path = self._plot_surface_2d(P1, P2, error_surface, param1, param2)
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
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args()

    pricer = SensitivityPricer2D(Path(args.config), verbose=not args.quiet)
    pricer.run()


if __name__ == "__main__":
    main()
