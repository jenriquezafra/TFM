from __future__ import annotations

from typing import Sequence

from scipy.optimize import OptimizeResult, differential_evolution
import torch

from src.calibration.objective_func import (
    MarketInputs,
    Regularization,
    calibration_objective_vectorized,
)


def run_de_calibration(
    *,
    model: torch.nn.Module,
    market_inputs: MarketInputs,
    bounds: Sequence[tuple[float, float]],
    normalization_stats: dict | None = None,
    lambda_reg: float = 0.0,
    regularization: Regularization = "l2",
    maxiter: int = 1000,
    popsize: int = 10,
    tol: float = 0.01,
    mutation: tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int | None = None,
    polish: bool = False,
    device: str | torch.device | None = None,
    invalid_value: float = 1.0e12,
) -> OptimizeResult:
    """
    Run Differential Evolution over the ANN-based calibration objective.

    Inputs
    - model: trained ANN forward model (8 -> 1 IV)
    - market_inputs: observed market quotes
    - bounds: parameter bounds [(low, high), ...]
    - lambda_reg/regularization: regularization in J(theta)
    - maxiter/popsize/tol/mutation/recombination: DE controls
      Note: in SciPy, `popsize` is a multiplier, not absolute population size.
    - seed: deterministic DE seed
    - polish: if True, run final local optimization
    - device: torch device used by model inference
    - invalid_value: penalty for invalid candidates

    Output
    - scipy OptimizeResult (best parameters in result.x, objective in result.fun)
    """

    bounds_list = [tuple(map(float, pair)) for pair in bounds]
    if len(bounds_list) == 0:
        raise ValueError("bounds cannot be empty")

    n_model_params = len(bounds_list)

    def _objective(theta_matrix):
        return calibration_objective_vectorized(
            theta_matrix,
            model=model,
            market_inputs=market_inputs,
            normalization_stats=normalization_stats,
            lambda_reg=lambda_reg,
            regularization=regularization,
            n_model_params=n_model_params,
            device=device,
            invalid_value=invalid_value,
        )

    return differential_evolution(
        func=_objective,
        bounds=bounds_list,
        strategy="best1bin",
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=float(tol),
        mutation=mutation,
        recombination=float(recombination),
        seed=seed,
        polish=bool(polish),
        vectorized=True,
        updating="deferred",
    )
