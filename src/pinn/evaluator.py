from __future__ import annotations

from pathlib import Path


def evaluate_pinn_run(*, run_dir: Path, evaluation_config: dict) -> dict:
    """
    Evaluate a trained PINN run.
    """
    raise NotImplementedError(
        "PINN scaffold only: evaluation logic not implemented yet."
    )

