from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PINNLossTerms:
    data: float
    pde: float
    boundary: float
    no_arbitrage: float

    @property
    def total(self) -> float:
        return self.data + self.pde + self.boundary + self.no_arbitrage


def compute_weighted_pinn_loss(*, loss_config: dict, batch_payload: dict) -> PINNLossTerms:
    """
    Compute weighted multi-objective PINN loss terms.
    """
    raise NotImplementedError(
        "PINN scaffold only: weighted loss computation not implemented yet."
    )

