from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


DEFAULT_FEATURE_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]


@dataclass(frozen=True)
class GreekIndexSpec:
    idx_spot: int
    idx_vol: int | None = None
    idx_tau: int | None = None
    idx_rate: int | None = None


def parse_feature_order(
    feature_order: Sequence[str] | None,
    *,
    fallback: Sequence[str] = DEFAULT_FEATURE_ORDER,
) -> list[str]:
    if feature_order is None:
        return list(fallback)
    out = [str(x).strip() for x in feature_order if str(x).strip()]
    if not out:
        raise ValueError("feature_order cannot be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"feature_order has duplicated names: {out}")
    return out


def feature_index(feature_order: Sequence[str], name: str, *, required: bool = True) -> int | None:
    name = str(name).strip()
    order = list(feature_order)
    try:
        return order.index(name)
    except ValueError:
        if required:
            raise KeyError(
                f"Feature '{name}' not found in feature_order={order}"
            ) from None
        return None


def build_greek_index_spec(
    feature_order: Sequence[str],
    *,
    spot_feature: str = "moneyness",
    vol_feature: str | None = None,
    tau_feature: str | None = "tau",
    rate_feature: str | None = "r",
) -> GreekIndexSpec:
    idx_spot = feature_index(feature_order, spot_feature, required=True)
    idx_vol = feature_index(feature_order, vol_feature, required=False) if vol_feature else None
    idx_tau = feature_index(feature_order, tau_feature, required=False) if tau_feature else None
    idx_rate = feature_index(feature_order, rate_feature, required=False) if rate_feature else None
    return GreekIndexSpec(
        idx_spot=idx_spot,
        idx_vol=idx_vol,
        idx_tau=idx_tau,
        idx_rate=idx_rate,
    )


def greek_column_names(spec: GreekIndexSpec) -> list[str]:
    cols = ["delta", "gamma"]
    if spec.idx_vol is not None:
        cols.append("vega")
    if spec.idx_tau is not None:
        cols.append("theta")
    if spec.idx_rate is not None:
        cols.append("rho")
    return cols


__all__ = [
    "DEFAULT_FEATURE_ORDER",
    "GreekIndexSpec",
    "parse_feature_order",
    "feature_index",
    "build_greek_index_spec",
    "greek_column_names",
]
