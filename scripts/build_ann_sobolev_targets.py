from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.greeks.heston_cf_greeks import HestonCFGreeksSettings
from src.sobolev.ann_iv import (
    DEFAULT_ANN_IV_FEATURE_COLUMNS,
    compute_ann_iv_sobolev_targets,
    robust_derivative_scales,
)


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _split_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _sample_df(df: pd.DataFrame, *, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(df), size=int(max_rows), replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def _process_one(
    *,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    split_seed: int,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    sampled = _sample_df(df, max_rows=args.max_rows, seed=split_seed)
    cf_settings = HestonCFGreeksSettings(
        u_min=float(args.cf_u_min),
        u_max=float(args.cf_u_max),
        n_u=int(args.cf_n_u),
    )
    targets = compute_ann_iv_sobolev_targets(
        sampled,
        iv_column=args.iv_column,
        feature_columns=_split_list(args.feature_columns),
        option_type=args.option_type,
        strike=float(args.strike),
        cf_settings=cf_settings,
        vega_floor=float(args.vega_floor),
        keep_invalid=bool(args.keep_invalid),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(output_path, index=False)
    valid_count = int(targets["valid"].sum()) if "valid" in targets.columns else len(targets)
    invalid_count = int((~targets["valid"]).sum()) if "valid" in targets.columns else 0
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "source_rows": int(len(df)),
        "sampled_rows": int(len(sampled)),
        "written_rows": int(len(targets)),
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
    }
    print(
        f"{input_path.name}: sampled={len(sampled)} written={len(targets)} "
        f"valid={valid_count} invalid={invalid_count} -> {output_path}"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build first-order ANN-IV Sobolev derivative targets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-parquet", default=None, help="Single input parquet split.")
    source.add_argument("--input-dir", default=None, help="Directory containing split parquet files.")

    parser.add_argument("--output-parquet", default=None, help="Output parquet for single input mode.")
    parser.add_argument("--output-dir", default="data/sobolev/ann_iv_v01")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits for --input-dir.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows per split.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--iv-column", default="IV")
    parser.add_argument(
        "--feature-columns",
        default=",".join(DEFAULT_ANN_IV_FEATURE_COLUMNS),
        help="Comma-separated feature columns in ANN order.",
    )
    parser.add_argument("--option-type", default="put", choices=["put", "call"])
    parser.add_argument("--strike", type=float, default=1.0)
    parser.add_argument("--vega-floor", type=float, default=1.0e-8)
    parser.add_argument("--keep-invalid", action="store_true")

    parser.add_argument("--cf-u-min", type=float, default=1.0e-6)
    parser.add_argument("--cf-u-max", type=float, default=200.0)
    parser.add_argument("--cf-n-u", type=int, default=1200)
    parser.add_argument("--scale-quantile", type=float, default=0.90)
    parser.add_argument("--scale-floor", type=float, default=1.0e-8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    summaries: list[dict[str, object]] = []
    output_dir = _resolve_path(args.output_dir)

    if args.input_parquet is not None:
        input_path = _resolve_path(args.input_parquet)
        if args.output_parquet is not None:
            output_path = _resolve_path(args.output_parquet)
        else:
            output_path = output_dir / input_path.name
        summaries.append(
            _process_one(
                input_path=input_path,
                output_path=output_path,
                args=args,
                split_seed=int(args.seed),
            )
        )
    else:
        input_dir = _resolve_path(args.input_dir)
        for offset, split in enumerate(_split_list(args.splits)):
            summaries.append(
                _process_one(
                    input_path=input_dir / f"{split}.parquet",
                    output_path=output_dir / f"{split}.parquet",
                    args=args,
                    split_seed=int(args.seed) + offset,
                )
            )

    manifest = {
        "builder": "scripts/build_ann_sobolev_targets.py",
        "iv_column": args.iv_column,
        "feature_columns": _split_list(args.feature_columns),
        "option_type": args.option_type,
        "strike": float(args.strike),
        "vega_floor": float(args.vega_floor),
        "cf_settings": {
            "u_min": float(args.cf_u_min),
            "u_max": float(args.cf_u_max),
            "n_u": int(args.cf_n_u),
        },
        "summaries": summaries,
    }

    scale_frames = []
    for item in summaries:
        out = Path(str(item["output"]))
        if out.exists():
            scale_frames.append(pd.read_parquet(out))
    if scale_frames:
        scale_df = pd.concat(scale_frames, axis=0, ignore_index=True)
        manifest["derivative_scales"] = robust_derivative_scales(
            scale_df,
            quantile=float(args.scale_quantile),
            floor=float(args.scale_floor),
        )

    manifest_path = output_dir / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
