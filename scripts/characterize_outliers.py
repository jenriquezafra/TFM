from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ("val", "test")
EPS = 1.0e-12

TYPE_ORDER = [
    "negative_iv_prediction",
    "floor_iv_mismatch",
    "short_low_m_high_sensitivity",
    "short_low_m",
    "high_sensitivity_elsewhere",
    "other_outlier",
]

TYPE_COLORS = {
    "negative_iv_prediction": "#6f1d1b",
    "floor_iv_mismatch": "#bb3e03",
    "short_low_m_high_sensitivity": "#ee9b00",
    "short_low_m": "#e9d8a6",
    "high_sensitivity_elsewhere": "#0a9396",
    "other_outlier": "#5c677d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize outliers into interpretable types and plot compact summaries."
    )
    parser.add_argument("--model-dir", default="Liu_like_v01")
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--tau-threshold", type=float, default=0.25)
    parser.add_argument("--moneyness-threshold", type=float, default=0.8)
    parser.add_argument("--sensitivity-quantile", type=float, default=0.95)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _parse_splits(raw: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not out:
        raise ValueError("splits must not be empty")
    allowed = {"train", "val", "test"}
    invalid = [x for x in out if x not in allowed]
    if invalid:
        raise ValueError(f"invalid splits: {invalid}; allowed: {sorted(allowed)}")
    return out


def _resolve_analysis_dir(args: argparse.Namespace) -> Path:
    if args.analysis_dir:
        p = Path(args.analysis_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return PROJECT_ROOT / "outputs" / "runs" / args.model_dir / "outliers_analysis"


def _resolve_out_dir(args: argparse.Namespace, analysis_dir: Path) -> Path:
    if args.out_dir:
        p = Path(args.out_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return analysis_dir / "characterization"


def _load_global_summary(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "all_splits_outlier_stability_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    return pd.read_csv(path)


def _sensitivity_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in ("abs_theta", "abs_vega", "abs_delta", "abs_rho") if c in df.columns]
    if not cols:
        raise ValueError("missing expected sensitivity columns in outlier/reference data")
    return cols


def _classify_outlier(
    row: pd.Series,
    *,
    tau_threshold: float,
    moneyness_threshold: float,
) -> str:
    negative_pred = bool(row.get("is_negative_iv_pred", False))
    floor_iv = bool(row.get("is_floor_iv", False))
    in_short_low_m = bool(row.get("is_short_low_m_region", False))
    high_sens = bool(row.get("high_sensitivity", False))

    if negative_pred:
        return "negative_iv_prediction"
    if floor_iv:
        return "floor_iv_mismatch"
    if in_short_low_m and high_sens:
        return "short_low_m_high_sensitivity"
    if in_short_low_m:
        return "short_low_m"
    if high_sens:
        return "high_sensitivity_elsewhere"
    return "other_outlier"


def _build_type_summary(
    *,
    df_char: pd.DataFrame,
    split: str,
    n_total: int,
    global_mse: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    n_out = len(df_char)
    if n_out == 0:
        return pd.DataFrame()

    grouped = df_char.groupby("outlier_type", observed=False)
    for outlier_type, grp in grouped:
        sq_sum = float(grp["sq_error"].sum())
        mse_contrib = sq_sum / max(n_total, 1)
        rows.append(
            {
                "split": split,
                "outlier_type": str(outlier_type),
                "n_outliers": int(len(grp)),
                "share_outliers_pct": 100.0 * float(len(grp)) / max(n_out, 1),
                "mse_contrib": mse_contrib,
                "mse_share_global_pct": 100.0 * mse_contrib / max(global_mse, EPS),
                "median_abs_error": float(np.nanmedian(grp["abs_error"])),
                "p90_abs_error": float(np.nanquantile(grp["abs_error"], 0.90)),
                "median_sensitivity_score": float(np.nanmedian(grp["sensitivity_score"])),
                "floor_count": int(grp["is_floor_iv"].sum()),
                "negative_pred_count": int(grp["is_negative_iv_pred"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["type_order"] = out["outlier_type"].map({t: i for i, t in enumerate(TYPE_ORDER)}).fillna(999)
    return out.sort_values(["split", "type_order", "mse_contrib"], ascending=[True, True, False]).drop(
        columns=["type_order"]
    )


def _plot_type_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    type_order = [t for t in TYPE_ORDER if t in plot_df["outlier_type"].unique()]
    splits = list(plot_df["split"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    metrics = [("n_outliers", "Outlier count"), ("mse_share_global_pct", "MSE share (%)")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(type_order))
        width = 0.35 if len(splits) == 2 else 0.25
        for i, split in enumerate(splits):
            sub = plot_df[plot_df["split"] == split].set_index("outlier_type")
            y = np.array([float(sub.loc[t, metric]) if t in sub.index else 0.0 for t in type_order], dtype=np.float64)
            offset = (i - (len(splits) - 1) / 2) * width
            ax.bar(
                x + offset,
                y,
                width=width,
                label=split,
                color=[TYPE_COLORS.get(t, "#999999") for t in type_order],
                alpha=0.85 if i == 0 else 0.5,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(type_order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Outlier type characterization", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_surface_by_type(char_df: pd.DataFrame, out_path: Path) -> None:
    if char_df.empty:
        return
    splits = list(char_df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6.0 * len(splits), 4.6), sharex=True, sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        sub = char_df[char_df["split"] == split].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No outliers for {split}", ha="center", va="center")
            ax.set_axis_off()
            continue
        for outlier_type in TYPE_ORDER:
            grp = sub[sub["outlier_type"] == outlier_type]
            if grp.empty:
                continue
            ax.scatter(
                grp["tau"],
                grp["moneyness"],
                s=45 + 600.0 * grp["mse_contribution"].to_numpy(dtype=np.float64),
                alpha=0.86,
                color=TYPE_COLORS.get(outlier_type, "#999999"),
                edgecolors="black",
                linewidths=0.4,
                label=outlier_type,
            )
        ax.axvline(0.25, color="#1d3557", linestyle="--", linewidth=1.0)
        ax.axhline(0.8, color="#1d3557", linestyle="--", linewidth=1.0)
        ax.set_title(split)
        ax.set_xlabel("tau")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("moneyness")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        fig.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper center",
            ncol=3,
            fontsize=8,
            bbox_to_anchor=(0.5, 1.05),
        )
    fig.suptitle("Outliers on surface by type (size ~ MSE contribution)", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def characterize_split(
    *,
    analysis_dir: Path,
    split: str,
    summary_row: pd.Series,
    tau_threshold: float,
    moneyness_threshold: float,
    sensitivity_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_path = analysis_dir / f"{split}_outliers_detailed.parquet"
    ref_path = analysis_dir / f"{split}_reference_non_outliers.parquet"
    if not out_path.exists():
        raise FileNotFoundError(f"missing outlier file: {out_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"missing reference file: {ref_path}")

    df_out = pd.read_parquet(out_path).copy()
    df_ref = pd.read_parquet(ref_path).copy()
    if df_out.empty:
        return pd.DataFrame(), pd.DataFrame()

    sens_cols = _sensitivity_columns(df_out)
    ref_thr = {
        c: float(np.nanquantile(df_ref[c].to_numpy(dtype=np.float64), sensitivity_quantile))
        for c in sens_cols
    }
    ref_thr = {k: max(v, EPS) for k, v in ref_thr.items()}

    for c in sens_cols:
        df_out[f"{c}_norm_refq"] = df_out[c].to_numpy(dtype=np.float64) / ref_thr[c]

    norm_cols = [f"{c}_norm_refq" for c in sens_cols]
    norm_arr = df_out[norm_cols].to_numpy(dtype=np.float64)
    max_idx = np.argmax(norm_arr, axis=1)
    df_out["sensitivity_score"] = np.max(norm_arr, axis=1)
    df_out["sensitivity_driver"] = [sens_cols[i].replace("abs_", "") for i in max_idx]
    df_out["high_sensitivity"] = df_out["sensitivity_score"] >= 1.0

    df_out["is_negative_iv_pred"] = df_out["iv_pred"] < 0.0
    df_out["is_short_low_m_region"] = (df_out["tau"] < tau_threshold) & (
        df_out["moneyness"] < moneyness_threshold
    )
    df_out["outlier_type"] = df_out.apply(
        _classify_outlier,
        axis=1,
        tau_threshold=tau_threshold,
        moneyness_threshold=moneyness_threshold,
    )

    n_total = int(summary_row["n_total"])
    global_mse = float(summary_row["global_mse"])
    out_sq_sum = float(df_out["sq_error"].sum())
    df_out["mse_contribution"] = df_out["sq_error"] / max(n_total, 1)
    df_out["mse_share_global_pct"] = 100.0 * df_out["mse_contribution"] / max(global_mse, EPS)
    df_out["mse_share_outliers_pct"] = 100.0 * df_out["sq_error"] / max(out_sq_sum, EPS)
    df_out["sq_error_rank"] = df_out["sq_error"].rank(ascending=False, method="dense").astype(int)
    df_out["split"] = split

    type_summary = _build_type_summary(df_char=df_out, split=split, n_total=n_total, global_mse=global_mse)
    return df_out, type_summary


def main() -> None:
    args = parse_args()
    splits = _parse_splits(args.splits)
    if not (0.0 < args.sensitivity_quantile < 1.0):
        raise ValueError("sensitivity-quantile must be in (0,1)")

    analysis_dir = _resolve_analysis_dir(args)
    out_dir = _resolve_out_dir(args, analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_summary = _load_global_summary(analysis_dir).set_index("split")

    char_frames: list[pd.DataFrame] = []
    type_sum_frames: list[pd.DataFrame] = []
    for split in splits:
        if split not in global_summary.index:
            continue
        df_char, df_type_sum = characterize_split(
            analysis_dir=analysis_dir,
            split=split,
            summary_row=global_summary.loc[split],
            tau_threshold=float(args.tau_threshold),
            moneyness_threshold=float(args.moneyness_threshold),
            sensitivity_quantile=float(args.sensitivity_quantile),
        )
        if df_char.empty:
            continue
        char_frames.append(df_char)
        type_sum_frames.append(df_type_sum)
        df_char.to_csv(out_dir / f"{split}_outliers_characterized.csv", index=False)
        df_char.to_parquet(out_dir / f"{split}_outliers_characterized.parquet", index=False)
        df_type_sum.to_csv(out_dir / f"{split}_outlier_type_summary.csv", index=False)

    if not char_frames:
        raise RuntimeError("No characterized outliers produced; check split files/inputs")

    all_char = pd.concat(char_frames, ignore_index=True)
    all_type_sum = pd.concat(type_sum_frames, ignore_index=True)
    all_char.to_csv(out_dir / "all_splits_outliers_characterized.csv", index=False)
    all_type_sum.to_csv(out_dir / "all_splits_outlier_type_summary.csv", index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_type_bars(all_type_sum, fig_dir / "01_outlier_type_summary.png")
    _plot_surface_by_type(all_char, fig_dir / "02_outlier_surface_by_type.png")

    print(f"Characterization written to: {out_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
