from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "pinn_greek_ablation_suite.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pinn" / "ablation_suites"
GREEKS = ("delta", "gamma", "vega", "theta", "rho")
GREEK_LABELS = {
    "delta": "Delta",
    "gamma": "Gamma",
    "vega": "Vega",
    "theta": "Theta",
    "rho": "Rho",
}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary in {path}, got {type(payload)!r}")
    return payload


def _resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _suite_id(manifest: dict) -> str:
    return str(manifest.get("suite_id", manifest.get("global", {}).get("suite_id", "pinn_greek_ablation_suite")))


def _variant_key(variant: dict) -> str:
    key = variant.get("key")
    if not key:
        raise KeyError(f"Variant missing required key: {variant}")
    return str(key)


def _enabled_variants(manifest: dict) -> list[dict]:
    variants = manifest.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("Ablation manifest must include a non-empty 'variants' list.")
    return [dict(v) for v in variants if bool(v.get("enabled", True))]


def _run_dir(manifest: dict, variant: dict) -> Path:
    suite_id = _suite_id(manifest)
    raw = variant.get("run_dir")
    if raw:
        return _resolve_path(raw)
    return PROJECT_ROOT / "outputs" / "pinn" / suite_id / _variant_key(variant)


def _diagnostics_subdir(manifest: dict, variant: dict) -> str:
    diagnostics = manifest.get("diagnostics", {})
    default = "baseline_diagnostics"
    if isinstance(diagnostics, dict):
        default = str(diagnostics.get("output_subdir", default))
    return str(variant.get("diagnostics_subdir", default))


def _paths(manifest: dict, variant: dict) -> dict[str, Path]:
    run_dir = _run_dir(manifest, variant)
    diag_dir = run_dir / "greeks" / _diagnostics_subdir(manifest, variant)
    return {
        "run_dir": run_dir,
        "diagnostics_dir": diag_dir,
        "metrics": diag_dir / str(variant.get("metrics_file", "metrics_by_region.csv")),
        "points": diag_dir / str(variant.get("points_file", "points_baseline_diagnostics.csv")),
        "train_summary": run_dir / str(variant.get("train_summary", "train/metrics/train_summary.yaml")),
    }


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    df = pd.read_csv(path)
    required = {"region", "variable"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Metrics file {path} missing columns {sorted(missing)}")
    return df


def _metric(df: pd.DataFrame, *, region: str, variable: str, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    row = df[(df["region"] == region) & (df["variable"] == variable)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def _same_metric_grid(row: pd.Series, baseline: pd.Series, columns: list[str]) -> bool:
    for col in columns:
        value = float(row.get(col, float("nan")))
        base_value = float(baseline.get(col, float("nan")))
        if np.isfinite(value) and np.isfinite(base_value) and int(value) != int(base_value):
            return False
    return True


def _load_train_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return _load_yaml(path)


def _build_comparison_table(manifest: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for variant in _enabled_variants(manifest):
        metrics = _read_metrics(_paths(manifest, variant)["metrics"])
        train_summary = _load_train_summary(_paths(manifest, variant)["train_summary"])
        row = {
            "key": _variant_key(variant),
            "label": str(variant.get("label", _variant_key(variant))),
            "role": str(variant.get("role", "ablation")),
            "metric_focus": str(variant.get("metric_focus", "greeks")),
            "run_dir": str(_run_dir(manifest, variant)),
            "price_n_points": _metric(metrics, region="full", variable="price", column="n_points"),
            "hard_price_n_points": _metric(metrics, region="hard", variable="price", column="n_points"),
            "price_rmse": _metric(metrics, region="full", variable="price", column="rmse"),
            "hard_price_rmse": _metric(metrics, region="hard", variable="price", column="rmse"),
            "non_hard_price_rmse": _metric(metrics, region="non_hard", variable="price", column="rmse"),
            "short_maturity_price_rmse": _metric(metrics, region="short_maturity", variable="price", column="rmse"),
            "atm_price_rmse": _metric(metrics, region="atm", variable="price", column="rmse"),
            "hard_p99_gamma_error": _metric(metrics, region="hard", variable="gamma", column="p99_abs_error"),
            "pde_residual_rmse": _metric(metrics, region="full", variable="pde_residual", column="rmse"),
            "hard_pde_residual_rmse": _metric(metrics, region="hard", variable="pde_residual", column="rmse"),
            "training_time_seconds": float(
                train_summary.get("total_training_seconds", train_summary.get("n_steps", float("nan")))
            ),
            "best_val_total": float(train_summary.get("best_val_total", train_summary.get("best_loss", float("nan")))),
        }
        for greek in GREEKS:
            row[f"{greek}_rmse"] = _metric(metrics, region="full", variable=greek, column="rmse")
            row[f"hard_{greek}_rmse"] = _metric(metrics, region="hard", variable=greek, column="rmse")
            row[f"{greek}_p99_abs_error"] = _metric(metrics, region="full", variable=greek, column="p99_abs_error")
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No enabled variants found in manifest.")
    return _add_scores(table=table, manifest=manifest)


def _safe_ratio(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0.0:
        return float("nan")
    return float(value / baseline)


def _geom_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v) and v > 0.0], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def _add_scores(*, table: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    baseline_key = str(manifest.get("baseline_key", "abl00_baseline"))
    baseline = table[table["key"] == baseline_key]
    if baseline.empty:
        raise KeyError(f"Baseline key '{baseline_key}' not found in comparison table.")
    b = baseline.iloc[0]
    out = table.copy()

    global_cols = [f"{g}_rmse" for g in GREEKS]
    hard_cols = ["hard_delta_rmse", "hard_gamma_rmse", "hard_p99_gamma_error"]
    price_cols = ["price_rmse", "hard_price_rmse"]
    optional_ratio_cols = ["non_hard_price_rmse", "short_maturity_price_rmse", "atm_price_rmse"]
    grid_cols = ["price_n_points", "hard_price_n_points"]
    out["comparison_grid_ok"] = [
        _same_metric_grid(row, b, grid_cols)
        for _, row in out.iterrows()
    ]

    for col in global_cols + hard_cols + price_cols + optional_ratio_cols:
        out[f"ratio_{col}"] = [
            _safe_ratio(float(row[col]), float(b[col])) if bool(row["comparison_grid_ok"]) else float("nan")
            for _, row in out.iterrows()
        ]

    out["global_score"] = [
        _geom_mean([float(row[f"ratio_{col}"]) for col in global_cols])
        for _, row in out.iterrows()
    ]
    out["hard_score"] = [
        _geom_mean([float(row[f"ratio_{col}"]) for col in hard_cols])
        for _, row in out.iterrows()
    ]
    out["price_score"] = [
        _geom_mean([float(row[f"ratio_{col}"]) for col in price_cols])
        for _, row in out.iterrows()
    ]

    comparison = manifest.get("comparison", {})
    thresholds = comparison.get("thresholds", {}) if isinstance(comparison, dict) else {}
    global_cfg = thresholds.get("global_improvement", {}) if isinstance(thresholds, dict) else {}
    hard_cfg = thresholds.get("hard_specialist", {}) if isinstance(thresholds, dict) else {}
    pricing_cfg = thresholds.get("pricing_specialist", {}) if isinstance(thresholds, dict) else {}
    global_score_max = float(global_cfg.get("global_score_max", 0.95))
    price_guard_max = float(global_cfg.get("price_score_max", 1.10))
    hard_guard_max = float(global_cfg.get("hard_score_max", 1.25))
    greek_ratio_max = float(global_cfg.get("greek_ratio_max", 1.20))
    specialist_hard_max = float(hard_cfg.get("hard_score_max", 0.90))
    specialist_price_max = float(hard_cfg.get("price_score_max", 1.10))
    specialist_global_max = float(hard_cfg.get("global_score_max", 1.10))
    pricing_full_improvement_max = float(pricing_cfg.get("full_price_ratio_max", 0.95))
    pricing_hard_guard_max = float(pricing_cfg.get("hard_price_ratio_max", 1.10))
    pricing_hard_specialist_max = float(pricing_cfg.get("hard_price_specialist_ratio_max", 0.90))
    pricing_full_guard_max = float(pricing_cfg.get("full_price_guard_ratio_max", 1.10))
    pricing_neutral_full_max = float(pricing_cfg.get("neutral_full_price_ratio_max", 1.10))
    pricing_neutral_hard_max = float(pricing_cfg.get("neutral_hard_price_ratio_max", 1.25))

    statuses: list[str] = []
    for _, row in out.iterrows():
        if row["key"] == baseline_key:
            statuses.append("baseline")
            continue
        if not bool(row["comparison_grid_ok"]):
            statuses.append("incompatible_grid")
            continue
        metric_focus = str(row.get("metric_focus", "greeks")).strip().lower()
        if metric_focus in {"price", "pricing", "pricing_only", "price_only"}:
            full_price_ratio = float(row["ratio_price_rmse"])
            hard_price_ratio = float(row["ratio_hard_price_rmse"])
            if (
                full_price_ratio <= pricing_full_improvement_max
                and hard_price_ratio <= pricing_hard_guard_max
            ):
                statuses.append("pricing_improvement")
            elif (
                hard_price_ratio <= pricing_hard_specialist_max
                and full_price_ratio <= pricing_full_guard_max
            ):
                statuses.append("pricing_hard_specialist")
            elif (
                full_price_ratio <= pricing_neutral_full_max
                and hard_price_ratio <= pricing_neutral_hard_max
            ):
                statuses.append("pricing_neutral")
            else:
                statuses.append("pricing_regression")
            continue
        global_ok = float(row["global_score"]) <= global_score_max
        price_ok = float(row["price_score"]) <= price_guard_max
        hard_guard = float(row["hard_score"]) <= hard_guard_max
        greek_regression = any(float(row[f"ratio_{col}"]) > greek_ratio_max for col in global_cols)
        hard_specialist = (
            float(row["hard_score"]) <= specialist_hard_max
            and float(row["price_score"]) <= specialist_price_max
            and float(row["global_score"]) <= specialist_global_max
        )
        if global_ok and price_ok and hard_guard and not greek_regression:
            statuses.append("global_improvement")
        elif hard_specialist:
            statuses.append("hard_specialist")
        elif not price_ok or greek_regression:
            statuses.append("regression")
        else:
            statuses.append("neutral")
    out["status"] = statuses
    return out


def _format_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        ("key", "Experiment"),
        ("status", "Status"),
        ("comparison_grid_ok", "Grid OK"),
        ("global_score", "Global"),
        ("hard_score", "Hard"),
        ("price_score", "Price"),
        ("price_rmse", "Price RMSE"),
        ("non_hard_price_rmse", "Non-Hard Price RMSE"),
        ("hard_price_rmse", "Hard Price RMSE"),
        ("gamma_rmse", "Gamma RMSE"),
        ("hard_gamma_rmse", "Hard Gamma RMSE"),
        ("training_time_seconds", "Train s"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in cols) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(cols) - 2)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for key, _label in cols:
            value = row[key]
            if key in {"key", "status"}:
                cells.append(str(value))
            elif key == "comparison_grid_ok":
                cells.append("yes" if bool(value) else "no")
            else:
                cells.append("" if pd.isna(value) else f"{float(value):.6g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _read_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing points diagnostics file: {path}")
    df = pd.read_csv(path)
    missing = {"tau", "moneyness"} - set(df.columns)
    if missing:
        raise KeyError(f"Points file {path} missing columns {sorted(missing)}")
    return df


def _pivot_matrix(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.sort(df["tau"].dropna().unique())
    y_values = np.sort(df["moneyness"].dropna().unique())
    pivot = df.pivot_table(index="moneyness", columns="tau", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=y_values, columns=x_values)
    return pivot.to_numpy(dtype=np.float64), x_values, y_values


def _row_norm(values: list[np.ndarray]) -> LogNorm | None:
    valid_parts = []
    for arr in values:
        valid = arr[np.isfinite(arr) & (arr > 0.0)]
        if valid.size:
            valid_parts.append(valid)
    if not valid_parts:
        return None
    flat = np.concatenate(valid_parts)
    vmin = max(float(np.nanpercentile(flat, 5.0)), float(np.finfo(np.float64).tiny))
    vmax = float(np.nanpercentile(flat, 95.0))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _add_hard_region(ax: plt.Axes, *, epsilon_m: float, epsilon_tau: float) -> None:
    y0 = float(np.exp(-epsilon_m))
    y1 = float(np.exp(epsilon_m))
    rect = Rectangle(
        (0.0, y0),
        width=float(epsilon_tau),
        height=y1 - y0,
        fill=False,
        edgecolor="cyan",
        linewidth=1.2,
        linestyle="--",
    )
    ax.add_patch(rect)


def _save_pairwise_figure(
    *,
    baseline_points: pd.DataFrame,
    variant_points: pd.DataFrame,
    baseline_label: str,
    variant_label: str,
    variant_key: str,
    output_path: Path,
    epsilon_m: float,
    epsilon_tau: float,
) -> None:
    fig, axes = plt.subplots(
        len(GREEKS),
        2,
        figsize=(11.5, 17.0),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f2f2f2")

    for row_idx, greek in enumerate(GREEKS):
        col = f"abs_error_{greek}"
        if col not in baseline_points.columns or col not in variant_points.columns:
            for ax in axes[row_idx, :]:
                ax.axis("off")
            continue
        base_matrix, base_x, base_y = _pivot_matrix(baseline_points, col)
        var_matrix, var_x, var_y = _pivot_matrix(variant_points, col)
        norm = _row_norm([base_matrix, var_matrix])

        for ax, matrix, x_values, y_values, title in (
            (axes[row_idx, 0], base_matrix, base_x, base_y, baseline_label),
            (axes[row_idx, 1], var_matrix, var_x, var_y, variant_label),
        ):
            plot_matrix = matrix
            if norm is not None:
                plot_matrix = np.where(np.isfinite(matrix), np.maximum(matrix, norm.vmin), np.nan)
            im = ax.imshow(
                plot_matrix,
                origin="lower",
                aspect="auto",
                extent=[float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max())],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            _add_hard_region(ax, epsilon_m=epsilon_m, epsilon_tau=epsilon_tau)
            ax.set_title(f"{GREEK_LABELS[greek]} | {title}", fontsize=10)
            ax.set_xlabel(r"$\tau$")
            ax.set_ylabel(r"$m$")
        fig.colorbar(im, ax=axes[row_idx, :].tolist(), shrink=0.76, label=f"|{greek} error|")

    fig.suptitle(f"Greek absolute-error maps: {variant_key} vs baseline", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _save_pairwise_figures(*, manifest: dict, output_dir: Path) -> dict[str, str]:
    variants = _enabled_variants(manifest)
    baseline_key = str(manifest.get("baseline_key", "abl00_baseline"))
    baseline_variant = next((v for v in variants if _variant_key(v) == baseline_key), None)
    if baseline_variant is None:
        raise KeyError(f"Baseline key '{baseline_key}' not found in manifest variants.")

    diagnostics = manifest.get("diagnostics", {})
    comparison = manifest.get("comparison", {})
    hard_region = {}
    if isinstance(diagnostics, dict):
        hard_region = diagnostics.get("hard_region", {})
    if isinstance(comparison, dict):
        hard_region = comparison.get("hard_region", hard_region)
    epsilon_m = float(hard_region.get("epsilon_m", 0.03))
    epsilon_tau = float(hard_region.get("epsilon_tau", 0.05))

    baseline_points = _read_points(_paths(manifest, baseline_variant)["points"])
    baseline_label = str(baseline_variant.get("label", baseline_key))
    out: dict[str, str] = {}
    for variant in variants:
        key = _variant_key(variant)
        if key == baseline_key:
            continue
        variant_points = _read_points(_paths(manifest, variant)["points"])
        path = output_dir / f"{key}_greek_error_maps_vs_baseline.png"
        _save_pairwise_figure(
            baseline_points=baseline_points,
            variant_points=variant_points,
            baseline_label=baseline_label,
            variant_label=str(variant.get("label", key)),
            variant_key=key,
            output_path=path,
            epsilon_m=epsilon_m,
            epsilon_tau=epsilon_tau,
        )
        out[key] = str(path)
    return out


def _save_score_scatter(*, table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    colors = {
        "baseline": "#333333",
        "global_improvement": "#2E7D32",
        "hard_specialist": "#1565C0",
        "pricing_improvement": "#00897B",
        "pricing_hard_specialist": "#00796B",
        "pricing_neutral": "#607D8B",
        "pricing_regression": "#C62828",
        "incompatible_grid": "#6D4C41",
        "neutral": "#777777",
        "regression": "#B71C1C",
    }
    for _, row in table.iterrows():
        color = colors.get(str(row["status"]), "#777777")
        if not np.isfinite(float(row["global_score"])) or not np.isfinite(float(row["hard_score"])):
            continue
        ax.scatter(float(row["global_score"]), float(row["hard_score"]), s=58, color=color)
        ax.annotate(str(row["key"]), (float(row["global_score"]), float(row["hard_score"])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("global_score (lower is better)")
    ax.set_ylabel("hard_score (lower is better)")
    ax.set_title("PINN Greek ablations: global vs hard score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_ablation_study(*, manifest_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    manifest = _load_yaml(manifest_path)
    suite_id = _suite_id(manifest)
    if output_dir is None:
        comparison = manifest.get("comparison", {})
        raw_output_dir = comparison.get("output_dir") if isinstance(comparison, dict) else None
        output_dir = _resolve_path(raw_output_dir) if raw_output_dir else DEFAULT_OUTPUT_ROOT / suite_id
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    pairwise_dir = figures_dir / "pairwise"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    table = _build_comparison_table(manifest)
    scores_path = tables_dir / "ablation_scores.csv"
    table.to_csv(scores_path, index=False)
    markdown_path = tables_dir / "ablation_scores.md"
    markdown_path.write_text(_format_markdown_table(table), encoding="utf-8")

    scatter_path = figures_dir / "score_scatter_hard_vs_global.png"
    _save_score_scatter(table=table, output_path=scatter_path)
    pairwise = _save_pairwise_figures(manifest=manifest, output_dir=pairwise_dir)

    summary = {
        "suite_id": suite_id,
        "manifest": str(manifest_path),
        "baseline_key": manifest.get("baseline_key", "abl00_baseline"),
        "outputs": {
            "scores_csv": str(scores_path),
            "scores_markdown": str(markdown_path),
            "score_scatter": str(scatter_path),
            "pairwise_dir": str(pairwise_dir),
        },
        "pairwise_figures": pairwise,
        "best_global_score": str(table.loc[table["global_score"].idxmin(), "key"]),
        "best_hard_score": str(table.loc[table["hard_score"].idxmin(), "key"]),
        "best_price_score": str(table.loc[table["price_score"].idxmin(), "key"]),
    }
    summary_path = output_dir / "ablation_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return {
        "scores_csv": scores_path,
        "scores_markdown": markdown_path,
        "summary": summary_path,
        "score_scatter": scatter_path,
        "pairwise_dir": pairwise_dir,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manifest-driven PINN Greek ablation study.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to ablation suite YAML manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for ablation tables and figures.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    output_dir = None
    if args.output_dir is not None:
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outputs = build_ablation_study(manifest_path=manifest_path, output_dir=output_dir)
    print("PINN ablation study completed")
    print(f"Scores CSV: {outputs['scores_csv']}")
    print(f"Scores Markdown: {outputs['scores_markdown']}")
    print(f"Summary: {outputs['summary']}")
    print(f"Score scatter: {outputs['score_scatter']}")
    print(f"Pairwise dir: {outputs['pairwise_dir']}")


if __name__ == "__main__":
    main()
