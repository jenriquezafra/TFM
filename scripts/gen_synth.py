import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
config_path = PROJECT_ROOT / "configs" / "synth.yaml"

from src.datasets.make_synth import generate_all
from src.datasets.splits import dataframes_splits, dataframes_splits_stratified_quantiles
from src.solvers.heston_cos import COS_solver_scalar
from src.solvers.implied_vol import IV_Brent, IV_LM


PARAM_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0"]
FEATURE_COLS = ["rho", "kappa", "gamma", "bar_v", "v0", "moneyness", "tau", "r"]
MASTER_COLS = FEATURE_COLS + [
    "price_cos",
    "iv_brent",
    "feller_slack",
    "feller_ok",
    "brent_success",
    "brent_abs_residual",
]


def as_bounds(value, name):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lower = np.float64(value[0])
        upper = np.float64(value[1])
        if lower > upper:
            raise ValueError(f"Invalid bounds for {name}: {lower} > {upper}")
        if lower == upper:
            upper = np.nextafter(lower, np.inf)
        return np.array([lower, upper], dtype=np.float64)
    lower = np.float64(value)
    upper = np.nextafter(lower, np.inf)
    return np.array([lower, upper], dtype=np.float64)


def _format_seconds(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_candidate_dataframe(
    n_samples: int,
    sample_seed: int,
    params_bounds: np.ndarray,
    grid_bounds: np.ndarray,
    r_bounds: np.ndarray,
    fixed_params: dict,
) -> pd.DataFrame:
    all_samples = generate_all(
        n_samples=n_samples,
        param_ranges=params_bounds,
        grid_bounds=grid_bounds,
        r_bounds=r_bounds,
        seed=sample_seed,
    )
    X_params, X_grid, X_r = all_samples
    X = np.hstack([X_params, X_grid, X_r])

    synth_df = pd.DataFrame(data=np.nan, index=range(n_samples), columns=FEATURE_COLS)
    synth_df.iloc[:, :] = X

    for param_name, param_value in fixed_params.items():
        synth_df.loc[:, param_name] = np.float64(param_value)

    return synth_df


def _compute_iv_and_filters_for_chunk(
    synth_df: pd.DataFrame,
    *,
    round_idx: int,
    rootfinder: str,
    K: float,
    cos_params: np.ndarray,
    cos_interval_rule: str,
    opt_type: str,
    residual_warn_abs: np.float64,
    residual_keep_abs: np.float64 | None,
    drop_sigma0_hits: bool,
    feller_enabled: bool,
    feller_slack_tol: np.float64,
    keep_all_rows: bool,
    progress_every: int,
    brent_cfg: dict | None,
    lm_cfg: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_chunk = len(synth_df)
    prefix = f"[round {round_idx}]"

    tau_keep_mask = synth_df["tau"].to_numpy(dtype=np.float64) > 0.0
    tau_violations = int((~tau_keep_mask).sum())
    if tau_violations > 0:
        print(
            f"{prefix} Tau validity filter: invalid tau rows={tau_violations}/{n_chunk} "
            f"({100.0 * tau_violations / n_chunk:.4f}%)"
        )

    feller_slack = (
        2.0 * synth_df["kappa"].to_numpy(dtype=np.float64) * synth_df["bar_v"].to_numpy(dtype=np.float64)
        - synth_df["gamma"].to_numpy(dtype=np.float64) ** 2
    )
    feller_ok_mask = np.isfinite(feller_slack) & (feller_slack >= 0.0)
    feller_filter_mask = np.isfinite(feller_slack) & (feller_slack >= -feller_slack_tol)
    if feller_enabled:
        feller_violations = int((~feller_filter_mask).sum())
        print(
            f"{prefix} Feller filter enabled (tol={feller_slack_tol:.3e}): "
            f"violations={feller_violations}/{n_chunk} "
            f"({100.0 * feller_violations / n_chunk:.4f}%)"
        )

    iv_values = np.full(shape=n_chunk, fill_value=np.nan, dtype=np.float64)
    price_cos_values = np.full(shape=n_chunk, fill_value=np.nan, dtype=np.float64)
    brent_abs_residual = np.full(shape=n_chunk, fill_value=np.nan, dtype=np.float64)
    brent_success = np.zeros(shape=n_chunk, dtype=bool)

    lm_sigma0 = None if lm_cfg is None else np.float64(lm_cfg["sigma0"])

    chunk_start_time = time.time()
    print(f"{prefix} Generating IVs with '{rootfinder}' for {n_chunk} samples")

    for i in range(n_chunk):
        params_heston = synth_df.iloc[i, :5].to_numpy(dtype=np.float64)
        S0 = np.float64(synth_df.iloc[i]["moneyness"])
        tau = np.float64(synth_df.iloc[i]["tau"])
        r = np.float64(synth_df.iloc[i]["r"])

        if not tau_keep_mask[i]:
            iv_values[i] = np.nan
            brent_abs_residual[i] = np.nan
            brent_success[i] = False
            price_cos_values[i] = np.nan
        else:
            try:
                if rootfinder == "brent_iv":
                    if brent_cfg is None:
                        raise ValueError("Missing Brent configuration")
                    iv, details = IV_Brent(
                        params_Heston=params_heston,
                        S0=S0,
                        K=np.float64(K),
                        tau=tau,
                        r=r,
                        COS_params=cos_params,
                        cos_interval_rule=cos_interval_rule,
                        opt_type=opt_type,
                        iv_bounds=brent_cfg["iv_bounds"],
                        tol=brent_cfg["tol"],
                        max_iter=brent_cfg["max_iter"],
                        return_details=True,
                    )
                    brent_success[i] = True
                elif rootfinder == "LM":
                    if lm_cfg is None:
                        raise ValueError("Missing LM configuration")
                    iv, details = IV_LM(
                        params_Heston=params_heston,
                        S0=S0,
                        K=np.float64(K),
                        tau=tau,
                        r=r,
                        COS_params=cos_params,
                        cos_interval_rule=cos_interval_rule,
                        opt_type=opt_type,
                        sigma0=lm_cfg["sigma0"],
                        return_details=True,
                    )
                    brent_success[i] = bool(details["success"])
                else:
                    raise ValueError(f"Root finder method '{rootfinder}' is not supported")

                iv_values[i] = np.float64(iv)
                brent_abs_residual[i] = np.float64(details.get("price_residual_abs", np.nan))
                price_cos_values[i] = np.float64(details.get("target_price", np.nan))
            except Exception:
                iv_values[i] = np.nan
                brent_abs_residual[i] = np.nan
                brent_success[i] = False
                try:
                    price_cos_values[i] = np.float64(
                        COS_solver_scalar(
                            params_Heston=params_heston,
                            S0=S0,
                            K=np.float64(K),
                            tau=tau,
                            r=r,
                            COS_params=cos_params,
                            interval_rule=cos_interval_rule,
                            opt_type=opt_type,
                        )
                    )
                except Exception:
                    price_cos_values[i] = np.nan

        if ((i + 1) % progress_every == 0) or (i == n_chunk - 1):
            processed = i + 1
            elapsed = time.time() - chunk_start_time
            rate = processed / max(elapsed, 1e-12)
            eta = (n_chunk - processed) / max(rate, 1e-12)
            pct = 100.0 * processed / n_chunk
            valid = brent_abs_residual[: i + 1]
            valid = valid[np.isfinite(valid)]
            mean_res = float(valid.mean()) if len(valid) > 0 else np.nan
            bad_count = int((valid > residual_warn_abs).sum()) if len(valid) > 0 else 0
            msg = (
                f"{prefix} [{rootfinder}] {processed}/{n_chunk} ({pct:5.1f}%) | "
                f"elapsed={_format_seconds(elapsed)} | "
                f"eta={_format_seconds(eta)} | "
                f"mean|BS(IV)-V_tgt|={mean_res:.3e} | "
                f"bad(>{residual_warn_abs:.1e})={bad_count}"
            )
            if rootfinder == "LM":
                sigma0_hits = int(np.isclose(iv_values[: i + 1], lm_sigma0, atol=1e-12, rtol=0).sum())
                msg = f"{msg} | IV==sigma0({lm_sigma0:.3f})={sigma0_hits}"
            print(msg)

    synth_df = synth_df.copy()
    synth_df.loc[:, "price_cos"] = price_cos_values
    synth_df.loc[:, "iv_brent"] = iv_values
    synth_df.loc[:, "feller_slack"] = feller_slack
    synth_df.loc[:, "feller_ok"] = feller_ok_mask
    synth_df.loc[:, "brent_success"] = brent_success
    synth_df.loc[:, "brent_abs_residual"] = brent_abs_residual

    valid_residual = brent_abs_residual[np.isfinite(brent_abs_residual)]
    if len(valid_residual) > 0:
        print(f"{prefix} IV quality summary")
        print(f"{prefix}   valid residuals: {len(valid_residual)}/{n_chunk}")
        print(f"{prefix}   mean |BS(IV)-V_tgt|: {float(valid_residual.mean()):.3e}")
        print(f"{prefix}   p90  |BS(IV)-V_tgt|: {float(np.percentile(valid_residual, 90)):.3e}")
        print(f"{prefix}   p99  |BS(IV)-V_tgt|: {float(np.percentile(valid_residual, 99)):.3e}")
        print(f"{prefix}   max  |BS(IV)-V_tgt|: {float(valid_residual.max()):.3e}")
        print(f"{prefix}   bad count (>{residual_warn_abs:.1e}): {int((valid_residual > residual_warn_abs).sum())}")
    else:
        print(f"{prefix} IV quality summary: no valid residuals found")

    keep_for_training_mask = np.isfinite(iv_values)
    keep_for_training_mask &= tau_keep_mask
    if feller_enabled:
        keep_for_training_mask &= feller_filter_mask
    if residual_keep_abs is not None:
        keep_for_training_mask &= np.isfinite(brent_abs_residual) & (brent_abs_residual <= residual_keep_abs)
    if rootfinder == "LM" and drop_sigma0_hits:
        keep_for_training_mask &= ~np.isclose(iv_values, lm_sigma0, atol=1e-12, rtol=0)

    if keep_all_rows:
        selected_mask = np.ones(n_chunk, dtype=bool)
    else:
        selected_mask = keep_for_training_mask

    selected = int(selected_mask.sum())
    removed = int(n_chunk - selected)
    if removed > 0:
        print(
            f"{prefix} Selection before aggregation: "
            f"selected={selected} removed={removed} ({100.0 * removed / n_chunk:.3f}%)"
        )

    quality_df = pd.DataFrame(
        {
            "row_local": np.arange(n_chunk, dtype=np.int64),
            "iv_brent": iv_values,
            "price_cos": price_cos_values,
            "brent_abs_residual": brent_abs_residual,
            "brent_success": brent_success,
            "tau_keep": tau_keep_mask,
            "feller_slack": feller_slack,
            "feller_ok": feller_ok_mask,
            "feller_keep": feller_filter_mask,
            "keep_for_training": keep_for_training_mask,
            "selected_for_final_dataset": selected_mask,
            "generation_round": np.full(n_chunk, round_idx, dtype=np.int64),
        }
    )
    if rootfinder == "LM":
        quality_df["iv_equals_sigma0"] = np.isclose(iv_values, lm_sigma0, atol=1e-12, rtol=0)
    quality_float_cols = ["iv_brent", "price_cos", "brent_abs_residual", "feller_slack"]
    for col in quality_float_cols:
        quality_df[col] = quality_df[col].astype(np.float32, copy=False)

    selected_df = synth_df.loc[selected_mask].copy()
    for col in MASTER_COLS:
        selected_df[col] = selected_df[col].astype(np.float32, copy=False)
    selected_df["row_local"] = np.flatnonzero(selected_mask).astype(np.int64)
    selected_df["generation_round"] = np.int64(round_idx)
    selected_df = selected_df.reset_index(drop=True)

    return selected_df, quality_df


####################################### LOAD CONFIGS ########################################
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

seed = int(config["meta"]["seed"])
data_cfg = config["data"]
target_n = int(data_cfg["n_samples"])

splits_cfg = data_cfg.get("splits", {})
split_train = float(splits_cfg.get("train", 0.8))
split_val = float(splits_cfg.get("val", 0.1))
split_test = float(splits_cfg.get("test", 0.1))
stratify_cfg = splits_cfg.get("stratify", {})
stratify_enabled = bool(stratify_cfg.get("enabled", False))
stratify_target_col = str(stratify_cfg.get("target_column", "iv_brent"))
stratify_n_bins = int(stratify_cfg.get("n_bins", 20))

params_cfg = data_cfg["heston_params"]
params_bounds = np.vstack([as_bounds(params_cfg[param], param) for param in PARAM_ORDER])
grid_cfg = data_cfg["grid"]
grid_bounds = np.vstack(
    [
        as_bounds(grid_cfg["moneyness"], "moneyness"),
        as_bounds(grid_cfg["tau"], "tau"),
    ]
)
r_bounds = as_bounds(config["market"]["r"], "r")

fixed_params = {}
for name in PARAM_ORDER:
    value = params_cfg[name]
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] != value[1]:
        continue
    fixed_params[name] = value[0] if isinstance(value, (list, tuple)) else value

cos_params_cfg = config["cos_solver"]
cos_params = np.array([np.float64(cos_params_cfg["N"]), np.float64(cos_params_cfg["L"])], dtype=np.float64)
cos_interval_rule = cos_params_cfg.get("interval_rule", "sqrt_t")
opt_type = config["market"]["option_type"]
K = np.float64(config["market"]["K"])

rootfinder = config["root_finder"]["method"]
brent_cfg = None
lm_cfg = None
if rootfinder == "brent_iv":
    raw = config["root_finder"]["methods"]["brent_iv"]
    brent_cfg = {
        "iv_bounds": np.array(raw["iv_bounds"], dtype=np.float64),
        "tol": np.float64(raw["tol"]),
        "max_iter": int(raw["max_iter"]),
    }
elif rootfinder == "LM":
    raw = config["root_finder"]["methods"]["LM"]
    lm_cfg = {"sigma0": np.float64(raw["sigma0"])}
else:
    raise ValueError(f"Root finder method '{rootfinder}' is not supported")

quality_cfg = config.get("quality_check", {})
residual_warn_abs = np.float64(quality_cfg.get("residual_warn_abs", 1.0e-6))
residual_keep_abs = quality_cfg.get("residual_keep_abs", None)
if residual_keep_abs is not None:
    residual_keep_abs = np.float64(residual_keep_abs)
drop_sigma0_hits = bool(quality_cfg.get("drop_sigma0_hits", False))
keep_all_rows = bool(quality_cfg.get("keep_all_rows", False))

progress_every = int(quality_cfg.get("progress_every", 1000))
if progress_every <= 0:
    progress_every = 1000

feller_cfg = quality_cfg.get("feller_filter", {})
feller_enabled = bool(feller_cfg.get("enabled", False))
feller_slack_tol = np.float64(feller_cfg.get("slack_tol", 0.0))

force_target_after_filters = bool(quality_cfg.get("force_target_after_filters", False))
generation_chunk_size = int(quality_cfg.get("generation_chunk_size", target_n))
if generation_chunk_size <= 0:
    raise ValueError("quality_check.generation_chunk_size must be > 0")
max_generation_rounds = int(quality_cfg.get("max_generation_rounds", 20 if force_target_after_filters else 1))
if max_generation_rounds <= 0:
    raise ValueError("quality_check.max_generation_rounds must be > 0")

print(
    "Synthetic generation setup | "
    f"target_samples={target_n} | rootfinder={rootfinder} | "
    f"keep_all_rows={keep_all_rows} | "
    f"force_target_after_filters={force_target_after_filters}"
)
if force_target_after_filters:
    print(
        f"Chunk generation enabled | chunk_size={generation_chunk_size} | "
        f"max_rounds={max_generation_rounds}"
    )

rng = np.random.default_rng(seed)
kept_chunks = []
quality_chunks = []

kept_total = 0
generated_total = 0
generation_start = time.time()

for round_idx in range(1, max_generation_rounds + 1):
    if (not force_target_after_filters) and round_idx > 1:
        break
    if force_target_after_filters and kept_total >= target_n:
        break

    remaining = max(0, target_n - kept_total)
    if force_target_after_filters:
        if keep_all_rows:
            chunk_n = min(generation_chunk_size, remaining) if remaining > 0 else generation_chunk_size
        else:
            chunk_n = max(generation_chunk_size, remaining)
    else:
        chunk_n = target_n

    sample_seed = int(rng.integers(0, 2**32 - 1))
    print(
        f"\n=== Generation round {round_idx} | candidates={chunk_n} | "
        f"sample_seed={sample_seed} ==="
    )

    synth_candidates = _build_candidate_dataframe(
        n_samples=chunk_n,
        sample_seed=sample_seed,
        params_bounds=params_bounds,
        grid_bounds=grid_bounds,
        r_bounds=r_bounds,
        fixed_params=fixed_params,
    )
    kept_df, quality_df = _compute_iv_and_filters_for_chunk(
        synth_candidates,
        round_idx=round_idx,
        rootfinder=rootfinder,
        K=K,
        cos_params=cos_params,
        cos_interval_rule=cos_interval_rule,
        opt_type=opt_type,
        residual_warn_abs=residual_warn_abs,
        residual_keep_abs=residual_keep_abs,
        drop_sigma0_hits=drop_sigma0_hits,
        feller_enabled=feller_enabled,
        feller_slack_tol=feller_slack_tol,
        keep_all_rows=keep_all_rows,
        progress_every=progress_every,
        brent_cfg=brent_cfg,
        lm_cfg=lm_cfg,
    )

    row_offset = generated_total
    quality_df["row_id"] = quality_df["row_local"] + row_offset
    quality_df = quality_df.drop(columns=["row_local"])

    if not kept_df.empty:
        kept_df["row_id"] = kept_df["row_local"] + row_offset
        kept_df = kept_df.drop(columns=["row_local"])
        kept_chunks.append(kept_df)

    quality_chunks.append(quality_df)

    generated_total += chunk_n
    kept_total += len(kept_df)

    elapsed = time.time() - generation_start
    acceptance = kept_total / max(generated_total, 1)
    msg = (
        f"Round {round_idx} summary | kept_in_round={len(kept_df)} / {chunk_n} "
        f"({100.0 * len(kept_df) / chunk_n:.2f}%) | total_kept={kept_total} | "
        f"target={target_n} | cumulative_acceptance={100.0 * acceptance:.2f}% | "
        f"elapsed={_format_seconds(elapsed)}"
    )
    if force_target_after_filters and kept_total < target_n and acceptance > 0:
        expected_remaining_raw = int(np.ceil((target_n - kept_total) / acceptance))
        eta_sec = expected_remaining_raw / max(generated_total / max(elapsed, 1e-12), 1e-12)
        msg = f"{msg} | eta~{_format_seconds(eta_sec)}"
    print(msg)

if kept_total == 0:
    raise RuntimeError("No samples were selected for the final dataset.")

if force_target_after_filters and kept_total < target_n:
    raise RuntimeError(
        "Could not reach requested post-filter target. "
        f"kept={kept_total}, target={target_n}, rounds={max_generation_rounds}. "
        "Increase quality_check.max_generation_rounds or quality_check.generation_chunk_size."
    )

synth_df = pd.concat(kept_chunks, axis=0, ignore_index=True)
quality_df = pd.concat(quality_chunks, axis=0, ignore_index=True)

if force_target_after_filters and len(synth_df) > target_n:
    selected_idx = rng.permutation(len(synth_df))[:target_n]
    synth_df = synth_df.iloc[selected_idx].reset_index(drop=True)

if keep_all_rows and len(synth_df) == generated_total and len(quality_df) == generated_total:
    quality_df["selected_for_final_dataset"] = True
else:
    selected_row_ids = set(synth_df["row_id"].tolist())
    quality_df["selected_for_final_dataset"] = quality_df["row_id"].isin(selected_row_ids)

print(
    "\nFinal selection summary | "
    f"generated_total={generated_total} | kept_total={kept_total} | "
    f"final_selected={len(synth_df)}"
)

final_df = synth_df.loc[:, MASTER_COLS].reset_index(drop=True)

output_cfg = config.get("output", {})
output_dtype = str(output_cfg.get("dtype", "float32")).strip().lower()
if output_dtype == "float32":
    final_df = final_df.astype(np.float32, copy=False)
elif output_dtype == "float64":
    final_df = final_df.astype(np.float64, copy=False)
else:
    raise ValueError(f"Unsupported output.dtype='{output_dtype}'. Use 'float32' or 'float64'.")

print(final_df.head())

################################# SAVE DATASET #####################################
name = config["meta"]["name"]
out_name = f"{name}.parquet"
run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

OUT_PATH = PROJECT_ROOT / "data" / "synth" / run_id
OUT_PATH.mkdir(parents=True, exist_ok=True)

shutil.copy(PROJECT_ROOT / "configs" / "synth.yaml", OUT_PATH / "synth_copy.yaml")

final_df.to_parquet(OUT_PATH / out_name, engine="pyarrow", index=False)
print(f"Final dataset saved on: {OUT_PATH / out_name}")

quality_df.to_parquet(OUT_PATH / "iv_quality.parquet", engine="pyarrow", index=False)
print(f"IV quality report saved on: {OUT_PATH / 'iv_quality.parquet'}")

################################# BUILD SPLITS #####################################
if stratify_enabled:
    print(
        f"Building splits with IV quantile stratification: "
        f"target='{stratify_target_col}', n_bins={stratify_n_bins}"
    )
    train_df, val_df, test_df = dataframes_splits_stratified_quantiles(
        final_df,
        split_train,
        split_val,
        split_test,
        seed=seed,
        target_col=stratify_target_col,
        n_bins=stratify_n_bins,
    )
else:
    print("Building random train/val/test splits (no stratification)")
    train_df, val_df, test_df = dataframes_splits(
        final_df,
        split_train,
        split_val,
        split_test,
        seed=seed,
    )

train_df.to_parquet(OUT_PATH / "train.parquet", engine="pyarrow", index=False)
val_df.to_parquet(OUT_PATH / "val.parquet", engine="pyarrow", index=False)
test_df.to_parquet(OUT_PATH / "test.parquet", engine="pyarrow", index=False)
print(f"Splits saved on: {OUT_PATH}")
