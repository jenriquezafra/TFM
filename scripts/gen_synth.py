import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
config_path = PROJECT_ROOT / "configs" / "synth.yaml"


from src.solvers.implied_vol import IV_Brent, IV_LM
from src.datasets.make_synth import generate_all


####################################### LOAD CONFIGS ########################################

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

seed = config["meta"]["seed"]
data_cfg = config["data"]

N = int(data_cfg["n_samples"])


params_cfg = data_cfg["heston_params"]
PARAM_ORDER = ["rho", "kappa", "gamma", "bar_v", "v0"]

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

params_bounds = np.vstack([as_bounds(params_cfg[param], param) for param in PARAM_ORDER])

grid_cfg = data_cfg["grid"]

grid_bounds = np.array([grid_cfg["moneyness"], grid_cfg["tau"]])
r_bounds = np.array([config["market"]["r"][0], config["market"]["r"][1]])


cos_params_cfg = config["cos_solver"]
cos_params = np.array([
    np.float64(cos_params_cfg["N"]),
    np.float64(cos_params_cfg["L"])
])

opt_type = config["market"]["option_type"]


rootfinder = config["root_finder"]["method"]
if rootfinder == "brent_iv":
    rootfinder_params_cfg = config["root_finder"]["methods"]["brent_iv"]
    iv_bounds = np.array(np.float64(rootfinder_params_cfg["iv_bounds"]))
    brent_tol = np.float64(rootfinder_params_cfg["tol"])
    brent_maxiter = np.float64(rootfinder_params_cfg["max_iter"])
elif rootfinder == "LM":
    rootfinder_params_cfg = config["root_finder"]["methods"]["LM"]
    LM_sigma0 = np.float64(rootfinder_params_cfg["sigma0"])
else:
    raise ValueError(f"Root finder method '{rootfinder}' is not supported")

quality_cfg = config.get("quality_check", {})
residual_warn_abs = np.float64(quality_cfg.get("residual_warn_abs", 1.0e-6))
progress_every = int(quality_cfg.get("progress_every", 10000))
if progress_every <= 0:
    progress_every = 10000



K = config["market"]["K"]

################################# GENERATE ALL THE PARAMS #####################################
all_samples = generate_all(n_samples=N,
                           param_ranges=params_bounds,
                           grid_bounds=grid_bounds,
                           r_bounds=r_bounds,
                           seed=seed)

    

################################# CREATE DATASET #####################################

cols = [
    "rho", "kappa", "gamma", "bar_v", "v0",
    "moneyness", "tau", "r",
    "IV"
]


synth_df = pd.DataFrame(data=np.nan,
                       index=range(N),
                       columns=cols,
                       )

# unpack the tuples
X_params, X_grid, X_r = all_samples
X = np.hstack([X_params, X_grid, X_r])

synth_df.iloc[:, :-1] = X

fixed = {}
variable = {}

# vemos cuáles params son fijos y cuáles variables
for name in PARAM_ORDER:
    value = params_cfg[name]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if value[0] == value[1]:
            fixed[name] = value[0]
        else:
            variable[name] = value
    else:
        fixed[name] = value

for param_name, param_value in fixed.items():
    synth_df.loc[:, param_name] = param_value


################################# COMPUTE IVs #####################################
# recorrer el dataset por fila e ir calculando las IVs
iv_values = np.full(shape=N, fill_value=np.nan, dtype=np.float64)
price_residual_abs = np.full(shape=N, fill_value=np.nan, dtype=np.float64)
solver_success = np.zeros(shape=N, dtype=bool)
gen_start_time = time.time()
print(
    f"Generating IVs with '{rootfinder}' "
    f"for {N} samples"
)

if rootfinder == "brent_iv":
    for i in range(0, N):
        try:
            iv, details = IV_Brent(
                params_Heston=synth_df.iloc[i,:5],
                S0=synth_df.loc[i, "moneyness"],          # m=S0/K, but K=1 so S0=m
                K=np.float64(K),                          # K=1 fixed
                tau=synth_df.loc[i, "tau"],
                r=synth_df.loc[i, "r"],
                COS_params=cos_params,
                opt_type=opt_type,
                iv_bounds=iv_bounds,
                tol=brent_tol,
                max_iter=brent_maxiter,
                return_details=True,
            )
            iv_values[i] = iv
            price_residual_abs[i] = details["price_residual_abs"]
            solver_success[i] = True
        except Exception:
            iv_values[i] = np.nan
            price_residual_abs[i] = np.nan
            solver_success[i] = False

        if ((i + 1) % progress_every == 0) or (i == N - 1):
            processed = i + 1
            elapsed = time.time() - gen_start_time
            rate = processed / max(elapsed, 1e-12)
            eta = (N - processed) / max(rate, 1e-12)
            pct = 100.0 * processed / N
            residual_slice = price_residual_abs[:i+1]
            valid = residual_slice[np.isfinite(residual_slice)]
            mean_res = float(valid.mean()) if len(valid) > 0 else np.nan
            bad_count = int((valid > residual_warn_abs).sum()) if len(valid) > 0 else 0
            print(
                f"[{rootfinder}] {processed}/{N} ({pct:5.1f}%) | "
                f"elapsed={_format_seconds(elapsed)} | "
                f"eta={_format_seconds(eta)} | "
                f"mean|BS(IV)-V_tgt|={mean_res:.3e} | "
                f"bad(>{residual_warn_abs:.1e})={bad_count}"
            )

elif rootfinder == "LM":
    for i in range(0, N):
        try:
            iv, details = IV_LM(
                params_Heston=synth_df.iloc[i,:5],
                S0=synth_df.loc[i, "moneyness"],          # m=S0/K, but K=1 so S0=m
                K=np.float64(K),                          # K=1 fixed
                tau=synth_df.loc[i, "tau"],
                r=synth_df.loc[i, "r"],
                COS_params=cos_params,
                opt_type=opt_type,
                sigma0=LM_sigma0,
                return_details=True,
            )
            iv_values[i] = iv
            price_residual_abs[i] = details["price_residual_abs"]
            solver_success[i] = details["success"]
        except Exception:
            iv_values[i] = np.nan
            price_residual_abs[i] = np.nan
            solver_success[i] = False

        if ((i + 1) % progress_every == 0) or (i == N - 1):
            processed = i + 1
            elapsed = time.time() - gen_start_time
            rate = processed / max(elapsed, 1e-12)
            eta = (N - processed) / max(rate, 1e-12)
            pct = 100.0 * processed / N
            residual_slice = price_residual_abs[:i+1]
            valid = residual_slice[np.isfinite(residual_slice)]
            mean_res = float(valid.mean()) if len(valid) > 0 else np.nan
            bad_count = int((valid > residual_warn_abs).sum()) if len(valid) > 0 else 0
            sigma0_hits = int(np.isclose(iv_values[:i+1], LM_sigma0, atol=1e-12, rtol=0).sum())
            print(
                f"[{rootfinder}] {processed}/{N} ({pct:5.1f}%) | "
                f"elapsed={_format_seconds(elapsed)} | "
                f"eta={_format_seconds(eta)} | "
                f"mean|BS(IV)-V_tgt|={mean_res:.3e} | "
                f"bad(>{residual_warn_abs:.1e})={bad_count} | "
                f"IV==sigma0({LM_sigma0:.3f})={sigma0_hits}"
            )

synth_df.loc[:, "IV"] = iv_values

valid_residual = price_residual_abs[np.isfinite(price_residual_abs)]
if len(valid_residual) > 0:
    print("IV quality summary")
    print(f"  valid residuals: {len(valid_residual)}/{N}")
    print(f"  mean |BS(IV)-V_tgt|: {float(valid_residual.mean()):.3e}")
    print(f"  p90  |BS(IV)-V_tgt|: {float(np.percentile(valid_residual, 90)):.3e}")
    print(f"  p99  |BS(IV)-V_tgt|: {float(np.percentile(valid_residual, 99)):.3e}")
    print(f"  max  |BS(IV)-V_tgt|: {float(valid_residual.max()):.3e}")
    print(f"  bad count (>{residual_warn_abs:.1e}): {int((valid_residual > residual_warn_abs).sum())}")
else:
    print("IV quality summary: no valid residuals found")


print(synth_df.head())
################################# SAVE DATASET #####################################
name = config["meta"]["name"]
out_name = f"{name}.parquet"
run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

OUT_PATH = PROJECT_ROOT / "data" / "synth" / run_id 
OUT_PATH.mkdir(parents=True, exist_ok=True)

# guardar en parquet
synth_df.to_parquet(OUT_PATH / out_name, engine="pyarrow", index=False)
print(f"Final dataset saved on: {OUT_PATH/out_name}")

# save IV quality diagnostics (separate file to keep train schema unchanged)
quality_df = pd.DataFrame({
    "row_id": np.arange(N, dtype=np.int64),
    "IV": iv_values,
    "price_residual_abs": price_residual_abs,
    "solver_success": solver_success,
})
if rootfinder == "LM":
    quality_df["iv_equals_sigma0"] = np.isclose(iv_values, LM_sigma0, atol=1e-12, rtol=0)
quality_df.to_parquet(OUT_PATH / "iv_quality.parquet", engine="pyarrow", index=False)
print(f"IV quality report saved on: {OUT_PATH / 'iv_quality.parquet'}")

# also split and save
from src.datasets.splits import dataframes_splits

train_df, val_df, test_df = dataframes_splits(
    synth_df, 0.8, 0.1, 0.1, seed=seed     # TODO: ponerlo con config mas adelante
)

train_df.to_parquet(OUT_PATH / "train.parquet", engine="pyarrow", index=False)
val_df.to_parquet(OUT_PATH / "val.parquet", engine="pyarrow", index=False)
test_df.to_parquet(OUT_PATH / "test.parquet", engine="pyarrow", index=False)
print(f"Splits saved on: {OUT_PATH}") 
