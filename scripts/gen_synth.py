import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

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
if rootfinder == "brent_iv":
    for i in range(0, N):
        print(synth_df.iloc[i,:])
        iv = IV_Brent(
            params_Heston=synth_df.iloc[i,:5],
            S0=synth_df.loc[i, "moneyness"],          # m=S0/K, but K=1 so S0=m
            K= np.float64(K),                        # K=1 fixed
            tau=synth_df.loc[i, "tau"],
            r=synth_df.loc[i,"r"],
            COS_params=cos_params,
            opt_type=opt_type,
            iv_bounds=iv_bounds,
            tol=brent_tol,
            max_iter=brent_maxiter 
        )
        synth_df.loc[i, "IV"] = iv

elif rootfinder == "LM":
    for i in range(0, N):
        iv = IV_LM(
            params_Heston=synth_df.iloc[i,:5],
            S0=synth_df.loc[i, "moneyness"],          # m=S0/K, but K=1 so S0=m
            K= np.float64(K),                        # K=1 fixed
            tau=synth_df.loc[i, "tau"],
            r=synth_df.loc[i,"r"],
            COS_params=cos_params,
            opt_type=opt_type,
            sigma0=LM_sigma0
        )
        synth_df.loc[i, "IV"] = iv
        print(synth_df.iloc[i,:])



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

# also split and save
from src.datasets.splits import dataframes_splits

train_df, val_df, test_df = dataframes_splits(
    synth_df, 0.8, 0.1, 0.1, seed=seed     # TODO: ponerlo con config mas adelante
)

train_df.to_parquet(OUT_PATH / "train.parquet", engine="pyarrow", index=False)
val_df.to_parquet(OUT_PATH / "val.parquet", engine="pyarrow", index=False)
test_df.to_parquet(OUT_PATH / "test.parquet", engine="pyarrow", index=False)
print(f"Splits saved on: {OUT_PATH}") 
