import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
config_path = PROJECT_ROOT / "configs" / "synth.yaml"


from src.solvers.implied_vol import IV_Brent
from src.datasets.make_synth import generate_all


####################################### LOAD CONFIGS ########################################

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

seed = config["meta"]["seed"]
data_cfg = config["data"]
N = int(data_cfg["n_samples"])


params_cfg = config["data"]["heston_params"]
rho_bounds = np.array([params_cfg["rho"][0], params_cfg["rho"][1]])
kappa_bounds = np.array([params_cfg["kappa"][0], params_cfg["kappa"][1]])
gamma_bounds = np.array([params_cfg["gamma"][0], params_cfg["gamma"][1]])
bar_v_bounds = np.array([params_cfg["bar_v"][0], params_cfg["bar_v"][1]])
v0_bounds = np.array([params_cfg["v0"][0], params_cfg["v0"][1]])
params_bounds = np.array([rho_bounds, kappa_bounds, gamma_bounds, bar_v_bounds, v0_bounds])

grid_cfg = config["data"]["grid"]

grid_bounds = np.array([grid_cfg["moneyness"], grid_cfg["tau"]])
r_bounds = np.array([data_cfg["market"]["r"][0], data_cfg["market"]["r"][1]])


cos_params_cfg = config["cos_solver"]
cos_params = np.array([
    np.float64(cos_params_cfg["N"]),
    np.float64(cos_params_cfg["L"])
])

opt_type = data_cfg["market"]["option_type"]

rootfinder_params_cfg = config["brent_iv"]
iv_bounds = np.array(
    np.float64(rootfinder_params_cfg["iv_bounds"])
)

brent_tol = np.float64(rootfinder_params_cfg["tol"])
brent_maxiter = np.float64(rootfinder_params_cfg["max_iter"])

K = data_cfg["market"]["K"]

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

# fijamos kappa y v0 (como hace Liu)
if params_cfg["fixed"]==True:           # NOTE: si no lo fijo, hay probkemas con Brent
    # the values from the TFG
    synth_df.loc[:, "kappa"] = 0.9
    synth_df.loc[:, "v0"] = 0.36



################################# COMPUTE IVs #####################################
# recorrer el dataset por fila e ir calculando las IVs
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