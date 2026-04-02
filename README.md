# TFM
Computational algorithms for pricing financial derivatives under local volatility models


## Repository structure

(may not be totally up to date)
```text

.
├── figures/                 # Generated figures (plots, final results)
├── .venv/                   # Local virtual environment (not versioned)
├── configs/                 # Experiment configuration files
│   ├── synth.yaml              # Synthetic data generation (Heston + COS)
│   └── train.yaml              # Neural network training configuration
│
├── data/
│   ├── market/                 # Market data (when used)
│   └── synth/                  # Synthetic data generated with COS
│
├── notebooks/               # Exploratory notebooks (non-core)
│   ├── 01_toy_NN.ipynb
│   ├── 02_Heston_solver.ipynb
│   ├── 03_NN_surrogate.ipynb
│   ├── figures.ipynb
│   └── some_checks.ipynb
│
├── scripts/                 # Executable entry points
│   ├── gen_synth.py            # Synthetic data generation
│   ├── train_pricer.py         # Surrogate pricer training
│   └── eval_pricer.py          # COS vs NN evaluation
│
├── src/                     # Reusable project code
│   ├── datasets/
│   │   ├── make_synth.py           # Synthetic dataset generation logic
│   │   └── market.py               # Market data ingestion/preprocessing (future)
│   ├── models/
│   │   ├── pricer_nn.py            # Surrogate pricer architecture (TF/Keras)
│   │   └── calibrator_nn.py        # NN architecture for calibration
│   ├── solvers/
│   │   └── heston_cos.py           # Heston COS solver (ground truth)
│   ├── eval.py                 # Metrics and benchmarks
│   ├── train.py                # Generic training loops
│   └── utils.py                # Common utilities (checks, seeds, helpers)
│
├── artifacts/               # Data from the models
│   ├── metrics/            
│   ├── models/                 # trained NN models and related artifacts
│       ├── pricer_nn/
│       │   ├── baseline/           # reference experiment used as benchmark
│       │   │   ├── checkpoints/        # intermediate models saved during training via callbacks 
│       │   │   └── final/              # selected final model and associated configuration
│       │   └── v2/                 # alternative experiment (TBD)
│       └── calibrator_nn/ 
│
├── tests/                   # Minimal tests (sanity checks)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```
Note:
- Core logic lives in 'src/'
- 'scripts/' only orchestrade experiments and ensure reproducibility
- Notebooks are used for exploration only

## PINN scaffold

A PINN scaffold has been added to start the PINN workstream:

- Core package: `src/pinn/`
- Entry point: `scripts/run_pinn_pipeline.py`
- Configs:
  - `configs/pinn_pipeline.yaml`
  - `configs/pinn_model_architecture.yaml`
  - `configs/pinn_training.yaml`
- VSCode:
  - `.vscode/tasks.json`
  - `.vscode/launch.json`

Run scaffold validation:

```bash
.venv/bin/python scripts/run_pinn_pipeline.py \
  --config configs/pinn_pipeline.yaml \
  --stage all \
  --dry-run \
  --dump-plan
```

Plan output is written to:
- `outputs/pinn/PINN_v01/pipeline_plan.yaml`

Design notes:
- `summary.yaml` and `quotes_comparison.parquet` from CaNN are treated as mandatory inputs.
- `price_market` is a mandatory target column for the PINN dataset build. There is no fallback to IV columns.
- Stage status:
  - `prepare_dataset`: implemented
  - `train` (supervised baseline, MSE): implemented
  - `evaluate` (all/train/val metrics): implemented

Run execution:

```bash
.venv/bin/python scripts/run_pinn_pipeline.py \
  --config configs/pinn_pipeline.yaml \
  --stage all \
  --dump-plan
```

Execution writes:
- `outputs/pinn/PINN_v01/pipeline_execution.yaml`
- `outputs/pinn/PINN_v01/data/supervised_dataset.npz`
- `outputs/pinn/PINN_v01/train/checkpoints/model_best.pt`

### PINN Greeks (autodiff)

Compute value + Jacobian/Hessian Greeks directly from a trained PINN checkpoint:

```bash
.venv/bin/python scripts/run_pinn_greeks.py \
  --run-dir PINN_mix_scaled_param \
  --features "0.5,1.0,0.04,-0.45,1.5,0.4,0.25,0.02" \
  --output-csv outputs/greeks/greeks_pinn_smoke.csv
```

Notes:
- Default feature order is inferred from PINN artifacts (`train_summary` / collocation manifest).
- If `spot_feature=moneyness`, use `--strike K` to convert \(\Delta,\Gamma\) from moneyness-space to spot-space.
- When a Greek name collides with an input column name (e.g., `gamma`, `rho`), output columns are prefixed as `greek_gamma`, `greek_rho`.

## Optimizer experiment logs (MIX / ADAM)

To build optimizer-specific experiment logs from `outputs/runs/*`:

```bash
.venv/bin/python scripts/build_optimizer_experiment_logs.py
```

Generated files:
- `outputs/experiment_logs/mix_experiments.csv`
- `outputs/experiment_logs/adam_experiments.csv`

The logs include:
- Training metrics (`best_val_loss`, `best_epoch`, `val_last_vs_best_pct`, etc.).
- Latest calibration metrics (`calib_weighted_mse`, `calib_residual_rmse`, `calib_objective_fun`) when available.
- Relative comparison versus historical best for each optimizer family:
  - `train_vs_best_hist_pct`
  - `calib_vs_best_hist_pct`

`scripts/train_pricer.py` now runs post-training calibration automatically for the new run, refreshes these logs, and prints the run-vs-history summary in terminal.

## Evaluate an existing trained pricer (`eval_pricer.py`)

This evaluates an already-trained ANN run (no retraining), computes global and regional metrics, and saves plots.

Main command:
```bash
.venv/bin/python scripts/eval_pricer.py --model-dir MIX_v04 --device cpu
```

With explicit data directory:
```bash
.venv/bin/python scripts/eval_pricer.py \
  --model-dir MIX_v04 \
  --data-dir data/synth/1M_Feller \
  --batch-size 8192
```

Important:
- There is no dedicated `eval` config file for this script.
- By default it loads settings from the selected run:
  - `outputs/runs/<RUN>/model_training_copy.yaml` for `data.dir` and evaluation bins.
  - `outputs/runs/<RUN>/metrics/normalization_stats.yaml` for input/output normalization.
- CLI flags can override defaults (`--data-dir`, `--tau-bins`, `--moneyness-bins`, `--splits`, etc.).

Artifacts written to `outputs/runs/<RUN>/`:
- `metrics/eval_global.csv` and `.parquet`
- `metrics/eval_by_region.csv` and `.parquet`
- `metrics/eval_summary.yaml`
- `figures/eval_global_mse.png`
- `figures/eval_global_mae.png`
- `figures/eval_region_mse.png`
- `figures/eval_region_mae.png`

Useful flags:
- `--checkpoint-name model_best.pt` (default)
- `--splits train,val,test`
- `--target-col IV`
- `--no-plots` (compute metrics only)

See all options:
```bash
.venv/bin/python scripts/eval_pricer.py --help
```


## Calibration (CaNN + Differential Evolution)

This project supports Liu-style calibration:
- Train a forward ANN once (offline).
- Calibrate parameters online with Differential Evolution (DE) over the frozen ANN.

### What is CLI?
CLI means **Command Line Interface**: running scripts from terminal with flags like:
`--model-dir`, `--maxiter`, `--quotes`, etc.

### Required market quotes file
`scripts/calibrate_cann.py` expects a CSV/Parquet with columns:
- `moneyness`
- `tau`
- `r`
- `iv_market`
- `weight` (optional)

Columns can be remapped in `configs/calibration.yaml`.

### List available trained models
```bash
.venv/bin/python scripts/calibrate_cann.py --list-models
```

### Run calibration for a specific model
```bash
.venv/bin/python scripts/calibrate_cann.py \
  --config configs/calibration.yaml \
  --model ADAM_v05 \
  --quotes data/market/market_quotes.csv
```

### Exact command (your current setup)
```bash
cd /Users/jenriquezafra/Proyectos/Dev/python/TFM/TFM

.venv/bin/python scripts/calibrate_cann.py \
  --config configs/calibration.yaml \
  --model-dir ADAM_v06 \
  --quotes data/market/market_quotes_liu_35.csv \
  --maxiter 1500 \
  --popsize 10
```

Important:
- Do not use `data/market/smoke_quotes.csv` unless you want a smoke test.
- Use `market_quotes_liu_35.csv` (or your real market file) for meaningful calibration.

Notes:
- `--model` is an alias of `--model-dir`.
- If no model is provided, config value is used (`model.model_dir`, default: `latest`).
- Output directory is always organized as:
  `outputs/calibration/<MODEL_RUN_NAME>/Calibration_<N>/`.
- `--tag` is optional metadata stored in `summary.yaml/json` (it does not change folder naming).
- `--theta-true RHO KAPPA GAMMA BAR_V V0` enables parameter error artifacts from CLI.
- `--truth-file <path.yaml|json>` loads `theta_true` from file (CLI override).

### Synthetic calibration with parameter error table/plots
Without editing YAML, you can pass ground truth directly:
```bash
.venv/bin/python scripts/calibrate_cann.py \
  --config configs/calibration.yaml \
  --model-dir ADAM_v06 \
  --quotes data/market/market_quotes_liu_35.csv \
  --theta-true -0.45 1.0 0.35 0.20 0.22
```

Or pass a truth sidecar file:
```bash
.venv/bin/python scripts/calibrate_cann.py \
  --config configs/calibration.yaml \
  --model-dir ADAM_v06 \
  --quotes data/market/market_quotes_liu_35.csv \
  --truth-file data/market/market_quotes_liu_35.truth.yaml
```

### Increase DE iterations
`maxiter` controls the number of DE generations.

Example with more iterations:
```bash
.venv/bin/python scripts/calibrate_cann.py \
  --config configs/calibration.yaml \
  --model ADAM_v05 \
  --quotes data/market/market_quotes.csv \
  --maxiter 1500 \
  --popsize 15
```

Tradeoff:
- higher `maxiter` -> slower but more chance of convergence (`success: true`)
- lower `maxiter` -> faster but more chance of early stop (`Maximum number of iterations has been exceeded`)

### Output artifacts
Each run writes under:
`outputs/calibration/<MODEL_RUN_NAME>/Calibration_<N>/`

Example:
`outputs/calibration/ADAM_v05/Calibration_1/`

Files saved in each calibration run:
- `summary.yaml`
- `summary.json`
- `quotes_comparison.parquet` (and csv optionally)
- `market_quotes_input.<ext>` (exact quotes file used)
- `calibration_config_source.yaml` (original config passed)
- `calibration_config_used.yaml` (effective config actually used, including CLI overrides)

### Evaluate Jacobian/Hessian at calibrated optimum (`theta_star`)
You can evaluate the calibration objective curvature directly at a saved optimum:

```bash
.venv/bin/python scripts/eval_calibration_curvature.py \
  --calibration-dir outputs/calibration/MIX_v05/Calibration_1 \
  --device cpu \
  --dtype float64
```

The script writes:
- `curvature_theta_star.json` (objective value, gradient, Hessian, eigenvalue diagnostics)
- `hessian_theta_star.csv` (Hessian matrix with parameter labels)
- `jacobian_theta_star.png` (bar plot of gradient components)
- `hessian_theta_star.png` (heatmap of Hessian entries)

Useful variants:
- Latest calibration automatically:
```bash
.venv/bin/python scripts/eval_calibration_curvature.py
```
- Use `theta_true` from summary (synthetic setups):
```bash
.venv/bin/python scripts/eval_calibration_curvature.py --theta-key theta_true
```
- Manual parameter vector:
```bash
.venv/bin/python scripts/eval_calibration_curvature.py \
  --summary-path outputs/calibration/MIX_v05/Calibration_1/summary.json \
  --theta "-0.33,1.66,0.55,0.20,0.23"
```
- Skip plots:
```bash
.venv/bin/python scripts/eval_calibration_curvature.py --no-plots
```

If `synthetic_truth.theta_true` is set in `configs/calibration.yaml`, extra parameter-error artifacts are generated:
- `parameter_errors.csv`
- `parameter_error_abs_bar.png`
- `parameter_error_abs_heatmap.png`
- `parameter_error_rel_heatmap.png`

Example in config:
```yaml
synthetic_truth:
  theta_true: [-0.45, 1.0, 0.35, 0.20, 0.22]  # [rho, kappa, gamma, bar_v, v0]
```

Alternative (recommended for synthetic market CSV):
```yaml
synthetic_truth:
  theta_true: null
  truth_file: data/market/market_quotes_liu_35.truth.yaml
```

Example `data/market/market_quotes_liu_35.truth.yaml`:
```yaml
theta_true:
  rho: -0.45
  kappa: 1.00
  gamma: 0.35
  bar_v: 0.20
  v0: 0.22
```

Auto-detection order for `theta_true`:
- `synthetic_truth.theta_true` in config
- columns in quotes file (`rho,kappa,gamma,bar_v,v0`) if present and constant
- `synthetic_truth.truth_file`
- sidecar file next to quotes:
  `<quotes>.truth.yaml|yml|json` or `<quotes_stem>_truth.yaml|yml|json`
