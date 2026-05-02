# TODO - Multi-Output Greek-Consistency PINN

## 0. Goal

- [ ] Improve hard-region Greek accuracy without using COS Greek labels.
- [ ] Keep the method transferable to non-vanilla payoffs.
- [ ] Treat Greek heads as internal consistency fields, not supervised reference labels.

## 1. Model

- [X] Add experimental multi-output model:

  \[
  f_\theta(\tau,m,v,\Theta)
  =
  (V_\theta,\Delta_\theta,\Gamma_\theta,\mathrm{Vega}_\theta)
  \]

- [X] Keep `forward()` price-only for compatibility with existing PDE residuals and diagnostics.
- [X] Add `forward_all()` for training-time Greek-head consistency.
- [X] Initialize from baseline scalar checkpoint with flexible warm-start:
  - [X] copy compatible hidden layers
  - [X] copy scalar price head into the first output row
  - [X] leave new Greek heads randomly initialized

## 2. Loss

- [X] Keep existing PDE, terminal, and lower-boundary losses.
- [X] Add no-label consistency losses:

  \[
  \mathcal{L}_{\Delta}
  =
  \|\Delta_\theta - \partial_m V_\theta\|^2
  \]

  \[
  \mathcal{L}_{\Gamma}
  =
  \|\Gamma_\theta - \partial_m \Delta_\theta\|^2
  \]

  \[
  \mathcal{L}_{Vega}
  =
  \|\mathrm{Vega}_\theta - \partial_v V_\theta\|^2
  \]

- [X] Add training history columns:
  - [X] `train_greek_delta`, `val_greek_delta`
  - [X] `train_greek_gamma`, `val_greek_gamma`
  - [X] `train_greek_vega`, `val_greek_vega`

## 3. First Experiment

- [X] Save config:
  - [X] `configs/pinn_model_architecture_multi_output_greek_consistency.yaml`
  - [X] `configs/pinn_training_multi_output_greek_consistency.yaml`
  - [X] `configs/pinn_pipeline_multi_output_greek_consistency.yaml`
  - [X] `configs/pinn_baseline_diagnostics_multi_output_greek_consistency.yaml`

- [ ] Train:

  ```bash
  .venv/bin/python scripts/run_pinn_pipeline.py --config configs/pinn_pipeline_multi_output_greek_consistency.yaml
  ```

- [ ] Run diagnostics:

  ```bash
  .venv/bin/python scripts/run_pinn_baseline_diagnostics.py --config configs/pinn_baseline_diagnostics_multi_output_greek_consistency.yaml
  ```

## 4. Evaluation

- [ ] Compare against baseline PINN on:
  - [ ] hard Delta RMSE
  - [ ] hard Gamma RMSE
  - [ ] hard Vega RMSE
  - [ ] hard p99 Gamma error
  - [ ] full price RMSE
  - [ ] PDE residual maps
  - [ ] terminal/lower boundary violations

- [ ] Compare autodiff Greeks against head Greeks:
  - [ ] Delta head vs autodiff Delta
  - [ ] Gamma head vs autodiff Gamma
  - [ ] Vega head vs autodiff Vega
  - [ ] head Greek errors vs Heston-CF benchmark

## 5. Decision

- [ ] Accept only if hard-region Gamma or Delta improves without unacceptable price or boundary degradation.
- [ ] If unstable, test lower head-consistency weights.
- [ ] If heads are better than autodiff Greeks, consider a separate inference adapter that can expose head Greeks explicitly.
