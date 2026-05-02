# TODO — Greek Error Improvement Experiments

## 0. Goal

- [ ] Improve Greek accuracy near the hard region:
  - [ ] Short maturity: \(\tau \to 0\)
  - [ ] At-the-money region: \(m \approx 1\)

- [X] Avoid training methods that require COS Greek labels.
- [X] Use COS only as an external evaluation benchmark for the vanilla Heston case.
- [ ] Keep the methodology scalable to more complex derivatives.
- [ ] Prioritize methods based on structure, PDE consistency and payoff information.

---

## 1. Sanity Checks and Baseline Diagnostics

### 1.1 PDE convention

- [X] Verify time convention:
  - [X] calendar time \(t\)
  - [X] time-to-maturity \(\tau = T-t\)
  - [X] sign of the time derivative

- [X] Verify Heston PDE terms:
  - [X] drift term in \(S\)
  - [X] variance drift term
  - [X] discounting term
  - [X] second derivative in \(S\)
  - [X] second derivative in \(v\)
  - [X] mixed derivative term
  - [X] correlation term

### 1.2 Greek definitions

- [X] Verify Delta is computed with respect to \(S\), not \(m\).
- [X] Verify Gamma is computed with respect to \(S\), not \(m\).
- [X] Verify Vega is computed with respect to the intended variable:
  - [X] variance \(v\)
  - [ ] volatility \(\sqrt{v}\)
  - [ ] initial variance \(v_0\)

- [X] Verify Theta convention:
  - [X] derivative with respect to calendar time \(t\)
  - [X] derivative with respect to time-to-maturity \(\tau\)

### 1.3 Autodiff validation

- [X] Compare autodiff Delta against finite differences of the trained model.
- [X] Compare autodiff Gamma against finite differences of the trained model.
- [X] Compare autodiff Vega against finite differences of the trained model.
- [ ] Check numerical stability near:
  - [X] \(\tau \to 0\)
  - [X] \(m \approx 1\)
  - [ ] low variance
  - [ ] high variance

### 1.4 Current diagnostic plots

- [X] Generate current error maps:
  - [X] price error over \((m,\tau)\)
  - [X] Delta error over \((m,\tau)\)
  - [X] Gamma error over \((m,\tau)\)
  - [X] Vega error over \((m,\tau)\)
  - [X] PDE residual over \((m,\tau)\)

- [X] Store all baseline diagnostic figures.

## 2. Define the Hard Region

### 2.1 Log-moneyness

- [X] Define log-moneyness:

  \[
  x = \log(m)
  \]

### 2.2 Hard region

- [X] Define the hard region:

  \[
  \mathcal{D}_{hard}
  =
  \{(m,\tau): |\log(m)| < \epsilon_m,\ \tau < \epsilon_\tau\}
  \]

- [X] Start with:
  - [X] \(\epsilon_m = 0.03\)
  - [X] \(\epsilon_\tau = 0.05\)

- [X] Create masks for:
  - [X] full domain
  - [X] short-maturity region
  - [X] ATM region
  - [X] hard region: ATM + short maturity

---

## 3. Baseline PINN

### 3.1 Freeze baseline

- [X] Freeze current baseline PINN configuration.
- [X] Save:
  - [X] architecture
  - [X] activation functions
  - [X] input variables
  - [X] output variable
  - [X] loss weights
  - [X] optimizer
  - [X] learning rate schedule
  - [X] training domain
  - [X] number of collocation points
  - [X] random seed

- [X] Save experiment as:
  - [X] `experiment_id: baseline_pinn`

### 3.2 Baseline metrics

- [X] Reproduce global metrics:
  - [X] price RMSE
  - [X] price MAE
  - [X] price \(R^2\)
  - [X] Delta RMSE
  - [X] Gamma RMSE
  - [X] Vega RMSE
  - [X] Theta RMSE, if available

- [X] Compute metrics separately on:
  - [X] full domain
  - [X] hard region
  - [X] short-maturity region
  - [X] ATM region

### 3.3 Baseline plots

- [X] Generate baseline heatmaps:
  - [X] price error
  - [X] Delta error
  - [X] Gamma error
  - [X] Vega error
  - [X] PDE residual

- [X] Save all baseline results as reference for the ablation study.

## 4. Payoff-Aware PINN

### 4.1 Motivation

- [X] Introduce a PINN parametrization that uses the known terminal payoff structure.
- [X] Avoid using COS Greek labels during training.
- [X] Enforce better behavior near \(\tau = 0\) by construction.

### 4.2 Payoff-aware ansatz

- [X] Implement:

  \[
  V_\theta(\tau, x, v, \Theta)
  =
  g_\epsilon(x)
  +
  a(\tau)N_\theta(\tau, x, v, \Theta)
  \]

- [X] Choose \(a(\tau)\) such that:

  \[
  a(0)=0
  \]

- [ ] Test candidates:
  - [X] \(a(\tau)=\tau\)
  - [ ] \(a(\tau)=\sqrt{\tau+\varepsilon}-\sqrt{\varepsilon}\)

### 4.3 Smoothed payoff

- [X] Implement smoothed call payoff:

  \[
  g_\epsilon(S)
  =
  \epsilon \log\left(1+\exp\left(\frac{S-K}{\epsilon}\right)\right)
  \]

- [ ] Test smoothing values:
  - [X] \(\epsilon = 10^{-3}\)
  - [ ] \(\epsilon = 5 \times 10^{-3}\)
  - [ ] \(\epsilon = 10^{-2}\)

- [X] Check whether payoff smoothing introduces unacceptable bias near maturity.
- [X] Compare smoothed payoff against true payoff.

### 4.3b Experimental adaptive payoff smoothing

- [X] Add an explicitly experimental smoothing mode:

  \[
  \epsilon(\tau,v)
  =
  c\sqrt{v\tau+\epsilon_0}
  \]

- [X] Keep fixed smoothing as the default production/payoff-aware behavior.
- [X] Mark adaptive smoothing as experimental in code and configs.
- [X] Add clipping parameters:
  - [X] minimum smoothing width
  - [X] maximum smoothing width
- [X] Save experiment config as:
  - [X] `experiment_id: payoff_adaptive_smoothing_experimental`
- [X] Train and evaluate this variant.
- [X] Reject unless hard-region Delta/Gamma improve without unacceptable price degradation.
- [X] First result:
  - [X] hard price RMSE: 0.00999
  - [X] hard Delta RMSE: 0.1595
  - [X] hard Gamma RMSE: 39.57
  - [X] hard Gamma p99: 149.8
  - [X] hard Vega RMSE: 0.1834
- [X] Decision:
  - [X] Reject current adaptive smoothing settings as final candidate.
  - [X] Keep implementation as experimental because the formulation remains transferable to non-vanilla payoffs, but the current scale/clipping over-smooths or mis-scales the short-maturity ATM layer.

### 4.4 Training

- [X] Train payoff-aware PINN on the same domain as baseline.
- [X] Keep architecture as close as possible to baseline.
- [X] Use PDE residual and boundary/terminal structure only.
- [X] Do not use COS Greek labels in training.
- [X] Save experiment as:
  - [X] `experiment_id: payoff_aware_pinn`

### 4.5 Evaluation

- [X] Evaluate price errors against COS.
- [X] Evaluate Greek errors against COS only for benchmarking.
- [X] Compare against baseline PINN.
- [X] Check whether payoff-aware parametrization improves:
  - [X] Delta near ATM-short maturity
  - [X] Gamma near ATM-short maturity
  - [X] Vega near ATM-short maturity
  - [X] Theta near short maturity
  - [X] global price accuracy
  - [X] PDE residual stability


## 5. Payoff-Aware + Boundary-Layer Coordinates

### 5.1 Motivation

- [X] Improve representation of the short-maturity ATM layer.
- [X] Account for the fact that the difficult region shrinks approximately like \(\sqrt{\tau}\).
- [X] Avoid adding unnecessary architectural complexity.

### 5.2 Coordinate transformation

- [X] Implement log-moneyness:

  \[
  x = \log(m)
  \]

- [X] Implement boundary-layer coordinate:

  \[
  z = \frac{x}{\sqrt{\tau+\varepsilon}}
  \]

- [X] Add transformed time variables:

  \[
  \sqrt{\tau+\varepsilon}
  \]

  \[
  \log(\tau+\varepsilon)
  \]

- [ ] Test \(\varepsilon\) values:
  - [X] \(10^{-6}\)
  - [ ] \(10^{-5}\)
  - [ ] \(10^{-4}\)

### 5.3 Input set

- [X] Replace or augment original inputs with:
  - [X] \(z\)
  - [X] \(x\), optionally
  - [X] \(\sqrt{\tau+\varepsilon}\)
  - [X] \(\log(\tau+\varepsilon)\)
  - [X] \(v_0\)
  - [X] \(\kappa\)
  - [X] \(\theta\)
  - [X] \(\gamma\)
  - [X] \(\rho\)
  - [X] \(r\)

### 5.4 Greek computation

- [X] Verify that Greeks are still computed with respect to the original financial variables.
- [X] Do not interpret derivatives with respect to \(z\) as financial Greeks.
- [X] Validate chain rule/autodiff consistency.
- [X] Compare model Greeks against finite differences of the model.

### 5.5 Training

- [X] Train payoff-aware PINN with boundary-layer inputs.
- [X] Keep loss weights comparable to payoff-aware PINN.
- [X] Save experiment as:
  - [X] `experiment_id: payoff_aware_boundary_layer_pinn`

### 5.6 Evaluation

- [X] Compare against:
  - [X] baseline PINN
  - [X] payoff-aware PINN

- [X] Focus on:
  - [X] Gamma hard-region RMSE
  - [X] Delta hard-region RMSE
  - [X] Vega hard-region RMSE
  - [X] p90/p99 Greek errors
  - [X] global price degradation, if any
  - [X] PDE residual maps

---

## 6. Boundary-Layer Coordinates Without Payoff Ansatz

### 6.1 Motivation

- [X] Isolate whether the degradation comes from:
  - [X] the payoff-aware ansatz \(g_\epsilon + a(\tau)N_\theta\)
  - [X] the boundary-layer coordinate transform itself

- [X] Keep the original PINN output form:

  \[
  V_\theta(\tau,m,v,\Theta)=N_\theta(\phi(\tau,m),v,\Theta)
  \]

- [X] Use the same PDE residual, boundary losses and collocation domain as the baseline.
- [X] Do not impose the payoff by construction in this variant.

### 6.2 Coordinate Transform

- [X] Implement model input transform only:

  \[
  x=\log(m)
  \]

  \[
  z=\frac{x}{\sqrt{\tau+\varepsilon}}
  \]

  \[
  s_\tau=\sqrt{\tau+\varepsilon}
  \]

  \[
  \ell_\tau=\log(\tau+\varepsilon)
  \]

- [ ] Network input candidates:
  - [X] \([z,x,s_\tau,\ell_\tau,v,\rho,\kappa,\gamma,\bar v,r]\)
  - [ ] \([z,s_\tau,\ell_\tau,v,\rho,\kappa,\gamma,\bar v,r]\)
  - [ ] original scaled inputs plus \([z,x,s_\tau,\ell_\tau]\)

- [X] Start with:
  - [X] \(\varepsilon=10^{-6}\)
  - [X] \(z\)-clip \(=50\)

### 6.3 Implementation Checks

- [X] Keep raw dataset feature order unchanged:
  - [X] \([\tau,m,v,\rho,\kappa,\gamma,\bar v,r]\)

- [X] Apply the transform inside the model forward pass.
- [X] Keep PDE residual derivatives with respect to raw financial variables.
- [X] Verify no Greek is computed with respect to \(z\).
- [X] Verify autodiff Greeks against finite differences of the model.
- [X] Verify checkpoint loading works with diagnostics scripts.
- [X] Verify the model can run with existing collocation manifests.

### 6.4 Training

- [X] Train on the same domain as baseline.
- [X] Keep architecture depth/width comparable to baseline.
- [X] Keep loss weights comparable to baseline.
- [X] Use no COS Greek labels during training.
- [X] Save experiment as:
  - [X] `experiment_id: boundary_layer_pinn_no_payoff`

### 6.5 Evaluation

- [X] Compare against:
  - [X] baseline PINN
  - [X] payoff-aware PINN
  - [X] payoff-aware + boundary-layer PINN

- [X] Focus on:
  - [X] hard-region Gamma RMSE
  - [X] hard-region Delta RMSE
  - [X] hard-region Vega RMSE
  - [X] p90/p99 Greek errors
  - [X] global price RMSE degradation
  - [X] PDE residual maps

- [ ] Accept this variant only if:
  - [ ] hard-region Gamma improves over baseline, or
  - [ ] hard-region Gamma does not degrade while price/PDE residual remains comparable

### 6.6 Payoff Ansatz Sanity Variant

- [ ] If boundary-layer without payoff is stable, test payoff-aware again with:

  \[
  a(\tau)=\sqrt{\tau+\varepsilon}-\sqrt{\varepsilon}
  \]

- [ ] Keep all other settings fixed.
- [ ] Save experiment as:
  - [ ] `experiment_id: payoff_aware_boundary_layer_sqrt_time`

- [ ] Reject if validation PDE residual remains orders of magnitude above baseline.

---

## 7. Payoff-Aware + Adaptive Collocation

### 7.1 Motivation

- [X] Improve local accuracy without using COS Greek labels.
- [X] Add more collocation points where the PDE residual is large.
- [X] Focus training effort near difficult regions.

### 7.2 Initial model

- [X] Start only from a stable model:
  - [ ] baseline PINN, or
  - [X] boundary-layer no-payoff PINN, if it beats or matches baseline

- [X] Train initial model with uniform or quasi-uniform collocation.
- [X] Save initial residual maps.

### 7.3 Residual-based sampling

- [X] Generate a large candidate pool of collocation points.
- [X] Evaluate PDE residual:

  \[
  |\mathcal{N}[V_\theta]|
  \]

- [ ] Optionally evaluate residual-gradient magnitude:

  \[
  |\nabla \mathcal{N}[V_\theta]|
  \]

- [X] Select additional collocation points from:
  - [X] highest PDE residual regions
  - [X] hard region \(\mathcal{D}_{hard}\)
  - [X] short-maturity region
  - [X] ATM region

### 7.4 Adaptive schedule

- [X] Define adaptive loop:
  - [X] train
  - [X] evaluate residual
  - [X] resample
  - [X] fine-tune
  - [ ] repeat

- [ ] Test number of adaptive rounds:
  - [ ] 1 round
  - [ ] 2 rounds
  - [ ] 3 rounds

- [ ] Test resampling ratio:
  - [ ] 20% adaptive / 80% uniform
  - [X] 40% adaptive / 60% uniform
  - [ ] 60% adaptive / 40% uniform

### 7.5 Training

- [X] Fine-tune model with adaptive collocation points.
- [X] Keep validation/evaluation grid fixed.
- [ ] Save experiments:
  - [X] `experiment_id: adaptive_collocation`

### 7.6 Evaluation

- [X] Compare residual maps before and after adaptive collocation.
- [X] Compare Greek error maps before and after adaptive collocation.
- [X] Check whether improvements are local or global.
- [X] Check whether adaptive collocation overfits the hard region.
- [X] Check whether global price accuracy deteriorates.

### 7.7 Decision after first run

- [X] First adaptive run:
  - [X] base model: `boundary_layer_pinn_no_payoff`
  - [X] candidate pool: 120000
  - [X] adaptive ratio: 40%
  - [X] selected adaptive points: 16000
  - [X] selected counts: hard 7273, short 4364, ATM 4363

- [X] Positive result:
  - [X] selected-point PDE residual RMSE improved from 106.06 to 1.18.
  - [X] hard-region Gamma RMSE improved versus boundary-layer no-payoff from 16.06 to 11.85.
  - [X] hard-region Gamma p99 improved versus boundary-layer no-payoff from 54.47 to 29.82.

- [X] Negative result:
  - [X] full price RMSE degraded from 0.0219 to 0.2387 versus boundary-layer no-payoff.
  - [X] hard price RMSE degraded from 0.0367 to 0.3185.
  - [X] hard Delta RMSE degraded from 0.312 to 0.407.
  - [X] original-domain mean absolute PDE residual degraded from 0.0029 to 0.0755.
  - [X] terminal MSE degraded from approximately \(2\times10^{-6}\) to 0.105.
  - [X] lower-boundary MSE degraded from approximately \(9\times10^{-7}\) to 0.032.

- [X] Decision:
  - [X] Reject current adaptive-collocation run as final candidate.
  - [X] Keep implementation as diagnostic/prototype.
  - [X] Next adaptive variant should use lower adaptive ratio, residual clipping/winsorization, a real uniform quota, stronger terminal/lower weights, lower learning rate, and early stopping.

---

## 8. Payoff-Aware + Derivative-Consistency Residual

### 8.1 Motivation

- [X] Use this as an advanced experiment after adaptive collocation.
- [X] Improve Greek consistency without requiring reference Greek labels.
- [X] Penalize derivatives of the PDE residual.

### 8.2 PDE residual

- [X] Define the Heston PDE operator:

  \[
  \mathcal{N}[V_\theta] = 0
  \]

- [X] Verify implementation:
  - [X] signs
  - [X] time convention
  - [X] discounting term
  - [X] variance derivatives
  - [X] mixed derivative term

### 8.3 Differential residuals

- [X] Implement:

  \[
  \partial_x \mathcal{N}[V_\theta]
  \]

  \[
  \partial_v \mathcal{N}[V_\theta]
  \]

- [X] Avoid \(\partial_{xx}\mathcal{N}[V_\theta]\) initially because it may require very high-order derivatives.
- [X] Use chain rule for \(x=\log(m)\):

  \[
  \partial_x \mathcal{N} = m\,\partial_m \mathcal{N}
  \]

### 8.4 Loss function

- [X] Add derivative-consistency loss:

  \[
  \mathcal{L}_{dPDE}
  =
  \|\partial_x \mathcal{N}[V_\theta]\|^2
  +
  \|\partial_v \mathcal{N}[V_\theta]\|^2
  \]

- [X] Total loss:

  \[
  \mathcal{L}
  =
  \lambda_{PDE}\mathcal{L}_{PDE}
  +
  \lambda_{dPDE}\mathcal{L}_{dPDE}
  +
  \lambda_T\mathcal{L}_T
  +
  \lambda_B\mathcal{L}_B
  \]

- [ ] Test \(\lambda_{dPDE}\):
  - [X] \(10^{-3}\)
  - [ ] \(10^{-2}\)
  - [ ] \(10^{-1}\)

### 8.5 Training

- [X] Train only from a stable base model:
  - [ ] boundary-layer no-payoff PINN, if stable
  - [X] baseline PINN, otherwise
- [ ] Optionally combine with adaptive collocation.
- [ ] Monitor:
  - [ ] training instability
  - [ ] exploding higher-order derivatives
  - [ ] training time
  - [ ] memory usage
  - [ ] degradation in price RMSE

- [ ] Save experiment as:
  - [X] `experiment_id: derivative_consistency`

### 8.6 Evaluation

- [X] Check whether derivative-consistency improves:
  - [X] Greek smoothness
  - [X] hard-region Delta error
  - [X] hard-region Gamma error
  - [X] hard-region Vega error
  - [X] PDE residual stability

- [X] Check whether price accuracy deteriorates.
- [X] Compare training time and inference time.

### 8.7 Decision after first run

- [X] First derivative-consistency run:
  - [X] base model: `PINN_mix_scaled_param`
  - [X] \(\lambda_{dPDE}=10^{-3}\)
  - [X] epochs: 500
  - [X] optimizer: Adam fine-tuning

- [X] Positive result:
  - [X] Training validation total loss improved from \(1.60\times10^{-5}\) to \(1.36\times10^{-5}\).
  - [X] Training validation PDE term improved from \(1.33\times10^{-5}\) to \(5.86\times10^{-6}\).
  - [X] Full price RMSE improved slightly from 0.00427 to 0.00396.
  - [X] Training remained numerically stable.

- [X] Negative result:
  - [X] Hard Delta RMSE worsened from 0.0873 to 0.0995.
  - [X] Hard Gamma RMSE worsened from 4.99 to 5.58.
  - [X] Hard Gamma p99 worsened from 17.64 to 18.55.
  - [X] Hard Vega RMSE worsened from 0.0835 to 0.0917.
  - [X] Hard-region diagnostic PDE residual worsened from 0.0505 to 0.0852.

- [X] Decision:
  - [X] Reject current derivative-consistency run as final candidate.
  - [X] Keep implementation because it is stable and useful for ablations.

- [ ] Possible future variants:
  - [ ] Try \(\lambda_{dPDE}=10^{-4}\) before testing larger weights.
  - [ ] Avoid spending much time on \(10^{-2}\) and \(10^{-1}\) unless used as quick stress tests.
  - [ ] Apply derivative-consistency only after a warm-up or only for a short fine-tuning window.
  - [ ] Localize derivative-consistency to short-maturity/ATM points instead of the full domain.
  - [ ] Normalize or clip derivative residual terms to avoid over-regularizing Gamma.
  - [ ] Track derivative residual maps separately from Greek benchmark maps.
  - [ ] Combine with a gentler adaptive-collocation variant only after terminal/lower losses are protected.

## 9. Evaluation Protocol

### 9.1 Global metrics

- [X] Compute:
  - [X] price RMSE
  - [X] price MAE
  - [X] price \(R^2\)
  - [X] Delta RMSE
  - [X] Gamma RMSE
  - [X] Vega RMSE
  - [X] Theta RMSE, if available
  - [X] p90 absolute Greek error
  - [X] p99 absolute Greek error

### 9.2 Hard-region metrics

- [X] Compute on \(\mathcal{D}_{hard}\):
  - [X] price RMSE
  - [X] Delta RMSE
  - [X] Gamma RMSE
  - [X] Vega RMSE
  - [X] Theta RMSE, if available
  - [X] p90 Greek error
  - [X] p99 Greek error

### 9.3 Stabilized relative Greek error

- [X] Compute stabilized relative error:

  \[
  e_G^{rel}
  =
  \frac{|G_\theta-G_{ref}|}{1+|G_{ref}|}
  \]

- [X] Compute this for:
  - [X] Delta
  - [X] Gamma
  - [X] Vega
  - [X] Theta, if available

### 9.4 No-reference diagnostics

- [X] Add diagnostics that do not require COS Greeks:
  - [X] PDE residual maps
  - [X] derivative-residual maps, if implemented
  - [X] finite-difference consistency of model Greeks
  - [X] monotonicity checks
  - [X] convexity checks
  - [X] boundary-condition violations
  - [X] smoothness of Greeks over \((m,\tau)\)

- [X] Use COS only as external benchmark for vanilla Heston.
- [X] Document which diagnostics would remain available for exotic derivatives.

---

## 10. Ablation Study

### 10.1 Models to compare

- [X] Compare the following models:
  - [X] Baseline PINN
  - [X] Payoff-aware PINN
  - [X] Payoff-aware + boundary-layer coordinates
  - [X] Boundary-layer coordinates without payoff ansatz
  - [X] Adaptive collocation, only if the base model is stable
  - [X] Derivative-consistency residual, only if the base model is stable
  - [X] ACV zero-residual control variate with best reliable-floor gate

### 10.2 Comparison table

- [X] Build table with:

| Model | Price RMSE | Delta RMSE | Gamma RMSE | Hard Delta RMSE | Hard Gamma RMSE | p99 Gamma Error | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline PINN | 0.004269 | 0.033808 | 0.534829 | 0.087287 | 4.992100 | 17.642700 | 1151.0 |
| Payoff-aware PINN | 0.027946 | 0.264658 | 18.686300 | 0.294878 | 50.654100 | 239.591000 | 2124.7 |
| Payoff-aware + boundary-layer | 0.015703 | 0.151225 | 10.016900 | 0.164745 | 37.681900 | 187.877000 | 2265.5 |
| Boundary-layer no-payoff | 0.021929 | 0.247039 | 1.830180 | 0.312323 | 16.056400 | 54.473100 | 2802.0 |
| Adaptive collocation | 0.238723 | 0.307172 | 1.787600 | 0.406953 | 11.853400 | 29.816700 | 320.3 |
| Derivative-consistency | 0.003958 | 0.035797 | 0.579619 | 0.099513 | 5.580890 | 18.550700 | 442.6 |
| ACV control variate | 0.004235 | 0.032691 | 0.404042 | 0.012883 | 0.644620 | 1.612340 | 0.0 |

### 10.3 Plots

- [X] Generate comparison heatmaps for:
  - [X] price error
  - [X] Delta error
  - [X] Gamma error
  - [X] Vega error
  - [X] PDE residual

- [X] Generate hard-region zoom plots.
- [X] Generate error distribution plots:
  - [X] global
  - [X] hard region
  - [X] short maturity
  - [X] ATM region

### 10.4 Ablation Decision

- [X] Generated artifacts:
  - [X] `outputs/pinn/ablation_study/tables/ablation_comparison.csv`
  - [X] `outputs/pinn/ablation_study/tables/ablation_comparison.md`
  - [X] `outputs/pinn/ablation_study/figures/`
  - [X] `outputs/pinn/ablation_study/ablation_summary.yaml`

- [X] Main conclusion:
  - [X] ACV zero-residual control variate is now best on full Delta RMSE, full Gamma RMSE, hard Delta RMSE, hard Gamma RMSE, and hard p99 Gamma error.
  - [X] Derivative-consistency keeps the best full price RMSE, but worsens hard-region Greeks.
  - [X] Adaptive collocation reduces some hard Gamma metrics versus boundary-layer variants, but price degradation is too large.
  - [X] Current final candidate should be `acv_hard_patch_control_variate_best_gate_tau_floor_5e4`.

---

## 10b. Experimental Multi-Output Greek-Consistency PINN

- [X] Create dedicated TODO list:
  - [X] `docs/multi_output_greek_consistency_TODOs.md`
- [X] Implement experimental multi-output architecture:
  - [X] price head
  - [X] Delta head
  - [X] Gamma head
  - [X] Vega head
- [X] Keep model `forward()` price-only for backward compatibility.
- [X] Add `forward_all()` for head-consistency losses.
- [X] Add no-label consistency losses:
  - [X] Delta head vs \(\partial_m V_\theta\)
  - [X] Gamma head vs \(\partial_m \Delta_\theta\)
  - [X] Vega head vs \(\partial_v V_\theta\)
- [X] Add flexible warm-start from scalar baseline checkpoint.
- [X] Add experiment configs:
  - [X] `configs/pinn_model_architecture_multi_output_greek_consistency.yaml`
  - [X] `configs/pinn_training_multi_output_greek_consistency.yaml`
  - [X] `configs/pinn_pipeline_multi_output_greek_consistency.yaml`
  - [X] `configs/pinn_baseline_diagnostics_multi_output_greek_consistency.yaml`
- [ ] Train and evaluate `experiment_id: multi_output_greek_consistency`.
- [ ] Compare autodiff Greeks and head Greeks.

---

## 10c. Experimental ACV-HardPatch

- [X] Add dedicated TODO list:
  - [X] `docs/acv_hard_patch_TODOs.md`
- [X] Implement separate experimental model path:
  - [X] frozen baseline \(V_B\)
  - [X] local PUT Black-Scholes control variate
  - [X] residual patch \(P^{locBS}+\tau R_\theta\)
  - [X] smooth gate \(\chi(x,\tau)\)
- [X] Implement residual features:
  - [X] \(x=\log(m)\)
  - [X] \(q=\sqrt{v\tau+\epsilon_q}\)
  - [X] \(z=x/q\)
  - [X] Fourier features only on \(z\)
- [X] Implement normalized Heston PDE residual in \((\tau,x,v)\).
- [X] Implement losses:
  - [X] baseline distillation
  - [X] terminal payoff
  - [X] interface value/\(x\)-derivative/\(v\)-derivative matching
  - [X] global replay
  - [X] optional hard price labels
  - [X] optional price-stencil loss
- [X] Add scripts/configs:
  - [X] `src/pinn/acv_hard_patch.py`
  - [X] `scripts/run_acv_hard_patch.py`
  - [X] `configs/acv_hard_patch_experimental.yaml`
  - [X] `scripts/run_hard_gamma_reference_audit.py`
  - [X] `configs/hard_gamma_reference_audit.yaml`
  - [X] `scripts/run_acv_hard_patch_diagnostics.py`
  - [X] `configs/acv_hard_patch_diagnostics.yaml`
- [ ] Run Stage 0 reference audit.
- [X] Run Stage 0 reference audit.
- [X] Train Stage 1 + Stage 2.
- [X] Run ACV diagnostics.
- [X] Stage 2 result:
  - [X] full price RMSE: `0.004263`
  - [X] hard price RMSE: `0.001609`
  - [X] hard Delta RMSE: `0.03133`
  - [X] hard Gamma RMSE: `15.8483`
  - [X] hard Gamma p99: `45.2636`
  - [X] hard PDE residual RMSE: `0.3300`
- [X] Stage 2 decision:
  - [X] Reject no-label ACV as final candidate.
  - [X] Continue with price-stencil curvature supervision.
- [ ] Compare against baseline acceptance criteria:
  - [ ] hard Gamma RMSE < 3.5
  - [ ] hard p99 Gamma error < 12
  - [ ] full price RMSE <= 0.0047
  - [ ] hard Delta RMSE <= 0.10
  - [ ] hard PDE residual RMSE <= 0.075
- [X] Decide whether to enable Stage 3 price labels.
- [X] Decide whether to enable Stage 4 price-stencil loss.
- [X] Add stencil continuation config:
  - [X] `configs/acv_hard_patch_stencil_experimental.yaml`
  - [X] `configs/acv_hard_patch_stencil_diagnostics.yaml`
- [X] Train and evaluate stencil continuation.
- [X] Stencil continuation result:
  - [X] hard price RMSE: `0.001420`
  - [X] hard Delta RMSE: `0.03292`
  - [X] hard Gamma RMSE: `31.9840`
  - [X] hard Gamma p99: `107.555`
  - [X] hard PDE residual RMSE: `0.6098`
- [X] Stencil continuation decision:
  - [X] Reject as final candidate.
  - [X] Price-stencil curvature supervision created larger Gamma spikes.
- [X] Test zero-residual control variate:
  - [X] `configs/acv_hard_patch_control_variate.yaml`
  - [X] `configs/acv_hard_patch_control_variate_diagnostics.yaml`
- [X] Zero-residual control variate result:
  - [X] full price RMSE: `0.004243`
  - [X] full Gamma RMSE: `0.4146`
  - [X] hard price RMSE: `0.000438`
  - [X] hard Delta RMSE: `0.01339`
  - [X] hard Gamma RMSE: `0.7356`
  - [X] hard Gamma p99: `1.6943`
  - [X] hard PDE residual RMSE: `0.02440`
- [X] Current best candidate:
  - [X] `acv_hard_patch_control_variate`
- [ ] Run gate sensitivity for the accepted zero-residual control variate.
- [X] Add zero-residual ACV to ablation study table.
- [X] Implement extreme-short diagnostic tooling:
  - [X] `scripts/run_acv_extreme_short_diagnostics.py`
  - [X] `configs/acv_extreme_short_diagnostics.yaml`
- [X] Implement gate-sensitivity tooling:
  - [X] `scripts/run_acv_gate_sensitivity.py`
  - [X] `configs/acv_gate_sensitivity.yaml`
- [X] Run full extreme-short diagnostic and review tau buckets.
- [X] Run full gate sweep.
- [X] Extreme-short result:
  - [X] ACV improves hard Gamma RMSE vs baseline over \(\tau\in[10^{-4},0.05]\): `30.2445 -> 17.2595`
  - [X] ACV improves hard Delta RMSE: `0.2221 -> 0.0300`
  - [X] ACV improves hard price RMSE: `0.005468 -> 0.000191`
  - [X] ACV improves hard PDE residual RMSE: `0.04736 -> 0.01836`
- [X] Tau-bucket result:
  - [X] ACV is strong for \(\tau\ge5\cdot10^{-4}\).
  - [X] Bucket \([10^{-4},5\cdot10^{-4})\) dominates the remaining Gamma error.
  - [X] CF analytic Gamma appears unstable there, including negative put Gamma values.
- [X] Gate sweep result:
  - [X] No gate passes strict criteria when \(\tau=10^{-4}\) is included.
  - [X] Best tested gate by score: `x_center=0.08`, `tau_center=0.10`, `delta_x=0.01`, `delta_tau=0.015`.
- [X] Add reliable-floor gate sweep config:
  - [X] `configs/acv_gate_sensitivity_tau_floor_5e4.yaml`
- [X] Run reliable-floor gate sweep with \(\tau_{\min}=5\cdot10^{-4}\).
- [X] Reliable-floor sweep result:
  - [X] local hard Gamma RMSE improves versus baseline: `14.3531 -> 0.8072`
  - [X] local hard Delta RMSE improves versus baseline: `0.1824 -> 0.00731`
  - [X] local hard price RMSE improves versus baseline: `0.005290 -> 0.000137`
  - [X] best gate: `x_center=0.08`, `tau_center=0.10`, `delta_x=0.01`, `delta_tau=0.015`
- [X] Add best reliable-floor gate configs:
  - [X] `configs/acv_hard_patch_control_variate_best_gate_tau_floor_5e4.yaml`
  - [X] `configs/acv_hard_patch_best_gate_tau_floor_5e4_diagnostics.yaml`
- [X] Run standard diagnostics for best reliable-floor gate.
- [X] Best reliable-floor gate standard result:
  - [X] full price RMSE: `0.004235`
  - [X] full Gamma RMSE: `0.4040`
  - [X] hard price RMSE: `0.000270`
  - [X] hard Delta RMSE: `0.01288`
  - [X] hard Gamma RMSE: `0.6446`
  - [X] hard Gamma p99: `1.6123`
  - [X] hard PDE residual RMSE: `0.01893`
- [ ] Validate \(\tau<5\cdot10^{-4}\) Gamma reference before making final claims in that bucket.
- [X] Select final gate:
  - [X] `acv_hard_patch_control_variate_best_gate_tau_floor_5e4`
- [X] Rebuild ablation study outputs:
  - [X] `outputs/pinn/ablation_study/tables/ablation_comparison.csv`
  - [X] `outputs/pinn/ablation_study/tables/ablation_comparison.md`
  - [X] `outputs/pinn/ablation_study/ablation_summary.yaml`
- [X] Updated ablation conclusion:
  - [X] ACV control variate is best on full Delta RMSE, full Gamma RMSE, hard Delta RMSE, hard Gamma RMSE, and hard p99 Gamma error.

---

## 11. Thesis Writing Tasks

### 11.1 New section

- [ ] Add section:

  Localized Greek Error Near the Short-Maturity ATM Region

- [ ] Explain:
  - [ ] payoff kink
  - [ ] Gamma concentration near ATM
  - [ ] short-maturity boundary layer
  - [ ] why price accuracy does not guarantee Greek accuracy

### 11.2 Methodology sections

- [ ] Add subsection:

  Payoff-Aware PINN Formulation

- [ ] Add subsection:

  Boundary-Layer Coordinates

- [ ] Add subsection:

  Boundary-Layer Coordinates Without Payoff Ansatz

- [ ] Add subsection:

  Adaptive Collocation Strategy

- [ ] Add subsection:

  Derivative-Consistency Residuals

### 11.3 Scalability argument

- [ ] Clearly state:
  - [ ] COS is used only for evaluation in vanilla Heston.
  - [ ] COS Greek labels are not used for training.
  - [ ] The proposed training methodology relies on payoff, PDE structure and residuals.
  - [ ] This makes the approach more transferable to complex derivatives.

---

## 12. Final Decision Criteria

- [ ] Keep a method only if it improves hard-region Greek accuracy without unacceptable global degradation.
- [ ] Prefer methods that do not require reference Greek labels.
- [ ] Prefer methods compatible with more complex derivatives.
- [ ] Prefer lower complexity if performance is similar.
- [ ] Reject methods that improve global price metrics but worsen hard-region Greeks.

---

## 13. Final Priority Ranking

- [ ] 1. Baseline PINN diagnostics
- [ ] 2. Payoff-aware PINN
- [ ] 3. Payoff-aware + boundary-layer coordinates
- [ ] 4. Boundary-layer coordinates without payoff ansatz
- [ ] 5. Adaptive collocation, only after a stable base model
- [ ] 6. Derivative-consistency residual, only after a stable base model
- [ ] 7. Optional: weak-form PINN
- [ ] 8. Optional: neural operator / DeepONet / Transformer-style model

---

## 14. Final Candidate Model

- [ ] Select final model among:
  - [ ] Baseline PINN
  - [ ] Payoff-aware PINN
  - [ ] Payoff-aware + boundary-layer coordinates
  - [ ] Boundary-layer coordinates without payoff ansatz
  - [ ] Adaptive collocation
  - [ ] Derivative-consistency residual

- [ ] Justify final choice based on:
  - [ ] hard-region Greek accuracy
  - [ ] global price accuracy
  - [ ] PDE consistency
  - [ ] scalability to exotic derivatives
  - [ ] implementation complexity
  - [ ] training/inference cost
