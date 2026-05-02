# TODO — ACV-HardPatch Experimental Greek Improvement

## 0. Scope

- [X] Treat this as a separate experimental path, not as another global PINN regularizer.
- [X] Keep the global baseline frozen.
- [X] Target the hard region:
  - [X] \(|\log(m)| < 0.03\)
  - [X] \(\tau < 0.05\)
- [X] Use COS/Heston CF only for vanilla-reference evaluation or optional price labels.
- [X] Do not use COS Greek labels as the core method.

## 1. Method

- [X] Implement ACV-HardPatch:

  \[
  V_{\theta}
  =
  (1-\chi)V_B
  +
  \chi\left(P^{locBS}+\tau R_{\theta}\right)
  \]

- [X] Use the PUT local Black-Scholes control variate, matching the current dataset payoff:

  \[
  P^{locBS}
  =
  e^{-r\tau}\Phi(-d_2)-m\Phi(-d_1)
  \]

  with:

  \[
  d_1=
  \frac{\log(m)+(r+\frac12v)\tau}{\sqrt{v\tau+\epsilon}},
  \qquad
  d_2=d_1-\sqrt{v\tau+\epsilon}.
  \]

- [X] Use local variables:
  - [X] \(x=\log(m)\)
  - [X] \(q=\sqrt{v\tau+\epsilon_q}\)
  - [X] \(z=x/q\)
- [X] Use Fourier features only on \(z\).
- [X] Initialize the final residual layer at zero so the patch starts from the control variate.
- [X] Mark the implementation as experimental in code and config.

## 2. Code Artifacts

- [X] Add model/loss module:
  - [X] `src/pinn/acv_hard_patch.py`
- [X] Add training script:
  - [X] `scripts/run_acv_hard_patch.py`
- [X] Add config:
  - [X] `configs/acv_hard_patch_experimental.yaml`
- [X] Add reference Gamma audit:
  - [X] `scripts/run_hard_gamma_reference_audit.py`
  - [X] `configs/hard_gamma_reference_audit.yaml`
- [X] Add ACV diagnostics:
  - [X] `scripts/run_acv_hard_patch_diagnostics.py`
  - [X] `configs/acv_hard_patch_diagnostics.yaml`

## 3. Training Stages

- [X] Stage 1: distill local patch to frozen baseline.
- [X] Stage 2: train with normalized local PDE, terminal, interface, and global replay losses.
- [X] Stage 3: optional hard price labels.
- [X] Stage 4: optional price-stencil loss.

Initial execution plan:

```bash
.venv/bin/python scripts/run_hard_gamma_reference_audit.py --config configs/hard_gamma_reference_audit.yaml
.venv/bin/python scripts/run_acv_hard_patch.py --config configs/acv_hard_patch_experimental.yaml
.venv/bin/python scripts/run_acv_hard_patch_diagnostics.py --config configs/acv_hard_patch_diagnostics.yaml
```

Continuation after Stage 2 failed Gamma:

```bash
.venv/bin/python scripts/run_acv_hard_patch.py --config configs/acv_hard_patch_stencil_experimental.yaml
.venv/bin/python scripts/run_acv_hard_patch_diagnostics.py --config configs/acv_hard_patch_stencil_diagnostics.yaml
```

## 3.1 Stage 2 Result

- [X] Reference Gamma audit:
  - [X] stable enough at \(h_x=5\times10^{-4}\)
  - [X] RMSE vs CF analytic Gamma: `0.01066`
  - [X] p99 abs error: `0.03346`
- [X] Stage 1 + Stage 2 completed:
  - [X] `n_steps: 4000`
  - [X] best stage: `stage2_pde_terminal_interface`
- [X] Stage 2 diagnostics:
  - [X] full price RMSE: `0.004263`
  - [X] hard price RMSE: `0.001609`
  - [X] hard Delta RMSE: `0.03133`
  - [X] hard Gamma RMSE: `15.8483`
  - [X] hard Gamma p99: `45.2636`
  - [X] hard PDE residual RMSE: `0.3300`
- [X] Decision:
  - [X] Reject Stage 2 no-label ACV as final candidate.
  - [X] Keep the result as evidence that price/Delta can improve while Gamma deteriorates.
  - [X] Continue directly with price-stencil curvature supervision.

## 3.2 Stencil Continuation

- [X] Add warm-start support in `scripts/run_acv_hard_patch.py`.
- [X] Add stencil continuation config:
  - [X] `configs/acv_hard_patch_stencil_experimental.yaml`
- [X] Add stencil diagnostics config:
  - [X] `configs/acv_hard_patch_stencil_diagnostics.yaml`
- [X] Train `acv_hard_patch_stencil_experimental`.
- [X] Evaluate stencil continuation.
- [X] Compare against acceptance criteria.
- [X] Stencil continuation diagnostics:
  - [X] full price RMSE: `0.004262`
  - [X] hard price RMSE: `0.001420`
  - [X] hard Delta RMSE: `0.03292`
  - [X] hard Gamma RMSE: `31.9840`
  - [X] hard Gamma p99: `107.555`
  - [X] hard PDE residual RMSE: `0.6098`
- [X] Decision:
  - [X] Reject stencil continuation as final candidate.
  - [X] It improves hard price but creates larger Gamma spikes than Stage 2.
  - [X] Treat this as evidence that naive finite-difference curvature supervision can destabilize the patch.

## 3.3 Control Variate Only

- [X] Test zero-residual ACV:
  - [X] \(R_\theta=0\)
  - [X] frozen baseline \(V_B\)
  - [X] local PUT Black-Scholes control variate
  - [X] same smooth gate \(\chi(x,\tau)\)
- [X] Add reproducible config:
  - [X] `configs/acv_hard_patch_control_variate.yaml`
  - [X] `configs/acv_hard_patch_control_variate_diagnostics.yaml`
- [X] Save checkpoint-only artifact:
  - [X] `outputs/pinn/acv_hard_patch_control_variate/checkpoints/model_best.pt`
- [X] Diagnostics:
  - [X] full price RMSE: `0.004243`
  - [X] full Gamma RMSE: `0.4146`
  - [X] hard price RMSE: `0.000438`
  - [X] hard Delta RMSE: `0.01339`
  - [X] hard Gamma RMSE: `0.7356`
  - [X] hard Gamma p99: `1.6943`
  - [X] hard PDE residual RMSE: `0.02440`
- [X] Decision:
  - [X] Promote control-variate-only ACV to current best candidate.
  - [X] Do not train the residual unless a much more constrained residual objective is designed.
  - [X] Interpret failed Stage 2/Stage 4 as residual learning destroying the analytically correct singular structure.

## 4. Acceptance Criteria

- [X] Hard Gamma RMSE below `3.5`.
- [X] Prefer hard Gamma RMSE below `3.0`.
- [X] Hard p99 Gamma error below `12`.
- [X] Full price RMSE no worse than `0.0047`.
- [X] Hard Delta RMSE no worse than `0.10`.
- [X] Hard PDE residual RMSE no worse than `0.075`.

Accepted candidate:

- [X] `acv_hard_patch_control_variate`

## 5. Future Switches

- [X] If Stage 2 improves PDE but not Gamma, enable `price_labels.enabled=true` and Stage 3.
- [X] If Stage 3 improves price/Delta but not Gamma, enable `stencil_labels.enabled=true` and Stage 4.
- [X] If Stage 4 only works with curvature-mode stencil, report it as price-stencil Sobolev proxy, not as label-free PINN.
- [ ] If global price degrades, increase interface/global replay weights or narrow the gate.
- [ ] If PDE residual explodes near the gate, try `pde.target: "patch"` inside the hard core as an ablation.
- [ ] Run gate sensitivity around the accepted zero-residual control variate:
  - [ ] \(x_c \in \{0.04, 0.05, 0.06, 0.08\}\)
  - [ ] \(\tau_c \in \{0.05, 0.08, 0.10\}\)
  - [ ] \(\delta_x,\delta_\tau\) sensitivity
- [X] Add extreme-short diagnostic tooling:
  - [X] `scripts/run_acv_extreme_short_diagnostics.py`
  - [X] `configs/acv_extreme_short_diagnostics.yaml`
- [X] Add gate-sensitivity tooling:
  - [X] `scripts/run_acv_gate_sensitivity.py`
  - [X] `configs/acv_gate_sensitivity.yaml`
- [ ] Run full extreme-short diagnostic:

  ```bash
  .venv/bin/python scripts/run_acv_extreme_short_diagnostics.py --config configs/acv_extreme_short_diagnostics.yaml
  ```

- [ ] Run full gate sensitivity:

  ```bash
  .venv/bin/python scripts/run_acv_gate_sensitivity.py --config configs/acv_gate_sensitivity.yaml
  ```

- [X] Run full extreme-short diagnostic.
- [X] Run full gate sensitivity.
- [X] Extreme-short diagnostic result:
  - [X] Baseline hard Gamma RMSE over \(\tau\in[10^{-4},5\cdot10^{-2}]\): `30.2445`
  - [X] ACV hard Gamma RMSE over \(\tau\in[10^{-4},5\cdot10^{-2}]\): `17.2595`
  - [X] Baseline hard Delta RMSE: `0.2221`
  - [X] ACV hard Delta RMSE: `0.0300`
  - [X] Baseline hard price RMSE: `0.005468`
  - [X] ACV hard price RMSE: `0.000191`
  - [X] Baseline hard PDE RMSE: `0.04736`
  - [X] ACV hard PDE RMSE: `0.01836`
- [X] Extreme-short tau buckets for ACV hard region:
  - [X] \([10^{-4},5\cdot10^{-4})\): Gamma RMSE `33.4519`, p99 `85.1577`
  - [X] \([5\cdot10^{-4},10^{-3})\): Gamma RMSE `1.4369`, p99 `4.2165`
  - [X] \([10^{-3},5\cdot10^{-3})\): Gamma RMSE `0.4319`, p99 `0.8430`
  - [X] \([5\cdot10^{-3},10^{-2})\): Gamma RMSE `0.5559`, p99 `0.8596`
  - [X] \([10^{-2},5\cdot10^{-2})\): Gamma RMSE `0.6539`, p99 `1.4592`
- [X] Gate sensitivity result over \(\tau\in[10^{-4},5\cdot10^{-2}]\):
  - [X] No gate satisfies the strict acceptance criteria because the \([10^{-4},5\cdot10^{-4})\) bucket dominates Gamma RMSE/p99.
  - [X] Best score among tested gates:
    - [X] `x_center=0.08`
    - [X] `tau_center=0.10`
    - [X] `delta_x=0.01`
    - [X] `delta_tau=0.015`
- [X] Inspect `tau ~= 1e-4` metrics carefully:
  - [X] CF analytic Gamma shows negative values for vanilla puts at \(\tau=10^{-4}\) away from ATM, which is inconsistent with convexity and indicates reference instability.
  - [X] Treat \(\tau < 5\cdot10^{-4}\) as not yet reliable for final Gamma claims.
- [ ] Validate \(\tau < 5\cdot10^{-4}\) against finite differences of CF prices or a wider/quadrature-stabilized reference before using it in final acceptance.
- [X] Add reliable-floor gate sweep config:
  - [X] `configs/acv_gate_sensitivity_tau_floor_5e4.yaml`
- [X] Run a gate sweep with reliable floor \(\tau_{\min}=5\cdot10^{-4}\):

  ```bash
  .venv/bin/python scripts/run_acv_gate_sensitivity.py --config configs/acv_gate_sensitivity_tau_floor_5e4.yaml
  ```

- [X] Reliable-floor gate sweep result:
  - [X] Baseline local hard Gamma RMSE: `14.3531`
  - [X] Baseline local hard Delta RMSE: `0.1824`
  - [X] Baseline local hard price RMSE: `0.005290`
  - [X] Best gate by score:
    - [X] `x_center=0.08`
    - [X] `tau_center=0.10`
    - [X] `delta_x=0.01`
    - [X] `delta_tau=0.015`
  - [X] Best gate local hard price RMSE: `0.000137`
  - [X] Best gate local hard Delta RMSE: `0.007306`
  - [X] Best gate local hard Gamma RMSE: `0.8072`
  - [X] Best gate local hard Gamma p99: `3.5855`
  - [X] Best gate local hard PDE RMSE: `0.01722`
- [X] Decision:
  - [X] The `accepted=false` flag is caused by strict criteria inherited from the global diagnostic, especially p99/full-Gamma thresholds that are not appropriate for this local extreme-short grid.
  - [X] Treat `x_center=0.08`, `tau_center=0.10` as the performance winner for \(\tau\ge5\cdot10^{-4}\).
  - [X] Compare it against the conservative `x_center=0.06`, `tau_center=0.08` gate on the standard full-surface diagnostic before finalizing the thesis candidate.
- [X] Add reproducible best-gate configs:
  - [X] `configs/acv_hard_patch_control_variate_best_gate_tau_floor_5e4.yaml`
  - [X] `configs/acv_hard_patch_best_gate_tau_floor_5e4_diagnostics.yaml`
- [X] Save checkpoint for the best reliable-floor gate:

  ```bash
  .venv/bin/python scripts/run_acv_hard_patch.py --config configs/acv_hard_patch_control_variate_best_gate_tau_floor_5e4.yaml
  ```

- [X] Run standard full-surface diagnostics for the best reliable-floor gate.
- [X] Best reliable-floor gate standard diagnostics:
  - [X] full price RMSE: `0.004235`
  - [X] full Gamma RMSE: `0.4040`
  - [X] hard price RMSE: `0.000270`
  - [X] hard Delta RMSE: `0.01288`
  - [X] hard Gamma RMSE: `0.6446`
  - [X] hard Gamma p99: `1.6123`
  - [X] hard PDE residual RMSE: `0.01893`
- [X] Final gate decision:
  - [X] Promote `acv_hard_patch_control_variate_best_gate_tau_floor_5e4` over the initial control-variate gate.
  - [X] It improves the standard hard Gamma RMSE from `0.7356` to `0.6446` and the hard PDE residual RMSE from `0.02440` to `0.01893`.
- [X] Add `acv_hard_patch_control_variate_best_gate_tau_floor_5e4` to the ablation comparison table.
- [X] Ablation table result:
  - [X] ACV is now best by full Delta RMSE.
  - [X] ACV is now best by full Gamma RMSE.
  - [X] ACV is now best by hard Delta RMSE.
  - [X] ACV is now best by hard Gamma RMSE.
  - [X] ACV is now best by hard p99 Gamma error.
- [ ] Write thesis interpretation:
  - [ ] singularity subtraction works
  - [ ] residual training can destroy the hard-region Greek structure
  - [ ] no Greek labels were needed for the accepted candidate
