# PINN Pricing Ablation Results Log

## 2026-06-15

### Current reliable pricing reference

`abl13_log_core_acv_pricing` is the current best pricing-oriented variant.

- Full price RMSE: `0.055836`
- Full price MAE: `0.048416`
- Non-hard price RMSE: `0.055952`
- Non-hard price MAE: `0.048609`
- Hard price RMSE: `0.002094`
- Hard price MAE: `0.001793`
- Short-maturity price RMSE: `0.001192`
- ATM price RMSE: `0.058670`
- Negative price violations: `0` in full, non-hard, hard, short-maturity and ATM regions.

### Disabled positivity ablation

`abl12_log_core_bounded_price` is disabled. It enforces price bounds and removed negative price violations, but the pricing error degraded too much for the current objective.

### New planned extra ablation

`abl14_acv_log_learned_weights` adds learned log-variance loss weights on top of `abl13`.

Design choice: learn `pde` and `low` only. The ACV model already hard-anchors the terminal payoff through its architecture, so learning a terminal log-weight would create a near-zero-loss degeneracy rather than a meaningful balance between constraints.

### Full result: `abl14_acv_log_learned_weights`

The full run completed with 4000 epochs and `total_training_seconds = 5964.663`.

Learned loss weights saturated at the configured clamp:

- `loss_weight_pde = 403.428793`
- `loss_weight_low = 403.428793`
- `loss_log_var_pde` and `loss_log_var_low` were clamped at `-6.0`.

Pricing metrics:

- Full price RMSE: `0.043731`
- Full price MAE: `0.036829`
- Non-hard price RMSE: `0.043821`
- Non-hard price MAE: `0.036976`
- Hard price RMSE: `0.001379`
- Hard price MAE: `0.001178`
- Short-maturity price RMSE: `0.000861`
- ATM price RMSE: `0.046357`

Compared with `abl13_log_core_acv_pricing`, the pricing RMSE improves in all tracked regions:

- Full: `0.055836 -> 0.043731`
- Non-hard: `0.055952 -> 0.043821`
- Hard: `0.002094 -> 0.001379`
- Short-maturity: `0.001192 -> 0.000861`
- ATM: `0.058670 -> 0.046357`

### Correction: comparison against canonical baseline

The suite-local `abl00_baseline` diagnostics currently contain a smoke-sized grid (`961` full points, `5` hard points), while `abl13` and `abl14` use the full grid (`32761` full points, `135` hard points). Therefore the `ablation_scores.csv` comparison against that suite-local baseline is not a valid pricing comparison.

The honest pricing reference is the canonical `PINN_mix_scaled_param` diagnostics:

- Full price RMSE: `0.004269`
- Full price MAE: `0.003745`
- Hard price RMSE: `0.004780`
- Hard price MAE: `0.004712`
- Short-maturity price RMSE: `0.002251`
- ATM price RMSE: `0.005615`

Against this canonical baseline, the ACV pricing variants are worse in global/ATM pricing:

- `abl13` full price RMSE: `0.055836`, ratio `13.08x` worse.
- `abl14` full price RMSE: `0.043731`, ratio `10.24x` worse.
- `abl13` ATM price RMSE: `0.058670`, ratio `10.45x` worse.
- `abl14` ATM price RMSE: `0.046357`, ratio `8.26x` worse.

They are better only in the short/hard pricing regions:

- `abl13` hard price RMSE: `0.002094` vs canonical `0.004780`.
- `abl14` hard price RMSE: `0.001379` vs canonical `0.004780`.
- `abl13` short-maturity price RMSE: `0.001192` vs canonical `0.002251`.
- `abl14` short-maturity price RMSE: `0.000861` vs canonical `0.002251`.

Conclusion: `abl14` improves over `abl13`, but neither beats the canonical baseline for global pricing.

### Planned stability check: `abl15_acv_log_learned_weights_prior`

Motivation: `abl14` improved pricing, but both learned weights saturated at the clamp. This suggests that the gain may partly come from a large global rescaling of `L_PDE` and `L_low`, not necessarily from a stable interior optimum of the log-variance variables.

Implementation:

- Same ACV pricing architecture as `abl13` and `abl14`.
- Learn `pde` and `low` log-variance weights only.
- Use a narrower clamp: `min_log_var=-3`, `max_log_var=3`, so the maximum effective multiplier is `exp(3) = 20.0855`.
- Add a quadratic prior on the raw log-variance parameter: `0.2 * (s - 0)^2`.

Bibliography checked:

- Kendall, Gal and Cipolla (2017), "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics": motivates learned homoscedastic task uncertainty and the `sigma^{-2} L + log sigma` structure.
- Wang, Teng and Perdikaris (2020), "Understanding and mitigating gradient pathologies in physics-informed neural networks": motivates care with unbalanced gradients in composite PINN losses.
- Bischof and Kraus (2021/2025), "Multi-Objective Loss Balancing for Physics-Informed Deep Learning": motivates treating PINN training as multi-objective loss balancing rather than relying on fixed manual weights.
