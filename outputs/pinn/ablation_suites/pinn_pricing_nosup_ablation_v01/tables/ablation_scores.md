| Experiment | Status | Grid OK | Global | Hard | Price | Price RMSE | Non-Hard Price RMSE | Hard Price RMSE | Gamma RMSE | Hard Gamma RMSE | Train s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| price00_baseline | baseline | yes | 1 | 1 | 1 | 0.113442 | 0.113491 | 0.101063 | 1.94945 | 11.8355 | 1623.77 |
| price01_right_vzero | pricing_improvement | yes | 1.00388 | 1.04674 | 0.188399 | 0.0474778 | 0.0475727 | 0.00857111 | 2.08433 | 12.3039 | 3086.47 |
| price02_guard_derivatives | pricing_improvement | yes | 0.92692 | 1.02337 | 0.287757 | 0.0403203 | 0.0403752 | 0.023545 | 2.06498 | 12.3396 | 956.722 |
| price03_payoff_sqrt_time | pricing_improvement | yes | 2.14345 | 2.87852 | 0.293364 | 0.072677 | 0.072822 | 0.0135765 | 20.5929 | 50.4807 | 398.744 |
| price04_pde_normalized | pricing_improvement | yes | 0.846272 | 0.862451 | 0.598614 | 0.0586774 | 0.0586259 | 0.0700152 | 1.63686 | 11.5464 | 370.695 |
| price05_log_core | pricing_improvement | yes | 2.41579 | 9.72074 | 0.44542 | 0.0461797 | 0.0461666 | 0.0492558 | 9.41794 | 126.568 | 271.564 |
| price06_learned_weights | pricing_improvement | yes | 0.858075 | 0.943458 | 0.299382 | 0.0470933 | 0.0471698 | 0.0218203 | 1.84154 | 11.663 | 369.633 |
| price07_combo_guard_norm_learned | pricing_improvement | yes | 0.996813 | 0.972149 | 0.397915 | 0.0439088 | 0.0439191 | 0.0413428 | 2.01949 | 12.1222 | 1032.27 |
| price08_combo_payoff_guard_norm_learned | pricing_improvement | yes | 1.75157 | 2.87662 | 0.185957 | 0.0472951 | 0.0473898 | 0.00838263 | 20.5952 | 50.4786 | 1165.38 |
| price09_combo_log_guard_norm_learned | pricing_regression | yes | 21.7641 | 299.813 | 1.49651 | 0.136625 | 0.136373 | 0.187932 | 540.687 | 7721.85 | 851.737 |
