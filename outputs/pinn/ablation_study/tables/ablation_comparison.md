| Model | Price RMSE | Delta RMSE | Gamma RMSE | Hard Delta RMSE | Hard Gamma RMSE | Hard p99 Gamma Error | Training Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline PINN | 0.00426945 | 0.0338084 | 0.534829 | 0.0872874 | 4.9921 | 17.6427 | 1151.0 |
| Payoff-aware PINN | 0.0279462 | 0.264658 | 18.6863 | 0.294878 | 50.6541 | 239.591 | 2124.7 |
| Payoff-aware + boundary-layer | 0.0157027 | 0.151225 | 10.0169 | 0.164745 | 37.6819 | 187.877 | 2265.5 |
| Boundary-layer no-payoff | 0.0219286 | 0.247039 | 1.83018 | 0.312323 | 16.0564 | 54.4731 | 2802.0 |
| Adaptive collocation | 0.238723 | 0.307172 | 1.7876 | 0.406953 | 11.8534 | 29.8167 | 320.3 |
| Derivative-consistency | 0.00395818 | 0.0357967 | 0.579619 | 0.0995134 | 5.58089 | 18.5507 | 442.6 |
| ACV control variate | 0.00423462 | 0.0326907 | 0.404042 | 0.0128829 | 0.64462 | 1.61234 | 0.0 |
