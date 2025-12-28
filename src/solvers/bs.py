# Solver of the Black-Scholes model for vanilla options

import numpy as np
from scipy.stats import norm


def BS_solver(S0, K, tau, sigma, r, t0=0, opt_type="put"):
    """
    Pointwise, not vectors
     
     :param S0: spot proce
     :param K: strike price
     :param tau: time to maturity
     :param sigma: volatility
     :param r: risk-free rate 
     :param t0: actual time
     :param opt_type: option type
     """

    # some formatting
    S0 = np.float64(S0)
    K = np.float64(K)

    # some constants
    T = tau + t0

    # compute d1 and d2
    d1 = (np.log(S0 / K) + (r+0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # compute the option price
    if opt_type == "call":
        V = S0*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    elif opt_type == "put":
        V = K*np.exp(-r*tau)*norm.cdf(-d2) - S0*norm.cdf(-d1) 
    return V

