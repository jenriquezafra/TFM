# Solver of the Black-Scholes model for vanilla options

import numpy as np
from scipy.stats import norm


def BS_solver(S0, K_array, tau, sigma, r, t0=0, opt_type="put"):
    """
     Docstring para BS_solver
     
     :param S0: Descripción
     :param K_array: Descripción
     :param tau: Descripción
     :param sigma: Descripción
     :param r: Descripción
     :param t0: Descripción
     :param opt_type: Descripción
     """

    # some formatting
    S0 = np.array(S0, dtype=float)
    K_array = np.array(K_array, dtype=float)

    # some constants
    T = tau + t0

    # compute d1 and d2
    d1 = (np.log(S0 / K_array) + (r+0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # compute the option price
    if opt_type == "call":
        V = S0*norm.cdf(d1) - K_array*np.exp(-r*tau)*norm.cdf(d2)
    elif opt_type == "put":
        V = K_array*np.exp(-r*tau)*norm.cdf(-d2) - S0*norm.cdf(-d1) 
    return V

