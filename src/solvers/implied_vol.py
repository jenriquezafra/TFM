# Obtain the IV given a price using Brent's method

import numpy as np
from scipy.optimize import brentq

from src.solvers.bs import BS_solver
from src.solvers.heston_cos import COS_solver


################################### To compute a vector of IVs ########################################

def IV_Brent(
        params_Heston,
        S0,
        K_array, 
        tau,
        COS_params,
        opt_type="put",
        iv_bounds=(1e-6, 5.0),
        tol=1e-6,
        max_iter=100,
        ):
    """
    From a tau and a vector of K, compute the IV using Brent's method
    
    :param K_array: array of strikes
    :param tau: float
    """

    # target price from Heston model
    target_price = COS_solver(params_Heston=params_Heston, 
                              S0=S0, 
                              K_array = K_array,
                              tau=tau,
                              COS_params=COS_params,
                              opt_type=opt_type
                              )
    
    r = params_Heston[-1]
    low_iv, high_iv = iv_bounds

    iv = np.empty_like(target_price, dtype=float)

    for j, K in enumerate(K_array):
        V_tgt = float(target_price[j])

        def f(sigma):
            V_bs = BS_solver(
                S0=S0, 
                K_array=K,
                tau=float(tau),
                sigma=float(sigma),
                r=r,
                opt_type=opt_type
            )
            return float(np.asarray(V_bs)) - V_tgt
        
        iv[j] = brentq(f,
                       low_iv, 
                       high_iv, 
                       xtol=tol, 
                       maxiter=max_iter)
    
    return iv


################################ To compute a IV surface ########################################

def IV_surface(
        params_Heston,
        S0,
        K_array,
        tau_array,
        COS_params,
        opt_type="put"
        ):

    """
    Compute the implied volatility surface using Brent's method
    :param K_array: array of strikes
    :param tau_array: array of maturities
    """

    iv_surface = np.empty((len(tau_array), len(K_array)), dtype=float)

    for i, tau in enumerate(tau_array):
        iv_surface[i, :] = IV_Brent(
            params_Heston=params_Heston,
            S0=S0,
            K_array=K_array,
            tau=tau,
            COS_params=COS_params,
            opt_type=opt_type
        )

    return iv_surface