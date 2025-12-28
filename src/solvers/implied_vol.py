# Obtain the IV given a price using Brent's method

import numpy as np
from scipy.optimize import brentq

from src.solvers.bs import BS_solver
from src.solvers.heston_cos import COS_solver_scalar


################################### To compute a single IV ########################################
# NOTE: this is the one we are using 
def IV_Brent(
        params_Heston,
        S0,
        K, 
        tau,
        r,
        COS_params,
        opt_type="put",
        iv_bounds=(1e-6, 5.0),
        tol=1e-6,
        max_iter=100,
        ):
    """
    From a tau and a K, compute the IV using Brent's method
    
    :param K: float
    :param tau: float
    """
    # formatting
    params_Heston = np.array(params_Heston, dtype=np.float64)
    COS_params = np.array(COS_params, dtype=int)
    iv_bounds = np.array(iv_bounds, dtype=np.float64)

    S0 = np.float64(S0)
    K = np.float64(K)
    tau = np.float64(tau)
    r = np.float64(r)

    tol = np.float64(tol)
    max_iter = int(max_iter)

    # target price from Heston model
    target_price = COS_solver_scalar(params_Heston=params_Heston, 
                              S0=S0, 
                              K=K,
                              tau=tau,
                              r=r,
                              COS_params=COS_params,
                              opt_type=opt_type
                              )
    
    low_iv, high_iv = iv_bounds

    V_tgt = np.float64(target_price)

    def f(sigma):
        sigma=np.float64(sigma)
        V_bs = BS_solver(
            S0=S0,
            K=K,
            tau=tau, 
            sigma=sigma,
            r=r,
            opt_type=opt_type
        )
        return np.float64(V_bs)-V_tgt
    
    
    
    iv = brentq(
        f,
        low_iv,
        high_iv,
        xtol=tol,
        maxiter=max_iter
    )
    
    return np.float64(iv)

################################### To compute a vector of IVs ########################################
# NOTE: deprecated
def IV_Brent_vect(
        params_Heston,
        S0,
        K_array, 
        tau,
        r,
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
                              r=r,
                              COS_params=COS_params,
                              opt_type=opt_type
                              )
    
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
# NOTE: Deprecated; not used anymore

def IV_surface(
        params_Heston,
        S0,
        K_array,
        tau_array,
        r,
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
            r=r,
            COS_params=COS_params,
            opt_type=opt_type
        )

    return iv_surface