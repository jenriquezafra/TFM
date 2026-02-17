# Obtain the IV given a price using Brent's method

import numpy as np
from scipy.optimize import brentq, least_squares

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
        cos_interval_rule="sqrt_t",
        iv_bounds=(1e-6, 5.0),
        tol=1e-6,
        max_iter=100,
        return_details=False,
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
                              interval_rule=cos_interval_rule,
                              opt_type=opt_type
                              )
    
    V_tgt = np.float64(target_price)
    if (not np.isfinite(V_tgt)) or (V_tgt < 0):
        raise ValueError(f"Invalid target price from COS solver: {V_tgt}")

    low_iv, high_iv = iv_bounds
    low_iv = np.float64(low_iv)
    high_iv = np.float64(high_iv)
    if low_iv <= 0:
        low_iv = np.float64(1.0e-12)
    if high_iv <= low_iv:
        raise ValueError(f"Invalid IV bounds: low={low_iv}, high={high_iv}")

    # funcion of the residual V_BS - V_Heston
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
    
    f_low = np.float64(f(low_iv))
    f_high = np.float64(f(high_iv))
    if (not np.isfinite(f_low)) or (not np.isfinite(f_high)):
        raise ValueError(
            f"Non-finite residual on initial bracket: f(low)={f_low}, f(high)={f_high}"
        )

    # Expand upper bound if the initial bracket does not bracket a root.
    # This avoids false negatives on high-IV edge cases.
    max_high_iv = np.float64(20.0)
    while (f_low * f_high > 0.0) and (high_iv < max_high_iv):
        high_iv = np.float64(min(max_high_iv, high_iv * 2.0))
        f_high = np.float64(f(high_iv))
        if not np.isfinite(f_high):
            break

    if np.isclose(f_low, 0.0, atol=tol):
        iv = low_iv
    elif np.isclose(f_high, 0.0, atol=tol):
        iv = high_iv
    elif f_low * f_high > 0.0:
        # Fallback: if one endpoint is already very close in price, keep it.
        abs_low = np.float64(abs(f_low))
        abs_high = np.float64(abs(f_high))
        best_abs = min(abs_low, abs_high)
        if best_abs <= np.float64(max(1.0e-10, 10.0 * tol)):
            iv = low_iv if abs_low <= abs_high else high_iv
        else:
            raise ValueError(
                "Brent failed to bracket root: "
                f"f(low={low_iv:.3e})={f_low:.3e}, "
                f"f(high={high_iv:.3e})={f_high:.3e}, "
                f"target_price={V_tgt:.3e}"
            )
    else:
        iv = brentq(
            f,
            low_iv,
            high_iv,
            xtol=tol,
            maxiter=max_iter
        )

    iv = np.float64(iv)
    if not return_details:
        return iv

    bs_at_iv = np.float64(BS_solver(S0=S0, K=K, tau=tau, sigma=iv, r=r, opt_type=opt_type))
    price_residual = np.float64(bs_at_iv - V_tgt)
    details = {
        "target_price": V_tgt,
        "bs_price_at_iv": bs_at_iv,
        "price_residual": price_residual,
        "price_residual_abs": np.float64(abs(price_residual)),
    }
    return iv, details


################################ To compute a IV surface ########################################
# NOTE: Deprecated; not used anymore

def IV_surface(
        params_Heston,
        S0,
        K_array,
        tau_array,
        r,
        COS_params,
        cos_interval_rule="sqrt_t",
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
            cos_interval_rule=cos_interval_rule,
            opt_type=opt_type
        )

    return iv_surface

################################ To compute IV using LM ########################################

def IV_LM(
        params_Heston,
        S0,
        K,
        tau, 
        r,
        COS_params,
        opt_type="put",
        cos_interval_rule="sqrt_t",
        sigma0=0.2,
        return_details=False,
        ):
    """
    Compute the IV using Levenberg-Marquardt method
    :param K: float
    :param tau: float
    """

    # formatting
    params_Heston = np.array(params_Heston, dtype=np.float64)
    COS_params = np.array(COS_params, dtype=int)
    S0 = np.float64(S0)
    K = np.float64(K)
    tau = np.float64(tau)
    r = np.float64(r)


    # target price from Heston model
    target_price = COS_solver_scalar(
        params_Heston=params_Heston,
        S0=S0,
        K=K,
        tau=tau,
        r=r,
        COS_params=COS_params,
        interval_rule=cos_interval_rule,
        opt_type=opt_type,
    )

    V_tgt = np.float64(target_price)


    # residual (BS - Heston)
    def residual(x):
        sigma = np.float64(x[0])
        dif = np.array(BS_solver(S0=S0, K=K, tau=tau, sigma=sigma, r=r, opt_type=opt_type) - V_tgt)
        return dif
    
    res = least_squares(residual, x0=np.array([sigma0]), method="lm")
    sigma_hat = np.float64(res.x[0])
    iv = sigma_hat if sigma_hat > 0 else np.nan

    if not return_details:
        return iv

    if np.isnan(iv):
        bs_at_iv = np.nan
        price_residual = np.nan
        price_residual_abs = np.nan
    else:
        bs_at_iv = np.float64(BS_solver(S0=S0, K=K, tau=tau, sigma=iv, r=r, opt_type=opt_type))
        price_residual = np.float64(bs_at_iv - V_tgt)
        price_residual_abs = np.float64(abs(price_residual))

    details = {
        "target_price": V_tgt,
        "bs_price_at_iv": bs_at_iv,
        "price_residual": price_residual,
        "price_residual_abs": price_residual_abs,
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "cost": np.float64(res.cost),
    }
    return iv, details
    
