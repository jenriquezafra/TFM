# Vanila options pricer under Heston using the COS method to use as ground truth

import numpy as np
from numpy import cos, exp, pi, sin, sqrt
import torch


def _heston_chf_numpy(u, tau, r, rho, kappa, gamma, bar_nu, nu0):
    u = np.asarray(u, dtype=np.complex128)

    d1 = np.sqrt((kappa - gamma * rho * 1j * u) ** 2 + (u**2 + 1j * u) * gamma**2)
    g = (kappa - gamma * rho * 1j * u - d1) / (kappa - gamma * rho * 1j * u + d1)
    cc = kappa - 1j * rho * gamma * u - d1

    exp_term = exp(-d1 * tau)
    c1 = exp(1j * u * tau * r + (nu0 / gamma**2) * (((1 - exp_term) / (1 - g * exp_term)) * cc))
    c2 = exp((kappa * bar_nu / gamma**2) * (tau * cc - 2 * np.log((1 - g * exp_term) / (1 - g))))
    return c1 * c2


def _heston_chf_torch(u, tau, r, rho, kappa, gamma, bar_nu, nu0):

    d1 = torch.sqrt((kappa - gamma * rho * 1j * u) ** 2 + (u**2 + 1j * u) * gamma**2)
    g = (kappa - gamma * rho * 1j * u - d1) / (kappa - gamma * rho * 1j * u + d1)
    cc = kappa - 1j * rho * gamma * u - d1

    exp_term = torch.exp(-d1 * tau)
    c1 = torch.exp(
        1j * u * tau * r + (nu0 / gamma**2) * (((1 - exp_term) / (1 - g * exp_term)) * cc)
    )
    c2 = torch.exp(
        (kappa * bar_nu / gamma**2) * (tau * cc - 2 * torch.log((1 - g * exp_term) / (1 - g)))
    )
    return c1 * c2


def heston_cumulants_autodiff(params_Heston, tau, r):
    """
    Compute (c1, c2, c4) cumulants of log-return X = log(S_T / S_0)
    with autodiff over k(u)=log(phi(-i*u)).
    """

    rho, kappa, gamma, bar_nu, nu0 = [float(x) for x in params_Heston]
    tau = float(tau)
    r = float(r)

    dtype_r = torch.float64
    dtype_c = torch.complex128

    rho_t = torch.tensor(rho, dtype=dtype_r)
    kappa_t = torch.tensor(kappa, dtype=dtype_r)
    gamma_t = torch.tensor(gamma, dtype=dtype_r)
    bar_nu_t = torch.tensor(bar_nu, dtype=dtype_r)
    nu0_t = torch.tensor(nu0, dtype=dtype_r)
    tau_t = torch.tensor(tau, dtype=dtype_r)
    r_t = torch.tensor(r, dtype=dtype_r)

    t = torch.tensor(0.0, dtype=dtype_r, requires_grad=True)
    u = torch.complex(torch.zeros_like(t), -t).to(dtype_c)  # u = -i*t

    phi = _heston_chf_torch(
        u=u,
        tau=tau_t,
        r=r_t,
        rho=rho_t,
        kappa=kappa_t,
        gamma=gamma_t,
        bar_nu=bar_nu_t,
        nu0=nu0_t,
    )
    k_fn = torch.log(phi).real

    c1 = torch.autograd.grad(k_fn, t, create_graph=True)[0]
    c2 = torch.autograd.grad(c1, t, create_graph=True)[0]
    c3 = torch.autograd.grad(c2, t, create_graph=True)[0]
    c4 = torch.autograd.grad(c3, t, create_graph=False)[0]

    return float(c1.detach()), float(c2.detach()), float(c4.detach())


def _cos_integration_bounds(params_Heston, tau, r, L, t0=0.0, interval_rule="sqrt_t"):
    T = float(np.float64(tau) + np.float64(t0))
    T_nonneg = max(T, 0.0)
    fallback_a = -L * np.sqrt(T_nonneg)
    fallback_b = L * np.sqrt(T_nonneg)

    if interval_rule in {"sqrt_t", "legacy"}:
        return fallback_a, fallback_b

    if interval_rule in {"cumulant_autodiff", "cumulants"}:
        try:
            c1, c2, c4 = heston_cumulants_autodiff(params_Heston=params_Heston, tau=tau, r=r)
            c4_pos = max(c4, 0.0)
            spread_inner = c2 + np.sqrt(c4_pos)
            if (not np.isfinite(spread_inner)) or spread_inner <= 0.0:
                return fallback_a, fallback_b
            spread = L * np.sqrt(spread_inner)
            a = c1 - spread
            b = c1 + spread
            if np.isfinite(a) and np.isfinite(b) and b > a:
                return a, b
            return fallback_a, fallback_b
        except Exception:
            return fallback_a, fallback_b

    raise ValueError(
        f"Unsupported interval_rule '{interval_rule}'. "
        "Use 'sqrt_t' or 'cumulant_autodiff'."
    )

# NOTE: deprecated
def COS_solver(params_Heston,
               S0, 
               K_array,
               tau, 
               r,
               COS_params, 
               t0=0, 
               opt_type="put",
               interval_rule="sqrt_t"):
    """
    
    Params:
    - params_Heston: array of params
        [rho, kappa, gamma, bar_nu, nu_0]
    - S0: float
        Spot price at time t0
    - K_array: array of strikes
    - tau: float
        Time left to maturity time
    - r: float
        risk-free rate
    - t0: float
        current time
    - COS_params: array of params
        [N (terms on the truncation), L (tolerance)]
        L needs to be 6 ≤ L ≤ 12 
    - opt_type: str
        "call" or "put"
        
    Returns:
    - V: value of the option
    """

    # unpack the parameters
    [rho, kappa, gamma, bar_nu, nu0] = params_Heston
    [N, L] = COS_params
    N = int(N)

    # definition of the maturity time (we will use tau = T)
    T = tau + t0

    # integration range
    a, b = _cos_integration_bounds(
        params_Heston=params_Heston,
        tau=tau,
        r=r,
        L=L,
        t0=t0,
        interval_rule=interval_rule,
    )

    # define the Characteristic Function of Heston
    def ChF_Heston(u, tau):
        return _heston_chf_numpy(
            u=u,
            tau=tau,
            r=r,
            rho=rho,
            kappa=kappa,
            gamma=gamma,
            bar_nu=bar_nu,
            nu0=nu0,
        )
    
    def payoff_coeff(k):
        def chi_coeff(c,d):
            chi = 1/(1+(pi*k/(b-a))**2) * (cos(pi*k*(d-a)/(b-a))*exp(d) - cos(pi*k*(c-a)/(b-a))*exp(c) + pi*k/(b-a)*sin(pi*k*(d-a)/(b-a))*exp(d) - pi*k/(b-a)*sin(pi*k*(c-a)/(b-a))*exp(c) )
            return chi
        
        def psi_coeff(c,d):
            if k==0:
                psi = d-c
            else:
                psi =(b-a)/(pi*k) * (sin(pi*k*(d-a)/(b-a)) - sin(pi*k*(c-a)/(b-a)))
            return psi
        
        if opt_type=="call":
            H = 2/(b-a) * (chi_coeff(0,b) - psi_coeff(0,b))
        elif opt_type == "put":
            H = 2/(b-a) * (psi_coeff(a,0) - chi_coeff(a,0))

        return H
    
    # some computations for the final expression
    ## some vectors
    u_array = np.array([pi*k/(b-a) for k in range(0,N)])
    U_array = np.array([payoff_coeff(k) for k in range(0,N)])
    ChF_array = ChF_Heston(u_array, tau)
    m_array = np.log(S0/K_array)
    exp_array = np.array([exp(1j*pi*k*(m_array-a)/(b-a)) for k in range(0,N)]) 


    ## the sum
    cos_terms = ChF_array[1:, None] * U_array[1:, None] * exp_array[1:]
    cos_sum_0 = 0.5 * ChF_array[0] * U_array[0] * exp_array[0]
    cos_sum = cos_sum_0 + np.sum(cos_terms, axis=0)

    # compute the value of the options
    V = K_array * exp(-r*tau) * np.real(cos_sum)

    return V


###############
# NOTE: using this right now

def COS_solver_scalar(params_Heston,
                      S0,
                      K,
                      tau,
                      r,
                      COS_params,
                      t0=0,
                      opt_type="put",
                      interval_rule="sqrt_t"):

    rho, kappa, gamma, bar_nu, nu0 = params_Heston
    N, L = COS_params
    N = int(N)
    
    S0 = np.float64(S0)
    tau = np.float64(tau)
    r = np.float64(r)

    T = tau + t0
    T = np.float64(T)

    a, b = _cos_integration_bounds(
        params_Heston=params_Heston,
        tau=tau,
        r=r,
        L=L,
        t0=t0,
        interval_rule=interval_rule,
    )

    def ChF_Heston(u, tau):
        return _heston_chf_numpy(
            u=u,
            tau=tau,
            r=r,
            rho=rho,
            kappa=kappa,
            gamma=gamma,
            bar_nu=bar_nu,
            nu0=nu0,
        )

    def payoff_coeff(k):
        def chi_coeff(c, d):
            denom = 1 + (pi*k/(b-a))**2
            term1 = cos(pi*k*(d-a)/(b-a)) * exp(d) - cos(pi*k*(c-a)/(b-a)) * exp(c)
            term2 = (pi*k/(b-a)) * (sin(pi*k*(d-a)/(b-a)) * exp(d) - sin(pi*k*(c-a)/(b-a)) * exp(c))
            return (term1 + term2) / denom

        def psi_coeff(c, d):
            if k == 0:
                return d - c
            return (b-a)/(pi*k) * (sin(pi*k*(d-a)/(b-a)) - sin(pi*k*(c-a)/(b-a)))

        if opt_type == "call":
            return 2/(b-a) * (chi_coeff(0, b) - psi_coeff(0, b))
        elif opt_type == "put":
            return 2/(b-a) * (psi_coeff(a, 0) - chi_coeff(a, 0))
        else:
            raise ValueError("Option type must be `call` or `put`.")


    # --- parte escalar ---
    S0 = float(S0)
    K = float(K)
    tau = float(tau)
    r = float(r)

    u = pi * np.arange(N) / (b - a)                 # (N,)
    U = np.array([payoff_coeff(k) for k in range(N)], dtype=complex)  # (N,)
    phi = ChF_Heston(u, tau)                        # (N,)

    m = np.log(S0 / K)                              # escalar
    exp_terms = exp(1j * pi * np.arange(N) * (m - a) / (b - a))    # (N,)

    cos_sum = 0.5 * phi[0] * U[0] * exp_terms[0] + np.sum(phi[1:] * U[1:] * exp_terms[1:])

    V = K * exp(-r * tau) * np.real(cos_sum)
    return float(V)
