# Vanila options pricer under Heston using the COS method to use as ground truth

import numpy as np 
from numpy import exp, pi, sqrt, cos, sin


def COS_solver(params_Heston,
               S0, 
               K_array,
               tau, 
               r,
               COS_params, 
               t0=0, 
               opt_type="put"):
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
    a, b = -L*sqrt(T), L*sqrt(T)     # TODO: implement the best version with cumulants    NECESARIO

    # define the Characteristic Function of Heston
    def ChF_Heston(u, tau):
        # first we need to define some coefficients
        def D1(u):
            d1 = sqrt((kappa-gamma*rho*1j*u)**2 + (u**2+1j*u)*gamma**2)
            return d1
    
        def g(u):
            gc = (kappa-gamma*rho*1j*u-D1(u)) / (kappa-gamma*rho*1j*u+D1(u))
            return gc

        cc = kappa - 1j*rho*gamma*u-D1(u)
        c1 =  exp(1j*u*tau*r + nu0/gamma**2 *((1-exp(-D1(u)*tau))/(1-g(u)*exp(-D1(u)*tau)))* cc)
        c2 =  exp(kappa*bar_nu/gamma**2 * (tau*cc - 2*np.log((1-g(u)*exp(-D1(u)*tau))/(1-g(u)))))
        phi = c1*c2
        return phi
    
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

def COS_solver_scalar(params_Heston,
                      S0,
                      K,
                      tau,
                      r,
                      COS_params,
                      t0=0,
                      opt_type="put"):

    rho, kappa, gamma, bar_nu, nu0 = params_Heston
    N, L = COS_params
    N = int(N)

    T = np.float64(tau + t0)

    a, b = -L * np.sqrt(T), L * np.sqrt(T)

    def ChF_Heston(u, tau):
        u = np.asarray(u, dtype=complex)

        def D1(u):
            return np.sqrt((kappa - gamma * rho * 1j * u) ** 2 + (u**2 + 1j*u) * gamma**2)

        def g(u):
            return (kappa - gamma * rho * 1j * u - D1(u)) / (kappa - gamma * rho * 1j * u + D1(u))

        cc = kappa - 1j * rho * gamma * u - D1(u)

        c1 = exp(1j*u*tau*r + (nu0/gamma**2) * (((1 - exp(-D1(u)*tau)) / (1 - g(u)*exp(-D1(u)*tau))) * cc))
        c2 = exp((kappa*bar_nu/gamma**2) * (tau*cc - 2*np.log((1 - g(u)*exp(-D1(u)*tau)) / (1 - g(u)))))

        return c1 * c2

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