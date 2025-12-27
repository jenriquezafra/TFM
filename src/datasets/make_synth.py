# Generate synthetic dataset
#   - samplear parámetros de Heston + (K,tau,r)
#   - compute IV surface 
#   - saves in standard format (parquet?) (save parameters + IVs)

from scipy.stats import qmc


################################# GENERATE THE PARAMS #####################################
# Generate the Heston parameters
def generate_params_Heston(n_samples, param_ranges, seed=None):
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    X_unit = sampler.random(n=n_samples)
    X = qmc.scale(X_unit, param_ranges[:,0], param_ranges[:,1])
    return X

# Generate the m, tau grid
def generate_grid(n_samples, bounds, seed=None):
    """
    - bounds: np.array of shape (2,2) with [[m_min, tau_min], [m_max, tau_max]]
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    X_unit = sampler.random(n=n_samples)
    X = qmc.scale(X_unit, bounds[:,0], bounds[:,1])
    return X


# Generate r
def generate_r(n_samples, bounds, seed):
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    X_unit = sampler.random(n=n_samples)
    X = qmc.scale(X_unit, bounds[0], bounds[1])
    return X

# Generate all at once (LHS over 8 dimensions) 
#NOTE: this is the one we use now

def generate_all(n_samples, param_ranges, grid_bounds, r_bounds, seed=None):
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    X_unit = sampler.random(n=n_samples)
    
    # scale params
    X_params = qmc.scale(X_unit[:,:5], param_ranges[:,0], param_ranges[:,1])
    
    # scale grid
    X_grid = qmc.scale(X_unit[:,5:7], grid_bounds[:,0], grid_bounds[:,1])
    
    # scale r
    X_r = qmc.scale(X_unit[:,7:], r_bounds[0], r_bounds[1])
    
    return X_params, X_grid, X_r