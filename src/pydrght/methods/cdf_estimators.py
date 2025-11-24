import numpy as np
from scipy.stats import norm
from scipy.stats import gamma as sp_gamma

def cdfglo(x, para):
    """
    Compute the cumulative distribution function (CDF) of the Generalized Logistic (Log-Logistic) distribution.

    Parameters
    ----------
    x : array-like
        Input values where the CDF should be evaluated.
    para : list or tuple of length 3
        Distribution parameters:
            xi    : location parameter
            alpha : scale parameter
            kappa : shape parameter

    Returns
    -------
    np.ndarray
        Array of CDF values corresponding to the input x.
    
    Notes
    -----
    Handles the special case when kappa ~ 0 (reduces to standard logistic distribution).
    Values outside the domain of the distribution are returned as 0 or 1 depending on kappa.
    NaN values in x are preserved in the output.
    """
    x = np.array(x, dtype=float)
    xi, alpha, kappa = para
    cdf = np.empty_like(x)
    SMALL = 1e-15
    for i, val in enumerate(x):
        if np.isnan(val):
            cdf[i] = np.nan
            continue
        Y = (val - xi) / alpha
        if abs(kappa) < SMALL:
            # kappa ~ 0, standard logistic
            cdf[i] = 1 / (1 + np.exp(-Y))
        else:
            ARG = 1 - kappa * Y
            if ARG > SMALL:
                Y2 = -np.log(ARG) / kappa
                cdf[i] = 1 / (1 + np.exp(-Y2))
            elif kappa < 0:
                cdf[i] = 0
            else:
                cdf[i] = 1
    return cdf

def cdfgam(x, para):
    """
    Compute CDF of 2- or 3-parameter gamma-like distribution.
    
    Parameters
    ----------
    x : array-like
        Values at which to compute the CDF
    para : dict
        {'para': [alpha, beta]} or {'para': [MU, SIGMA, NU]}
    
    Returns
    -------
    np.ndarray
        CDF values
    """
    x = np.asarray(x)
    params = para
    
    # 2-parameter gamma
    if len(params) == 2:
        alpha, beta = params
        f = np.where(x <= 0, 0, sp_gamma.cdf(x, a=alpha, scale=beta))
        return f

    # 3-parameter generalized gamma
    elif len(params) == 3:
        MU, SIGMA, NU = params
        f = np.zeros_like(x, dtype=float)
        B = SIGMA * abs(NU)
        z = (x / MU) ** NU
        
        theta = 1 / (SIGMA**2 * NU**2)
        mask_normal = (~np.isfinite(theta)) | (abs(NU) < 1e-6)
        
        # Normal approximation
        if np.any(mask_normal):
            f[mask_normal] = norm.cdf(z[mask_normal], loc=np.log(MU), scale=SIGMA)
        
        # Gamma CDF
        mask_gamma = ~mask_normal
        if np.any(mask_gamma):
            f[mask_gamma] = sp_gamma.cdf(z[mask_gamma], a=1/B**2, scale=B**2)
        
        # Flip for negative NU
        if NU < 0:
            f = 1 - f
        
        return f

    else:
        raise ValueError("para['para'] must have length 2 or 3")


def cdfpe3(x, para):
    """
    Pearson Type III CDF (PE3), equivalent to lmomco::cdfpe3 in R.

    Parameters
    ----------
    x : array-like
        Input data.
    para : dict
        PE3 parameters as returned by `parpe3`, i.e.,
        {
            'type': 'pe3',
            'para': [MU, SIGMA, GAMMA],
            'source': 'parpe3'
        }

    Returns
    -------
    np.ndarray
        CDF values (0..1)
    """
    x = np.array(x, dtype=float)
    MU, SIGMA, GAMMA = para
    
    SMALL = np.sqrt(np.finfo(float).eps)
    
    cdf = np.empty_like(x)
    
    # If skewness is effectively zero → normal distribution
    if abs(GAMMA) <= SMALL:
        cdf = norm.cdf((x - MU) / SIGMA)
        return cdf
    
    # Skewed Pearson III
    ALPHA = 4.0 / (GAMMA**2)
    BETA  = 0.5 * SIGMA * abs(GAMMA)
    XI    = MU - 2.0 * SIGMA / GAMMA

    if GAMMA > 0:
        # Gamma CDF
        cdf = sp_gamma.cdf((x - XI) / BETA, a=ALPHA)
    else:
        # Negative skew
        cdf = 1.0 - sp_gamma.cdf((XI - x) / BETA, a=ALPHA)
    
    return cdf
