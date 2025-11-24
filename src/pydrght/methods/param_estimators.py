import math
import numpy as np
from scipy.special import gammaln

def parglo(lmom, small=1e-6):
    """
    Fit 3-parameter log-logistic (Generalized Logistic) distribution
    from L-moments (SPEI / parglo style).
    
    Parameters
    ----------
    lmom : dict
        Dictionary with keys:
            'lambdas' : list/array of L-moments [L1, L2, L3,...]
            'ratios'  : list/array of L-moment ratios [NA, tau2, tau3,...]
    small : float
        Threshold to treat kappa as 0 (2-parameter logistic)
        
    Returns
    -------
    dict
        {'type': 'glo', 'para': [xi, alpha, kappa]}
    """
    L1 = lmom['lambdas'][0]
    L2 = lmom['lambdas'][1]
    tau3 = lmom['ratios'][2]
    
    K = -tau3
    
    if abs(K) <= small:
        # effectively 2-parameter logistic
        xi = L1
        alpha = L2
        kappa = 0
        return {'type':'glo', 'para':[xi, alpha, kappa]}
    
    # full 3-parameter log-logistic
    KK = K * math.pi / math.sin(K * math.pi)
    A = L2 / KK
    xi = L1 - A * (1 - KK) / K
    alpha = A
    kappa = K
    
    return {'type':'glo', 'para':[xi, alpha, kappa]}

def pargam(lmom):
    """
    Python translation of lmomco::pargam (case p = 2).

    Parameters
    ----------
    lmom : dict
        {
            'lambdas': [L1, L2, ...],
            'ratios' : [...],
        }

    Returns
    -------
    dict with:
        {
            'type': 'gam',
            'para': [alpha, beta]
        }
    """

    L1 = lmom["lambdas"][0]
    L2 = lmom["lambdas"][1]

    if L1 <= 0 or L2 <= 0:
        raise ValueError("Invalid L-moments for Gamma")

    LCV = L2 / L1     # L-coefficient of variation

    # Constants from lmomco (minimax rational approximation)
    A1 = -0.3080
    A2 = -0.05812
    A3 =  0.01765
    B1 =  0.7213
    B2 = -0.5947
    B3 = -2.1817
    B4 =  1.2113

    if LCV >= 0.5:
        TT = 1 - LCV
        ALPHA = TT * (B1 + TT*B2) / (1 + TT*(B3 + TT*B4))
    else:
        TT = math.pi * LCV * LCV
        ALPHA = (1 + A1*TT) / (TT * (1 + TT*(A2 + TT*A3)))

    # Parameters identical to lmomco:
    # para[1] = ALPHA
    # para[2] = L1 / ALPHA
    alpha = ALPHA
    beta  = L1 / ALPHA

    return {"type": "gam", "para": [alpha, beta]}

def parpe3(lmom):
    """
    Fit Pearson Type III (PE3) distribution from L-moments.
    
    Parameters
    ----------
    lmom : dict
        L-moments with keys:
            'L1'   : mean
            'L2'   : L-scale
            'TAU3' : L-skewness
    checklmom : bool
        Whether to check validity of L-moments (not implemented)
    
    Returns
    -------
    dict
        {
            'type': 'pe3',
            'para': [mu, sigma, gamma],
            'source': 'parpe3'
        }
    """

    para = [np.nan, np.nan, np.nan]  # mu, sigma, gamma

    # Constants from lmomco
    C1, C2, C3 = 0.2906, 0.1882, 0.0442
    D1, D2, D3, D4, D5, D6 = 0.36067, -0.59567, 0.25361, -2.78861, 2.56096, -0.77045
    PI3 = 3 * math.pi
    ROOTPI = math.sqrt(math.pi)
    SMALL = 1e-6

    L1 = lmom['L1']
    L2 = lmom['L2']
    T3 = abs(lmom['TAU3'])

    # Zero skewness: reduce to normal approximation
    if T3 <= SMALL:
        para[0] = L1
        para[1] = L2 * ROOTPI
        para[2] = 0
        return {'type': 'pe3', 'para': para, 'source': 'parpe3'}

    # Rational approximations for ALPHA
    if T3 >= 1/3:
        T = 1 - T3
        ALPHA = T * (D1 + T*(D2 + T*D3)) / (1 + T*(D4 + T*(D5 + T*D6)))
    else:
        T = PI3 * T3**2
        ALPHA = (1 + C1*T) / (T * (1 + T*(C2 + T*C3)))

    RTALPH = math.sqrt(ALPHA)
    BETA = ROOTPI * L2 * math.exp(gammaln(ALPHA) - gammaln(ALPHA + 0.5))

    para[0] = L1
    para[1] = BETA * RTALPH
    para[2] = 2 / RTALPH

    # Correct for negative skew
    if lmom['TAU3'] < 0:
        para[2] = -para[2]

    return {'type': 'pe3', 'para': para, 'source': 'parpe3'}
