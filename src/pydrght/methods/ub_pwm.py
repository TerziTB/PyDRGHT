import numpy as np
from scipy.special import comb

def ub_pwm(x, orders=(0,1,2)):
    """
    Compute unbiased probability weighted moments (UB-PWM)
    exactly like TLMoments::PWM(x, order=0:2).

    Parameters
    ----------
    x : array-like
        Input data.
    orders : tuple
        PWM orders to compute (default: 0,1,2)

    Returns
    -------
    dict with "betas" array and "source"
    """
    x = np.array(x, dtype=float)
    x = np.sort(x)        

    n = len(x)
    betas = []

    for r in orders:
        num = 0.0
        denom = comb(n - 1, r)

        for j in range(1, n+1):    # 1..n
            num += comb(j - 1, r) * x[j - 1]

        beta_r = num / (n * denom)
        betas.append(beta_r)

    return {"betas": np.array(betas)}

def pwm2lmom(pwm):
    """
    Convert probability weighted moments (PWM) to L-moments.
    
    Parameters
    ----------
    pwm : list or np.array
        PWM values, e.g. [beta0, beta1, beta2, ...]
        
    Returns
    -------
    dict
        {
            'lambdas': L-moments [L1, L2, L3, ...],
            'ratios' : L-moment ratios [NA, tau2, tau3, ...],
            'source' : 'pwm2lmom'
        }
    """
    
    pwm = np.array(pwm, dtype=float)
    nmom = len(pwm)
    
    L = np.zeros(nmom)
    R = np.full(nmom, np.nan)
    
    for i in range(nmom):
        r = i
        s = 0.0
        for k in range(r+1):
            weight = (-1)**(r-k) * comb(r, k) * comb(r+k, k)
            s += weight * pwm[k]
        L[i] = s
    
    if nmom >= 2:
        R[1] = L[1] / L[0]
    if nmom >= 3:
        for r in range(2, nmom):
            R[r] = L[r] / L[1]
    
    return {'lambdas': L, 'ratios': R}
