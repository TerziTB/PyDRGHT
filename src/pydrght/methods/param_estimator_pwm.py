from .ub_pwm import ub_pwm, pwm2lmom
from .param_estimators import pargam, parglo, parpe3

def estimate_params_pwm(data, dist_type):
    """
    Wrapper for UB-PWM → L-moments → parameter estimation.
    
    Parameters
    ----------
    data : array-like
        Input time series.
    dist_type : str
        Distribution type: 'gam', 'pe3', 'glo'
    
    Returns
    -------
    dict
        {'type': ..., 'para': [...]}
    """
    pwm = ub_pwm(data)
    lmom_raw = pwm2lmom(pwm["betas"])
    
    # Map to what the param estimators expect
    lmom_mapped = {
        "L1": lmom_raw["lambdas"][0],
        "L2": lmom_raw["lambdas"][1],
        "TAU3": lmom_raw["ratios"][2]
    }
    
    if dist_type.lower() == "gam":
        return pargam(lmom_raw)
    elif dist_type.lower() == "pe3":
        return parpe3(lmom_mapped)
    elif dist_type.lower() == "glo":
        return parglo(lmom_raw)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
