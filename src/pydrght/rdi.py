import pandas as pd
import numpy as np
from .utils import accu

class RDI:
    """
    Reconnaissance Drought Index (RDI) for drought monitoring.

    Calculates both the **Normalized RDI (RDIₙ)** and the 
    **Standardized RDI (RDIst)** following Tsakiris & Vangelis (2005).

    Both global (all months combined) and month-wise (seasonally adjusted)
    RDI can be calculated.

    Definitions
    -----------
    Let:
        P = accumulated precipitation over the selected timescale
        PET = accumulated potential evapotranspiration
        alpha = P / PET   (initial index)

    - Normalized RDI (RDIₙ):
        RDIₙ = (alpha / mean(alpha)) - 1

    - Standardized RDI (RDIst):
        y = ln(alpha)
        RDIst = (y - mean(y)) / std(y)

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation series (no datetime index required).
    pet : pd.Series
        Monthly potential evapotranspiration series (same length as precip).
    ts : int, default=1
        Accumulation period in months.

    Returns
    -------
    pd.DataFrame
        Global RDI: columns 'RDI_normalized', 'RDI_standardized'
        Month-wise RDI: columns 'RDI_normalized_month', 'RDI_standardized_month'

    References
    ----------
    - Tsakiris, G., & Vangelis, H. (2005). 
    *Establishing a drought index incorporating evapotranspiration*. 
    European Water, 9/10, 3–11.
    """

    def __init__(self, precip: pd.Series, pet: pd.Series, ts: int = 1):
        if len(precip) != len(pet):
            raise ValueError("Precipitation and PET series must have the same length.")
        self.precip = precip.dropna()
        self.pet = pet.dropna()
        self.ts = ts
        self.results = None

    def calculate(self) -> pd.DataFrame:
        P_acc = accu(self.precip, self.ts)
        PET_acc = accu(self.pet, self.ts)

        alpha = P_acc / PET_acc.replace(0, np.nan)

        rdi_normalized = alpha / alpha.mean() - 1

        log_alpha = np.log(alpha.replace(0, np.nan))
        rdi_standardized = (log_alpha - log_alpha.mean()) / log_alpha.std(ddof=1)

        self.results = pd.DataFrame({
            "RDI_normalized": rdi_normalized,
            "RDI_standardized": rdi_standardized
        }, index=P_acc.index)

        return self.results

    def calculate_monthwise(self) -> pd.DataFrame:
        P_acc = accu(self.precip, self.ts)
        PET_acc = accu(self.pet, self.ts)

        months = (np.arange(len(P_acc)) % 12) + 1
        rdi_norm_month = pd.Series(index=P_acc.index, dtype=float)
        rdi_st_month = pd.Series(index=P_acc.index, dtype=float)

        for m in range(1, 13):
            idx = months == m
            P_m = P_acc[idx]
            PET_m = PET_acc[idx]
            if P_m.empty:
                continue

            alpha_m = P_m / PET_m.replace(0, np.nan)

            rdi_norm_month[idx] = alpha_m / alpha_m.mean() - 1

            log_alpha_m = np.log(alpha_m.replace(0, np.nan))
            rdi_st_month[idx] = (log_alpha_m - log_alpha_m.mean()) / log_alpha_m.std(ddof=1)

        self.results = pd.DataFrame({
            "RDI_normalized_month": rdi_norm_month,
            "RDI_standardized_month": rdi_st_month
        }, index=P_acc.index)

        return self.results