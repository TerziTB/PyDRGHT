import numpy as np
import pandas as pd
from .utils import accu

class CZI:
    """
    China Z Index (CZI) for drought monitoring.

    Calculates the China Z Index for a monthly precipitation series using
    standardized variates (Z-scores) and the original formula from
    Wu et al., 2001.
    
    The Modified CZI (MCZI) can also be calculated by using the **median** 
    precipitation instead of the mean in the Z-score calculation.

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation series. No datetime index is required.
    ts : int, default=1
        Accumulation period in months. 

    Returns
    -------
    pd.Series
        Continuous time series of CZI values.

    References
    ----------
    - Wu, H., Hayes, M. J., Weiss, A., & Hu, Q. (2001). 
    *An evaluation of the Standardized Precipitation Index, the China-Z Index and the statistical Z-Score*. 
    International Journal of Climatology, 21(6), 745–758. 
    [DOI: 10.1002/joc.658](https://doi.org/10.1002/joc.658)
    """
    
    def __init__(self, precip: pd.Series, ts: int = 1):
        self.precip = precip.dropna()
        self.ts = ts
        self.results = None
    
    def _compute_monthly_czi(self, group: pd.Series) -> pd.Series:
        P = group.values.astype(float)
        n = len(P)
        
        mean_P = P.mean()
        std_P = P.std(ddof=1)
        
        Z = (P - mean_P) / std_P
        
        Csi = np.sum((P - mean_P)**3 / std_P**3) / len(P)
        
        CZI = (6 / Csi) * (((Csi / 2) * Z + 1)**(1/3)) - (6 / Csi) + (Csi / 6)
        
        return pd.Series(CZI, index=group.index)
    
    def calculate(self) -> pd.Series:
        P_acc = accu(self.precip, self.ts)
        
        months = (np.arange(len(P_acc)) % 12) + 1
        
        czi_parts = []
        for m in range(1, 13):
            gr = P_acc[months == m]
            if not gr.empty:
                czi_parts.append(self._compute_monthly_czi(gr))
        
        self.results = pd.concat(czi_parts).sort_index()
        self.results.name = f"CZI-{self.ts}"
        return self.results