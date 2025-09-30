import pandas as pd
import numpy as np
from .utils import accu 

class PNI:
    """
    Percent of Normal Index (PNI) for drought monitoring.

    The Percent of Normal Index is one of the simplest precipitation-based
    drought indicators. It compares observed precipitation to the long-term
    mean precipitation (the "normal") over a given period:

        PNI = (P_obs / P_normal) * 100

    - PNI = 100 : precipitation equals the normal.
    - PNI < 100 : below-normal (dry conditions).
    - PNI > 100 : above-normal (wet conditions).

    Notes
    -----
    - "Normal" is typically the 30-year mean precipitation for the same
      time period.
    - PNI is simple but assumes a normal distribution of precipitation,
      which is often not true (the mean ≠ median).
    - Best used for a single location and for seasonal or annual scales.

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation series (no datetime index required).
    ts : int, default=1
        Accumulation period in months.

    Returns
    -------
    pd.Series
        Global PNI: "PNI_global" (% of normal across entire series).
        Month-wise PNI: "PNI_monthwise" (% of normal relative to each month).

    References
    ----------
    - Hayes, M. J. (1999). 
    *Drought Indices*. 
    National Drought Mitigation Center, University of Nebraska-Lincoln.
    """

    def __init__(self, precip: pd.Series, ts: int = 1):
        self.precip = precip.dropna()
        self.ts = ts
        self.results = None

    def calculate(self) -> pd.Series:
        P_acc = accu(self.precip, self.ts)
        mean_precip = P_acc.mean()

        pni = (P_acc / mean_precip) * 100
        self.results = pd.Series(pni, index=P_acc.index, name="PNI_global")
        return self.results

    def calculate_monthwise(self) -> pd.Series:
        P_acc = accu(self.precip, self.ts)
        months = (np.arange(len(P_acc)) % 12) + 1
        pni = pd.Series(index=P_acc.index, dtype=float)

        for m in range(1, 13):
            month_vals = P_acc[months == m]
            if month_vals.empty:
                continue

            mean_month = month_vals.mean()
            pni.loc[month_vals.index] = (month_vals / mean_month) * 100

        self.results = pni
        self.results.name = "PNI_monthwise"
        return self.results