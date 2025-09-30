import pandas as pd
import numpy as np
from .utils import accu

class DI:
    """
    Deciles Index (DI) for drought monitoring.

    Computes drought deciles for a monthly precipitation series using 
    the method originally proposed by Gibbs and Maher (1967). The 
    DI can be calculated globally (all months combined) or 
    month-wise (seasonally adjusted, each calendar month ranked separately).

    The DI assigns each month a decile (1–10) based on the empirical 
    cumulative distribution of precipitation values, where 1 indicates 
    extremely dry conditions and 10 indicates extremely wet conditions.

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation series.
    ts : int, default=1
        Accumulation period in months.

    Returns
    -------
    pd.Series
        Time series of DI values (1–10). The `calculate()` method returns
        global DI, while `calculate_monthwise()` returns seasonally adjusted 
        month-wise DI.

    References
    ----------
    - Gibbs, W. J., & Maher, J. V. (1967). 
    *Rainfall deciles as drought indicators*. 
    Bureau of Meteorology, Australia.
    """
    def __init__(self, precip: pd.Series, ts: int = 1):
        self.precip = precip.dropna()
        self.ts = ts
        self.results = None

    def calculate(self) -> pd.Series:
        P_acc = accu(self.precip, self.ts)

        ranks = P_acc.rank(method="average")
        n = len(P_acc)
        percentiles = (ranks - 1) / n

        deciles = np.ceil(percentiles * 10).astype(int)
        deciles[deciles == 0] = 1

        self.results = pd.Series(deciles, index=P_acc.index, name="DI")
        return self.results

    def calculate_monthwise(self) -> pd.Series:
        P_acc = accu(self.precip, self.ts)
        months = (np.arange(len(P_acc)) % 12) + 1
        di = pd.Series(index=P_acc.index, dtype=int)

        for m in range(1, 13):
            month_vals = P_acc[months == m]
            if month_vals.empty:
                continue

            ranks = month_vals.rank(method="average")
            n = len(month_vals)
            percentiles = (ranks - 1) / n
            deciles_month = np.ceil(percentiles * 10).astype(int)
            deciles_month[deciles_month == 0] = 1

            di.loc[month_vals.index] = deciles_month

        self.results = di
        self.results.name = "DI_monthwise"
        return self.results