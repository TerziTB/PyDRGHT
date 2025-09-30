import pandas as pd
import numpy as np
from .utils import accu

class RAI:
    """
    Rainfall Anomaly Index (RAI) and Modified Rainfall Anomaly Index (mRAI) for drought monitoring.

    Computes the RAI (van Rooy, 1965) and the monthly-based mRAI (Gibbs & Maher, 1967; Springer, 2015)
    for a monthly precipitation series. Supports arbitrary accumulation periods.

    The RAI is calculated using the mean of the series and the average of the 10 largest
    and 10 smallest values to scale positive and negative anomalies.

    The mRAI is calculated month-wise using the median precipitation and the
    top/bottom 10% extremes of each month group to scale anomalies.

    Parameters
    ----------
    series : pd.Series
        Monthly precipitation series. No datetime index is required.
    ts : int, default=1
        Accumulation period in months (1 = monthly, 3 = seasonal, etc.).

    Returns
    -------
    pd.Series
        Continuous time series of RAI or mRAI values.

    References
    ----------
    - van Rooy, M. P. (1965). 
    *A rainfall anomaly index (RAI) independent of time and space*. 
    Notos, 14, 43–48.

    - Hänsel, S., Schucknecht, A., & Matschullat, J. (2016). 
    *The Modified Rainfall Anomaly Index (mRAI)—is this an alternative to the Standardised Precipitation Index (SPI) in evaluating future extreme precipitation characteristics?* 
    Theoretical and Applied Climatology, 123, 827–844. 
    [DOI: 10.1007/s00704-015-1389-y](https://doi.org/10.1007/s00704-015-1389-y)
    """

    def __init__(self, precip: pd.Series, ts: int = 1):
        self.series = precip.dropna()
        self.ts = ts
        self.results = None

    def RAI(self, scale: float = 3) -> pd.Series:
        P_acc = accu(self.series, self.ts)
        pm = P_acc.mean()

        mean_top10 = P_acc.nlargest(10).mean()
        mean_bottom10 = P_acc.nsmallest(10).mean()

        rai = pd.Series(np.nan, index=P_acc.index, dtype=float)

        above_mean = P_acc > pm
        below_mean = ~above_mean


        rai[above_mean] = scale * (P_acc[above_mean] - pm) / (mean_top10 - pm)
        
        rai[below_mean] = -scale * (P_acc[below_mean] - pm) / (mean_bottom10 - pm)

        self.results = rai
        return self.results

    def mRAI(self, scale: float = 1.7) -> pd.Series:
        P_acc = accu(self.series, self.ts)
        mrai = pd.Series(np.nan, index=P_acc.index, dtype=float)

        months = (np.arange(len(P_acc)) % 12) + 1

        for m in range(1, 13):
            month_vals = P_acc[months == m]
            if month_vals.empty:
                continue

            median_month = month_vals.median()

            threshold_wet = month_vals.quantile(0.9)
            threshold_dry = month_vals.quantile(0.1)

            wet_mask = month_vals >= median_month
            dry_mask = month_vals < median_month

            if wet_mask.any():
                mrai.loc[month_vals.index[wet_mask]] = scale * (
                    month_vals[wet_mask] - median_month
                ) / (month_vals[month_vals >= threshold_wet].mean() - median_month)

            if dry_mask.any():
                mrai.loc[month_vals.index[dry_mask]] = -scale * (
                    month_vals[dry_mask] - median_month
                ) / (month_vals[month_vals <= threshold_dry].mean() - median_month)

        self.results = mrai
        return self.results