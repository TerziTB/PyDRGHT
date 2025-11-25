import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple
from .utils import accu, tri_emp

class TSDI:
    """
    TSDI (Trivariate Standardized Drought Index) Class

    This class calculates a trivariate standardized drought index for three 
    hydrological or meteorological variables (e.g., precipitation, streamflow, soil moisture).

    Attributes
    ----------
    X : pd.Series
        Time series of the first variable.
    Y : pd.Series
        Time series of the second variable.
    Z : pd.Series
        Time series of the third variable.
    ts : int
        Time scale for accumulation (e.g., 3-month, 6-month sums).
    XA, YA, ZA : np.ndarray
        Accumulated values over the specified time scale.
    n : int
        Length of the accumulated series.
    index : pd.Index
        Index of the original time series.
    """

    def __init__(self, X: pd.Series, Y: pd.Series, Z: pd.Series, ts: int):
        if not all(isinstance(s, pd.Series) for s in (X, Y, Z)):
            raise TypeError("X, Y, and Z must all be pandas Series")
        if len(X) != len(Y) or len(X) != len(Z):
            raise ValueError("X, Y, and Z must have the same length")
        
        self.X = X
        self.Y = Y
        self.Z = Z
        self.ts = ts
        self.index = X.index

        # Accumulated series
        self.XA = accu(X, ts)
        self.YA = accu(Y, ts)
        self.ZA = accu(Z, ts)
        self.n = len(self.XA)

    def parametric(self, copula_family: str = "Gaussian") -> Tuple[pd.Series, Dict[str, float]]:
        """
        Placeholder for parametric TSDI using 3-variate copulas.
        Will be implemented after development of 3-variate copula functions.

        Returns
        -------
        msdi_values : pd.Series
            Currently empty, will contain parametric TSDI values.
        metrics : dict
            Currently empty, will store AIC, BIC, log-likelihood after copula fitting.
        """
        # TODO: Implement parametric 3-variate copula method
        msdi_values = pd.Series(np.full(self.n, np.nan), index=self.index)
        metrics = {"AIC": np.nan, "BIC": np.nan, "NLL": np.nan}
        return msdi_values, metrics

    def empirical(self, method: str = "Gringorten") -> pd.Series:
        """
        Compute trivariate TSDI using empirical joint probabilities.

        Parameters
        ----------
        method : str, optional
            Method for empirical plotting position.
            Options:
            - "Gringorten" (default)
            - "Weibull"
            
        Returns
        -------
        tsdi_values : pd.Series
            Trivariate standardized index aligned with the original index.
        """
        tsdi_values = {}
        for month in range(1, 13):
            xa_slice = self.XA.values[month-1::12]
            ya_slice = self.YA.values[month-1::12]
            za_slice = self.ZA.values[month-1::12]

            prob = tri_emp(xa_slice, ya_slice, za_slice, method=method)
            tsdi_values[month] = norm.ppf(prob)

        reordered = np.full(self.n, np.nan)
        for month in range(1, 13):
            idx = np.arange(month-1, self.n, 12)
            reordered[idx] = tsdi_values[month]

        return pd.Series(reordered)