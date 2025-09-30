import numpy as np
from .utils import accu, uni_emp, multi_emp
from scipy.stats import norm
import pandas as pd
from typing import Dict, Tuple
from .copulas import (
    ClaytonCopula, FrankCopula, GumbelCopula,
    GaussianCopula,
    GalambosCopula, PlackettCopula
)

class MSDI:
    def __init__(self, X: pd.Series, Y: pd.Series, ts: int):
        """
        MSDI (Multivariate Standardized Drought Index) Class

        This class calculates a bivariate standardized drought index for two dependent
        hydrological or meteorological variables (e.g., precipitation and streamflow). 
        The MSDI framework extends the univariate SPI/SPEI concept to account for 
        dependencies between two variables and allows flexible modeling with copulas 
        or empirical approaches.

        Attributes:
        -----------
        X : pd.Series
            Time series of the first variable (e.g., precipitation or streamflow).
        Y : pd.Series
            Time series of the second variable.
        ts : int
            Time scale for accumulation (e.g., 3-month, 6-month sums).
        XA : np.ndarray
            Accumulated values of X over the specified time scale.
        YA : np.ndarray
            Accumulated values of Y over the specified time scale.
        n : int
            Length of the accumulated series.
        index : pd.Index
            Index of the original time series.
        metrics : dict
            Stores goodness-of-fit metrics (AIC, BIC, negative log-likelihood) 
            after parametric copula fitting.

        Raises:
        -------
        TypeError
            If X or Y are not pandas Series.
        ValueError
            If X and Y have different lengths or an unsupported copula family is specified.

        References:
        -----------
        - Hao, Z., & AghaKouchak, A. (2013). 
        *Multivariate standardized drought index: a parametric multi-index model*. 
        Advances in Water Resources, 57, 12–18. 
        [DOI: 10.1016/j.advwatres.2013.03.009](https://doi.org/10.1016/j.advwatres.2013.03.009)

        - Hao, Z., & AghaKouchak, A. (2014). 
        *A nonparametric multivariate multi-index drought monitoring framework*. 
        Journal of Hydrometeorology, 15(1), 89–101. 
        [DOI: 10.1175/jhm-d-12-0160.1](https://doi.org/10.1175/jhm-d-12-0160.1)
        """
        if not isinstance(X, pd.Series) or not isinstance(Y, pd.Series):
            raise TypeError("X and Y must be pandas Series")

        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")

        self.X = X
        self.Y = Y
        self.ts = ts
        self.index = X.index

        # Accumulate using numpy arrays internally, but keep as Series
        self.XA = accu(X, ts)
        self.YA = accu(Y, ts)
        self.n = len(self.XA)
        
    def _get_copula_object(self, copula_family: str):
        """Return copula object based on family name."""
        mapping = {
            "gaussian": GaussianCopula,
            "clayton": ClaytonCopula,
            "frank": FrankCopula,
            "gumbel": GumbelCopula,
            "galambos": GalambosCopula,
            "plackett": PlackettCopula
        }
        try:
            return mapping[copula_family.lower()]()
        except KeyError:
            raise ValueError(f"Unsupported copula family: {copula_family}")
    
    def parametric(self, copula_family: str = "Gaussian", method: str = "Gringorten") -> Tuple[pd.Series, Dict[str, float]]:
        """
        Compute parametric MSDI using a copula object.
        Returns a pandas Series aligned with the original index.
        """
        u = uni_emp(self.XA, method)
        v = uni_emp(self.YA, method)
        data_combined = np.column_stack((u, v))

        copula = self._get_copula_object(copula_family)
        copula.fit(data_combined)
        cdf_values = copula.cdf(data_combined)
        msdi_values = pd.Series(norm.ppf(cdf_values))

        try:
            pdf_values = copula.pdf(data_combined)
            loglik = np.sum(np.log(pdf_values))
            k = getattr(copula, "theta", 1).size if hasattr(copula, "theta") else 1
            aic = 2*k - 2*loglik
            bic = np.log(len(data_combined))*k - 2*loglik
        except Exception:
            aic = bic = loglik = np.nan

        self.metrics = {"AIC": aic, "BIC": bic, "NLL": loglik}
        return msdi_values
    
    def empirical(self) -> pd.Series:
        """
        Compute bivariate MSDI using empirical copula.
        Returns a pandas Series aligned with the original index.
        """
        msdi_values = {}
        for month in range(1, 13):
            xa_slice = self.XA.values[month-1::12]
            ya_slice = self.YA.values[month-1::12]

            prob = multi_emp(xa_slice, ya_slice)
            msdi_values[month] = norm.ppf(prob)

        reordered = np.full(self.n, np.nan)
        for month in range(1, 13):
            idx = np.arange(month-1, self.n, 12)
            reordered[idx] = msdi_values[month]

        return pd.Series(reordered)