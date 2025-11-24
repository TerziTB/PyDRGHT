import numpy as np
import pandas as pd
import scipy.stats as stats
from .utils import accu, uni_emp 
from .methods import estimate_params_pwm
from .methods import cdfgam, cdfglo, cdfpe3

class SI:
    """
    Standardized Index for Drought Analysis (SPI-based methodology).

    This class computes standardized indices based on the SPI (Standardized Precipitation Index)
    methodology, which transforms a time series into a normalized index representing
    deviations from long-term climatology. While SPI was originally developed for precipitation,
    the same methodology can be applied to other hydrometeorological variables:

    - **SPI**: precipitation
    - **SPEI**: climatic water balance (precipitation minus potential evapotranspiration)
    - **SSFI**: streamflow or river discharge
    - **SGI**: groundwater levels
    - **SSMI**: soil moisture

    The general procedure is:
    1. Aggregate the time series over a chosen timescale (`ts` months) to capture
       short-term or long-term drought patterns.
    2. Fit a parametric distribution (e.g., gamma, generalized extreme value) or
       compute empirical cumulative probabilities.
    3. Transform the probabilities to standard normal quantiles, yielding a
       standardized index with mean 0 and variance 1.

    Parameters
    ----------
    data : pd.Series
        Input 1D time series of the variable of interest (precipitation, streamflow,
        soil moisture, groundwater, or water balance). Index should be datetime-like.
    ts : int, optional
        Aggregation timescale in months (default is 1). Values are summed or averaged
        over this window before standardization.

    Methods
    -------
    fit_parametric(distribution_func, is_2p=False)
        Fit a parametric distribution to the aggregated series and compute the
        standardized index. Can fix the location parameter to zero for two-parameter
        distributions.
    fit_empirical(method='Gringorten')
        Compute the standardized index using the empirical cumulative distribution
        function with a chosen plotting position formula (Gringorten or Weibull).

    References
    ----------
    - McKee, T. B., Doesken, N. J., & Kleist, J. (1993). 
    *The Relationship of Drought Frequency and Duration to Time Scales*. 
    Proceedings of the 8th Conference on Applied Climatology, 179–184.

    - Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). 
    *A multiscalar drought index sensitive to global warming: The Standardized Precipitation Evapotranspiration Index*. 
    Journal of Climate, 23(7), 1696–1718. 
    [DOI: 10.1175/2009JCLI2909.1](https://doi.org/10.1175/2009JCLI2909.1)

    - Farahmand, A., & AghaKouchak, A. (2015). 
    *A generalized framework for deriving nonparametric standardized drought indicators*. 
    Advances in Water Resources, 76, 140–145. 
    [DOI: 10.1016/j.advwatres.2014.11.012](https://doi.org/10.1016/j.advwatres.2014.11.012)

    - Shukla, S., & Wood, A. W. (2008). 
    *Use of a standardized runoff index for characterizing hydrologic drought*. 
    Geophysical Research Letters, 35(2), L02405. 
    [DOI: 10.1029/2007GL032487](https://doi.org/10.1029/2007GL032487)
    """

    def __init__(self, data: pd.Series, ts: int = 1):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        self.data = data
        self.ts = ts
        self.aggregated = accu(data, ts)

    def fit_parametric(self, distribution_func, is_2p: bool = False) -> pd.DataFrame:
        """
        Fit a parametric distribution (e.g., gamma, genextreme) to compute the standardized index.

        Parameters
        ----------
        distribution_func : scipy.stats distribution
            Continuous distribution to fit.
        is_2p : bool
            If True, fit two-parameter distribution (loc fixed at 0).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Index' and 'CDF'.
        """
        XA = pd.Series(self.aggregated, index=self.data.index[self.ts-1:])
        zero_count = (XA == 0).sum()
        prob_zero = zero_count / len(XA)

        monthly_data = {f'Month_{i}': XA[i-1::12] for i in range(1, 13)}

        params = {}
        cdf_values = {}

        for month, values in monthly_data.items():
            filtered = values[values > 0]
            if len(filtered) == 0:
                params[month] = None
                cdf_values[month] = np.full(len(values), np.nan)
                continue
            fitted_params = distribution_func.fit(filtered, floc=0) if is_2p else distribution_func.fit(filtered)
            params[month] = fitted_params
            cdf = distribution_func.cdf(values, *fitted_params)
            cdf_values[month] = prob_zero + (1 - prob_zero) * cdf

        eps = 1e-10
        sip_values = {month: stats.norm.ppf(np.clip(cdf, eps, 1 - eps))
                      for month, cdf in cdf_values.items()}

        n = len(XA)
        reordered_sip = np.full(n, np.nan)
        reordered_cdf = np.full(n, np.nan)

        for i in range(1, 13):
            idx = np.arange(i - 1, n, 12)
            month_key = f'Month_{i}'
            if month_key in sip_values:
                reordered_sip[idx] = sip_values[month_key]
            if month_key in cdf_values:
                reordered_cdf[idx] = cdf_values[month_key]

        return pd.DataFrame({
            'Index': reordered_sip,
            'CDF': reordered_cdf
        }, index=XA.index)

    def fit_parametric_pwm(self, dist_type: str):
        """
        Fit a parametric distribution using UB-PWM + L-moments.

        Parameters
        ----------
        dist_type : str
            Distribution type: 'gam', 'pe3', 'glo'.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Index' (standardized) and 'CDF'.
        """
        XA = pd.Series(self.aggregated, index=self.data.index[self.ts-1:])
        n = len(XA)
                
        zero_count = (XA == 0).sum()
        prob_zero = zero_count / len(XA)

        
        monthly_data = {f'Month_{i}': XA[i-1::12] for i in range(1, 13)}

        params = {}
        cdf_values = {}

        # Map dist_type to the corresponding CDF function
        cdf_func_map = {
            "gam": cdfgam,
            "pe3": cdfpe3,
            "glo": cdfglo
        }

        if dist_type.lower() not in cdf_func_map:
            raise ValueError(f"Unknown dist_type: {dist_type}")

        cdf_func = cdf_func_map[dist_type.lower()]
        
        for month, values in monthly_data.items():
            filtered = values.dropna()
            if len(filtered) == 0:
                params[month] = None
                cdf_values[month] = np.full(len(values), np.nan)
                continue
            pwm_result = estimate_params_pwm(filtered, dist_type=dist_type)
            params[month] = pwm_result["para"]
            raw_cdf = cdf_func(values.values, params[month])
            cdf_values[month] = prob_zero + (1 - prob_zero) * raw_cdf
        eps = 1e-10
        sip_values = {
            month: stats.norm.ppf(np.clip(cdf, eps, 1 - eps))
            for month, cdf in cdf_values.items()
        }

        reordered_sip = np.full(n, np.nan)
        reordered_cdf = np.full(n, np.nan)

        for i in range(1, 13):
            month_key = f"Month_{i}"
            values = monthly_data[month_key]

            sip = sip_values[month_key]
            cdf = cdf_values[month_key]

            idx = np.arange(i - 1, n, 12)[:len(sip)]
            reordered_sip[idx] = sip
            reordered_cdf[idx] = cdf

        return pd.DataFrame({
            "Index": reordered_sip,
            "CDF": reordered_cdf
        }, index=XA.index)
        
    def fit_empirical(self, method: str = 'Gringorten') -> pd.DataFrame:
        """
        Compute the standardized index using the empirical distribution function
        with a specified plotting position formula.

        Parameters
        ----------
        method : str
            Plotting position formula. Options: 'Gringorten', 'Weibull'.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Index' and 'CDF'.
        """
        XA = pd.Series(self.aggregated, index=self.data.index[self.ts-1:])
        n = len(XA)
        
        monthly_data = {f'Month_{i}': XA[i-1::12] for i in range(1, 13)}

        reordered_sip = np.full(n, np.nan)
        reordered_cdf = np.full(n, np.nan)

        for i in range(1, 13):
            month_key = f'Month_{i}'
            values = monthly_data[month_key]
            if values.dropna().empty:
                continue
            
            # Empirical CDF
            cdf = uni_emp(values, method=method)
            # Standardized index
            sip = stats.norm.ppf(cdf)
            
            # Only assign to as many values as returned
            idx = np.arange(i - 1, n, 12)[:len(sip)]
            reordered_sip[idx] = sip
            reordered_cdf[idx] = cdf

        return pd.DataFrame({
            'Index': reordered_sip,
            'CDF': reordered_cdf
        }, index=XA.index)