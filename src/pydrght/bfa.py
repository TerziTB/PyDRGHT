import numpy as np
import pandas as pd
from .dist import Dist

class BFA:
    """
    Bivariate frequency analysis (BFA) of hydrological or meteorological events using copulas.

    This class estimates joint return periods of two dependent variables (e.g., drought 
    duration and severity) by combining fitted univariate marginal distributions with 
    a copula to model their dependence structure. Both "OR" and "AND" return periods 
    can be computed simultaneously, providing insights into the probability of extreme 
    events occurring individually or jointly.

    Parameters
    ----------
    dist_x : Dist
        Fitted `Dist` object for the first variable (e.g., drought duration).
    dist_y : Dist
        Fitted `Dist` object for the second variable (e.g., drought severity).
    copula_family : Callable
        Copula class to use for modeling dependence between the variables. Supported 
        families include Archimedean copulas (Frank, Clayton, Gumbel), elliptical 
        (Gaussian), and extreme-value copulas (Galambos, Plackett).

    Attributes
    ----------
    copula : object
        Instance of the specified copula, fitted to the uniform-transformed marginals.
    data_combined : np.ndarray
        Array of uniform-transformed observed data used for copula fitting.

    Methods
    -------
    joint_return_period(T, interarrival)
        Computes both "OR" and "AND" joint return periods for a specified T-year event.

    Returns
    -------
    pd.Series
        Series with indices 'OR' and 'AND', containing the joint return periods in 
        the same units as `interarrival`.

    References
    ----------
    - Shiau, J. T. (2006). 
    *Fitting drought duration and severity with two-dimensional copulas*. 
    Water Resources Management, 20(5), 795–815. 
    [DOI: 10.1007/s11269-005-9008-9](https://doi.org/10.1007/s11269-005-9008-9)
    """
    def __init__(self, dist_x: Dist, dist_y: Dist, copula_family):
        self.dist_x = dist_x
        self.dist_y = dist_y
        self.copula_family = copula_family

        u = self.dist_x.cdf().values
        v = self.dist_y.cdf().values
        self.data_combined = np.column_stack((u, v))

        self.copula = copula_family()

    def joint_return_period(
        self,
        T: float,
        interarrival: float = 1.0,
    ) -> pd.Series:
        """
        Compute both 'OR' and 'AND' bivariate return periods for a given T-year event.

        This method calculates joint return periods of two dependent variables 
        (e.g., drought duration and severity) using the fitted marginal distributions 
        and copula. It returns the expected return periods for:

        - 'OR': at least one variable exceeds its threshold
        - 'AND': both variables exceed their thresholds simultaneously

        Parameters
        ----------
        T : float
            Return period of interest in the same units as `interarrival`.
        interarrival : float, default=1.0
            Expected interarrival time between events (e.g., 1 year). This scales the 
            return period to the observed data frequency.

        Returns
        -------
        pd.Series
            Series with indices ['OR', 'AND'] containing the joint return periods 
            in the same units as `interarrival`.

        Raises
        ------
        ValueError
            If `T` is not positive.
        """
        if T <= 0:
            raise ValueError("Return period T must be positive.")

        p_exceed = interarrival / T
        expected_cdf = 1 - p_exceed

        threshold_x = self.dist_x.ppf(expected_cdf)
        threshold_y = self.dist_y.ppf(expected_cdf)
        
        u_thresh = self.dist_x.cdf(x=threshold_x).iloc[0]
        v_thresh = self.dist_y.cdf(x=threshold_y).iloc[0]
        thresholds = np.column_stack((u_thresh, v_thresh))

        self.copula.fit(u=self.data_combined)
        joint_cdf = self.copula.cdf(u=thresholds)

        joint_or_prob = 1 - joint_cdf[0]
        joint_and_prob = 1 - u_thresh - v_thresh + joint_cdf[0]

        return pd.Series(
            [interarrival / joint_or_prob, interarrival / joint_and_prob],
            index=['OR', 'AND'],
            dtype=float
        )
