import pandas as pd
import numpy as np
from scipy.stats import rv_continuous, kstest, anderson


class Dist:
    """
    A wrapper around a SciPy continuous distribution fitted to data.

    This class provides methods to fit a continuous distribution to univariate
    data, compute the PDF, CDF, inverse CDF (percent point function), 
    goodness-of-fit tests, information criteria, and return period analysis.

    Parameters
    ----------
    data : pd.Series
        Input data to fit the distribution.
    dist : rv_continuous
        SciPy continuous distribution (e.g., `scipy.stats.norm`, `scipy.stats.gamma`).
    prob_zero : bool, default=False
        Whether to account for zero-inflation (probability of zero values in the dataset).
    floc0 : bool, default=False
        If True, fixes the location parameter (`loc`) to 0 when fitting the distribution.

    Attributes
    ----------
    loc : float
        Fitted location parameter of the distribution.
    scale : float
        Fitted scale parameter of the distribution.
    shape : list[float] or None
        Fitted shape parameters (if applicable), otherwise None.
    p0 : float
        Estimated probability of zero values if `prob_zero=True`.

    Methods
    -------
    cdf()
        Returns the cumulative distribution function (CDF) values of the fitted data.
    pdf()
        Returns the probability density function (PDF) values of the fitted data.
    ppf(q)
        Computes the percent point function (inverse CDF) at probability `q`.
    ks_test()
        Performs the Kolmogorov-Smirnov goodness-of-fit test.
    ad_test()
        Computes the Anderson-Darling test statistic for supported distributions.
    aic()
        Computes Akaike Information Criterion for the fitted distribution.
    bic()
        Computes Bayesian Information Criterion for the fitted distribution.
    return_period(T, interarrival=1.0)
        Returns the threshold value for a specified return period.

    References
    ----------
    - Kolmogorov, A. (1933). 
    *Sulla determinazione empirica di una legge di distribuzione*. 
    Giornale dell'Istituto Italiano degli Attuari, 4, 83–91. 

    - Smirnov, N. (1948). 
    *Table for estimating the goodness of fit of empirical distributions*. 
    Annals of Mathematical Statistics, 19, 279–281. 
    [DOI: 10.1214/aoms/1177730256](https://doi.org/10.1214/aoms/1177730256)

    - Anderson, T. W., & Darling, D. A. (1952). 
    *Asymptotic theory of certain "goodness-of-fit" criteria based on stochastic processes*. 
    Annals of Mathematical Statistics, 23, 193–212. 
    [DOI: 10.1214/aoms/1177729437](https://doi.org/10.1214/aoms/1177729437)

    - Akaike, H. (1974). 
    *A new look at the statistical model identification*. 
    IEEE Transactions on Automatic Control, 19(6), 716–723. 
    [DOI: 10.1109/TAC.1974.1100705](https://doi.org/10.1109/TAC.1974.1100705)

    - Schwarz, G. (1978). 
    *Estimating the dimension of a model*. 
    Annals of Statistics, 6(2), 461–464. 
    [DOI: 10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
    """

    def __init__(self, data: pd.Series, dist: rv_continuous, prob_zero: bool = False, floc0: bool = False):
        self.data = data.dropna()
        self.dist = dist
        self.prob_zero = prob_zero
        self.floc0 = floc0
        self.loc = None
        self.scale = None
        self.shape = None
        self.p0 = 0.0
        self._fit()

    def _fit(self):
        if self.floc0:
            fit = self.dist.fit(self.data, floc=0)
        else:
            fit = self.dist.fit(self.data)

        if len(fit) > 2:
            *shape, loc, scale = fit
            self.shape = shape
        else:
            loc, scale = fit
            self.shape = None

        self.loc = loc
        self.scale = scale

        if self.prob_zero:
            self.p0 = (self.data == 0.0).mean()

    def cdf(self, x: pd.Series = None) -> pd.Series:
        data = self.data if x is None else x
        if self.shape:
            cdf = self.dist.cdf(data, *self.shape, loc=self.loc, scale=self.scale)
        else:
            cdf = self.dist.cdf(data, loc=self.loc, scale=self.scale)

        if self.prob_zero:
            cdf = self.p0 + (1 - self.p0) * cdf
            cdf[self.data == 0.0] = self.p0

        return pd.Series(cdf, dtype=float)

    def pdf(self, x: pd.Series = None) -> pd.Series:
        data = self.data if x is None else x
        if self.shape:
            pdf = self.dist.pdf(data, *self.shape, loc=self.loc, scale=self.scale)
        else:
            pdf = self.dist.pdf(data, loc=self.loc, scale=self.scale)

        return pd.Series(pdf, dtype=float)

    def ppf(self, q: float) -> float:
        if self.shape:
            return self.dist.ppf(q, *self.shape, loc=self.loc, scale=self.scale)
        else:
            return self.dist.ppf(q, loc=self.loc, scale=self.scale)

    def ks_test(self) -> tuple[float, float]:
        args = (*self.shape, self.loc, self.scale) if self.shape else (self.loc, self.scale)
        return kstest(self.data, self.dist.cdf, args=args)

    def ad_test(self) -> tuple[float, float]:
        supported_dists = ["norm", "expon", "logistic", "gumbel", "gumbel_l", "gumbel_r", "extreme1", "weibull_min"]

        if self.dist.name not in supported_dists:
            raise NotImplementedError(
                f"Anderson-Darling only implemented for {supported_dists} in SciPy."
            )

        result = anderson(self.data, dist=self.dist.name)
        return result.statistic, result.critical_values

    def aic(self) -> float:
        ll = self._log_likelihood()
        k = len(self.shape) if self.shape is not None else 0
        k += 2  # loc, scale
        return 2 * k - 2 * ll

    def bic(self) -> float:
        ll = self._log_likelihood()
        n = len(self.data)
        k = len(self.shape) if self.shape is not None else 0
        k += 2  # loc, scale
        return np.log(n) * k - 2 * ll

    def _log_likelihood(self) -> float:
        if self.shape is not None:
            pdf_vals = self.dist.pdf(
                self.data,
                *([self.shape] if isinstance(self.shape, float) else self.shape),
                loc=self.loc, scale=self.scale
            )
        else:
            pdf_vals = self.dist.pdf(self.data, loc=self.loc, scale=self.scale)

        pdf_vals = np.where(pdf_vals <= 0, 1e-12, pdf_vals)  # avoid log(0)
        return np.sum(np.log(pdf_vals))
    
    def return_period(self, T: float, interarrival: float = 1.0) -> float:
        """
        Compute the threshold (return level) corresponding to a given return period.

        This is widely used in hydrology, climatology, and extreme value analysis
        to estimate the magnitude of an event expected once every T years.

        Parameters
        ----------
        T : float
            Return period in the same units as `interarrival` (e.g., years).
        interarrival : float, default=1.0
            Expected interarrival time between events.

        Returns
        -------
        float
            Threshold value corresponding to the specified return period.

        Raises
        ------
        ValueError
            If `T` is non-positive.

        Notes
        -----
        The exceedance probability is calculated as:
            p_exceed = interarrival / T
        The return level is then obtained using the inverse CDF (percent point function):
            x_T = PPF(1 - p_exceed)

        References
        ----------
        - Katz, R. W., Parlange, M. B., & Naveau, P. (2002). Statistics of extremes in hydrology.
        Advances in Water Resources, 25(8-12), 1287-1304.
        - Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values.
        Springer Series in Statistics.
        """
        if T <= 0:
            raise ValueError("Return period T must be positive.")
        p_exceed = interarrival / T
        threshold = self.ppf(1 - p_exceed)
        return threshold