from ..si import SI

class SGI(SI):
    """
    Standardized Groundwater Index (SGI).

    Default:
    - Uses empirical CDF (non-parametric)
    - Normal-scores transform
    """

    def __init__(self, data, ts=1):
        super().__init__(data, ts)

    def fit(self, method="Gringorten"):
        """
        Compute SGI using the empirical CDF (non-parametric normal scores).

        Parameters
        ----------
        method : str
            Plotting position: 'Gringorten' (default) or 'Weibull'.

        Returns
        -------
        pd.DataFrame
            Index + CDF columns.
        """
        return self.fit_empirical(method=method)
