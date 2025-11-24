from pydrght.si import SI

class SSMI(SI):
    """
    Standardized Soil Moisture Index (SSMI).

    Default:
    - Uses empirical CDF (non-parametric)
    - Normal-scores transform
    """

    def __init__(self, data, ts=1):
        super().__init__(data, ts)

    def fit(self, method="Gringorten"):
        """
        Compute SSMI using the empirical CDF (non-parametric normal scores).

        Parameters
        ----------
        method : str
            Plotting position: 'Gringorten' (default) or 'Weibull'.

        Returns
        -------
        pd.DataFrame
            Index + CDF columns.
        """
        # Beta distribution as default
        return self.fit_empirical(method=method)
