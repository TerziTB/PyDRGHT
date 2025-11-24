from pydrght.si import SI

class SSFI(SI):
    """
    Standardized Streamflow Index (SSFI).

    Default:
    - Uses Log-logistic distribution
    """

    def __init__(self, data, ts=1):
        super().__init__(data, ts)

    def fit(self):
        """
        Default SSFI computation using log-logistic distribution.
        """
        return self.fit_parametric_pwm(dist_type="glo")
