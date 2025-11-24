from pydrght.si import SI

class SPI(SI):
    """
    Standardized Precipitation Index (SPI).

    Default:
    - Uses Gamma distribution
    - Two-parameter fit
    """

    def __init__(self, data, ts=1):
        super().__init__(data, ts)

    def fit(self):
        """
        Default SPI computation:
        Gamma distribution (2-parameter).
        """
        return self.fit_parametric_pwm(dist_type="gam")
