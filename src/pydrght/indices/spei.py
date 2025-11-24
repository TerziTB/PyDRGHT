from pydrght.si import SI

class SPEI(SI):
    """
    Standardized Precipitation Evapotranspiration Index (SPEI).

    Default behavior:
    - Uses GLO distribution
    - Uses PWM (L-moment) parameter estimation
    """

    def __init__(self, data, ts=1):
        super().__init__(data, ts) 

    def fit(self):
        """
        Default SPEI fit method:
        - Uses GLO distribution
        - Uses PWM/L-moments

        Returns
        -------
        pd.DataFrame
            'Index' and 'CDF'
        """
        return self.fit_parametric_pwm(dist_type="glo")
