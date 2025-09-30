import numpy as np
from scipy.stats import norm, multivariate_normal
from .base import EllipticalCopula, CopulaError
from .utils import is_positive_definite, nearest_positive_definite

# -----------------------------
# Gaussian Copula
# -----------------------------
class GaussianCopula(EllipticalCopula):
    """Gaussian copula."""

    def __init__(self, Rho=None):
        self.Rho = Rho  # correlation matrix

    def pdf(self, u):
        if self.Rho is None:
            raise CopulaError("Copula must be fitted before calling pdf")

        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = norm.ppf(u)
        mvn = multivariate_normal(mean=np.zeros(u.shape[1]), cov=self.Rho)
        pdf = mvn.pdf(z) / np.prod(norm.pdf(z), axis=1)
        return pdf

    def cdf(self, u):
        if self.Rho is None:
            raise CopulaError("Copula must be fitted before calling cdf")

        u = np.clip(u, 0, 1)
        z = norm.ppf(u)
        return multivariate_normal.cdf(z, mean=np.zeros(u.shape[1]), cov=self.Rho)

    def fit(self, u):
        z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        self.Rho = np.corrcoef(z, rowvar=False)
        # Ensure positive definiteness
        if not is_positive_definite(self.Rho):
            self.Rho = nearest_positive_definite(self.Rho)
        return self