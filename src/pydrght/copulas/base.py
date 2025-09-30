import numpy as np

class CopulaError(Exception):
    """Custom exception for copula errors."""
    pass

class BaseCopula:
    """Abstract base class for all copulas."""

    def __init__(self, theta=None):
        self.theta = theta

    def fit(self, u: np.ndarray):
        """Estimate copula parameters from data."""
        raise NotImplementedError

    def pdf(self, u: np.ndarray):
        """Copula density function."""
        raise NotImplementedError

    def cdf(self, u: np.ndarray):
        """Copula cumulative distribution function."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(theta={self.theta})"


class ArchimedeanCopula(BaseCopula):
    """Base class for Archimedean copulas (Clayton, Frank, Gumbel)."""
    family = "archimedean"

    def __init__(self, theta=None):
        super().__init__(theta)


class EllipticalCopula(BaseCopula):
    """Base class for Elliptical copulas (Gaussian, t)."""
    family = "elliptical"

    def __init__(self, theta=None, nu=None):
        super().__init__(theta)
        self.nu = nu  # only relevant for t-Copula


class ExtremeCopula(BaseCopula):
    """Base class for Extreme-value copulas (Galambos, Plackett)."""
    family = "extreme"

    def __init__(self, theta=None):
        super().__init__(theta)
