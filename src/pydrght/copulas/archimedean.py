import numpy as np
from scipy.optimize import fminbound
from .base import ArchimedeanCopula, CopulaError
from .utils import bracket_1d

class ClaytonCopula(ArchimedeanCopula):
    """Clayton copula."""

    def pdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling pdf")
        theta = self.theta
        if theta <= 0:
            raise CopulaError("Theta must be > 0 for Clayton copula")

        u = np.clip(u, 1e-10, 1-1e-10)
        powu = u ** (-theta)
        sum_pow = np.sum(powu, axis=1)
        logC = (-1 / theta) * np.log(sum_pow - 1)
        pdf = (theta + 1) * np.exp((2 * theta + 1) * logC - np.sum((theta + 1) * np.log(u), axis=1))
        return pdf

    def cdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling cdf")
        theta = self.theta
        u = np.clip(u, 0, 1)
        sum_pow = np.sum(u ** (-theta), axis=1)
        return (sum_pow - 1) ** (-1/theta)

    def fit(self, u):
        def negloglike(theta):
            powu = u ** -theta
            sum_pow = np.sum(powu, axis=1)
            logC = (-1.0 / theta) * np.log(sum_pow - 1)
            logy = np.log(theta + 1) + (2 * theta + 1) * logC - (theta + 1) * np.sum(np.log(u), axis=1)
            return -np.sum(logy)

        lower, upper = bracket_1d(negloglike, 1e-6, 5.0)
        if not np.isfinite(upper):
            raise CopulaError("Failed to bracket minimizer for Clayton copula")
        self.theta = fminbound(negloglike, lower, upper, xtol=1e-6, maxfun=200)
        return self

class FrankCopula(ArchimedeanCopula):
    """Frank copula (bivariate)."""

    def pdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling pdf")
        theta = self.theta

        u1, u2 = u[:, 0], u[:, 1]
        diff_u = u1 - u2
        cosh_term = np.cosh(theta * diff_u / 2)
        sum_u = u1 + u2
        exp_term1 = np.exp(theta * (sum_u - 2) / 2)
        exp_term2 = np.exp(-theta * sum_u / 2)

        denominator = (2 * cosh_term - exp_term1 - exp_term2) ** 2
        return theta * (1 - np.exp(-theta)) / denominator

    def cdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling cdf")
        theta = self.theta
        u = np.clip(u, 1e-10, 1-1e-10)  # avoid log(0)
        if theta == 0:
            return np.prod(u, axis=1)
        # Vectorized computation
        exp_theta_u = np.exp(-theta * u)
        sum_exp_theta_u = np.sum(exp_theta_u, axis=1)
        numerator = np.exp(-theta) + (np.exp(-theta * np.sum(u, axis=1)) - sum_exp_theta_u)
        denominator = np.expm1(-theta)  # exp(-theta) - 1, numerically stable
        p = -np.log(numerator / denominator) / theta
        return p

    def fit(self, u):
        def negloglike(theta):
            pdf_vals = self.__class__(theta).pdf(u)
            if np.any(pdf_vals <= 0):
                return np.inf
            return -np.sum(np.log(pdf_vals))

        lower, upper = bracket_1d(negloglike, 1e-6, 10.0)
        self.theta = fminbound(negloglike, lower, upper, xtol=1e-6, maxfun=500)
        return self

# -----------------------------
# Gumbel Copula
# -----------------------------
class GumbelCopula(ArchimedeanCopula):
    """Gumbel copula."""

    def pdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling pdf")
        theta = self.theta
        if theta < 1:
            raise CopulaError("Theta must be >= 1 for Gumbel copula")

        u = np.clip(u, 1e-10, 1-1e-10)
        v = -np.log(u)
        vmin, vmax = np.min(v, axis=1), np.max(v, axis=1)
        nlogC = vmax * (1 + (vmin / vmax) ** theta) ** (1/theta)
        term1 = theta - 1 + nlogC
        term2 = np.sum((theta - 1) * np.log(v) + v, axis=1)
        term3 = (1 - 2 * theta) * np.log(nlogC)
        return term1 * np.exp(-nlogC + term2 + term3)

    def cdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling cdf")
        theta = self.theta
        u = np.clip(u, 0, 1)
        v = -np.log(u)
        vmin, vmax = np.min(v, axis=1), np.max(v, axis=1)
        return np.exp(-vmax * (1 + (vmin / vmax) ** theta) ** (1/theta))

    def fit(self, u):
        def negloglike(theta):
            pdf_vals = self.__class__(theta).pdf(u)
            if np.any(pdf_vals <= 0):
                return np.inf
            return -np.sum(np.log(pdf_vals))

        lower, upper = bracket_1d(negloglike, 1.01, 5.0)
        self.theta = fminbound(negloglike, lower, upper, xtol=1e-6, maxfun=200)
        return self