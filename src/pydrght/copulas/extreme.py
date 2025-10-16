import numpy as np
from scipy.optimize import fminbound
from .base import ExtremeCopula, CopulaError
from .utils import bracket_1d

# -----------------------------
# Galambos Copula (bivariate)
# -----------------------------
class GalambosCopula(ExtremeCopula):
    """Galambos extreme-value copula (bivariate)."""

    def __init__(self, theta=None):
        self.theta = theta

    def pdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling pdf")
        alpha = self.theta
        u1, u2 = u[:, 0], u[:, 1]

        expr1 = np.log(u1)
        expr3 = u1 * u2
        expr4 = np.log(expr3)
        expr7 = expr4**2
        expr11 = expr1 / expr4
        expr12 = 1 - expr11
        expr13 = -alpha
        expr16 = expr12**expr13 + expr11**expr13
        expr18 = expr16**(2 / alpha)
        expr20 = expr1 - expr4
        expr24 = expr16**(1 / alpha)
        expr31 = 2 * alpha

        pdf_values = (
            ((expr1 * (-expr1 + expr4) / expr7)**alpha) *
            (2 * expr1 * expr18 * expr20 - expr7 + expr24 * expr4 * (1 + alpha + expr4)) +
            expr24 * (
                expr12**expr31 * expr20 * (expr1 * expr24 - expr4) +
                expr1 * expr11**expr31 * (expr24 * expr20 + expr4)
            )
        ) / (
            expr3**expr16**(-1 / alpha) * expr1 * expr18 * (expr12**alpha + expr11**alpha)**2 * expr20
        )
        return pdf_values

    def cdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling cdf")
        alpha = self.theta
        u1, u2 = u[:, 0], u[:, 1]
        expr1 = u1 * u2
        expr4 = np.log(u1) / np.log(expr1)
        expr6 = -alpha
        return expr1 ** (1 - ((1 - expr4) ** expr6 + expr4 ** expr6) ** (-1 / alpha))

    def fit(self, u):
        def negloglike(alpha):
            pdf_vals = self.__class__(alpha).pdf(u)
            pdf_vals = np.clip(pdf_vals, 1e-10, None)
            return -np.sum(np.log(pdf_vals))

        lower, upper = bracket_1d(negloglike, 0, 5)
        self.theta = fminbound(negloglike, lower, upper, xtol=1e-6, maxfun=200)
        return self

# -----------------------------
# Plackett Copula (bivariate)
# -----------------------------
class PlackettCopula(ExtremeCopula):
    """Plackett copula (bivariate)."""

    def __init__(self, theta=None):
        self.theta = theta

    def pdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling pdf")
        alpha = self.theta
        u1, u2 = u[:, 0], u[:, 1]

        if alpha == 1:
            return np.ones(len(u1))

        eta = alpha - 1
        u1_u2 = u1 + u2
        tu12 = 2 * u1 * u2
        t1 = eta * (u1_u2 * (2 + eta * u1_u2) - 2 * alpha * tu12)
        t2 = eta * (u1_u2 - tu12)
        return (1 + t1) ** (-1.5) * alpha * (1 + t2)

    def cdf(self, u):
        if self.theta is None:
            raise CopulaError("Copula must be fitted before calling cdf")
        alpha = self.theta
        u1, u2 = u[:, 0], u[:, 1]
        eta = alpha - 1.0
        if alpha == 1.0:
            return u1 * u2
        Ieuu = 1 + eta * (u1 + u2)
        return (0.5 / eta) * (Ieuu - np.sqrt(Ieuu**2 - 4 * alpha * eta * u1 * u2))

    def fit(self, u):
        def negloglike(alpha):
            pdf_vals = self.__class__(alpha).pdf(u)
            pdf_vals = np.clip(pdf_vals, 1e-10, None)
            return -np.sum(np.log(pdf_vals))

        try:
            lower, upper = bracket_1d(negloglike, 1e-3, 5)
            if not np.isfinite(lower) or not np.isfinite(upper):
            # If bracket_1d returns non-finite bounds, replace with fallback
                lower, upper = 0.01, 100
        except Exception:
            lower, upper = 0.01, 100
        
        self.theta = fminbound(negloglike, lower, upper, xtol=1e-6, maxfun=200)
        return self