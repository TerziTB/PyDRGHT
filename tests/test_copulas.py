import numpy as np
import math
import pytest
from pydrght.copulas import (
    ClaytonCopula, FrankCopula, GumbelCopula,
    GaussianCopula, GalambosCopula, PlackettCopula
)
from pydrght.copulas.base import BaseCopula, CopulaError
from pydrght.copulas.utils import is_positive_definite, nearest_positive_definite, bracket_1d

# Dummy uniform data
u_data = np.random.rand(20, 2)

# -----------------------------
# BaseCopula
# -----------------------------
def test_base_copula_repr():
    cop = BaseCopula(theta=1.5)
    assert repr(cop) == "BaseCopula(theta=1.5)"

def test_base_copula_methods_not_implemented():
    cop = BaseCopula()
    u = np.random.rand(5, 2)
    
    with pytest.raises(NotImplementedError):
        cop.fit(u)
    with pytest.raises(NotImplementedError):
        cop.pdf(u)
    with pytest.raises(NotImplementedError):
        cop.cdf(u)

# -----------------------------
# Test initialization and fit
# -----------------------------
@pytest.mark.parametrize("copula_cls", [
    ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, GalambosCopula, PlackettCopula
])
def test_copula_fit_and_eval(copula_cls):
    copula = copula_cls()
    
    # Before fitting, calling pdf/cdf should raise
    with pytest.raises(CopulaError):
        copula.pdf(u_data)
    with pytest.raises(CopulaError):
        copula.cdf(u_data)
    
    # Fit copula
    copula.fit(u_data)
    
    # After fitting, pdf and cdf should work
    pdf_vals = copula.pdf(u_data)
    cdf_vals = copula.cdf(u_data)
    
    assert np.all(np.isfinite(pdf_vals))
    assert np.all(np.isfinite(cdf_vals))

# -----------------------------
# Test parameter edge cases
# -----------------------------
def test_clayton_invalid_theta():
    copula = ClaytonCopula()
    copula.fit(u_data)
    copula.theta = -1  # invalid
    with pytest.raises(CopulaError):
        copula.pdf(u_data)

def test_gumbel_invalid_theta():
    copula = GumbelCopula()
    copula.fit(u_data)
    copula.theta = 0.5  # invalid
    with pytest.raises(CopulaError):
        copula.pdf(u_data)

# -----------------------------
# is_positive_definite
# -----------------------------
def test_is_positive_definite():
    A = np.array([[2, -1], [-1, 2]])
    B = np.array([[0, 1], [1, 0]])
    
    assert is_positive_definite(A) is True
    assert is_positive_definite(B) is False

# -----------------------------
# nearest_positive_definite
# -----------------------------
def test_nearest_positive_definite():
    B = np.array([[0, 1], [1, 0]])
    B_pd = nearest_positive_definite(B)
    
    # Result must be symmetric
    assert np.allclose(B_pd, B_pd.T)
    # Result must be positive definite
    assert is_positive_definite(B_pd)
    
def test_nearest_positive_definite_already_pd():
    A = np.array([[2, -1], [-1, 2]])
    A_pd = nearest_positive_definite(A)
    # Already PD → should not change much
    assert np.allclose(A, A_pd)

def test_nearest_positive_definite_loop():
    # Construct a matrix that is definitely not PD
    B = np.array([[0, 10], [10, 0]])
    B_pd = nearest_positive_definite(B)
    assert is_positive_definite(B_pd)

# -----------------------------
# bracket_1d
# -----------------------------
def test_bracket_1d_basic():
    # Simple convex function with minimum at x=1
    f = lambda x: (x-1)**2
    near, far = bracket_1d(f, near_bnd=0.5, far_start=2)
    
    assert near <= 1 <= far
    assert np.isfinite(far)

def test_bracket_1d_far_nan():
    # Function decreases slowly → far bound exceeds upper_lim → should return NaN
    f = lambda x: -x  # keeps decreasing
    near, far = bracket_1d(f, near_bnd=0, far_start=1)
    assert math.isnan(far)