import pytest
import numpy as np
import pandas as pd
from scipy.stats import lognorm, gamma
from pydrght.dist import Dist
from pydrght.bfa import BFA
from pydrght.copulas import FrankCopula

# --- Fixtures for marginal distributions ---
@pytest.fixture
def dist_x():
    data = pd.Series(np.random.gamma(2, 1, size=100))
    return Dist(data, gamma, floc0=True)

@pytest.fixture
def dist_y():
    data = pd.Series(np.random.lognormal(mean=0.5, sigma=0.8, size=100))
    return Dist(data, lognorm, floc0=True)

# --- Basic BFA initialization ---
def test_bfa_initialization(dist_x, dist_y):
    bfa = BFA(dist_x, dist_y, FrankCopula)
    assert hasattr(bfa, "copula")
    assert isinstance(bfa.data_combined, np.ndarray)
    assert bfa.data_combined.shape[1] == 2

# --- Test joint return periods ---
def test_joint_return_period(dist_x, dist_y):
    bfa = BFA(dist_x, dist_y, FrankCopula)
    T = 50  # 50-year return period
    result = bfa.joint_return_period(T)
    assert "OR" in result.index
    assert "AND" in result.index
    # Probabilities scaled to interarrival
    assert all(result.values > 0)

# --- Invalid return period ---
def test_joint_return_period_invalid(dist_x, dist_y):
    bfa = BFA(dist_x, dist_y, FrankCopula)
    with pytest.raises(ValueError):
        bfa.joint_return_period(0)
