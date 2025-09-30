import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def prec():
    """Synthetic monthly precipitation series (no zeros/nans)."""
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    values = np.random.gamma(2, 2, size=len(dates))  # positive values
    return pd.Series(values, index=dates)

@pytest.fixture
def prec_with_zeros():
    """Synthetic series with some zeros included."""
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    values = np.random.gamma(2, 2, size=len(dates))
    values[::5] = 0  # insert zeros every 5th element
    return pd.Series(values, index=dates)

@pytest.fixture
def prec_with_nans():
    """Synthetic series with NaNs included."""
    dates = pd.date_range("2000-01-01", periods=36, freq="MS")
    values = np.random.gamma(2, 2, size=len(dates))
    values[::7] = np.nan  # insert NaN every 7th element
    return pd.Series(values, index=dates)

@pytest.fixture
def pet():
    """Synthetic monthly PET series."""
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    values = np.random.uniform(50, 150, size=len(dates))  # reasonable PET values
    return pd.Series(values, index=dates)

@pytest.fixture
def evap():
    """Synthetic monthly evapotranspiration series."""
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    values = np.random.gamma(1.5, 1.5, size=len(dates))  # positive values
    return pd.Series(values, index=dates)

@pytest.fixture
def positive_prec():
    """Positive-only series suitable for Dist fitting tests."""
    np.random.seed(123)
    dates = pd.date_range("2000-01-01", periods=100, freq="MS")
    values = np.random.gamma(2.5, 1.8, size=len(dates))
    return pd.Series(values, index=dates)

@pytest.fixture
def prec_for_dist_with_zeros():
    """Series for Dist testing with some zeros."""
    np.random.seed(123)
    dates = pd.date_range("2000-01-01", periods=100, freq="MS")
    values = np.random.gamma(2.5, 1.8, size=len(dates))
    values[::12] = 0
    return pd.Series(values, index=dates)
