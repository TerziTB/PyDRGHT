import pytest
import pandas as pd
import numpy as np
from pydrght.pet import hargreaves, thornthwaite

# --- Fixtures for synthetic temperature data ---
@pytest.fixture
def monthly_temps():
    n_months = 24
    tmin = pd.Series(np.random.uniform(5, 15, n_months))
    tmax = pd.Series(np.random.uniform(15, 30, n_months))
    tmean = (tmin + tmax) / 2
    return tmin, tmax, tmean

@pytest.fixture
def monthly_tmean_only():
    n_months = 24
    tmean = pd.Series(np.random.uniform(5, 25, n_months))
    return tmean

# --- Test Hargreaves PET ---
def test_hargreaves_pet(monthly_temps):
    tmin, tmax, tmean = monthly_temps
    latitude = 45.0
    start_date = "2000-01"
    df = hargreaves(start_date, tmin, tmax, latitude, tmean=tmean)
    assert "MonthlyPET" in df.columns
    assert all(df["MonthlyPET"] >= 0)
    assert df.shape[0] == len(tmin)

# --- Test Thornthwaite PET ---
def test_thornthwaite_pet(monthly_tmean_only):
    tmean = monthly_tmean_only
    latitude = 45.0
    start_date = "2000-01"
    df = thornthwaite(start_date, tmean, latitude)
    assert "MonthlyPET" in df.columns
    assert all(df["MonthlyPET"] >= 0)
    assert df.shape[0] == len(tmean)
