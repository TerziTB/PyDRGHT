import pytest
import math
import numpy as np
import pandas as pd
from pydrght.utils.utils import uni_emp, accu, multi_emp
from pydrght.pet.utils import (
    monthly_mean_daylen, solar_declination, month_lengths,
    mid_month_days, tan_lat_rad, inverse_rel_dist,
    extraterrestrial_radiation, sunset_hour_angle
)

# -------------------------
# Tests for uni_emp
# -------------------------
def test_uni_emp_basic():
    data = pd.Series([1, 2, 3, 4])
    cdf = uni_emp(data)
    assert isinstance(cdf, pd.Series)
    assert all((cdf > 0) & (cdf < 1))  # values strictly between 0 and 1

def test_uni_emp_methods():
    data = [5, 10, 15, 20]
    cdf_g = uni_emp(data, method='Gringorten')
    cdf_w = uni_emp(data, method='Weibull')
    assert not cdf_g.equals(cdf_w)  # different formulas give different results

def test_uni_emp_invalid_method():
    with pytest.raises(ValueError):
        uni_emp([1,2,3], method='invalid')

def test_uni_emp_with_nan():
    data = pd.Series([1, np.nan, 2, 3])
    cdf = uni_emp(data)
    assert len(cdf) == 3  # nan should be dropped

# -------------------------
# Tests for accu
# -------------------------
def test_accu_basic():
    data = pd.Series(np.arange(1, 7))  # 1..6
    acc = accu(data, ts=3)
    assert isinstance(acc, pd.Series)
    assert len(acc) == len(data) - 3 + 1
    assert np.allclose(acc.values, [2.0,3.0,4.0,5.0])

def test_accu_short_data():
    data = pd.Series([1, 2])
    with pytest.raises(ValueError):
        accu(data, ts=3)

def test_accu_invalid_ts():
    data = pd.Series([1,2,3])
    with pytest.raises(ValueError):
        accu(data, ts=0)

def test_accu_array_input():
    data = np.array([1,2,3,4])
    acc = accu(data, ts=2)
    assert isinstance(acc, pd.Series)
    assert len(acc) == len(data) - 2 + 1

# -------------------------
# Tests for multi_emp
# -------------------------
def test_multi_emp_basic():
    X = pd.Series([1,2,3])
    Y = pd.Series([3,2,1])
    p = multi_emp(X,Y)
    assert isinstance(p, pd.Series)
    assert all((p > 0) & (p < 1))

def test_multi_emp_methods():
    X = [1,2,3]
    Y = [3,2,1]
    p_g = multi_emp(X,Y, method='Gringorten')
    p_w = multi_emp(X,Y, method='Weibull')
    assert not p_g.equals(p_w)

def test_multi_emp_invalid_method():
    with pytest.raises(ValueError):
        multi_emp([1,2],[3,4], method='invalid')

def test_multi_emp_with_nan():
    X = pd.Series([1,2,np.nan,3])
    Y = pd.Series([3,2,1, np.nan])
    p = multi_emp(X,Y)
    assert len(p) == 2  # rows with NaN removed


# -------------------------
# monthly_mean_daylen
# -------------------------
@pytest.mark.parametrize("lat, month", [
    (0, 1), (45, 6), (-45, 12), (90, 6), (-90, 12)
])
def test_monthly_mean_daylen(lat, month):
    val = monthly_mean_daylen(lat, month)
    assert 0 <= val <= 24

# -------------------------
# solar_declination
# -------------------------
@pytest.mark.parametrize("day", [1, 100, 200, 365])
def test_solar_declination(day):
    val = solar_declination(day)
    assert isinstance(val, float)
    assert -0.5 <= val <= 0.5  # reasonable range in radians

# -------------------------
# month_lengths
# -------------------------
def test_month_lengths():
    lengths = month_lengths(2024, 1, 12)
    assert len(lengths) == 12
    # February in leap year
    assert lengths[1] == 29
    # sum of lengths matches 366 for leap year
    assert sum(lengths) == 366

# -------------------------
# mid_month_days
# -------------------------
def test_mid_month_days():
    mids = mid_month_days()
    assert len(mids) == 12
    assert all(isinstance(d, (int, np.integer)) for d in mids)

# -------------------------
# tan_lat_rad
# -------------------------
def test_tan_lat_rad():
    val = tan_lat_rad(45)
    assert isinstance(val, float)

# -------------------------
# inverse_rel_dist
# -------------------------
@pytest.mark.parametrize("day", [1, 100, 200, 365])
def test_inverse_rel_dist(day):
    val = inverse_rel_dist(day)
    assert isinstance(val, float)
    assert 0.9 <= val <= 1.1

# -------------------------
# extraterrestrial_radiation
# -------------------------
def test_extraterrestrial_radiation():
    lat_rad = math.radians(45)
    sol_dec = 0.2
    sha = 1.0
    ird = 1.0
    ra = extraterrestrial_radiation(lat_rad, sol_dec, sha, ird)
    assert ra >= 0

# -------------------------
# sunset_hour_angle
# -------------------------
@pytest.mark.parametrize("lat_rad, sol_dec", [
    (0, 0), (math.radians(45), 0.2), (math.radians(-45), -0.2),
    (math.radians(90), 0.5), (math.radians(-90), -0.5)
])
def test_sunset_hour_angle(lat_rad, sol_dec):
    val = sunset_hour_angle(lat_rad, sol_dec)
    assert 0 <= val <= math.pi