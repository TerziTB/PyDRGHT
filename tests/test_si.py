import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm, expon
from pydrght.si import SI
import warnings

# --- Basic initialization tests ---
def test_si_initialization(prec):
    si = SI(prec, ts=1)
    assert isinstance(si.aggregated, pd.Series)
    assert len(si.aggregated) == len(prec)

# --- Parametric fitting tests ---
def test_fit_parametric_basic(prec):
    si = SI(prec, ts=1)
    result = si.fit_parametric(lognorm)
    assert "Index" in result.columns
    assert "CDF" in result.columns
    assert len(result) == len(prec)

def test_fit_parametric_2p_option(prec):
    si = SI(prec, ts=1)
    result = si.fit_parametric(expon, is_2p=True)
    assert "Index" in result.columns
    assert "CDF" in result.columns
    assert len(result) == len(prec)

def test_fit_parametric_with_zeros(prec_with_zeros):
    si = SI(prec_with_zeros, ts=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = si.fit_parametric(lognorm)
    assert "Index" in result.columns
    assert "CDF" in result.columns
    # Ensure that CDF values are between 0 and 1
    assert result["CDF"].dropna().between(0, 1).all()

# --- Empirical fitting tests ---
@pytest.mark.parametrize("method", ["Gringorten", "Weibull"])
def test_fit_empirical_methods(prec, method):
    si = SI(prec, ts=1)
    result = si.fit_empirical(method=method)
    assert "Index" in result.columns
    assert "CDF" in result.columns
    assert len(result) == len(prec)
    # Ensure that CDF values are between 0 and 1
    assert result["CDF"].dropna().between(0, 1).all()

def test_fit_empirical_with_nans(prec_with_nans):
    si = SI(prec_with_nans, ts=1)
    result = si.fit_empirical()
    assert "Index" in result.columns
    assert "CDF" in result.columns
    assert len(result) == len(prec_with_nans)
    # CDF values should be between 0 and 1, ignoring NaNs
    assert result["CDF"].dropna().between(0, 1).all()

# --- Tests with timescale aggregation ---
@pytest.mark.parametrize("ts", [3, 6, 12])
def test_si_initialization_with_timescale(prec, ts):
    si = SI(prec, ts=ts)
    assert isinstance(si.aggregated, pd.Series)
    # Aggregated length = len(prec) - ts + 1
    assert len(si.aggregated) == len(prec) - ts + 1

@pytest.mark.parametrize("ts", [3, 6, 12])
def test_fit_parametric_with_timescale(prec, ts):
    si = SI(prec, ts=ts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = si.fit_parametric(lognorm)
    assert "Index" in result.columns
    assert "CDF" in result.columns
    assert len(result) == len(prec) - ts + 1
    # CDF values between 0 and 1
    assert result["CDF"].dropna().between(0, 1).all()

@pytest.mark.parametrize("ts", [3, 6, 12])
@pytest.mark.parametrize("method", ["Gringorten", "Weibull"])
def test_fit_empirical_with_timescale(prec, ts, method):
    si = SI(prec, ts=ts)
    result = si.fit_empirical(method=method)
    assert "Index" in result.columns
    assert "CDF" in result.columns
    assert len(result) == len(prec) - ts + 1
    # CDF values between 0 and 1
    assert result["CDF"].dropna().between(0, 1).all()
