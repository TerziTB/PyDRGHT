# tests/test_rai.py
import numpy as np
import pandas as pd
import pytest
from pydrght.rai import RAI

def test_rai_basic(prec):
    rai = RAI(prec, ts=1)
    result = rai.RAI()
    # Check it's a pandas Series
    assert isinstance(result, pd.Series)
    # Length should match input
    assert len(result) == len(prec)
    # There should be some non-NaN values
    assert result.notna().any()

def test_rai_basic(prec):
    ts = 1
    rai = RAI(prec, ts=ts)
    result = rai.RAI()
    assert isinstance(result, pd.Series)
    assert len(result) == len(prec) - ts + 1
    assert result.notna().any()

def test_mrai_basic(prec):
    ts = 1
    rai = RAI(prec, ts=ts)
    result = rai.mRAI()
    assert isinstance(result, pd.Series)
    assert len(result) == len(prec) - ts + 1
    assert result.notna().any()

def test_rai_with_timescale(prec):
    ts = 3
    rai = RAI(prec, ts=ts)
    result = rai.RAI()
    assert isinstance(result, pd.Series)
    assert len(result) == len(prec) - ts + 1

def test_mrai_with_timescale(prec):
    ts = 3
    rai = RAI(prec, ts=ts)
    result = rai.mRAI()
    assert isinstance(result, pd.Series)
    assert len(result) == len(prec) - ts + 1

