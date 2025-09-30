import pytest
import pandas as pd
import numpy as np
from pydrght.di import DI

# --- Basic initialization tests ---
@pytest.mark.parametrize("ts", [1, 3, 6])
def test_di_initialization(prec, ts):
    di = DI(prec, ts=ts)
    assert isinstance(di.precip, pd.Series)
    # Aggregated length = len(prec) - ts + 1
    assert len(di.precip) == len(prec)

# --- Global deciles ---
@pytest.mark.parametrize("ts", [1, 3, 6])
def test_di_calculate(prec, ts):
    di = DI(prec, ts=ts)
    result = di.calculate()
    assert isinstance(result, pd.Series)
    assert result.name == "DI"
    # Length reduced by timescale
    assert len(result) == len(prec) - ts + 1
    # All values between 1 and 10
    assert result.dropna().between(1, 10).all()

# --- Month-wise deciles ---
@pytest.mark.parametrize("ts", [1, 3, 6])
def test_di_calculate_monthwise(prec, ts):
    di = DI(prec, ts=ts)
    result = di.calculate_monthwise()
    assert isinstance(result, pd.Series)
    assert result.name == "DI_monthwise"
    assert len(result) == len(prec) - ts + 1
    # All values between 1 and 10
    assert result.dropna().between(1, 10).all()
