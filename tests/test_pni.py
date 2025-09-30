import pytest
import pandas as pd
from pydrght.pni import PNI

def test_pni_global(prec):
    ts = 1
    pni = PNI(prec, ts=ts)
    result = pni.calculate()
    
    # Check output type and name
    assert isinstance(result, pd.Series)
    assert result.name == "PNI_global"
    
    # Length should match aggregated series
    assert len(result) == len(prec) - ts + 1
    
    # Values around 100 (rough check)
    assert result.mean() - 100 < 1e-6

def test_pni_monthwise(prec):
    ts = 1
    pni = PNI(prec, ts=ts)
    result = pni.calculate_monthwise()
    
    # Check output type and name
    assert isinstance(result, pd.Series)
    assert result.name == "PNI_monthwise"
    
    # Length matches aggregated series
    assert len(result) == len(prec) - ts + 1
    
    # Each month's mean should be ~100
    months = (range(len(result)))
    for m in range(1, 13):
        month_vals = result.iloc[m-1::12]
        if not month_vals.empty:
            assert abs(month_vals.mean() - 100) < 1e-6

def test_pni_with_timescale(prec):
    ts = 3
    pni = PNI(prec, ts=ts)
    
    global_result = pni.calculate()
    monthwise_result = pni.calculate_monthwise()
    
    # Check length adjusted for timescale
    assert len(global_result) == len(prec) - ts + 1
    assert len(monthwise_result) == len(prec) - ts + 1
