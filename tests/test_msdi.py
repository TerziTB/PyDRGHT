import pytest
import pandas as pd
import numpy as np
from scipy.stats import norm
from pydrght.msdi import MSDI

@pytest.mark.parametrize("ts", [1, 3, 6])
def test_msdi_initialization(prec, evap, ts):
    msdi = MSDI(prec, evap, ts=ts)
    assert isinstance(msdi.XA, pd.Series)
    assert len(msdi.XA) == len(prec) - ts + 1
    assert isinstance(msdi.YA, pd.Series)
    assert len(msdi.YA) == len(evap) - ts + 1

@pytest.mark.parametrize("copula_family", ["gaussian", "clayton", "frank", "gumbel"])
def test_msdi_parametric(prec, evap, copula_family):
    msdi = MSDI(prec, evap, ts=1)
    msdi_values = msdi.parametric(copula_family=copula_family)
    assert isinstance(msdi_values, pd.Series)
    assert len(msdi_values) == len(msdi.XA)
    # Check for finite MSDI values
    assert np.isfinite(msdi_values.dropna()).all()
    # Metrics dictionary exists
    assert hasattr(msdi, "metrics")
    assert all(k in msdi.metrics for k in ["AIC", "BIC", "NLL"])

def test_msdi_empirical(prec, evap):
    msdi = MSDI(prec, evap, ts=1)
    msdi_values = msdi.empirical()
    assert isinstance(msdi_values, pd.Series)
    assert len(msdi_values) == len(msdi.XA)
    assert np.isfinite(msdi_values.dropna()).all()
