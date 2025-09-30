import pytest
import pandas as pd
import numpy as np
from pydrght.czi import CZI

# --- Basic initialization ---
@pytest.mark.parametrize("ts", [1, 3, 6])
def test_czi_initialization(prec, ts):
    czi = CZI(prec, ts=ts)
    assert isinstance(czi.precip, pd.Series)
    assert len(czi.precip) == len(prec)

# --- Compute CZI ---
@pytest.mark.parametrize("ts", [1, 3, 6])
def test_czi_calculate(prec, ts):
    czi = CZI(prec, ts=ts)
    result = czi.calculate()
    assert isinstance(result, pd.Series)
    assert result.name == f"CZI-{ts}"
    # Length of result should match aggregated series length
    assert len(result) == len(prec) - ts + 1
    # Basic sanity check: CZI values should be finite
    assert np.isfinite(result.dropna()).all()
