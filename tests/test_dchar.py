import pandas as pd
import numpy as np
import pytest
from pydrght.dchar import DChar

@pytest.fixture
def ts_numeric():
    # Numeric index instead of datetime
    data = np.array([1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 1, 2, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3])
    return pd.Series(data[:24], index=np.arange(24))

def test_dchar_basic(ts_numeric):
    dchar = DChar(ts_numeric, onset_threshold=1, recovery_threshold=3, min_drought_duration=2)
    features = dchar.calculate()
    assert "Duration" in features.columns
    assert "Severity" in features.columns
    assert "Intensity" in features.columns
    assert "Date_Ini_Ev" in features.columns
    assert "Date_Fin_Ev" in features.columns
    assert "Interarrival" in features.columns
    assert "Frequency_of_Occ" in features.columns
    assert "Recovery_Duration" in features.columns

    duration_vals = features["Duration"]


    assert all(duration_vals > 0)
    assert all(features["Severity"] >= 0)
    assert all(features["Intensity"] >= 0)


def test_dchar_no_droughts():
    ts = pd.Series(np.ones(10), index=np.arange(10))
    dchar = DChar(ts, onset_threshold=2, recovery_threshold=3, min_drought_duration=2)
    features = dchar.calculate()
    # Should return empty DataFrame
    assert features.empty

def test_dchar_short_droughts():
    ts = pd.Series([0, 1, 0, 1, 0], index=np.arange(5))  # numeric index
    dchar = DChar(ts, onset_threshold=1, recovery_threshold=2, min_drought_duration=3)
    features = dchar.calculate()
    
    # Duration column should only contain droughts >= min_drought_duration
    assert all(d >= dchar.min_drought_duration for d in features["Duration"])

def test_dchar_no_droughts():
    ts = pd.Series([10, 10, 10, 10, 10], index=np.arange(5))  # high values, no droughts
    dchar = DChar(ts, onset_threshold=1, recovery_threshold=3, min_drought_duration=2)
    features = dchar.calculate()
    
    # Either no valid droughts or all durations >= min_drought_duration
    if not features.empty:
        assert all(d >= dchar.min_drought_duration for d in features["Duration"])

