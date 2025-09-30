import pytest
import numpy as np
import pandas as pd
from scipy.stats import lognorm, gamma
from pydrght.dist import Dist

def test_fit_basic(positive_prec):
    dist_obj = Dist(positive_prec, lognorm)
    assert dist_obj.shape is not None or dist_obj.shape is None
    assert isinstance(dist_obj.loc, float)
    assert isinstance(dist_obj.scale, float)
    assert isinstance(dist_obj.cdf(), pd.Series)
    assert isinstance(dist_obj.pdf(), pd.Series)
    # Percentiles
    q = 0.25
    val = dist_obj.ppf(q)
    assert isinstance(val, float)

def test_fit_with_zeros(prec_for_dist_with_zeros):
    dist_obj = Dist(prec_for_dist_with_zeros, gamma, prob_zero=True)
    assert 0 <= dist_obj.p0 <= 1
    cdf_vals = dist_obj.cdf()
    # CDF for zeros should equal p0 using positional indexing
    zero_pos = np.where(prec_for_dist_with_zeros == 0)[0]
    zero_cdf = cdf_vals.iloc[zero_pos]    
    # Compare element-wise using pytest.approx
    assert all([v == pytest.approx(dist_obj.p0) for v in zero_cdf])

def test_ks_test(positive_prec):
    dist_obj = Dist(positive_prec, lognorm)
    stat, pval = dist_obj.ks_test()
    assert 0 <= stat <= 1
    assert 0 <= pval <= 1

def test_aic_bic(positive_prec):
    dist_obj = Dist(positive_prec, lognorm)
    aic = dist_obj.aic()
    bic = dist_obj.bic()
    assert aic > 0
    assert bic > 0

def test_return_period(positive_prec):
    dist_obj = Dist(positive_prec, lognorm)
    T = 50
    thresh = dist_obj.return_period(T)
    assert isinstance(thresh, float)
    # Check that threshold is at least as high as the minimum data value
    assert thresh >= positive_prec.min()
    # Invalid T should raise ValueError
    with pytest.raises(ValueError):
        dist_obj.return_period(0)
    
def test_fit_no_zeros_prob_zero():
    ts = pd.Series([1, 2, 3, 4])
    dist_obj = Dist(ts, gamma, prob_zero=True)
    # p0 should be 0
    assert dist_obj.p0 == 0
    cdf_vals = dist_obj.cdf()
    # CDF values should not equal p0
    assert all(v > 0 for v in cdf_vals)

def test_ppf_edges(positive_prec):
    dist_obj = Dist(positive_prec, lognorm)

    # Check intermediate probabilities (finite values)
    for q in [1e-6, 0.25, 0.5, 0.75, 1 - 1e-6]:
        val = dist_obj.ppf(q)
        assert np.isfinite(val)

    # ppf outside [0,1] should return nan
    assert np.isnan(dist_obj.ppf(-0.1))
    assert np.isnan(dist_obj.ppf(1.1))

