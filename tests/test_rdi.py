import numpy as np
import pandas as pd
from pydrght.rdi import RDI

def test_rdi_calculate(prec, pet):
    rdi = RDI(prec, pet, ts=1)
    df = rdi.calculate()
    # Check columns exist
    assert "RDI_normalized" in df.columns
    assert "RDI_standardized" in df.columns
    # Length matches input
    assert len(df) == len(prec)
    # No NaNs for nonzero PET values
    assert df["RDI_normalized"].notna().all()

def test_rdi_calculate_monthwise(prec, pet):
    rdi = RDI(prec, pet, ts=1)
    df = rdi.calculate_monthwise()
    assert "RDI_normalized_month" in df.columns
    assert "RDI_standardized_month" in df.columns
    assert len(df) == len(prec)

def test_rdi_with_timescale(prec, pet):
    ts = 3  # 3-month accumulation
    rdi = RDI(prec, pet, ts=ts)
    df_global = rdi.calculate()
    df_monthwise = rdi.calculate_monthwise()

    # Check columns
    assert "RDI_normalized" in df_global.columns
    assert "RDI_standardized" in df_global.columns
    assert "RDI_normalized_month" in df_monthwise.columns
    assert "RDI_standardized_month" in df_monthwise.columns

    # Check length adjusted for timescale
    assert len(df_global) == len(prec) - ts + 1
    assert len(df_monthwise) == len(prec) - ts + 1

    # Check no NaNs for normalized values if PET nonzero
    assert df_global["RDI_normalized"].notna().all()
    assert df_monthwise["RDI_normalized_month"].notna().all()