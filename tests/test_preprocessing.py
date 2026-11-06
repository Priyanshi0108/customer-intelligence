import numpy as np
import pandas as pd
import pytest

from customer_intelligence.preprocessing.transform import (
    fit_preprocessor,
)

def create_sample_features():

    return pd.DataFrame(
        {
            "recency_days": [5, 10, 20, 30],
            "frequency": [1, 2, 5, 10],
            "monetary": [100, 500, 1000, 5000],
        },
        index=["101", "102", "103", "104"],
    )


def test_standard_scaler():
    df = create_sample_features()

    scaled,_ = fit_preprocessor(
        df, 
        scaler_name= "standard"
    )


    assert np.allclose(
        scaled.mean().values,
        0,
        atol = 1e-7,
    )

    assert list(scaled.index) == list(df.index)


def test_minmax_scaler():
    
    df = create_sample_features()

    scaled, _ = fit_preprocessor(
        df,
        scaler_name="minmax",
    )

    assert np.allclose(
        scaled.min().values,
        0,
    )  


    assert np.allclose(
        scaled.max().values,
        1,
    )  


def test_log_transformation():

    df = create_sample_features()

    scaled, _ = fit_preprocessor(
        df,
        scaler_name="standard",
        log_features=["monetary"],
    )

    assert scaled.shape == df.shape


def test_negative_values_rejected_for_log():

    df = create_sample_features()

    df.loc["101","monetary"] = -100

    with pytest.raises(ValueError):
        fit_preprocessor(
            df,
            scaler_name= "standard",
            log_features= ["monetary"]
        )



def test_invalid_scaler():
    
    df = create_sample_features()

    with pytest.raises(ValueError):
        fit_preprocessor(
            df,
            scaler_name="unknown",
        )


def __init__(self,columns:list[str] | None = None):
    self.columns = columns