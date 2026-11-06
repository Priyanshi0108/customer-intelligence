import pandas as pd
import pytest

from customer_intelligence.models.dbscan import train_dbscan


def create_sample_data():

    return pd.DataFrame(
        {
            "feature_1": [
                0.0, 0.1, 0.2,
                5.0, 5.1, 5.2,
                20.0,
            ],
            "feature_2": [
                0.0, 0.1, 0.2,
                5.0, 5.1, 5.2,
                20.0,
            ],
        }
    )


def test_dbscan_training():

    X = create_sample_data()

    model, labels = train_dbscan(
        X,
        eps=0.5,
        min_samples=2,
    )

    assert len(labels) == len(X)

    assert labels.nunique() >= 2

    assert model.eps == 0.5


def test_customer_index_preserved():

    X = create_sample_data()

    X.index = ["A", "B", "C", "D", "E", "F", "G"]

    _, labels = train_dbscan(
        X,
        eps=0.5,
        min_samples=2,
    )

    assert list(labels.index) == list(X.index)


def test_invalid_eps():

    X = create_sample_data()

    with pytest.raises(ValueError):

        train_dbscan(
            X,
            eps=0,
            min_samples=2,
        )


def test_invalid_min_samples():

    X = create_sample_data()

    with pytest.raises(ValueError):

        train_dbscan(
            X,
            eps=0.5,
            min_samples=1,
        )


def test_empty_dataset():

    X = pd.DataFrame()

    with pytest.raises(ValueError):

        train_dbscan(
            X,
            eps=0.5,
            min_samples=2,
        )