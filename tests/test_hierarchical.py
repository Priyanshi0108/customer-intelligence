import pandas as pd
import pytest

from customer_intelligence.models.hierarchical import (
    train_hierarchical,
)


def create_sample_data():

    return pd.DataFrame(
        {
            "feature_1": [
                0.0, 0.1, 0.2,
                5.0, 5.1, 5.2,
                10.0, 10.1, 10.2,
            ],
            "feature_2": [
                0.0, 0.1, 0.2,
                5.0, 5.1, 5.2,
                10.0, 10.1, 10.2,
            ],
        },
        index=[
            "A", "B", "C",
            "D", "E", "F",
            "G", "H", "I",
        ],
    )


def test_hierarchical_training():

    X = create_sample_data()

    model, labels = train_hierarchical(
        X,
        n_clusters=3,
        linkage="ward",
    )

    assert len(labels) == len(X)

    assert labels.nunique() == 3

    assert model.n_clusters == 3


def test_customer_index_preserved():

    X = create_sample_data()

    _, labels = train_hierarchical(
        X,
        n_clusters=3,
    )

    assert list(labels.index) == list(X.index)


def test_invalid_cluster_count():

    X = create_sample_data()

    with pytest.raises(ValueError):

        train_hierarchical(
            X,
            n_clusters=1,
        )


def test_invalid_linkage():

    X = create_sample_data()

    with pytest.raises(ValueError):

        train_hierarchical(
            X,
            n_clusters=3,
            linkage="invalid",
        )


def test_empty_dataset():

    X = pd.DataFrame()

    with pytest.raises(ValueError):

        train_hierarchical(
            X,
            n_clusters=2,
        )