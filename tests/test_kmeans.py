import pandas as pd
import pytest

from customer_intelligence.models.kmeans import train_kmeans
from customer_intelligence.evaluation.clustering_metrics import (
    evaluate_clustering,
)


def create_sample_data():

    return pd.DataFrame(
        {
            "frequency": [1, 2, 10, 12, 20, 22],
            "monetary": [100, 150, 1000, 1200, 5000, 5500],
        },
        index=["101", "102", "103", "104", "105", "106"],
    )


def test_kmeans_training():

    X = create_sample_data()

    model, labels = train_kmeans(
        X,
        n_clusters=3,
    )

    assert len(labels) == len(X)
    assert labels.nunique() == 3
    assert model.n_clusters == 3


def test_customer_index_preserved():

    X = create_sample_data()

    _, labels = train_kmeans(
        X,
        n_clusters=3,
    )

    assert list(labels.index) == list(X.index)


def test_clustering_metrics():

    X = create_sample_data()

    model, labels = train_kmeans(
        X,
        n_clusters=3,
    )

    metrics = evaluate_clustering(
        X,
        labels,
        model.inertia_,
    )

    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert "calinski_harabasz_score" in metrics


def test_invalid_cluster_count():

    X = create_sample_data()

    with pytest.raises(ValueError):

        train_kmeans(
            X,
            n_clusters=1,
        )


def test_empty_dataset():

    X = pd.DataFrame()

    with pytest.raises(ValueError):

        train_kmeans(
            X,
            n_clusters=2,
        )