
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_clustering(
    X: pd.DataFrame,
    labels: pd.Series,
    inertia: float | None = None,
) -> dict:
    """
    Calculate internal clustering evaluation metrics.
    """

    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        raise ValueError(
            "At least two clusters are required for evaluation."
        )

    if len(unique_clusters) >= len(X):
        raise ValueError(
            "Every sample cannot be its own cluster."
        )

    sample_size = min(2000, len(X))

    silhouette = silhouette_score(
        X,
        labels,
        sample_size=sample_size,
        random_state=42,
    )

    davies_bouldin = davies_bouldin_score(
        X,
        labels,
    )

    calinski_harabasz = calinski_harabasz_score(
        X,
        labels,
    )

    metrics = {
        "n_clusters": len(unique_clusters),
        "silhouette_score": float(silhouette),
        "davies_bouldin_score": float(davies_bouldin),
        "calinski_harabasz_score": float(calinski_harabasz),
    }

    if inertia is not None:
        metrics["inertia"] = float(inertia)

    return metrics
