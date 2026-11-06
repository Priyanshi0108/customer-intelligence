import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering


def validate_input(X: pd.DataFrame) -> None:

    if X.empty:
        raise ValueError("Input data cannot be empty.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.")


def train_hierarchical(
    X: pd.DataFrame,
    n_clusters: int,
    linkage: str = "ward",
) -> tuple[AgglomerativeClustering, pd.Series]:

    validate_input(X)

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters >= len(X):
        raise ValueError(
            "n_clusters must be smaller than the number of samples."
        )

    allowed_linkages = {
        "ward",
        "complete",
        "average",
    }

    if linkage not in allowed_linkages:
        raise ValueError(
            f"Unsupported linkage: {linkage}. "
            f"Choose from {allowed_linkages}."
        )

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean",
    )

    labels = model.fit_predict(X)

    cluster_labels = pd.Series(
        labels,
        index=X.index,
        name="cluster",
    )

    return model, cluster_labels