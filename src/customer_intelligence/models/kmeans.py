import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


def validate_input(X:pd.DataFrame) -> None:

    """Validate input data before clustering"""

    if X.empty:
        raise ValueError("Input data is not empty")


    if X.shape[1] < 1:
        raise ValueError("At least one feature is required.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.")


def train_kmeans(
        X:pd.DataFrame,
        n_clusters:int,
        random_state:int=42,
        n_init:int =20
)->tuple[KMeans,pd.Series]:

    """
    Train K-Means and return the fitted model and cluster labels.
    """

    validate_input(X)

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters >= len(X):
        raise ValueError(
            "n_clusters must be smaller than the number of samples."
        )

    model = KMeans(
        n_clusters = n_clusters,
        init = "k-means++",
        n_init = n_init,
        random_state= random_state,
        max_iter= 300
    )


    label = model.fit_predict(X)

    cluster_labels = pd.Series(
        label,
        index = X.index,
        name = "cluster"
    )


    return model , cluster_labels