import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN


def validate_input(X: pd.DataFrame) -> None:
    if X.empty:
        raise ValueError("Input data cannot be empty.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.")


def train_dbscan(
        X:pd.DataFrame,
        eps:float,
        min_samples:int,
) -> tuple[DBSCAN, pd.Series]:
    validate_input(X)

    if eps <= 0:
            raise ValueError("eps must be greater than zero.")

    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")

    model = DBSCAN(
         eps=eps,
         min_samples=min_samples,
         metric = "euclidean"
    )

    labels = model.fit_predict(X)

    cluster_labels = pd.Series(
        labels,
        index = X.index,
        name = "cluster"
    )


    return model, cluster_labels

