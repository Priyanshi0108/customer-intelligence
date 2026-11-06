
import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor


def validate_input(X: pd.DataFrame) -> None:
    if X.empty:
        raise ValueError("Input data cannot be empty.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.")


def detect_lof_anomalies(
    X: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.05,
):
    validate_input(X)

    if n_neighbors < 2 or n_neighbors >= len(X):
        raise ValueError("n_neighbors must be between 2 and len(X)-1.")

    if not 0 < contamination <= 0.5:
        raise ValueError("Invalid contamination value.")

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric="minkowski",
        p=2,
    )

    raw_labels = model.fit_predict(X)

    # 1 = normal, -1 = anomaly
    labels = pd.Series(
        np.where(raw_labels == -1, 1, 0),
        index=X.index,
        name="is_anomaly",
    )

    # Larger score means more anomalous
    scores = pd.Series(
        -model.negative_outlier_factor_,
        index=X.index,
        name="anomaly_score",
    )

    return model, labels, scores
