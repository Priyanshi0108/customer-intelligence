
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


def detect_anomalies(
    X: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
):
    """Fit Isolation Forest and return anomaly labels and scores."""

    if X.empty:
        raise ValueError("Input dataset cannot be empty.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.")

    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )

    model.fit(X)

    # sklearn: 1 = inlier, -1 = anomaly
    raw_labels = model.predict(X)

    labels = pd.Series(
        np.where(raw_labels == -1, 1, 0),
        index=X.index,
        name="is_anomaly",
    )

    # Higher values mean more anomalous.
    anomaly_scores = pd.Series(
        -model.score_samples(X),
        index=X.index,
        name="anomaly_score",
    )

    return model, labels, anomaly_scores
