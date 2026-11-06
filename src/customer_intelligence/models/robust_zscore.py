import numpy as np
import pandas as pd


def detect_robust_zscore(
    X: pd.DataFrame,
    threshold: float = 3.5,
):
    if X.empty:
        raise ValueError("Input cannot be empty")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values")

    if threshold <= 0:
        raise ValueError("Threshold must be positive")

    median = X.median()

    mad = X.sub(median).abs().median()

    # Avoid division by zero
    mad = mad.replace(0, np.nan)

    robust_z = (
        0.6745 * X.sub(median)
    ).div(mad)

    robust_z = robust_z.fillna(0)

    feature_flags = (
        robust_z.abs() > threshold
    ).astype(int)

    feature_flags["is_anomaly"] = (
        feature_flags.sum(axis=1) > 0
    ).astype(int)

    return robust_z, feature_flags, median, mad