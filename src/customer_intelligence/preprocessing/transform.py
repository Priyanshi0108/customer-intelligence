from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.pipeline import Pipeline


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log1p transformation to selected columns only.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)

        transformed = X.copy()

        for column in self.columns:
            if (transformed[column] < 0).any():
                raise ValueError(
                    f"Cannot apply log1p to negative values "
                    f"in column: {column}"
                )

            transformed[column] = np.log1p(
                transformed[column]
            )

        return transformed

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = set(self.columns) - set(X.columns)

        if missing:
            raise ValueError(
                f"Log transformation columns not found: {missing}"
            )


def get_scaler(scaler_name: str):
    """
    Return the requested sklearn scaler.
    """

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    if scaler_name not in scalers:
        raise ValueError(
            f"Unsupported scaler: {scaler_name}. "
            f"Choose from {list(scalers)}."
        )

    return scalers[scaler_name]


def validate_features(df: pd.DataFrame) -> None:
    """
    Validate customer-level numerical features.
    """

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate customer IDs found.")

    if df.isnull().any().any():
        raise ValueError("Missing values found in features.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in df.dtypes
    ):
        raise ValueError("All clustering features must be numeric.")

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Infinite values found in features.")


def build_preprocessor(
    scaler_name: str,
    log_features: list[str] | None = None,
) -> Pipeline:
    """
    Build a reusable preprocessing pipeline.
    """

    return Pipeline(
        steps=[
            (
                "log",
                LogTransformer(columns=log_features),
            ),
            (
                "scaler",
                get_scaler(scaler_name),
            ),
        ]
    )


def fit_preprocessor(
    df: pd.DataFrame,
    scaler_name: str,
    log_features: list[str] | None = None,
) -> tuple[pd.DataFrame, Pipeline]:
    """
    Fit preprocessing pipeline and transform customer features.

    Returns:
        scaled_features: transformed DataFrame
        preprocessor: fitted sklearn Pipeline
    """

    validate_features(df)

    preprocessor = build_preprocessor(
        scaler_name=scaler_name,
        log_features=log_features,
    )

    transformed_array = preprocessor.fit_transform(df)

    scaled_features = pd.DataFrame(
        transformed_array,
        index=df.index,
        columns=df.columns,
    )

    return scaled_features, preprocessor