import logging

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate input before applying PCA.
    """

    if df.empty:
        raise ValueError("Input dataset cannot be empty.")

    if df.shape[1] < 2:
        raise ValueError(
            "PCA requires at least two features."
        )

    if df.isnull().any().any():
        raise ValueError(
            "Missing values detected in input data."
        )

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            "Infinite values detected in input data."
        )


def fit_pca(
    df: pd.DataFrame,
    variance_threshold: float = 0.95,
) -> tuple[pd.DataFrame, PCA]:
    """
    Apply PCA while retaining the requested proportion
    of explained variance.

    Args:
        df: Scaled customer feature DataFrame.
        variance_threshold: Required cumulative explained variance.

    Returns:
        transformed_features: PCA-transformed DataFrame.
        pca: Fitted PCA model.
    """

    validate_input(df)

    if not 0 < variance_threshold <= 1:
        raise ValueError(
            "variance_threshold must be between 0 and 1."
        )

    pca = PCA(
        n_components=variance_threshold,
        svd_solver="full",
    )

    transformed_array = pca.fit_transform(df)

    component_names = [
        f"PC{i + 1}"
        for i in range(transformed_array.shape[1])
    ]

    transformed_features = pd.DataFrame(
        transformed_array,
        index=df.index,
        columns=component_names,
    )

    logger.info(
        "PCA completed. Original features: %d, "
        "Principal components: %d",
        df.shape[1],
        transformed_features.shape[1],
    )

    return transformed_features, pca


def get_explained_variance(
    pca: PCA,
) -> pd.DataFrame:
    """
    Return explained variance information.
    """

    explained_variance = pd.DataFrame({
        "component": [
            f"PC{i + 1}"
            for i in range(len(pca.explained_variance_ratio_))
        ],
        "explained_variance_ratio": (
            pca.explained_variance_ratio_
        ),
    })

    explained_variance[
        "cumulative_explained_variance"
    ] = (
        explained_variance["explained_variance_ratio"]
        .cumsum()
    )

    return explained_variance


def get_component_loadings(
    pca: PCA,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Show how strongly original features contribute
    to each principal component.
    """

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[
            f"PC{i + 1}"
            for i in range(pca.n_components_)
        ],
    )

    return loadings