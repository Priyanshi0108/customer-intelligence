import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture


def validate_input(X: pd.DataFrame) -> None:

    if X.empty:
        raise ValueError("Input dataset cannot be empty.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.")


def train_gmm(
    X: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    random_state: int = 42,
):

    validate_input(X)

    if n_components < 2 or n_components >= len(X):
        raise ValueError("Invalid number of components.")

    allowed_types = {
        "full",
        "tied",
        "diag",
        "spherical",
    }

    if covariance_type not in allowed_types:
        raise ValueError("Invalid covariance type.")

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=3,
        max_iter=200,
        reg_covar=1e-6,
    )

    model.fit(X)

    labels = model.predict(X)

    probabilities = model.predict_proba(X)

    labels = pd.Series(
        labels,
        index=X.index,
        name="cluster",
    )

    probabilities = pd.DataFrame(
        probabilities,
        index=X.index,
        columns=[
            f"cluster_{i}_probability"
            for i in range(n_components)
        ],
    )

    return model, labels, probabilities