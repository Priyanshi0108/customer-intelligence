import pandas as pd


def detect_iqr_anomalies(
    X: pd.DataFrame,
    columns: list[str] | None = None,
    multiplier: float = 1.5,
):
    if X.empty:
        raise ValueError("Input dataframe cannot be empty")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values")

    if multiplier <= 0:
        raise ValueError("Multiplier must be positive")

    if columns is None:
        columns = X.columns.tolist()

    results = pd.DataFrame(index=X.index)
    bounds = {}

    for column in columns:
        q1 = X[column].quantile(0.25)
        q3 = X[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        results[column] = (
            (X[column] < lower) |
            (X[column] > upper)
        ).astype(int)

        bounds[column] = {
            "lower": lower,
            "upper": upper,
        }

    results["is_anomaly"] = (
        results.sum(axis=1) > 0
    ).astype(int)

    return results, bounds