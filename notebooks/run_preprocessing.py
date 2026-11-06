from pathlib import Path

import joblib
import pandas as pd

from customer_intelligence.preprocessing.transform import (
    fit_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

STATISTICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "eda"
    / "feature_statistics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessors"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def select_log_features(
    features: pd.DataFrame,
) -> list[str]:
    """
    Select non-negative features with skewness > 1.
    """

    if not STATISTICS_PATH.exists():
        raise FileNotFoundError(
            "EDA statistics not found. Run EDA first."
        )

    statistics = pd.read_csv(
        STATISTICS_PATH,
        index_col=0,
    )

    skewed_features = statistics[
        statistics["skewness"] > 1
    ].index.tolist()

    log_features = []

    for column in skewed_features:

        if column not in features.columns:
            continue

        if (features[column] < 0).any():
            continue

        log_features.append(column)

    return log_features


def main():

    print("Loading customer features...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_PATH}"
        )

    features = pd.read_csv(
        INPUT_PATH,
        index_col=0,
    )

    features.index.name = "CustomerID"

    print("Original dataset shape:", features.shape)

    log_features = select_log_features(features)

    print("\nSelected log-transformed features:")
    print(log_features)

    scaler_names = [
        "standard",
        "minmax",
        "robust",
    ]

    for scaler_name in scaler_names:

        print(f"\nRunning {scaler_name} preprocessing...")

        scaled_features, preprocessor = fit_preprocessor(
            df=features,
            scaler_name=scaler_name,
            log_features=log_features,
        )

        output_path = (
            OUTPUT_DIR / f"customers_{scaler_name}.csv"
        )

        artifact_path = (
            ARTIFACT_DIR / f"{scaler_name}_preprocessor.joblib"
        )

        scaled_features.to_csv(output_path)

        joblib.dump(
            preprocessor,
            artifact_path,
        )

        print("Saved transformed data:", output_path)
        print("Saved fitted pipeline:", artifact_path)

        print(
            "Feature means after transformation and scaling:"
        )
        print(scaled_features.mean().round(3))

        print(
            "\nFeature standard deviations:"
        )
        print(scaled_features.std().round(3))

    print("\nPreprocessing experiments completed!")


if __name__ == "__main__":
    main()