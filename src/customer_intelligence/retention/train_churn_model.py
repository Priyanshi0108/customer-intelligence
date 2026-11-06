
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data" / "processed" / "churn_training_data.csv"

ARTIFACT_DIR = ROOT / "artifacts" / "retention"
REPORT_DIR = ROOT / "reports" / "retention"

MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"

# --------------------------------------------------
# Features and target
# --------------------------------------------------

FEATURES = [
    "frequency",
    "monetary",
    "total_quantity",
    "product_diversity",
    "average_order_value",
    "avg_purchase_interval",
    "recency_days",
    "customer_tenure_days",
]

TARGET = "churn_90d"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    required = FEATURES + [TARGET, "snapshot_date", "CustomerID"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# --------------------------------------------------
# Chronological train-test split
# --------------------------------------------------



def chronological_split(df):

    dates = sorted(
        pd.to_datetime(df["snapshot_date"].unique())
    )

    if len(dates) < 4:
        raise ValueError(
            "Not enough snapshot dates for a purged temporal split."
        )

    # Keep the last two snapshots for final testing.
    test_dates = dates[-2:]

    first_test_date = test_dates[0]

    # A training label must be fully observable before
    # the first test snapshot begins.
    train_dates = [
        date
        for date in dates[:-2]
        if date + pd.Timedelta(days=90) < first_test_date
    ]

    # Dates between training and testing are purged.
    purged_dates = [
        date
        for date in dates
        if date not in train_dates and date not in test_dates
    ]

    if len(train_dates) < 2:
        raise ValueError(
            "Insufficient training snapshots after applying "
            "the 90-day temporal purge."
        )

    train_df = df[
        df["snapshot_date"].isin(train_dates)
    ].copy()

    test_df = df[
        df["snapshot_date"].isin(test_dates)
    ].copy()

    print("\nPurged chronological split")
    print("--------------------------")
    print("Training dates:", train_dates)
    print("Purged dates:", purged_dates)
    print("Testing dates:", test_dates)

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    return train_df, test_df


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_model(X_train, y_train):

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
    }

    print("\nEvaluation metrics")
    print("------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Active", "Inactive"],
            zero_division=0,
        )
    )

    print("\nConfusion matrix")
    print(
        confusion_matrix(y_test, predictions)
    )

    return metrics, predictions, probabilities


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

def save_feature_importance(model):

    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": classifier.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    print("\nFeature importance")
    print(importance.to_string(index=False))


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading churn dataset...")

    df = load_data()

    print("Dataset shape:", df.shape)
    print("Snapshot dates:", df["snapshot_date"].nunique())

    train_df, test_df = chronological_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTraining Random Forest...")

    model = train_model(X_train, y_train)

    print("\nEvaluating model...")

    metrics, predictions, probabilities = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save metrics
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    # Save feature importance
    save_feature_importance(model)

    # Save test predictions
    results = test_df[
        ["CustomerID", "snapshot_date", TARGET]
    ].copy()

    results["predicted_churn"] = predictions
    results["churn_probability"] = probabilities

    results.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nTraining completed successfully!")
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    print("Feature importance:", IMPORTANCE_PATH)
    print("Predictions:", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
