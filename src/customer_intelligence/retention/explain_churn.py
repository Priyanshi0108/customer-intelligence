
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.customer_intelligence.retention.train_churn_model import (
    FEATURES,
    TARGET,
    chronological_split,
    load_data,
)


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "artifacts" / "retention" / "churn_model.joblib"
REPORT_DIR = ROOT / "reports" / "retention"
FIGURE_DIR = ROOT / "reports" / "figures" / "retention"

GLOBAL_IMPORTANCE_PATH = REPORT_DIR / "shap_global_importance.csv"
LOCAL_EXPLANATION_PATH = REPORT_DIR / "customer_shap_explanation.csv"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"

HIGH_RISK_THRESHOLD = 0.60
SHAP_BATCH_SIZE = 128
GLOBAL_SAMPLE_SIZE = 300


def get_positive_class_shap_values(values):
    """
    Extract SHAP contributions for class 1:
    churn_90d = 1 (inactive).
    Supports common SHAP output formats.
    """
    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        # samples x features x classes
        if values.shape[-1] == 2:
            values = values[:, :, 1]

        # samples x classes x features
        elif values.shape[1] == 2:
            values = values[:, 1, :]

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP values, received {values.shape}"
        )

    return values


def get_positive_class_base_value(base_values):
    """Extract baseline for the positive/inactive class."""

    base_values = np.asarray(base_values)

    if base_values.ndim == 0:
        return float(base_values)

    if base_values.shape[-1] == 2:
        return float(base_values[-1])

    return float(base_values.reshape(-1)[0])


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained churn model...")

    model = joblib.load(MODEL_PATH)
    df = load_data()

    train_df, test_df = chronological_split(df)

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    # The trained pipeline contains median imputation.
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]

    X_train_transformed = pd.DataFrame(
        imputer.transform(X_train),
        columns=FEATURES,
        index=X_train.index,
    )

    X_test_transformed = pd.DataFrame(
        imputer.transform(X_test),
        columns=FEATURES,
        index=X_test.index,
    )

    # --------------------------------------------------
    # 1. Predict churn probability for every test customer
    # --------------------------------------------------

    print("\nPredicting churn probabilities...")

    test_probabilities = model.predict_proba(X_test)[:, 1]

    predictions = test_df[
        ["CustomerID", "snapshot_date", TARGET]
    ].copy()

    predictions["churn_probability"] = test_probabilities

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("Total test customers:", len(predictions))

    # --------------------------------------------------
    # 2. Select high-risk customers
    # --------------------------------------------------

    high_risk_positions = np.flatnonzero(
        test_probabilities >= HIGH_RISK_THRESHOLD
    )

    if len(high_risk_positions) == 0:
        raise ValueError(
            "No customers reached the configured high-risk threshold. "
            "Review HIGH_RISK_THRESHOLD."
        )

    print(
        f"\nHigh-risk threshold: {HIGH_RISK_THRESHOLD:.0%}"
    )
    print(
        "Selected high-risk customers:",
        len(high_risk_positions),
    )

    # --------------------------------------------------
    # 3. Create SHAP explainer
    # --------------------------------------------------

    background = X_train_transformed.sample(
        n=min(100, len(X_train_transformed)),
        random_state=42,
    )

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )

    base_value = get_positive_class_base_value(
        explainer.expected_value
    )

    print("Baseline churn probability:", base_value)

    # --------------------------------------------------
    # 4. Global SHAP importance
    # --------------------------------------------------

    print("\nCalculating global SHAP importance...")

    global_data = X_test_transformed.sample(
        n=min(GLOBAL_SAMPLE_SIZE, len(X_test_transformed)),
        random_state=42,
    )

    global_output = explainer.shap_values(global_data)

    global_values = get_positive_class_shap_values(
        global_output
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_shap": np.abs(global_values).mean(axis=0),
        "mean_shap": global_values.mean(axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False,
    )

    print("\nGlobal SHAP importance:")
    print(importance.to_string(index=False))

    # Global summary plot
    plt.figure(figsize=(11, 7))

    shap.summary_plot(
        global_values,
        global_data,
        feature_names=FEATURES,
        show=False,
    )

    plt.title("Customer Churn — SHAP Summary")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_summary.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Global importance bar plot
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        global_values,
        global_data,
        feature_names=FEATURES,
        plot_type="bar",
        show=False,
    )

    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_global_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 5. Batch SHAP for all high-risk customers
    # --------------------------------------------------

    print("\nCalculating batch SHAP explanations...")

    high_risk_data = X_test_transformed.iloc[
        high_risk_positions
    ]

    explanation_rows = []

    total_customers = len(high_risk_data)

    for start in range(0, total_customers, SHAP_BATCH_SIZE):

        end = min(
            start + SHAP_BATCH_SIZE,
            total_customers,
        )

        batch_data = high_risk_data.iloc[start:end]

        batch_output = explainer.shap_values(batch_data)

        batch_values = get_positive_class_shap_values(
            batch_output
        )

        batch_original_positions = high_risk_positions[start:end]

        batch_customers = test_df.iloc[
            batch_original_positions
        ]

        for local_position, (_, customer) in enumerate(
            batch_customers.iterrows()
        ):

            customer_values = batch_values[local_position]

            for feature_position, feature in enumerate(FEATURES):

                shap_value = float(
                    customer_values[feature_position]
                )

                explanation_rows.append({
                    "CustomerID": customer["CustomerID"],
                    "snapshot_date": customer["snapshot_date"],
                    "churn_probability": test_probabilities[
                        batch_original_positions[local_position]
                    ],
                    "actual_churn": customer[TARGET],
                    "feature": feature,
                    "customer_value": batch_data.iloc[
                        local_position
                    ][feature],
                    "shap_value": shap_value,
                    "direction": (
                        "Increases churn risk"
                        if shap_value > 0
                        else "Decreases churn risk"
                    ),
                    "absolute_shap": abs(shap_value),
                })

        print(
            f"Processed {end}/{total_customers} customers"
        )

    local_explanation = pd.DataFrame(explanation_rows)

    local_explanation.to_csv(
        LOCAL_EXPLANATION_PATH,
        index=False,
    )

    # --------------------------------------------------
    # 6. Waterfall plot for the highest-risk customer
    # --------------------------------------------------

    highest_risk_local_position = int(
        np.argmax(
            test_probabilities[high_risk_positions]
        )
    )

    highest_risk_original_position = high_risk_positions[
        highest_risk_local_position
    ]

    customer_row = test_df.iloc[
        highest_risk_original_position
    ]

    customer_features = high_risk_data.iloc[
        highest_risk_local_position
    ]

    customer_shap_values = (
        get_positive_class_shap_values(
            explainer.shap_values(
                customer_features.to_frame().T
            )
        )[0]
    )

    waterfall_explanation = shap.Explanation(
        values=customer_shap_values,
        base_values=base_value,
        data=customer_features.values,
        feature_names=FEATURES,
    )

    plt.figure(figsize=(11, 7))

    shap.plots.waterfall(
        waterfall_explanation,
        max_display=len(FEATURES),
        show=False,
    )

    plt.title("Why This Customer Is Predicted to Churn")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "customer_waterfall.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------

    print("\nBATCH SHAP COMPLETED")
    print("--------------------")

    print("Total test customers:", len(test_df))
    print("High-risk customers:", total_customers)

    print("\nHighest-risk customer:")
    print("Customer ID:", customer_row["CustomerID"])
    print("Snapshot date:", customer_row["snapshot_date"])
    print("Actual churn:", customer_row[TARGET])

    print(
        "Predicted churn probability:",
        f"{test_probabilities[highest_risk_original_position]:.2%}",
    )

    print("\nFiles saved:")
    print("Predictions:", PREDICTIONS_PATH)
    print("Global importance:", GLOBAL_IMPORTANCE_PATH)
    print("Customer SHAP explanations:", LOCAL_EXPLANATION_PATH)
    print("Figures:", FIGURE_DIR)


if __name__ == "__main__":
    main()
