
from pathlib import Path

import pandas as pd
import joblib

from customer_intelligence.models.anamoly import detect_anomalies

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data/processed/customer_features.csv"

OUTPUT_DIR = ROOT / "reports/anomaly"
MODEL_DIR = ROOT / "artifacts/anomaly"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(FEATURES_PATH)

numeric_columns = [
    "recency_days",
    "frequency",
    "monetary",
    "total_quantity",
    "product_diversity",
    "average_order_value",
    "customer_tenure_days",
    "avg_purchase_interval",
]

X = features[numeric_columns].copy()

# Compare a few assumed anomaly proportions.
results = []

for contamination in [0.01, 0.03, 0.05]:
    model, labels, scores = detect_anomalies(
        X,
        contamination=contamination,
    )

    results.append({
        "contamination": contamination,
        "anomaly_count": int(labels.sum()),
        "anomaly_percentage": round(labels.mean() * 100, 2),
        "mean_anomaly_score": scores.mean(),
    })

    if contamination == 0.05:
        report = features.copy()
        report["is_anomaly"] = labels
        report["anomaly_score"] = scores

        report = report.sort_values(
            "anomaly_score",
            ascending=False,
        )

        report.to_csv(
            OUTPUT_DIR / "isolation_forest_anomalies.csv",
            index=False,
        )

        joblib.dump(
            model,
            MODEL_DIR / "isolation_forest_05.joblib",
        )

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "contamination_comparison.csv",
    index=False,
)

print("Contamination comparison:")
print(results_df.to_string(index=False))

print("\nTop 10 most anomalous customers:")
print(
    report[
        [
            "CustomerID",
            "recency_days",
            "frequency",
            "monetary",
            "product_diversity",
            "anomaly_score",
        ]
    ].head(10).to_string(index=False)
)
