import pandas as pd

from customer_intelligence.models.iqr import detect_iqr_anomalies

df = pd.read_csv(
    "data/processed/customer_features.csv",
    index_col=0
)

feature_columns = [
    "recency_days",
    "frequency",
    "monetary",
    "total_quantity",
    "product_diversity",
    "average_order_value",
    "customer_tenure_days",
    "avg_purchase_interval",
]

X = df[feature_columns]

for multiplier in [1.5, 2.0, 2.5, 3.0]:

    results, bounds = detect_iqr_anomalies(
        X,
        multiplier=multiplier
    )

    count = results["is_anomaly"].sum()

    print(
        f"Multiplier: {multiplier} | "
        f"Anomalies: {count} | "
        f"Percentage: {count / len(X) * 100:.2f}%"
    )