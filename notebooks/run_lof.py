
import numpy as np
import pandas as pd

from customer_intelligence.models.lof import detect_lof_anomalies
from sklearn.preprocessing import StandardScaler

# Load customer features
df = pd.read_csv(
    "data/processed/customer_features.csv",
    index_col=0,
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

X = df[feature_columns].copy()

# Log transformation
X_log = np.log1p(X)

# Standardization
scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X_log),
    columns=X_log.columns,
    index=X_log.index,
)

# Run LOF
model, labels, scores = detect_lof_anomalies(
    X_scaled,
    n_neighbors=20,
    contamination=0.05,
)

# Combine results
results = df.copy()

results["is_anomaly"] = labels
results["anomaly_score"] = scores

# Display summary
print("Total customers:", len(results))
print("Anomalies:", labels.sum())
print("Anomaly percentage:", labels.mean() * 100)

# Top 10 anomalies
top_anomalies = results.sort_values(
    "anomaly_score",
    ascending=False,
).head(10)

print("\nTop 10 LOF anomalies:")

print(
    top_anomalies[
        [
            "recency_days",
            "frequency",
            "monetary",
            "total_quantity",
            "product_diversity",
            "anomaly_score",
        ]
    ].to_string()
)

# Save results
results.to_csv(
    "reports/anomaly/lof_results.csv"
)
