import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load original customer features
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

X = df[feature_columns].copy()

# 1. Raw features
X_raw = X.copy()

# 2. Log transformation
X_log = np.log1p(X)

# 3. Log transformation + StandardScaler
scaler = StandardScaler()
X_log_scaled = pd.DataFrame(
    scaler.fit_transform(X_log),
    columns=X_log.columns,
    index=X_log.index
)

representations = {
    "raw": X_raw,
    "log": X_log,
    "log_scaled": X_log_scaled,
}

results = []

for name, data in representations.items():

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(data)

    labels = model.predict(data)
    scores = -model.score_samples(data)



    anomaly_count = (labels == -1).sum()

    results.append({
        "representation": name,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_count / len(data) * 100,
        "mean_anomaly_score": scores.mean(),
    })

    print(f"\nRepresentation: {name}")

    top_indices = np.argsort(scores)[-10:][::-1]

    

    top_customers = df.iloc[top_indices].copy()

    top_customers["anomaly_score"] = scores[top_indices]

    print(f"\nTop 10 anomalies: {name}")

    print(
    top_customers[
        [
            "monetary",
            "frequency",
            "total_quantity",
            "anomaly_score",
        ]
    ]
)

    print(
        df.iloc[top_indices][
            ["monetary", "frequency", "total_quantity"]
        ].assign(
            anomaly_score=scores[top_indices]
        )
    )

results_df = pd.DataFrame(results)

print("\nComparison:")
print(results_df)

results_df.to_csv(
    "reports/anomaly/representation_comparison.csv",
    index=False
)

anomaly_sets = {}

for name, data in representations.items():

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(data)

    labels = model.predict(data)

    anomaly_sets[name] = set(
        data.index[labels == -1]
    )

print("\nAnomaly overlap:")

print(
    "Raw vs Log:",
    len(anomaly_sets["raw"] & anomaly_sets["log"])
)

print(
    "Raw vs Log Scaled:",
    len(anomaly_sets["raw"] & anomaly_sets["log_scaled"])
)

print(
    "Log vs Log Scaled:",
    len(anomaly_sets["log"] & anomaly_sets["log_scaled"])
)



raw_only = anomaly_sets["raw"] - anomaly_sets["log"]
log_only = anomaly_sets["log"] - anomaly_sets["raw"]

print("Raw-only anomalies:", len(raw_only))
print("Log-only anomalies:", len(log_only))

print("\nCustomers detected only by RAW:")

print(
    df.loc[list(raw_only)][
        ["recency_days", "frequency", "monetary",
         "total_quantity", "product_diversity"]
    ].head(10)
)

print("\nCustomers detected only by LOG:")

print(
    df.loc[list(log_only)][
        ["recency_days", "frequency", "monetary",
         "total_quantity", "product_diversity"]
    ].head(10)
)