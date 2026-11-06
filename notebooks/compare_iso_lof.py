
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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

X = df[feature_columns]

# Same preprocessing for both algorithms
X_log = np.log1p(X)

X_scaled = pd.DataFrame(
    StandardScaler().fit_transform(X_log),
    columns=X.columns,
    index=X.index,
)

# 1. Isolation Forest
iforest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)

if_labels = iforest.fit_predict(X_scaled)

if_anomalies = set(
    X_scaled.index[if_labels == -1]
)

# 2. Local Outlier Factor
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
)

lof_labels = lof.fit_predict(X_scaled)

lof_anomalies = set(
    X_scaled.index[lof_labels == -1]
)

# 3. Compare anomaly sets
common = if_anomalies & lof_anomalies
if_only = if_anomalies - lof_anomalies
lof_only = lof_anomalies - if_anomalies

union = if_anomalies | lof_anomalies

print("Total customers:", len(df))
print("Isolation Forest anomalies:", len(if_anomalies))
print("LOF anomalies:", len(lof_anomalies))

print("\nAgreement:")
print("Common anomalies:", len(common))
print("IF only:", len(if_only))
print("LOF only:", len(lof_only))

print(
    "Jaccard similarity:",
    round(len(common) / len(union), 4)
)

# 4. Inspect common anomalies
print("\nCommon anomalous customers:")

print(
    df.loc[list(common)][
        ["recency_days", "frequency", "monetary",
         "total_quantity", "product_diversity"]
    ].head(15)
)

# 5. Save comparison
comparison = df.copy()

comparison["is_iforest_anomaly"] = (
    comparison.index.isin(if_anomalies)
).astype(int)

comparison["is_lof_anomaly"] = (
    comparison.index.isin(lof_anomalies)
).astype(int)

comparison["agreement"] = np.select(
    [
        comparison["is_iforest_anomaly"].eq(1)
        & comparison["is_lof_anomaly"].eq(1),

        comparison["is_iforest_anomaly"].eq(1),

        comparison["is_lof_anomaly"].eq(1),
    ],
    [
        "both",
        "iforest_only",
        "lof_only",
    ],
    default="normal",
)

comparison.to_csv(
    "reports/anomaly/iforest_lof_comparison.csv"
)

print("\nSaved comparison report.")


# Create agreement dataframe
agreement_df = pd.DataFrame(index=X_scaled.index)

agreement_df["iforest_anomaly"] = (
    X_scaled.index.isin(if_anomalies).astype(int)
)

agreement_df["lof_anomaly"] = (
    X_scaled.index.isin(lof_anomalies).astype(int)
)

agreement_df["agreement_score"] = (
    agreement_df["iforest_anomaly"]
    + agreement_df["lof_anomaly"]
)

# Add original customer features
agreement_df = agreement_df.join(X)

# Inspect customers flagged by both models
common_customers = agreement_df[
    agreement_df["agreement_score"] == 2
].sort_values("monetary", ascending=False)

print(common_customers.head(20))

# Save report
agreement_df.to_csv(
    "reports/anomaly/anomaly_agreement.csv"
)