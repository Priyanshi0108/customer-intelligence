
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from customer_intelligence.models.iqr import detect_iqr_anomalies
from customer_intelligence.models.robust_zscore import (
    detect_robust_zscore,
)


# 1. Load customer features
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


# 2. Preprocessing for distance/tree-based experiments
X_log = np.log1p(X)

X_scaled = pd.DataFrame(
    StandardScaler().fit_transform(X_log),
    columns=feature_columns,
    index=X.index,
)


# 3. Isolation Forest
iforest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)

if_labels = iforest.fit_predict(X_scaled)

if_anomalies = pd.Series(
    (if_labels == -1).astype(int),
    index=X.index,
    name="iforest_anomaly",
)


# 4. LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
)

lof_labels = lof.fit_predict(X_scaled)

lof_anomalies = pd.Series(
    (lof_labels == -1).astype(int),
    index=X.index,
    name="lof_anomaly",
)


# 5. IQR
iqr_results, iqr_bounds = detect_iqr_anomalies(
    X,
    multiplier=1.5,
)

iqr_anomalies = iqr_results["is_anomaly"].rename(
    "iqr_anomaly"
)


# 6. Robust Z-score
robust_z, robust_results, median, mad = (
    detect_robust_zscore(
        X,
        threshold=3.5,
    )
)

robust_anomalies = robust_results[
    "is_anomaly"
].rename("robust_zscore_anomaly")


# 7. Combine all methods
comparison = pd.concat(
    [
        if_anomalies,
        lof_anomalies,
        iqr_anomalies,
        robust_anomalies,
    ],
    axis=1,
)

comparison["total_agreement"] = comparison.sum(axis=1)

# Add original customer features
comparison = comparison.join(X)


# 8. Print agreement distribution
print("\nAgreement distribution:")

print(
    comparison["total_agreement"]
    .value_counts()
    .sort_index()
)


# 9. Customers flagged by all four methods
all_four = comparison[
    comparison["total_agreement"] == 4
]

print("\nCustomers flagged by all four methods:")
print(all_four)


# 10. Customers flagged by at least two methods
multiple_methods = comparison[
    comparison["total_agreement"] >= 2
].sort_values(
    "total_agreement",
    ascending=False,
)

print("\nTop 20 customers flagged by multiple methods:")
print(multiple_methods.head(20))
# Prioritize customers flagged by at least 3 methods

priority_customers = comparison[
    comparison["total_agreement"] >= 3
].copy()

priority_customers = priority_customers.sort_values(
    by=["total_agreement", "monetary"],
    ascending=[False, False],
)

display_columns = [
    "iforest_anomaly",
    "lof_anomaly",
    "iqr_anomaly",
    "robust_zscore_anomaly",
    "total_agreement",
    "frequency",
    "monetary",
    "total_quantity",
    "recency_days",
    "product_diversity",
]

print("\nPriority customer investigation list:")

print(
    priority_customers[display_columns]
    .to_string()
)

# 11. Save report
comparison.to_csv(
    "reports/anomaly/all_methods_comparison.csv"
)

print("\nUnified anomaly report saved successfully!")
