import pandas as pd

from customer_intelligence.models.robust_zscore import (
    detect_robust_zscore
)

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

robust_z, results, median, mad = detect_robust_zscore(
    X,
    threshold=3.5
)

print("\nFeature-wise outlier counts:")

print(
    results[feature_columns]
    .sum()
    .sort_values(ascending=False)
)

print("\nTotal anomalous customers:")

print(results["is_anomaly"].value_counts())

print("\nMedian values:")
print(median)

print("\nMAD values:")
print(mad)

print("\nTop anomalous customers:")

anomalous_ids = results[
    results["is_anomaly"] == 1
].index

print(df.loc[anomalous_ids].assign( max_robust_z=robust_z.loc[anomalous_ids].abs().max(axis=1)))
        
        
print("Feature-wise outlier counts:")

print(
    results[feature_columns]
    .sum()
    .sort_values(ascending=False)
)

print("\nTotal anomalous customers:")

print(results["is_anomaly"].value_counts())

print("\nCustomer 16446 feature-wise scores:")

print(
    robust_z.loc[16446].sort_values(
        key=abs,
        ascending=False
    )
)


results.to_csv(
    "reports/anomaly/robust_zscore_results.csv"
)

print("\nResults saved successfully!")