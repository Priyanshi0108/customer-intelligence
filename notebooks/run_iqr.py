import pandas as pd

from customer_intelligence.models.iqr import detect_iqr_anomalies


# Load customer features
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

# Run IQR
results, bounds = detect_iqr_anomalies(
    X,
    multiplier=1.5
)

# Feature-wise outlier counts
print("\nFeature-wise outlier counts:")

print(
    results[feature_columns]
    .sum()
    .sort_values(ascending=False)
)

# Total anomalous customers
print("\nTotal anomalous customers:")

print(results["is_anomaly"].value_counts())

# Display IQR bounds
print("\nIQR bounds:")

for feature, values in bounds.items():
    print(
        f"{feature}: "
        f"Lower={values['lower']:.2f}, "
        f"Upper={values['upper']:.2f}"
    )

# Save results
results.to_csv(
    "reports/anomaly/iqr_results.csv"
)

print("\nIQR results saved successfully!")