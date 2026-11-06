from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT / "data/processed/customer_features.csv"
)

LABELS_PATH = (
    ROOT / "data/processed/hierarchical/ward_k_2.csv"
)

OUTPUT_DIR = ROOT / "reports/hierarchical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original customer features
features = pd.read_csv(FEATURES_PATH)

# Load hierarchical cluster labels
labels = pd.read_csv(LABELS_PATH)

# Inspect both datasets
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Check whether customer IDs are available
print(features.columns.tolist())
print(labels.columns.tolist())

# Merge using customer ID
if "CustomerID" in features.columns and "CustomerID" in labels.columns:
    df = features.merge(
        labels[["CustomerID", "cluster"]],
        on="CustomerID",
        how="inner",
    )
else:
    # Use index alignment only if both files preserve the same row order.
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts do not match.")

    df = features.copy()
    df["cluster"] = labels["cluster"].values

# Profile clusters
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

mean_profile = df.groupby("cluster")[numeric_columns].mean().round(2)

median_profile = df.groupby("cluster")[numeric_columns].median().round(2)

cluster_sizes = df["cluster"].value_counts().sort_index()

cluster_percentages = (
    df["cluster"].value_counts(normalize=True).sort_index() * 100
).round(2)

print("\nCluster sizes:")
print(cluster_sizes)

print("\nCluster percentages:")
print(cluster_percentages)

print("\nMean profile:")
print(mean_profile)

print("\nMedian profile:")
print(median_profile)

# Save reports
mean_profile.to_csv(
    OUTPUT_DIR / "hierarchical_mean_profile.csv"
)

median_profile.to_csv(
    OUTPUT_DIR / "hierarchical_median_profile.csv"
)

cluster_sizes.to_csv(
    OUTPUT_DIR / "hierarchical_cluster_sizes.csv"
)

print("\nProfiling completed!")