from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data/processed/customer_features.csv"

LABELS_PATH = (
    ROOT / "data/processed/gmm/tied_k_2_labels.csv"
)

PROBABILITIES_PATH = (
    ROOT / "data/processed/gmm/tied_k_2_probabilities.csv"
)

OUTPUT_DIR = ROOT / "reports/gmm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load data
features = pd.read_csv(FEATURES_PATH)

labels = pd.read_csv(LABELS_PATH, index_col=0)

probabilities = pd.read_csv(
    PROBABILITIES_PATH,
    index_col=0,
)

print("Features:", features.shape)
print("Labels:", labels.shape)
print("Probabilities:", probabilities.shape)


# 2. Validate row alignment
# The PCA and GMM pipeline must preserve the original customer row order.
if not (
    len(features) == len(labels) == len(probabilities)
):
    raise ValueError(
        "Features, labels and probabilities have different row counts."
    )

if "cluster" not in labels.columns:
    raise ValueError("Cluster column not found in labels.")

# Assign by position because the original customer feature file
# and PCA-derived GMM files may have different index representations.
df = features.copy().reset_index(drop=True)

df["cluster"] = labels["cluster"].reset_index(drop=True)

probabilities = probabilities.reset_index(drop=True)


# 3. Cluster distribution
cluster_sizes = df["cluster"].value_counts().sort_index()

cluster_percentages = (
    df["cluster"].value_counts(normalize=True).sort_index() * 100
).round(2)

print("\nCluster sizes:")
print(cluster_sizes)

print("\nCluster percentages:")
print(cluster_percentages)


# 4. Customer profiling
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

mean_profile = (
    df.groupby("cluster")[numeric_columns]
    .mean()
    .round(2)
)

median_profile = (
    df.groupby("cluster")[numeric_columns]
    .median()
    .round(2)
)

print("\nMean profile:")
print(mean_profile)

print("\nMedian profile:")
print(median_profile)


# 5. GMM assignment confidence
df["assignment_confidence"] = probabilities.max(axis=1)

confidence_by_cluster = (
    df.groupby("cluster")["assignment_confidence"]
    .agg(["mean", "median", "min", "max"])
    .round(4)
)

print("\nAssignment confidence:")
print(confidence_by_cluster)


# 6. Identify uncertain customer assignments
df["assignment_confidence"] = probabilities.max(axis=1)

uncertain_customers = df[
    df["assignment_confidence"] < 0.60
]

print("\nCustomers with confidence below 60%:")
print(len(uncertain_customers))


# 7. Save reports
mean_profile.to_csv(
    OUTPUT_DIR / "gmm_mean_profile.csv"
)

median_profile.to_csv(
    OUTPUT_DIR / "gmm_median_profile.csv"
)

cluster_sizes.to_csv(
    OUTPUT_DIR / "gmm_cluster_sizes.csv"
)

confidence_by_cluster.to_csv(
    OUTPUT_DIR / "gmm_assignment_confidence.csv"
)

uncertain_customers.to_csv(
    OUTPUT_DIR / "gmm_uncertain_customers.csv",
    index=False,
)

print("\nGMM profiling completed!")