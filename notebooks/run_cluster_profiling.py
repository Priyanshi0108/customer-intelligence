import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "customer_features.csv"
)

CLUSTER_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "clustered"
    / "minmax_pca_best_clusters.csv"
)


REPORT_DIR = ROOT_DIR / "reports" / "cluster_profiling"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():
    
    # Load original customer-level features.
    features = pd.read_csv(
        FEATURE_PATH,
        index_col=0
    )

    # Load K-Means cluster assignments.
    clusters = pd.read_csv(
        CLUSTER_PATH,
        index_col=0
    )

    # Join original features with cluster labels.
    customer_data = features.join(
        clusters[["cluster"]],
        how="inner"
    )

    print("\n========== DATA OVERVIEW ==========")

    print("Total customers:", len(customer_data))

    print("\nCluster distribution:")

    print(
        customer_data["cluster"]
        .value_counts()
        .sort_index()
    )

    # Calculate average feature values for each cluster.
    cluster_profile = (
        customer_data
        .groupby("cluster")
        .mean(numeric_only=True)
    )

    print("\n========== CLUSTER PROFILES ==========")

    print(
        cluster_profile.round(2)
    )

    # Calculate median feature values.
    cluster_median = (
        customer_data
        .groupby("cluster")
        .median(numeric_only=True)
    )




    print("\n========== CLUSTER MEDIANS ==========")

    print(
        cluster_median.round(2)
    )



    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n========== COMPLETE CLUSTER PROFILES ==========")

    print(cluster_profile.T.round(2))

    print("\n========== COMPLETE CLUSTER MEDIANS ==========")

    print(cluster_median.T.round(2))

    # Calculate percentage distribution.
    cluster_percentage = (
        customer_data["cluster"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print("\n========== CLUSTER PERCENTAGES ==========")

    print(
        cluster_percentage.round(2)
    )

    # Save reports.
    cluster_profile.to_csv(
        REPORT_DIR / "cluster_mean_profile.csv"
    )

    cluster_median.to_csv(
        REPORT_DIR / "cluster_median_profile.csv"
    )

    cluster_percentage.to_csv(
        REPORT_DIR / "cluster_distribution.csv"
    )

    customer_data.to_csv(
        REPORT_DIR / "customer_segments.csv"
    )

    print("\nCluster profiling completed!")


if __name__ == "__main__":
    main()