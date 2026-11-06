from pathlib import Path

import pandas as pd


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
    / "dbscan"
    / "dbscan_eps_0.14_min_samples_15.csv"
)

REPORT_DIR = ROOT_DIR / "reports" / "dbscan"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    # Load original customer-level features.
    features = pd.read_csv(
        FEATURE_PATH,
        index_col=0,
    )

    # Load DBSCAN labels.
    labels = pd.read_csv(
        CLUSTER_PATH,
        index_col=0,
    )

    # Combine original features and cluster labels.
    customer_data = features.join(
        labels[["cluster"]],
        how="inner",
    )

    print("\n========== DATA OVERVIEW ==========")

    print("Total customers:", len(customer_data))

    # Calculate noise statistics.
    noise_count = (
        customer_data["cluster"] == -1
    ).sum()

    noise_percentage = (
        noise_count / len(customer_data)
    ) * 100

    print("Noise customers:", noise_count)

    print("Noise percentage:", round(noise_percentage, 2))

    print(
        "Clustered customers:",
        len(customer_data) - noise_count,
    )

    # Cluster distribution including noise.
    print("\n========== CLUSTER DISTRIBUTION ==========")

    print(
        customer_data["cluster"]
        .value_counts()
        .sort_index()
    )

    # Exclude noise for customer cluster profiling.
    actual_clusters = customer_data[
        customer_data["cluster"] != -1
    ]

    # Calculate cluster means.
    cluster_mean = (
        actual_clusters
        .groupby("cluster")
        .mean(numeric_only=True)
    )

    # Calculate cluster medians.
    cluster_median = (
        actual_clusters
        .groupby("cluster")
        .median(numeric_only=True)
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n========== CLUSTER MEAN PROFILE ==========")

    print(cluster_mean.T.round(2))

    print("\n========== CLUSTER MEDIAN PROFILE ==========")

    print(cluster_median.T.round(2))

    # Save reports.
    cluster_mean.to_csv(
        REPORT_DIR / "dbscan_cluster_mean.csv"
    )

    cluster_median.to_csv(
        REPORT_DIR / "dbscan_cluster_median.csv"
    )

    customer_data.to_csv(
        REPORT_DIR / "dbscan_customer_segments.csv"
    )

    print("\nDBSCAN profiling completed!")


if __name__ == "__main__":
    main()