from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "pca"
    / "customers_minmax_pca.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "reports"
    / "hierarchical"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    X = pd.read_csv(
        DATA_PATH,
        index_col=0,
    )

    # Use a reproducible sample for readable visualization.
    sample = X.sample(
        n=min(300, len(X)),
        random_state=42,
    )

    # Build the hierarchy using Ward linkage.
    hierarchy = linkage(
        sample,
        method="ward",
        metric="euclidean",
    )

    plt.figure(figsize=(16, 8))

    dendrogram(
        hierarchy,
        no_labels=True,
        leaf_rotation=90,
    )

    plt.title("Customer Hierarchical Clustering Dendrogram")

    plt.xlabel("Customers")

    plt.ylabel("Ward Linkage Distance")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "ward_dendrogram_sample.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("Dendrogram saved:", output_path)


if __name__ == "__main__":
    main()
    