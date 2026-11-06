from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "pca"
    / "customers_minmax_pca.csv"
)

REPORT_DIR = ROOT_DIR / "reports" / "dbscan"

FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

def plot_k_distance(X:pd.DataFrame,min_samples:int=10) ->None:

    if min_samples >len(X):
        raise ValueError(
            "min_samples. cannot exceed number of customers"
        )

    neighbors = NearestNeighbors(
        n_neighbors=min_samples,
        metric = "euclidean"
    )

    neighbors.fit(X)

    distances,_ = neighbors.kneighbors(X)

    k_distances = distances[:, -1]

    # Sort distances for visualization.
    sorted_distances = np.sort(k_distances)

    plt.figure(figsize=(10, 6))

    plt.plot(sorted_distances)

    plt.xlabel("Customers sorted by distance")

    plt.ylabel(
        f"Distance to {min_samples}th nearest neighbor"
    )

    plt.title(
        f"DBSCAN k-distance plot (min_samples={min_samples})"
    )

    plt.grid(True)

    output_path = (
        FIGURE_DIR
        / f"k_distance_min_samples_{min_samples}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {output_path}")

    print("\nDistance statistics:")

    print(
        pd.Series(k_distances).describe()
    )


def main():

    X = pd.read_csv(
        DATA_PATH,
        index_col=0,
    )

    print("Dataset shape:", X.shape)

    print("\nGenerating k-distance plots...")

    for min_samples in [5, 10, 15]:

        plot_k_distance(
            X,
            min_samples=min_samples,
        )

    print("\nDBSCAN diagnostics completed!")


if __name__ == "__main__":
    main()

