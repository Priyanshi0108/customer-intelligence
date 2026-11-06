from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from customer_intelligence.models.hierarchical import (
    train_hierarchical,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "pca"
    / "customers_minmax_pca.csv"
)

REPORT_DIR = ROOT_DIR / "reports" / "hierarchical"

ARTIFACT_DIR = ROOT_DIR / "artifacts" / "hierarchical"

CLUSTER_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "hierarchical"
)

for directory in [
    REPORT_DIR,
    ARTIFACT_DIR,
    CLUSTER_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def evaluate_clustering(X, labels):

    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        raise ValueError("At least two clusters are required.")

    sample_size = min(2000, len(X))

    return {
        "n_clusters": len(unique_clusters),

        "silhouette_score": silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42,
        ),

        "davies_bouldin_score": davies_bouldin_score(
            X,
            labels,
        ),

        "calinski_harabasz_score": calinski_harabasz_score(
            X,
            labels,
        ),
    }


def run_experiments():

    X = pd.read_csv(
        DATA_PATH,
        index_col=0,
    )

    print("Dataset shape:", X.shape)

    cluster_values = range(2, 9)

    linkage_methods = [
        "ward",
        "complete",
        "average",
    ]

    results = []

    for linkage in linkage_methods:

        for n_clusters in cluster_values:

            print(
                f"\nTraining hierarchical clustering: "
                f"linkage={linkage}, "
                f"n_clusters={n_clusters}"
            )

            model, labels = train_hierarchical(
                X,
                n_clusters=n_clusters,
                linkage=linkage,
            )

            metrics = evaluate_clustering(
                X,
                labels,
            )

            result = {
                "linkage": linkage,
                **metrics,
            }

            results.append(result)

            # Save cluster assignments.
            labels.to_frame().to_csv(
                CLUSTER_DIR
                / f"{linkage}_k_{n_clusters}.csv"
            )

            # Save the fitted model.
            joblib.dump(
                model,
                ARTIFACT_DIR
                / f"{linkage}_k_{n_clusters}.joblib",
            )

            print(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="silhouette_score",
        ascending=False,
    )

    results_df.to_csv(
        REPORT_DIR / "hierarchical_experiment_results.csv",
        index=False,
    )

    print("\n========== FINAL COMPARISON ==========")

    print(
        results_df.to_string(index=False)
    )

    print("\nHierarchical experiments completed!")


if __name__ == "__main__":
    run_experiments()