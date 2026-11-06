from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from customer_intelligence.models.dbscan import train_dbscan

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "pca"
    / "customers_minmax_pca.csv"
)

REPORT_DIR = ROOT_DIR / "reports" / "dbscan"

ARTIFACT_DIR = ROOT_DIR / "artifacts" / "dbscan"

CLUSTER_DIR = ROOT_DIR / "data" / "processed" / "dbscan"

for directory in [REPORT_DIR, ARTIFACT_DIR, CLUSTER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def evaluate_dbscan(X, labels):

    labels_array = labels.to_numpy()

    total_customers = len(labels_array)

    noise_mask = labels_array == -1

    noise_count = int(noise_mask.sum())

    noise_percentage = (
        noise_count / total_customers
    ) * 100

    # Exclude noise points for cluster quality metrics.
    non_noise_mask = ~noise_mask

    X_evaluation = X.loc[non_noise_mask]

    labels_evaluation = labels.loc[non_noise_mask]

    unique_clusters = labels_evaluation.nunique()

    result = {
        "n_clusters": unique_clusters,
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "silhouette_score": np.nan,
        "davies_bouldin_score": np.nan,
        "calinski_harabasz_score": np.nan,
    }

    # Internal clustering metrics require at least two clusters.
    if (
        unique_clusters >= 2
        and unique_clusters < len(X_evaluation)
    ):

        result["silhouette_score"] = silhouette_score(
            X_evaluation,
            labels_evaluation,
        )

        result["davies_bouldin_score"] = davies_bouldin_score(
            X_evaluation,
            labels_evaluation,
        )

        result["calinski_harabasz_score"] = (
            calinski_harabasz_score(
                X_evaluation,
                labels_evaluation,
            )
        )

    return result


def run_experiments():

    X = pd.read_csv(
        DATA_PATH,
        index_col=0,
    )

    print("Dataset shape:", X.shape)

    eps_values = [0.06, 0.08, 0.10, 0.12, 0.14]

    min_samples_values = [5, 10, 15]

    results = []

    for min_samples in min_samples_values:

        for eps in eps_values:

            print(
                f"\nRunning DBSCAN: eps={eps}, "
                f"min_samples={min_samples}"
            )

            model, labels = train_dbscan(
                X,
                eps=eps,
                min_samples=min_samples,
            )

            metrics = evaluate_dbscan(
                X,
                labels,
            )

            experiment_result = {
                "eps": eps,
                "min_samples": min_samples,
                **metrics,
            }

            results.append(experiment_result)

            # Save cluster assignments.
            labels.to_frame().to_csv(
                CLUSTER_DIR
                / f"dbscan_eps_{eps}_min_samples_{min_samples}.csv"
            )

            # Save trained model.
            joblib.dump(
                model,
                ARTIFACT_DIR
                / f"dbscan_eps_{eps}_min_samples_{min_samples}.joblib",
            )

            print(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="silhouette_score",
        ascending=False,
        na_position="last",
    )

    results_df.to_csv(
        REPORT_DIR / "dbscan_experiment_results.csv",
        index=False,
    )

    print("\n========== DBSCAN EXPERIMENT RESULTS ==========")

    print(
        results_df.to_string(index=False)
    )

    print("\nDBSCAN experiments completed!")


if __name__ == "__main__":
    run_experiments()