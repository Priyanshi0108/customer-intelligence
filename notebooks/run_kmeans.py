from pathlib import Path

import joblib
import pandas as pd

from customer_intelligence.models.kmeans import train_kmeans
from customer_intelligence.evaluation.clustering_metrics import (
    evaluate_clustering,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"
PCA_DIR = PROCESSED_DIR / "pca"

REPORT_DIR = PROJECT_ROOT / "reports" / "kmeans"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "kmeans"
ASSIGNMENT_DIR = PROCESSED_DIR / "clustered"

for directory in [
    REPORT_DIR,
    ARTIFACT_DIR,
    ASSIGNMENT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_dataset(
    dataset_name: str,
) -> pd.DataFrame:
    """Load one scaled or PCA-transformed dataset."""

    if dataset_name.endswith("_pca"):
        path = PCA_DIR / f"customers_{dataset_name}.csv"
    else:
        path = (
            PREPROCESSED_DIR
            / f"customers_{dataset_name}.csv"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path, index_col=0)
    df.index.name = "CustomerID"

    return df.astype(float)


def run_experiment(
    dataset_name: str,
    X: pd.DataFrame,
) -> list[dict]:
    """Evaluate K-Means for different values of K."""

    print(f"\n========== {dataset_name.upper()} ==========")

    results = []

    best_score = float("-inf")
    best_model = None
    best_labels = None
    best_k = None

    max_k = min(10, len(X) - 1)

    for k in range(2, max_k + 1):

        print(f"\nTraining K-Means with K={k}")

        model, labels = train_kmeans(
            X=X,
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        metrics = evaluate_clustering(
            X=X,
            labels=labels,
            inertia=model.inertia_,
        )

        metrics["dataset"] = dataset_name

        results.append(metrics)

        print(metrics)

        if metrics["silhouette_score"] > best_score:

            best_score = metrics["silhouette_score"]
            best_model = model
            best_labels = labels
            best_k = k

    # Save the best model for this representation.
    model_path = (
        ARTIFACT_DIR / f"{dataset_name}_best_model.joblib"
    )

    joblib.dump(best_model, model_path)

    # Save customer-to-cluster assignments.
    assignments = pd.DataFrame(
        {
            "CustomerID": X.index,
            "cluster": best_labels.to_numpy(),
        }
    )

    assignments.to_csv(
        ASSIGNMENT_DIR
        / f"{dataset_name}_best_clusters.csv",
        index=False,
    )

    # Save cluster sizes.
    cluster_sizes = (
        best_labels.value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="customer_count")
    )

    cluster_sizes.to_csv(
        REPORT_DIR / f"{dataset_name}_cluster_sizes.csv",
        index=False,
    )

    print(f"\nBest K for {dataset_name}: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")
    print(f"Model saved: {model_path}")

    return results


def main():

    dataset_names = [
        "standard",
        "minmax",
        "robust",
        "standard_pca",
        "minmax_pca",
        "robust_pca",
    ]

    all_results = []

    for dataset_name in dataset_names:

        print(f"\nLoading {dataset_name} dataset...")

        X = load_dataset(dataset_name)

        print("Dataset shape:", X.shape)

        results = run_experiment(
            dataset_name=dataset_name,
            X=X,
        )

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORT_DIR / "kmeans_experiment_results.csv",
        index=False,
    )

    print("\n========== FINAL COMPARISON ==========")

    print(
        results_df.sort_values(
            by="silhouette_score",
            ascending=False,
        ).to_string(index=False)
    )

    print(
        "\nAll K-Means experiments completed successfully!"
    )


if __name__ == "__main__":
    main()