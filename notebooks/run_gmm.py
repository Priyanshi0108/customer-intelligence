from pathlib import Path

import pandas as pd

from customer_intelligence.models.gmm import train_gmm
from customer_intelligence.evaluation.clustering_metrics import (
    evaluate_clustering,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/pca/customers_minmax_pca.csv"

MODEL_DIR = ROOT / "artifacts/gmm"
LABEL_DIR = ROOT / "data/processed/gmm"
REPORT_DIR = ROOT / "reports/gmm"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

X = pd.read_csv(DATA_PATH, index_col=0)

results = []

for covariance_type in ["full", "tied", "diag", "spherical"]:

    for n_components in range(2, 9):

        print(
            f"Training GMM: covariance={covariance_type}, "
            f"components={n_components}"
        )

        model, labels, probabilities = train_gmm(
            X,
            n_components=n_components,
            covariance_type=covariance_type,
        )

        metrics = evaluate_clustering(X, labels)

        result = {
            "covariance_type": covariance_type,
            "n_components": n_components,
            **metrics,
            "bic": model.bic(X),
            "aic": model.aic(X),
            "converged": model.converged_,
            "iterations": model.n_iter_,
        }

        results.append(result)

        filename = f"{covariance_type}_k_{n_components}"

        labels.to_csv(
            LABEL_DIR / f"{filename}_labels.csv"
        )

        probabilities.to_csv(
            LABEL_DIR / f"{filename}_probabilities.csv"
        )

        import joblib

        joblib.dump(
            model,
            MODEL_DIR / f"{filename}.joblib",
        )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="silhouette_score",
    ascending=False,
)

results_df.to_csv(
    REPORT_DIR / "gmm_experiment_results.csv",
    index=False,
)

print("\nTop GMM configurations:")
print(results_df.head(10).to_string(index=False))

print("\nGMM experiments completed!")