from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

from customer_intelligence.dimensionality_reduction.pca import (
    fit_pca,
    get_explained_variance,
    get_component_loadings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "pca"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "pca"
)

FIGURE_DIR = REPORT_DIR / "figures"

for directory in [
    OUTPUT_DIR,
    ARTIFACT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_scaled_features(
    scaler_name: str,
) -> pd.DataFrame:

    path = (
        INPUT_DIR
        / f"customers_{scaler_name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}. "
            "Run preprocessing first."
        )

    df = pd.read_csv(
        path,
        index_col=0,
    )

    df.index.name = "CustomerID"

    return df


def plot_explained_variance(
    variance_df: pd.DataFrame,
    scaler_name: str,
) -> None:

    plt.figure(figsize=(9, 5))

    plt.plot(
        variance_df["component"],
        variance_df["cumulative_explained_variance"],
        marker="o",
    )

    plt.axhline(
        y=0.95,
        linestyle="--",
        label="95% variance",
    )

    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")

    plt.title(
        f"PCA Explained Variance - {scaler_name}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{scaler_name}_explained_variance.png",
        dpi=150,
    )

    plt.close()


def plot_pca_2d(
    df: pd.DataFrame,
    scaler_name: str,
) -> None:

    if df.shape[1] < 2:
        print(
            f"Skipping 2D plot for {scaler_name}: "
            "fewer than two components retained."
        )
        return

    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["PC1"],
        df["PC2"],
        alpha=0.5,
        s=12,
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        f"Customer Distribution after PCA - {scaler_name}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{scaler_name}_pca_2d.png",
        dpi=150,
    )

    plt.close()


def run_experiment(scaler_name: str) -> None:

    print(f"\n========== {scaler_name.upper()} PCA ==========")

    features = load_scaled_features(scaler_name)

    print("Original shape:", features.shape)

    transformed, pca = fit_pca(
        features,
        variance_threshold=0.95,
    )

    print("Transformed shape:", transformed.shape)

    variance_df = get_explained_variance(pca)

    print("\nExplained variance:")
    print(variance_df)

    print(
        "\nTotal retained variance:",
        round(
            pca.explained_variance_ratio_.sum() * 100,
            2,
        ),
        "%",
    )

    # Save transformed data.
    transformed.to_csv(
        OUTPUT_DIR / f"customers_{scaler_name}_pca.csv"
    )

    # Save fitted PCA model.
    joblib.dump(
        pca,
        ARTIFACT_DIR / f"{scaler_name}_pca.joblib",
    )

    # Save explained variance report.
    variance_df.to_csv(
        REPORT_DIR / f"{scaler_name}_variance.csv",
        index=False,
    )

    # Save component loadings.
    loadings = get_component_loadings(
        pca,
        feature_names=features.columns.tolist(),
    )

    loadings.to_csv(
        REPORT_DIR / f"{scaler_name}_loadings.csv"
    )

    plot_explained_variance(
        variance_df,
        scaler_name,
    )

    # Create a separate two-component PCA for visualization.
    # This is not the same model as the 95%-variance PCA.
    if features.shape[1] >= 2:

        pca_2d = PCA(n_components=2)

        projected_2d = pd.DataFrame(
            pca_2d.fit_transform(features),
            index=features.index,
            columns=["PC1", "PC2"],
        )

        plot_pca_2d(
            projected_2d,
            scaler_name,
        )

    print("\nPCA experiment completed.")


def main():

    for scaler_name in [
        "standard",
        "minmax",
        "robust",
    ]:

        run_experiment(scaler_name)

    print("\nAll PCA experiments completed!")


if __name__ == "__main__":
    main()