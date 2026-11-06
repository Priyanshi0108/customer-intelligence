from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT /"data"/"processed"/"customer_features.csv"

REPORT_DIR = PROJECT_ROOT/"reports"/"eda"
FIGURE_DIR = PROJECT_ROOT / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_features() ->pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Customer feature file not found: {DATA_PATH}\n"
            "Run feature engineering before EDA."
        )

    df = pd.read_csv(DATA_PATH)

    df.index.name = "CustomerID"

    if df.empty:
        raise ValueError("Customer feature dataset is empty")

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    
    print("\n========== DATASET OVERVIEW ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five customers:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate customer IDs:")
    print(df.index.duplicated().sum())

    print("\nDuplicate feature rows:")
    print(df.duplicated().sum())

    print("\nBasic statistical summary:")
    print(df.describe().T)


def calculate_statistics(df:pd.DataFrame) -> pd.DataFrame:

    numeric_df = df.select_dtypes(include = np.number)

    if numeric_df.empty:
        raise ValueError("No numeric features available")

    statistics = numeric_df.describe().T

    statistics["skewness"] = numeric_df.skew()
    statistics["kurtosis"] = numeric_df.kurtosis()
    statistics["zero_count"] = (numeric_df ==0).sum()



    statistics.to_csv(
        REPORT_DIR / "feature_statistics.csv"
    )


    print("\n========== FEATURE STATISTICS ==========")
    print(statistics)

    return statistics



def detect_outliers(df:pd.DataFrame)->pd.DataFrame:

    numeric_df = df.select_dtypes(include = np.number)

    outlier_results = [] 

    for column in numeric_df.columns:

        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q1 + 1.5 * iqr

        outlier_mask = (
            (numeric_df[column] < lower_bound)
            | (numeric_df[column] > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        outlier_percentage = (
            (outlier_count)/len(numeric_df)
        )*100

        outlier_results.append({
            "feature": column,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
        })

    outlier_report = pd.DataFrame(outlier_results)    


    outlier_report.to_csv(
        REPORT_DIR / "outlier_report.csv",
        index = False
    )


    print("\n========== OUTLIER REPORT ==========")
    print(outlier_report)

    return outlier_report


def plot_distribution(df:pd.DataFrame)->None:
    numeric_df = df.select_dtypes(include=np.number)

    for column in numeric_df.columns:
        plt.figure(figsize=(9,5))

        sns.histplot(
            data = numeric_df,
            x= column,
            bins = 50 ,
            kde = True
        )

        plt.xlabel(column)
        plt.ylabel("Number of customers")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{column}_distribution.png",
            dpi=150
        )

        plt.close()

    print("\nDistribution plots saved.")


def plot_boxplots(df:pd.DataFrame) ->None:
    numeric_df = df.select_dtypes(include=np.number)

    for column in numeric_df.columns:
        plt.figure(figsize=(9,4))
        sns.boxplot(
            x = numeric_df[column]
        )

        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{column}_boxplot.png",
            dpi=150
        )

        plt.close()

    print("\nBoxplots saved.")


def analyze_correlations(df:pd.DataFrame) ->pd.DataFrame:
    numeric_df = df.select_dtypes(include= np.number)

    correlation_matrix = numeric_df.corr(method = "person")

    correlation_matrix.to_csv(
        REPORT_DIR/ "correlation_matrix.csv"
    )

    plt.figure(figsize = (11,8))

    sns.heatmap(
        correlation_matrix,
        annot = True,
        cmap = "coolwarm",
        fmt = ".2f",
        linewidths=0.5
    )

    plt.title("Customer Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi = 150
    )

    plt.close()

    print("CORRELATION MATRIX")
    print(correlation_matrix)


def explore_log_transform(
        df:pd.DataFrame,
        statistics:pd.DataFrame
)->None:
    numeric_df = df.select_dtypes(include= np.number)

    skewed_features = statistics[
        statistics["skewness"]>1
    ].index


    for column in skewed_features:

        if (numeric_df[column]<0).any():
            print(
                f"Skipping {column}: contains negative values"
            )

            continue

        transformed = np.log1p(numeric_df[column])

        plt.figure(figsize =(9,5))

        sns.histplot(
            transformed,
            bins=50,
            kde = True
        )


        plt.title(f"Log1p Distribution of {column}")
        plt.xlabel(f"log1p({column})")
        plt.ylabel("Number of customers")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{column}_log_distribution.png",
            dpi=150
        )

        plt.close()

    print("\nLog transformation exploration completed.")


# --------------------------------------------------
# 10. Main execution
# --------------------------------------------------

def main():

    print("Starting Customer Intelligence EDA...")

    df = load_features()

    inspect_dataset(df)

    statistics = calculate_statistics(df)

    detect_outliers(df)

    plot_distributions(df)

    plot_boxplots(df)

    analyze_correlation(df)

    explore_log_transform(df, statistics)

    print("\nEDA completed successfully!")

    print(f"\nReports saved in: {REPORT_DIR}")
    print(f"Figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()