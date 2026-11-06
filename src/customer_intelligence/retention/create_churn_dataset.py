
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

TRANSACTION_PATH = (
    ROOT / "data" / "processed" / "cleaned_transactions.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "churn_training_data.csv"
)

CHURN_HORIZON_DAYS = 90


# --------------------------------------------------
# Load and validate transactions
# --------------------------------------------------

def load_transactions() -> pd.DataFrame:
    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {TRANSACTION_PATH}"
        )

    df = pd.read_csv(TRANSACTION_PATH)

    required_columns = {
        "InvoiceNo",
        "StockCode",
        "InvoiceDate",
        "Quantity",
        "CustomerID",
        "Revenue",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )

    df["Revenue"] = pd.to_numeric(
        df["Revenue"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "CustomerID",
            "InvoiceDate",
            "InvoiceNo",
            "Revenue",
            "Quantity",
        ]
    ).copy()

    df["CustomerID"] = df["CustomerID"].astype("int64")

    df = df.sort_values("InvoiceDate")

    if df.empty:
        raise ValueError("No valid transactions found.")

    return df


# --------------------------------------------------
# Generate customer features using past transactions
# --------------------------------------------------

def build_customer_features(
    history: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:

    # One row per customer invoice
    invoices = (
        history.groupby(
            ["CustomerID", "InvoiceNo"],
            as_index=False,
        )
        .agg(
            invoice_date=("InvoiceDate", "max"),
            invoice_revenue=("Revenue", "sum"),
        )
    )

    # Customer-level historical features
    features = (
        history.groupby("CustomerID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            first_purchase=("InvoiceDate", "min"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Revenue", "sum"),
            total_quantity=("Quantity", "sum"),
            product_diversity=("StockCode", "nunique"),
        )
    )

    # Average invoice value
    average_order_value = (
        invoices.groupby("CustomerID")["invoice_revenue"]
        .mean()
        .rename("average_order_value")
    )

    features = features.join(average_order_value)

    # Average time between distinct purchase dates
    purchase_dates = (
        invoices.groupby("CustomerID")["invoice_date"]
        .apply(lambda dates: dates.sort_values().drop_duplicates())
    )

    def calculate_average_interval(dates):
        intervals = dates.diff().dt.days.dropna()

        if intervals.empty:
            return np.nan

        return intervals.mean()

    average_intervals = (
        purchase_dates.groupby(level=0)
        .apply(calculate_average_interval)
        .rename("avg_purchase_interval")
    )

    features = features.join(average_intervals)

    # Time-based features
    features["recency_days"] = (
        snapshot_date - features["last_purchase"]
    ).dt.days

    features["customer_tenure_days"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days

    features["snapshot_date"] = snapshot_date

    # Remove temporary date columns
    features = features.drop(
        columns=["last_purchase", "first_purchase"]
    )

    return features.reset_index()


# --------------------------------------------------
# Generate historical churn labels
# --------------------------------------------------

def create_churn_dataset(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    first_date = transactions["InvoiceDate"].min().normalize()
    last_date = transactions["InvoiceDate"].max().normalize()

    horizon = pd.Timedelta(days=CHURN_HORIZON_DAYS)

    # Only use snapshots with a complete future window
    first_snapshot = first_date + horizon
    last_snapshot = last_date - horizon

    snapshot_dates = pd.date_range(
        start=first_snapshot,
        end=last_snapshot,
        freq="MS",
    )

    if len(snapshot_dates) == 0:
        raise ValueError(
            "Not enough historical data for a complete "
            "90-day churn observation window."
        )

    print(f"Dataset starts: {first_date.date()}")
    print(f"Dataset ends: {last_date.date()}")
    print(f"Snapshots generated: {len(snapshot_dates)}")

    all_snapshots = []

    for snapshot_date in snapshot_dates:

        print(f"Processing snapshot: {snapshot_date.date()}")

        # Past data: used only for features
        history = transactions[
            transactions["InvoiceDate"] <= snapshot_date
        ].copy()

        # Future data: used only for churn labels
        future_end = snapshot_date + horizon

        future = transactions[
            (transactions["InvoiceDate"] > snapshot_date)
            & (transactions["InvoiceDate"] <= future_end)
        ]

        # Build features from historical behaviour
        customer_features = build_customer_features(
            history,
            snapshot_date,
        )

        # Customers who purchased in the next 90 days
        future_active_customers = set(
            future["CustomerID"].unique()
        )

        # 1 = no purchase in next 90 days
        # 0 = at least one purchase in next 90 days
        customer_features["churn_90d"] = (
            ~customer_features["CustomerID"].isin(
                future_active_customers
            )
        ).astype(int)

        all_snapshots.append(customer_features)

    result = pd.concat(
        all_snapshots,
        ignore_index=True,
    )

    # Basic validation
    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.sort_values(
        ["snapshot_date", "CustomerID"]
    ).reset_index(drop=True)

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading transactions...")

    transactions = load_transactions()

    print(f"Transactions loaded: {len(transactions):,}")

    print("\nGenerating churn training dataset...")

    churn_data = create_churn_dataset(transactions)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    churn_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDataset successfully created!")
    print(f"Saved at: {OUTPUT_PATH}")

    print("\nDataset shape:")
    print(churn_data.shape)

    print("\nColumns:")
    print(churn_data.columns.tolist())

    print("\nChurn distribution:")
    print(
        churn_data["churn_90d"]
        .value_counts()
        .rename(index={0: "Active", 1: "Inactive"})
    )

    print("\nChurn percentage:")
    print(
        (
            churn_data["churn_90d"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print("\nPreview:")
    print(churn_data.head())


if __name__ == "__main__":
    main()
