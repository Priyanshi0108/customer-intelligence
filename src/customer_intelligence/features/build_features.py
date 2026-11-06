import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "CustomerID",
    "InvoiceNo",
    "InvoiceDate",
    "Quantity",
    "UnitPrice",
    "StockCode",
}


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction-level retail data.

    Removes:
    - Missing customer identifiers
    - Invalid dates
    - Cancelled invoices
    - Non-positive quantities
    - Non-positive unit prices
    - Duplicate rows
    """

    df = df.copy()

    # Check whether all required columns exist.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    initial_rows = len(df)

    # Convert columns into expected data types.
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    # Remove rows containing missing or invalid values.
    df = df.dropna(
        subset=[
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "StockCode",
        ]
    )

    # Remove duplicate transaction rows.
    df = df.drop_duplicates()

    # Remove cancelled invoices.
    cancelled = (
        df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    )

    df = df[~cancelled]

    # Remove returned products and invalid prices.
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ]

    # Standardize CustomerID.
    df["CustomerID"] = (
        df["CustomerID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    # Calculate revenue for each transaction row.
    df["Revenue"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    removed_rows = initial_rows - len(df)

    logger.info(
        "Cleaning completed. Removed %d rows.",
        removed_rows
    )

    if df.empty:
        raise ValueError(
            "No valid transactions remain after cleaning."
        )

    return df


def build_customer_features(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Convert transaction-level data into customer-level features.

    Features:
    - Recency
    - Frequency
    - Monetary
    - Average Order Value
    - Total Quantity
    - Product Diversity
    - Customer Tenure
    - Average Purchase Interval
    """

    df = transactions.copy()

    if df.empty:
        raise ValueError(
            "Input transactions cannot be empty."
        )

    required_columns = REQUIRED_COLUMNS

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Calculate revenue if it does not already exist.
    if "Revenue" not in df.columns:
        df["Revenue"] = (
            df["Quantity"] * df["UnitPrice"]
        )

    # Reference date = one day after the latest transaction.
    snapshot_date = (
        df["InvoiceDate"].max().normalize()
        + pd.Timedelta(days=1)
    )

    # Calculate revenue for each customer invoice.
    invoice_revenue = (
        df.groupby(
            ["CustomerID", "InvoiceNo"]
        )["Revenue"]
        .sum()
        .reset_index()
    )

    # Calculate average order value for each customer.
    average_order_value = (
        invoice_revenue.groupby("CustomerID")["Revenue"]
        .mean()
        .rename("average_order_value")
    )

    # Aggregate transaction records into customer-level features.
    customer_features = (
        df.groupby("CustomerID")
        .agg(
            recency_days=(
                "InvoiceDate",
                lambda dates: (
                    snapshot_date - dates.max().normalize()
                ).days,
            ),

            frequency=(
                "InvoiceNo",
                "nunique",
            ),

            monetary=(
                "Revenue",
                "sum",
            ),

            total_quantity=(
                "Quantity",
                "sum",
            ),

            product_diversity=(
                "StockCode",
                "nunique",
            ),

            first_purchase=(
                "InvoiceDate",
                "min",
            ),

            last_purchase=(
                "InvoiceDate",
                "max",
            ),
        )
    )

    # Add average order value.
    customer_features = customer_features.join(
        average_order_value
    )

    # Calculate customer tenure.
    customer_features["customer_tenure_days"] = (
        customer_features["last_purchase"]
        - customer_features["first_purchase"]
    ).dt.days

    # Calculate average time between purchases.
    customer_features["avg_purchase_interval"] = (
        customer_features["customer_tenure_days"]
        / (customer_features["frequency"] - 1).clip(lower=1)
    )

    # Remove intermediate date columns.
    customer_features = customer_features.drop(
        columns=[
            "first_purchase",
            "last_purchase",
        ]
    )

    # Handle infinite and missing values.
    customer_features = (
        customer_features
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    if customer_features.isnull().any().any():
        raise ValueError(
            "Missing values detected in customer features."
        )

    logger.info(
        "Feature engineering completed for %d customers.",
        len(customer_features)
    )


    
      
    return customer_features, snapshot_date


