import pandas as pd

from customer_intelligence.features.build_features import (
    clean_transactions,
    build_customer_features,
)


def create_sample_data():

    return pd.DataFrame({
        "CustomerID": [101, 101, 101, 102, 102],

        "InvoiceNo": [
            "I001",
            "I001",
            "I002",
            "I003",
            "I004",
        ],

        "InvoiceDate": pd.to_datetime([
            "2025-01-01",
            "2025-01-01",
            "2025-01-10",
            "2025-01-05",
            "2025-01-20",
        ]),

        "Quantity": [2, 3, 1, 2, 4],

        "UnitPrice": [100, 100, 200, 50, 50],

        "StockCode": [
            "P1",
            "P2",
            "P3",
            "P1",
            "P2",
        ],
    })


def test_clean_transactions():

    df = create_sample_data()

    cleaned = clean_transactions(df)

    assert len(cleaned) == 5

    assert "Revenue" in cleaned.columns

    assert cleaned["Revenue"].sum() == 1000


def test_customer_feature_generation():

    df = create_sample_data()

    cleaned = clean_transactions(df)

    features, snapshot_date = build_customer_features(cleaned)

    assert len(features) == 2

    assert features.loc["101", "frequency"] == 2

    assert features.loc["101", "monetary"] == 700

    assert features.loc["101", "average_order_value"] == 350.0

    assert features.loc["102", "frequency"] == 2

    assert features.loc["102", "monetary"] == 300.0

    assert snapshot_date == pd.Timestamp("2025-01-21")


def test_one_order_customer():

    df = create_sample_data()

    cleaned = clean_transactions(df)

    # Keep only the first invoice of customer 101.
    customer = cleaned[
        (cleaned["CustomerID"] == "101") &
        (cleaned["InvoiceNo"] == "I001")
    ]

    features, _ = build_customer_features(customer)

    assert features.loc["101", "frequency"] == 1

    assert features.loc["101", "avg_purchase_interval"] == 0