import pandas as pd
import pytest

from customer_intelligence.pipeline.ingestion import load_dataset


def test_load_dataset_successfully(tmp_path):

    data = {
        "InvoiceNo": ["1001"],
        "StockCode": ["P101"],
        "Quantity": [2],
        "InvoiceDate": ["2025-01-01"],
        "UnitPrice": [10],
        "CustomerID": [101],
        "Country": ["India"],
    }

    file_path = tmp_path / "sample.csv"

    pd.DataFrame(data).to_csv(
        file_path,
        index=False
    )

    df = load_dataset(file_path)

    assert len(df) == 1


def test_missing_file():

    with pytest.raises(FileNotFoundError):

        load_dataset("non_existing.csv")