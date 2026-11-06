import pandas as pd


def validate_dataset(df:pd.DataFrame) -> dict :      

    report = {}

    report["total_rows"] = len(df)
    report["total_columns"] = len(df.columns)
    report["missing_values"] = (
        df.isnull().sum().to_dict()
    )
    report["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    report["negative_quantity"] = int(
        (df["Quantity"] <0).sum()
    )

    report["non_positive_quantity"] = int(
        (df["Quantity"] <=0).sum()
    )


    report["non_positive_price"] = int(
        (df["UnitPrice"] <= 0).sum()
    )

    report["unique_customers"] = int(
        df["CustomerID"].nunique()
    )

    report["unique_invoices"] = int(
        df["InvoiceNo"].nunique()
    )

    return report