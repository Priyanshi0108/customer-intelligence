
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/Online Retail.xlsx"

df = pd.read_excel(DATA_PATH)

# Customers identified by Isolation Forest
customer_ids = [
    14646, 14911, 18102, 17450, 14156,
    12415, 16446, 17511, 13089, 15311,
]

# Basic transaction cleaning
df["CustomerID"] = pd.to_numeric(
    df["CustomerID"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df = df.dropna(
    subset=["CustomerID", "Quantity", "UnitPrice", "InvoiceDate"]
)

df = df[
    (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
    & (~df["InvoiceNo"].astype(str).str.startswith("C"))
]

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Filter selected customers
selected = df[
    df["CustomerID"].isin(customer_ids)
].copy()

# Invoice-level analysis
invoice_summary = (
    selected.groupby(["CustomerID", "InvoiceNo"])
    .agg(
        invoice_date=("InvoiceDate", "min"),
        total_quantity=("Quantity", "sum"),
        invoice_revenue=("Revenue", "sum"),
        unique_products=("StockCode", "nunique"),
    )
    .reset_index()
)

# Customer-level analysis
customer_summary = (
    invoice_summary.groupby("CustomerID")
    .agg(
        invoice_count=("InvoiceNo", "nunique"),
        total_revenue=("invoice_revenue", "sum"),
        average_invoice_value=("invoice_revenue", "mean"),
        max_invoice_value=("invoice_revenue", "max"),
        total_quantity=("total_quantity", "sum"),
    )
    .sort_values("total_revenue", ascending=False)
)

print("\nCustomer transaction summary:")
print(customer_summary.to_string())

print("\nLargest individual invoices:")
print(
    invoice_summary.sort_values(
        "invoice_revenue",
        ascending=False,
    ).head(20).to_string(index=False)
)

OUTPUT_DIR = ROOT / "reports/anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

customer_summary.to_csv(
    OUTPUT_DIR / "investigated_customer_summary.csv"
)

invoice_summary.to_csv(
    OUTPUT_DIR / "investigated_invoice_details.csv",
    index=False,
)


invoice = df[
    (df["CustomerID"] == 16446)
    & (df["InvoiceNo"].astype(str) == "581483")
]

print(
    invoice[
        [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "UnitPrice",
            "Revenue",
            "InvoiceDate",
        ]
    ].to_string(index=False)
)
