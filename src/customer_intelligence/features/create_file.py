from pathlib import Path
import pandas as pd

from customer_intelligence.pipeline.ingestion import load_dataset
from customer_intelligence.features.build_features import clean_transactions

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data" / "raw" / "Online Retail.xlsx"

CLEANED_DATA_PATH = (
    ROOT / "data" / "processed" / "cleaned_transactions.csv"
)


def main():
    print(f"Loading dataset from: {DATA_PATH}")

    df = load_dataset(DATA_PATH)

    print("Cleaning transactions...")
    cleaned_df = clean_transactions(df)

    # Create directory if it doesn't exist
    CLEANED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned transaction-level data
    cleaned_df.to_csv(
        CLEANED_DATA_PATH,
        index=False
    )

    print(f"Cleaned transactions saved to: {CLEANED_DATA_PATH}")
    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print("Cleaning completed successfully!")


if __name__ == "__main__":
    main()
