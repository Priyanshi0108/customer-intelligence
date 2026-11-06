import logging

from customer_intelligence.pipeline.ingestion import load_dataset

from customer_intelligence.features.build_features import (
    clean_transactions,
    build_customer_features
)

logging.basicConfig(level = logging.INFO)

df = load_dataset(
    "data/raw/Online Retail.xlsx"
)



cleaned_df = clean_transactions(df)

customer_features , snapshot_date = (
    build_customer_features(cleaned_df)
)

print("\n Customer Features: ")
print(customer_features.head())

print("\n Dataset Shape: ")
print(customer_features.shape)

print("\n Snapshot Date: ")
print(snapshot_date)


customer_features.to_csv("data/processed/customer_features.csv")


print('\n Feature engineering completed')