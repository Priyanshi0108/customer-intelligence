
import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "data/processed/customer_features.csv",
    index_col=0,
)

feature_columns = [
    "recency_days",
    "frequency",
    "monetary",
    "total_quantity",
    "product_diversity",
    "average_order_value",
    "customer_tenure_days",
    "avg_purchase_interval",
]

X = df[feature_columns]
X_log = np.log1p(X)

X_scaled = pd.DataFrame(
    StandardScaler().fit_transform(X_log),
    columns=X.columns,
    index=X.index,
)

k_values = [5, 10, 20, 35, 50]

anomaly_sets = {}

for k in k_values:

    model = LocalOutlierFactor(
        n_neighbors=k,
        contamination=0.05,
    )

    labels = model.fit_predict(X_scaled)

    anomaly_sets[k] = set(
        X_scaled.index[labels == -1]
    )

# Compare every configuration with K=20
baseline = anomaly_sets[20]

print("Overlap with K=20:")

for k in k_values:

    common = baseline & anomaly_sets[k]

    print(
        f"K={k}: "
        f"{len(common)} common customers, "
        f"{len(common) / len(baseline) * 100:.2f}% overlap"
    )

# Compare consecutive configurations
print("\nConsecutive K comparison:")

for i in range(len(k_values) - 1):

    k1 = k_values[i]
    k2 = k_values[i + 1]

    common = anomaly_sets[k1] & anomaly_sets[k2]

    print(
        f"K={k1} vs K={k2}: "
        f"{len(common)} common customers"
    )
