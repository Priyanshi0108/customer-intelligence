
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

results = []

for k in [5, 10, 20, 35, 50]:

    model = LocalOutlierFactor(
        n_neighbors=k,
        contamination=0.05,
    )

    labels = model.fit_predict(X_scaled)

    scores = -model.negative_outlier_factor_

    results.append({
        "n_neighbors": k,
        "anomaly_count": (labels == -1).sum(),
        "mean_anomaly_score": scores.mean(),
        "median_anomaly_score": np.median(scores),
    })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    "reports/anomaly/lof_tuning.csv",
    index=False,
)
