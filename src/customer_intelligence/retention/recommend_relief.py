
from pathlib import Path

import pandas as pd

from src.customer_intelligence.retention.train_churn_model import (
    FEATURES,
    chronological_split,
    load_data,
)


ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = ROOT / "reports" / "retention" / "test_predictions.csv"
SHAP_PATH = ROOT / "reports" / "retention" / "customer_shap_explanation.csv"
OUTPUT_PATH = ROOT / "reports" / "retention" / "customer_relief_recommendations.csv"

HIGH_RISK_THRESHOLD = 0.60


class PersonalizedReliefEngine:

    def __init__(self, training_data):

        self.thresholds = {}

        # Learn behavioural thresholds from training data only.
        for feature in FEATURES:

            values = pd.to_numeric(
                training_data[feature],
                errors="coerce",
            ).dropna()

            self.thresholds[feature] = {
                "low": values.quantile(0.25),
                "high": values.quantile(0.75),
            }

    def is_low(self, feature, value):
        threshold = self.thresholds[feature]["low"]
        return pd.notna(value) and value <= threshold

    def is_high(self, feature, value):
        threshold = self.thresholds[feature]["high"]
        return pd.notna(value) and value >= threshold

    @staticmethod
    def positive_shap(shap_values, feature):
        return shap_values.get(feature, 0) > 0

    def recommend(self, customer, shap_values, churn_probability):

        monetary_actions = []
        non_monetary_actions = []

        def add_monetary(priority, action, reason, safeguard):
            monetary_actions.append({
                "priority": priority,
                "type": "Monetary relief",
                "action": action,
                "reason": reason,
                "safeguard": safeguard,
            })

        def add_non_monetary(priority, action, reason, safeguard):
            non_monetary_actions.append({
                "priority": priority,
                "type": "Non-monetary relief",
                "action": action,
                "reason": reason,
                "safeguard": safeguard,
            })

        # 1. Recency
        recency = customer["recency_days"]

        if (
            self.is_high("recency_days", recency)
            and self.positive_shap(shap_values, "recency_days")
        ):
            add_non_monetary(
                100,
                "Personalized comeback reminder",
                (
                    f"The customer has not purchased for {int(recency)} days, "
                    "and recency increases the model's inactivity prediction."
                ),
                "Send a limited number of relevant reminders.",
            )

        # 2. Historical monetary value
        monetary = customer["monetary"]

        if (
            self.is_low("monetary", monetary)
            and self.positive_shap(shap_values, "monetary")
        ):
            add_monetary(
                90,
                "Small conditional comeback voucher",
                "Low historical spending contributes positively to predicted inactivity.",
                "Use a minimum basket threshold and a maximum incentive budget.",
            )

        elif (
            self.is_high("monetary", monetary)
            and self.is_high("recency_days", recency)
            and self.positive_shap(shap_values, "recency_days")
        ):
            add_monetary(
                100,
                "Targeted VIP win-back reward",
                "Historically high customer value combined with elevated inactivity risk.",
                "Set a reward cap based on customer lifetime value.",
            )

        # 3. Purchase frequency
        frequency = customer["frequency"]

        if (
            self.is_low("frequency", frequency)
            and self.positive_shap(shap_values, "frequency")
        ):
            add_monetary(
                70,
                "Second-purchase loyalty reward",
                "Limited purchase frequency contributes to predicted inactivity.",
                "Reward a completed repeat purchase, not just a click.",
            )

            add_non_monetary(
                80,
                "Personalized repeat-purchase suggestions",
                "The customer has a limited purchasing history.",
                "Recommend relevant products, not random products.",
            )

        # 4. Product diversity
        diversity = customer["product_diversity"]

        if (
            self.is_low("product_diversity", diversity)
            and self.positive_shap(shap_values, "product_diversity")
        ):
            add_non_monetary(
                90,
                "Curated product discovery",
                (
                    f"The customer has purchased only {int(diversity)} distinct products, "
                    "and this feature increases predicted inactivity."
                ),
                "Base suggestions on previously purchased products.",
            )

        # 5. Average order value
        aov = customer["average_order_value"]

        if (
            self.is_low("average_order_value", aov)
            and self.positive_shap(shap_values, "average_order_value")
        ):
            add_monetary(
                60,
                "Bundle or spend-threshold offer",
                "Low average order value contributes to inactivity risk.",
                "Ensure the threshold is realistic and the offer remains profitable.",
            )

        # 6. Total quantity
        quantity = customer["total_quantity"]

        if (
            self.is_low("total_quantity", quantity)
            and self.positive_shap(shap_values, "total_quantity")
        ):
            add_non_monetary(
                50,
                "Relevant product bundle suggestions",
                "Low historical quantity contributes to inactivity risk.",
                "Suggest useful complementary products.",
            )

        # 7. Customer tenure
        tenure = customer["customer_tenure_days"]

        if (
            self.is_low("customer_tenure_days", tenure)
            and self.positive_shap(shap_values, "customer_tenure_days")
        ):
            add_non_monetary(
                40,
                "New-customer engagement journey",
                "The customer has limited established purchase history.",
                "Avoid treating a new customer as a loyal customer.",
            )

        # 8. Average purchase interval
        interval = customer["avg_purchase_interval"]

        if (
            frequency >= 2
            and pd.notna(interval)
            and interval > 0
            and self.is_high("avg_purchase_interval", interval)
            and self.positive_shap(shap_values, "avg_purchase_interval")
        ):
            add_non_monetary(
                60,
                "Expected reorder reminder",
                "The customer's historical purchase interval is relatively long.",
                "Time the reminder around the customer's usual cycle.",
            )

        # Select highest-priority actions.
        monetary_actions.sort(
            key=lambda item: item["priority"],
            reverse=True,
        )

        non_monetary_actions.sort(
            key=lambda item: item["priority"],
            reverse=True,
        )

        selected_monetary = monetary_actions[:1]
        selected_non_monetary = non_monetary_actions[:2]

        if not selected_monetary:
            selected_monetary = [{
                "priority": 0,
                "type": "Monetary relief",
                "action": "No monetary incentive recommended",
                "reason": "Available behavioural and SHAP signals do not justify a monetary offer.",
                "safeguard": "Reassess after new purchasing activity.",
            }]

        if not selected_non_monetary:
            selected_non_monetary = [{
                "priority": 0,
                "type": "Non-monetary relief",
                "action": "Personalized engagement message",
                "reason": "No specific behavioural trigger was identified.",
                "safeguard": "Avoid excessive communication.",
            }]

        return selected_monetary + selected_non_monetary


def main():

    print("Loading customer data...")

    df = load_data()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    train_df, _ = chronological_split(df)

    engine = PersonalizedReliefEngine(train_df)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    shap_df = pd.read_csv(SHAP_PATH)

    predictions["snapshot_date"] = pd.to_datetime(
        predictions["snapshot_date"]
    )

    shap_df["snapshot_date"] = pd.to_datetime(
        shap_df["snapshot_date"]
    )

    # --------------------------------------------------
    # 1. Select all high-risk customers
    # --------------------------------------------------

    high_risk = predictions[
        predictions["churn_probability"] >= HIGH_RISK_THRESHOLD
    ].copy()

    high_risk = high_risk.sort_values(
        "churn_probability",
        ascending=False,
    )

    if high_risk.empty:
        print("No customers crossed the high-risk threshold.")
        return

    print("\nHIGH-RISK CUSTOMER SELECTION")
    print("----------------------------")
    print("Risk threshold:", f"{HIGH_RISK_THRESHOLD:.0%}")
    print("Selected customers:", len(high_risk))

    # --------------------------------------------------
    # 2. Generate personalized recommendations
    # --------------------------------------------------

    output_rows = []

    for position, (_, prediction) in enumerate(
        high_risk.iterrows(),
        start=1,
    ):

        customer_id = prediction["CustomerID"]
        snapshot_date = prediction["snapshot_date"]
        churn_probability = float(prediction["churn_probability"])

        # Find the customer's exact historical snapshot.
        matched_customer = df[
            (df["CustomerID"] == customer_id)
            & (df["snapshot_date"] == snapshot_date)
        ]

        if matched_customer.empty:
            print(
                f"Skipping customer {customer_id}: "
                "matching feature row not found."
            )
            continue

        customer = matched_customer.iloc[0]

        # Retrieve only this customer's SHAP rows.
        customer_shap = shap_df[
            (shap_df["CustomerID"] == customer_id)
            & (shap_df["snapshot_date"] == snapshot_date)
        ]

        if customer_shap.empty:
            print(
                f"Skipping customer {customer_id}: "
                "SHAP explanation not found."
            )
            continue

        # Convert their local explanation into a feature dictionary.
        shap_values = dict(
            zip(
                customer_shap["feature"],
                customer_shap["shap_value"],
            )
        )

        # Identify the strongest local SHAP contributor.
        top_feature_row = customer_shap.loc[
            customer_shap["absolute_shap"].idxmax()
        ]

        top_feature = top_feature_row["feature"]
        top_shap_value = float(top_feature_row["shap_value"])

        recommendations = engine.recommend(
            customer,
            shap_values,
            churn_probability,
        )

        # Attach customer-specific metadata to every action.
        for item in recommendations:

            output_rows.append({
                "CustomerID": customer_id,
                "snapshot_date": snapshot_date,
                "churn_probability": churn_probability,
                "top_shap_feature": top_feature,
                "top_shap_value": top_shap_value,
                **item,
            })

        if position % 25 == 0 or position == len(high_risk):
            print(
                f"Processed {position}/{len(high_risk)} high-risk customers"
            )

    # --------------------------------------------------
    # 3. Save all customer-level action plans
    # --------------------------------------------------

    if not output_rows:
        raise ValueError(
            "No recommendations were generated. "
            "Check the customer IDs, dates, and SHAP report."
        )

    result = pd.DataFrame(output_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nRETENTION RECOMMENDATIONS COMPLETED")
    print("-----------------------------------")

    print("High-risk customers selected:", len(high_risk))
    print(
        "Customers with recommendations:",
        result["CustomerID"].nunique(),
    )
    print("Total action recommendations:", len(result))

    print("\nAction distribution:")
    print(
        result.groupby(["type", "action"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nRecommendations saved to:")
    print(OUTPUT_PATH)

    print("\nSample recommendations:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
