# Insight360 — Customer Intelligence Platform

**An end-to-end Machine Learning platform for customer segmentation, behavioral analysis, anomaly detection, churn prediction, and personalized retention recommendations.**

<p align="center">
  <strong>Turning transactional data into actionable customer intelligence.</strong>
</p>

---

## Overview

Insight360 is an end-to-end customer analytics platform built using Python, Machine Learning, Streamlit, and Plotly.

It analyzes historical retail transactions to understand customer purchasing behavior, identify valuable customer segments, detect unusual activity, predict customer inactivity, and recommend personalized retention strategies.

Instead of treating customer analytics as a collection of isolated ML models, Insight360 brings multiple analytical capabilities together in one interactive dashboard.

### Key Objectives

- Understand customer purchasing patterns.
- Identify high-value and low-engagement customers.
- Discover meaningful customer segments.
- Detect unusual customer behavior using multiple algorithms.
- Predict the probability of customer inactivity.
- Explain churn predictions using SHAP.
- Generate personalized retention recommendations.
- Translate ML predictions into actionable business strategies.

---

## Features

### 1. Customer Intelligence Overview

Provides a consolidated view of customer behavior and business performance.

- Total customers and revenue.
- Average purchase frequency.
- Revenue concentration analysis.
- Top customers by monetary value.
- Purchase frequency and recency distributions.
- Customer value versus purchase frequency.
- Feature correlation analysis.
- Interactive charts and global filters.

### 2. Customer Segmentation

Groups customers according to their purchasing behavior.

Customer features include:

- Recency
- Frequency
- Monetary value
- Total quantity
- Product diversity
- Average order value
- Customer tenure
- Average purchase interval

The dashboard provides:

- Customer distribution across segments.
- Revenue contribution by segment.
- Segment-level behavioral profiles.
- Feature comparison heatmaps.
- Radar-based customer profiling.
- Individual segment exploration.

### 3. Anomaly Detection

Uses multiple algorithms to identify unusual customer behavior.

| Algorithm | Purpose |
|---|---|
| Isolation Forest | Identifies observations that are easier to isolate |
| Local Outlier Factor | Detects locally unusual observations |
| IQR | Identifies statistical outliers using quartiles |
| Robust Z-Score | Detects unusual values using median and MAD |

The platform compares detection results using:

- Number of customers flagged by each method.
- Agreement between detection algorithms.
- Overlap between detection methods.
- Customer-level investigation.

**Important:** Agreement between algorithms indicates review priority, not the probability of fraud.

### 4. Customer Investigation

Provides an individual customer-level investigation interface.

For a selected customer, the dashboard displays:

- Revenue and purchase frequency.
- Recency and product diversity.
- Customer behavioral profile.
- Segment comparison.
- Anomaly detection status.
- Transaction history.
- Revenue per invoice.
- Cumulative revenue.
- Most valuable products.

### 5. Model Evaluation

Provides an experiment comparison interface for clustering algorithms.

Algorithms:

- K-Means
- DBSCAN
- Hierarchical Clustering
- Gaussian Mixture Model

Evaluation metrics:

| Metric | Interpretation | Preferred Direction |
|---|---|---|
| Silhouette Score | Cluster separation and cohesion | Higher |
| Davies-Bouldin Index | Similarity between clusters | Lower |
| Calinski-Harabasz Index | Between-cluster versus within-cluster dispersion | Higher |
| Inertia | Within-cluster sum of squared distances | Lower, considering cluster count |
| BIC | Penalized model likelihood | Lower |
| AIC | Penalized model likelihood | Lower |

The dashboard also supports experiment exploration, parameter comparisons, and CSV exports.

### 6. Retention Intelligence

The retention module extends customer analytics into predictive and prescriptive insights.

Its objective is to identify customers who may become inactive and recommend relevant retention interventions.

**Churn Prediction**

- Uses historical purchasing behavior.
- Predicts the probability of customer inactivity.
- Uses a 90-day inactivity definition.
- Applies a chronological train-test split.
- Uses a Random Forest classification pipeline.

Risk categories:

| Risk Category | Predicted Churn Probability |
|---|---|
| High | ≥ 80% |
| Medium | ≥ 60% and < 80% |
| Low | < 60% |

**Model Explainability**

SHAP (SHapley Additive exPlanations) is used to understand the factors influencing predictions.

The dashboard provides:

- Global feature importance.
- Customer-level SHAP explanations.
- Positive and negative feature contributions.
- Feature values associated with individual predictions.

**Personalized Retention Recommendations**

The recommendation engine generates customer-specific interventions using behavioral features and SHAP explanations.

Examples:

| Customer Behavior | Suggested Intervention |
|---|---|
| High recency | Personalized comeback reminder |
| Low monetary value | Small conditional comeback voucher |
| Low purchase frequency | Second-purchase reward |
| Low product diversity | Curated product discovery |
| Low average order value | Bundle or spending-threshold offer |
| Long purchase interval | Reorder reminder |
| Valuable but inactive customer | VIP win-back outreach |

Recommendations include:
- Action type.
- Suggested intervention.
- Priority score.
- Explanation for the recommendation.
- Business safeguard.

**Business safeguards** help prevent excessive discounts, irrelevant communication, and unnecessary incentives.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Explainability | SHAP |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Data Storage | CSV-based processed datasets and reports |
| Experiment Tracking | Saved experiment result files |
| Version Control | Git and GitHub |

---

## Dataset

**UCI Online Retail Dataset**

The project uses the Online Retail transactional dataset, containing historical transactions from a UK-based online retailer.

The dataset includes information such as:

- Invoice number
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

Revenue is calculated from transaction quantity and unit price.

The dataset is used to construct customer-level behavioral features and model-training datasets.

---

## Machine Learning Methodology

### Customer Feature Engineering

Transactional records are aggregated into customer-level features.

| Feature | Description |
|---|---|
| Recency | Days since the customer's last purchase |
| Frequency | Number of distinct invoices |
| Monetary | Total customer spending |
| Total Quantity | Total quantity purchased |
| Product Diversity | Number of distinct products |
| Average Order Value | Average revenue per invoice |
| Average Purchase Interval | Average time between purchases |
| Customer Tenure | Customer relationship duration |

### Churn Label Construction

The retention model uses a forward-looking 90-day inactivity definition.

For each historical snapshot:

1. Customer behavior is calculated using transactions available up to the snapshot date.
2. The following 90 days are examined.
3. If the customer makes no purchase during that period, the churn label is 1.
4. Otherwise, the label is 0.
5. Incomplete future observation windows are excluded.

This creates a time-aware supervised learning dataset.

### Temporal Validation

Random splitting can introduce temporal leakage in historical churn prediction.

Insight360 uses chronological validation with a purge period to prevent overlapping future-label windows from leaking into the test period.

The training period is separated from the test period so that training labels do not extend into the test period.

### Churn Model

The current implementation uses:

**Random Forest Classifier**

Pipeline:

1. Median imputation for missing numerical values.
2. Random Forest classification.
3. Predicted churn probabilities.
4. Customer-level risk categorization.
5. SHAP-based explanations.

Model parameters are configured in the retention training module.

---

## Project Structure

```text
Insight360/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   └── Online Retail Dataset
│   │
│   └── processed/
│       ├── customer_features.csv
│       ├── cleaned_transactions.csv
│       └── churn_training_data.csv
│
├── src/
│   └── customer_intelligence/
│       ├── retention/
│       │   ├── create_churn_dataset.py
│       │   ├── train_churn_model.py
│       │   ├── explain_churn.py
│       │   └── recommend_relief.py
│       │
│       ├── dimensionality_reduction/
│       └── other ML modules
│
├── artifacts/
│   └── retention/
│       └── churn_model.joblib
│
├── reports/
│   ├── cluster_profiling/
│   │   └── customer_segments.csv
│   │
│   ├── anomaly/
│   │   └── all_methods_comparison.csv
│   │
│   ├── kmeans/
│   ├── dbscan/
│   ├── hierarchical/
│   ├── gmm/
│   │
│   └── retention/
│       ├── model_metrics.json
│       ├── test_predictions.csv
│       ├── feature_importance.csv
│       ├── shap_global_importance.csv
│       ├── customer_shap_explanation.csv
│       └── customer_relief_recommendations.csv
│
└── README.md
```

*The structure above highlights the principal project files. Additional modules, reports, and generated artifacts may exist in the repository.*

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- pip.
- Git.
- The UCI Online Retail dataset.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Insight360.git

cd Insight360
```

Replace `YOUR_USERNAME` with your GitHub username and update the repository name if needed.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a requirements file has not yet been created, install the core dashboard dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn shap joblib
```

### 4. Prepare the datasets

Place the raw dataset in the project's expected raw-data directory and execute the project's preprocessing and feature-engineering modules.

Ensure the dashboard's required processed CSV files and experiment reports exist before launching the application.

### 5. Launch the dashboard

```bash
python -m streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Important Model Limitations

Insight360 is a machine-learning portfolio project built using historical transactional data.

Some limitations must be considered before using its predictions for real business decisions.

1. **Churn is a behavioral definition:** No purchase for 90 days does not necessarily mean a customer has permanently left.
2. **Historical dataset:** The model learns from past purchasing patterns and may not capture future market changes.
3. **Limited observation windows:** Monthly snapshots and purged temporal validation can leave relatively few independent time periods for training and testing.
4. **SHAP is not causal:** SHAP explains model predictions; it does not establish why a customer leaves or prove that an intervention will prevent churn.
5. **Intervention effectiveness:** Discounts, reminders, and other recommendations are hypotheses that require controlled A/B testing.
6. **Limited business context:** The dataset does not contain customer satisfaction, campaign outcomes, marketing costs, or shipping experience.

The current retention dashboard uses available held-out test predictions. It should not be interpreted as a live production scoring system for every customer.

---

## Future Improvements

- Live customer scoring using the latest available transactions.
- Model monitoring and prediction drift detection.
- Automated retention campaign workflows.
- A/B testing framework for intervention effectiveness.
- Customer lifetime value prediction.
- Uplift modeling to identify customers most likely to respond to an intervention.
- Real-time transaction ingestion.
- Database integration.
- API-based model serving using FastAPI.
- Model versioning and experiment tracking.
- Deployment using Docker and cloud infrastructure.

---

## Why Insight360?

Many analytics systems stop after identifying customer segments or predicting churn.

Insight360 aims to connect the complete analytical workflow:

**Transaction Data → Customer Features → Segmentation → Anomaly Detection → Churn Prediction → SHAP Explainability → Personalized Retention Actions**

The central idea is to move beyond asking:

*Which customers are likely to leave?*

And toward asking:

*Why might they become inactive, which customers should we prioritize, and what action could we test to improve retention?*

---

## Author

**Priyanshi Singh**

B.Tech — Computer Science and Engineering

Interested in Machine Learning, AI Engineering, Data Science, and intelligent decision-support systems.

---

<p align="center">
  <strong>Insight360 — From Customer Data to Intelligent Decisions.</strong>
</p>