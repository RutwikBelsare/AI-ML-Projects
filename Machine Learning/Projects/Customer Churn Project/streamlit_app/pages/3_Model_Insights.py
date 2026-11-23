import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.title("🧠 Model Insights")

st.markdown(
    """
    Technical and business-oriented information about the
    Customer Churn Prediction model.
    """
)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

if METADATA_PATH.exists():

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

else:

    st.error(
        "Model metadata file was not found."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# EXTRACT METADATA
# ============================================================

model_type = metadata.get(
    "model_type",
    "Unknown"
)

target = metadata.get(
    "target",
    "Churn"
)

random_state = metadata.get(
    "random_state",
    42
)

production_threshold = metadata.get(
    "production_threshold",
    0.30
)

best_cv_pr_auc = metadata.get(
    "best_cv_pr_auc",
    None
)

default_metrics = metadata.get(
    "test_metrics_default_threshold",
    {}
)

production_metrics = metadata.get(
    "test_metrics_production_threshold",
    {}
)

best_parameters = metadata.get(
    "best_parameters",
    {}
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📊 Dataset Overview")

total_customers = len(df)

total_features = len(
    [
        column
        for column in df.columns
        if column not in [
            "customerID",
            "Churn"
        ]
    ]
)

churned_customers = (
    df["Churn"] == "Yes"
).sum()

churn_rate = (
    churned_customers
    / total_customers
    * 100
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Features",
        total_features
    )


with col3:

    st.metric(
        "Churned",
        f"{churned_customers:,}"
    )


with col4:

    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


st.divider()


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.subheader("🤖 Model Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Model",
        model_type
    )


with col2:

    st.metric(
        "Target",
        target
    )


with col3:

    st.metric(
        "Production Threshold",
        f"{production_threshold:.0%}"
    )


with col4:

    st.metric(
        "Random State",
        random_state
    )


if best_cv_pr_auc is not None:

    st.info(
        f"Best cross-validation PR-AUC: "
        f"**{best_cv_pr_auc:.4f}**"
    )


st.divider()


# ============================================================
# TEST SET PERFORMANCE
# ============================================================

st.subheader("📈 Test Set Performance")

st.markdown(
    """
    The same trained XGBoost model can be evaluated using
    different classification thresholds.

    The probability ranking itself does not change when the
    threshold changes; what changes is the final Yes/No
    classification.
    """
)


performance_data = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC",
        ],

        "50% Threshold": [
            default_metrics.get("accuracy"),
            default_metrics.get("precision"),
            default_metrics.get("recall"),
            default_metrics.get("f1"),
            default_metrics.get("roc_auc"),
            default_metrics.get("pr_auc"),
        ],

        "30% Threshold": [
            production_metrics.get("accuracy"),
            production_metrics.get("precision"),
            production_metrics.get("recall"),
            production_metrics.get("f1"),
            None,
            None,
        ],
    }
)


st.dataframe(
    performance_data.style.format(
        {
            "50% Threshold": "{:.3f}",
            "30% Threshold": "{:.3f}",
        },
        na_rep="—"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# KEY METRICS
# ============================================================

st.subheader("Key Model Metrics")


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "ROC-AUC",
        f"{default_metrics.get('roc_auc', 0):.3f}"
    )


with col2:

    st.metric(
        "PR-AUC",
        f"{default_metrics.get('pr_auc', 0):.3f}"
    )


with col3:

    st.metric(
        "Precision @ 30%",
        f"{production_metrics.get('precision', 0):.3f}"
    )


with col4:

    st.metric(
        "Recall @ 30%",
        f"{production_metrics.get('recall', 0):.3f}"
    )


with col5:

    st.metric(
        "F1 @ 30%",
        f"{production_metrics.get('f1', 0):.3f}"
    )


st.divider()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

st.subheader("🎯 Classification Threshold Analysis")

st.markdown(
    """
    The model produces a churn probability.

    We then apply a business-defined threshold:

    **Probability ≥ threshold → predict churn**

    The production threshold is **30%** because the business
    objective is to identify more potential churners and therefore
    prioritize recall.
    """
)


threshold_data = pd.DataFrame(
    {
        "Threshold": [
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
        ],

        "Precision": [
            0.484,
            0.508,
            0.535,
            0.558,
            0.585,
            0.609,
            0.653,
            0.684,
            0.731,
            0.765,
            0.789,
        ],

        "Recall": [
            0.834,
            0.791,
            0.765,
            0.706,
            0.655,
            0.591,
            0.537,
            0.457,
            0.393,
            0.340,
            0.270,
        ],

        "F1": [
            0.613,
            0.619,
            0.629,
            0.623,
            0.618,
            0.600,
            0.589,
            0.548,
            0.511,
            0.470,
            0.402,
        ],
    }
)


st.dataframe(
    threshold_data.style.format(
        {
            "Threshold": "{:.0%}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PRECISION VS RECALL
# ============================================================

st.subheader("Precision vs Recall")


chart_df = threshold_data.set_index(
    "Threshold"
)[
    [
        "Precision",
        "Recall",
    ]
]


st.line_chart(
    chart_df
)


st.info(
    """
    **Why use a 30% threshold?**

    At the 30% threshold, recall is approximately **77%**.
    This means the model identifies a substantially larger
    proportion of actual churners than it would at the default
    50% threshold.

    The trade-off is lower precision and therefore more false
    positives.

    In a real production environment, the final threshold should
    be selected using retention campaign cost, customer lifetime
    value, intervention capacity, and the cost of missed churners.
    """
)


st.divider()


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔢 Confusion Matrix")

st.markdown(
    """
    The confusion matrix below uses the test-set results already
    obtained during model evaluation.

    **30% threshold:**

    - True Negatives: 773
    - False Positives: 262
    - False Negatives: 86
    - True Positives: 288
    """
)


# These values come from our previously evaluated test results.
production_cm = [
    [773, 262],
    [86, 288],
]


fig, ax = plt.subplots()

ax.imshow(
    production_cm
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    [
        "Predicted No Churn",
        "Predicted Churn"
    ]
)

ax.set_yticklabels(
    [
        "Actual No Churn",
        "Actual Churn"
    ]
)

ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title(
    "Confusion Matrix — Production Threshold 30%"
)


for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            production_cm[i][j],
            ha="center",
            va="center"
        )


st.pyplot(
    fig,
    clear_figure=True
)


st.divider()


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.subheader("💼 Business Interpretation")


st.markdown(
    """
    ### False Negatives

    A false negative occurs when the model predicts that a customer
    will stay, but the customer actually churns.

    For churn prediction, these cases are important because the
    retention team may miss an opportunity to intervene.

    ### False Positives

    A false positive occurs when the model predicts churn, but the
    customer actually stays.

    These customers may receive unnecessary retention outreach.

    ### Threshold Trade-off

    Moving from a 50% threshold to a 30% threshold increases recall
    from approximately **52% to 77%**.

    This means we catch more potential churners, but we also create
    more false positives.

    Therefore, the threshold is a **business decision**, not simply
    an ML parameter.
    """
)


st.divider()


# ============================================================
# TUNED XGBOOST PARAMETERS
# ============================================================

st.subheader("⚙️ Tuned XGBoost Parameters")


if best_parameters:

    parameter_data = pd.DataFrame(
        [
            {
                "Parameter": key,
                "Value": value,
            }
            for key, value in best_parameters.items()
        ]
    )

    st.dataframe(
        parameter_data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Tuned parameter metadata is not available."
    )


st.divider()


# ============================================================
# MODEL LIMITATIONS
# ============================================================

st.subheader("⚠️ Model Limitations")

st.markdown(
    """
    - The dataset represents historical customer behavior.
    - Historical relationships should not automatically be treated
      as causal relationships.
    - The 30% threshold is a business-oriented operating point,
      not a universal optimum.
    - Model performance may change as customer behavior changes.
    - A production system should monitor data drift, prediction
      drift, and model performance over time.
    - Retention actions should ultimately be validated using actual
      business outcomes.
    """
)