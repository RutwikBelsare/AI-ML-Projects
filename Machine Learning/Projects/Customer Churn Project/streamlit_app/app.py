import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Customer Churn Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("📊 Customer Churn Prediction Platform")

st.markdown(
    """
    ## Welcome

    This platform predicts the probability that a telecom customer
    will churn and provides business-oriented risk insights.

    ### Available Modules

    Use the sidebar to navigate between:

    - 📈 **Dashboard** — Explore customer and churn statistics
    - 👤 **Predict Churn** — Predict churn for an individual customer
    - 🧠 **Model Insights** — Understand model performance and drivers
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Prediction Type",
        value="Binary Classification"
    )

with col2:
    st.metric(
        label="Production Threshold",
        value="30%"
    )

with col3:
    st.metric(
        label="Model",
        value="XGBoost"
    )

st.divider()

st.info(
    "💡 The prediction threshold is intentionally set to 30% "
    "to prioritize identifying potential churners."
)