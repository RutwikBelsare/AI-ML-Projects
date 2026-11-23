from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


st.title("📈 Customer Churn Dashboard")

st.markdown(
    """
    Business overview of the telecom customer base and
    historical churn behavior.
    """
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Clean TotalCharges for analysis
    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", "0")
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


df = load_data()


# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

total_customers = len(df)

churned_customers = (
    df["Churn"] == "Yes"
).sum()

churn_rate = (
    churned_customers / total_customers
) * 100

avg_monthly_charge = df["MonthlyCharges"].mean()

avg_tenure = df["tenure"].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )


with col3:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


with col4:
    st.metric(
        "Avg Monthly Charges",
        f"${avg_monthly_charge:.2f}"
    )


st.divider()


# ---------------------------------------------------------
# CHURN DISTRIBUTION
# ---------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.subheader("Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        churn_counts.set_index("Churn")
    )


with col2:

    st.subheader("Churn Rate by Contract")

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.bar_chart(contract_churn)


st.divider()


# ---------------------------------------------------------
# TENURE
# ---------------------------------------------------------

st.subheader("Customer Tenure Distribution")

tenure_bins = [
    -1,
    12,
    24,
    48,
    72
]

tenure_labels = [
    "0–12 months",
    "13–24 months",
    "25–48 months",
    "49–72 months"
]

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels
)

tenure_churn = (
    df.groupby(
        "TenureGroup",
        observed=True
    )["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

st.bar_chart(tenure_churn)


st.divider()


# ---------------------------------------------------------
# INTERNET SERVICE
# ---------------------------------------------------------

st.subheader("Churn Rate by Internet Service")

internet_churn = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

st.bar_chart(internet_churn)


st.divider()


# ---------------------------------------------------------
# BUSINESS INTERPRETATION
# ---------------------------------------------------------

st.subheader("🔎 Business Observations")

st.markdown(
    f"""
    - Overall customer churn rate is **{churn_rate:.2f}%**.
    - Customers on **month-to-month contracts** show substantially
      higher historical churn than customers on longer contracts.
    - Customers with shorter tenure tend to show higher churn.
    - Monthly charges are generally higher among customers who churn.
    - Internet service type is also associated with different
      historical churn rates.
    """
)

st.caption(
    "These are historical associations in the dataset and should "
    "not be interpreted as causal relationships."
)