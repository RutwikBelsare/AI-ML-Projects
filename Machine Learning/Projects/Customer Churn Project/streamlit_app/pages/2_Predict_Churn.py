import pandas as pd
import streamlit as st

from streamlit_app.utils import predict_churn
from src.explain import get_top_contributions


st.title("👤 Customer Churn Prediction")

st.markdown(
    """
    Enter customer information below to estimate the probability
    that the customer will churn.
    """
)

st.divider()


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )


col1, col2, col3 = st.columns(3)

with col1:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col3:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# ============================================================
# SERVICES
# ============================================================

st.divider()

st.subheader("📱 Services")

col1, col2 = st.columns(2)

with col1:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


with col2:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.divider()

st.subheader("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=65.0,
        step=1.0
    )


total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    value=780.0,
    step=10.0
)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

st.subheader("🔮 Prediction")

predict_button = st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # CREATE CUSTOMER INPUT
    # --------------------------------------------------------

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


    # --------------------------------------------------------
    # CALL FASTAPI
    # --------------------------------------------------------

    with st.spinner("Running churn prediction..."):

        result = predict_churn(customer_data)


    if result:

        probability = result["churn_probability"]
        prediction = result["churn_prediction"]
        risk_level = result["risk_level"]
        threshold = result["threshold"]

        probability_percent = probability * 100


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )


        with col2:

            prediction_text = (
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
            )

            st.metric(
                "Prediction",
                prediction_text
            )


        with col3:

            st.metric(
                "Risk Level",
                risk_level
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.subheader("Churn Probability")

        st.progress(
            probability
        )

        st.caption(
            f"Prediction threshold: "
            f"{threshold * 100:.0f}%"
        )


        # ====================================================
        # BUSINESS RECOMMENDATION
        # ====================================================

        st.subheader("💼 Recommended Action")


        if risk_level == "HIGH":

            st.error(
                """
                **High-risk customer**

                Consider immediate retention intervention.

                Recommended actions:

                - Contact the customer proactively
                - Offer a personalized retention incentive
                - Review pricing and service issues
                - Consider contract upgrade options
                """
            )


        elif risk_level == "MEDIUM":

            st.warning(
                """
                **Medium-risk customer**

                Consider proactive engagement.

                Recommended actions:

                - Send a personalized retention offer
                - Monitor service usage
                - Identify dissatisfaction signals
                - Encourage migration to a longer-term contract
                """
            )


        else:

            st.success(
                """
                **Low-risk customer**

                No immediate retention intervention is required.

                Continue normal customer engagement and
                monitor future behavior.
                """
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 Why did the model make this prediction?"
        )

        st.markdown(
            """
            The following features had the strongest influence
            on this individual prediction.

            **Positive impact** → pushes the prediction toward churn.

            **Negative impact** → pushes the prediction away from churn.
            """
        )


        try:

            # ------------------------------------------------
            # GENERATE SHAP EXPLANATION
            # ------------------------------------------------

            customer_df = pd.DataFrame(
                [customer_data]
            )

            explanation_df = get_top_contributions(
                customer_df,
                top_n=8
            )


            # ------------------------------------------------
            # PREPARE TABLE
            # ------------------------------------------------

            display_df = explanation_df[
                [
                    "feature",
                    "direction",
                    "shap_value"
                ]
            ].copy()


            display_df["shap_value"] = (
                display_df["shap_value"]
                .round(3)
            )


            display_df.columns = [
                "Feature",
                "Impact",
                "SHAP Value"
            ]


            # ------------------------------------------------
            # DISPLAY SHAP TABLE
            # ------------------------------------------------

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # SHAP INTERPRETATION
            # ------------------------------------------------

            st.caption(
                """
                SHAP values indicate how individual features
                influence the model output. Positive values push
                the prediction toward churn, while negative values
                push it away from churn. SHAP values are not
                percentage-point changes in churn probability.
                """
            )


        except Exception as exc:

            st.warning(
                f"Unable to generate explanation: {exc}"
            )