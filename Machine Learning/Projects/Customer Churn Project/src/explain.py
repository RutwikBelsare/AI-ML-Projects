from pathlib import Path

import joblib
import pandas as pd
import shap


MODEL_PATH = Path("models/churn_pipeline.joblib")


def load_pipeline():
    """Load the trained ML pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def clean_feature_name(feature_name: str) -> str:
    """
    Convert transformed sklearn feature names into
    human-readable names.
    """

    name = feature_name

    if name.startswith("num__"):
        name = name.replace("num__", "")

        mapping = {
            "tenure": "Tenure",
            "MonthlyCharges": "Monthly Charges",
            "TotalCharges": "Total Charges",
        }

        return mapping.get(name, name)

    if name.startswith("cat__"):
        name = name.replace("cat__", "")

        parts = name.split("_", 1)

        if len(parts) == 2:
            feature, value = parts

            mapping = {
                "gender": "Gender",
                "SeniorCitizen": "Senior Citizen",
                "Partner": "Partner",
                "Dependents": "Dependents",
                "PhoneService": "Phone Service",
                "MultipleLines": "Multiple Lines",
                "InternetService": "Internet Service",
                "OnlineSecurity": "Online Security",
                "OnlineBackup": "Online Backup",
                "DeviceProtection": "Device Protection",
                "TechSupport": "Tech Support",
                "StreamingTV": "Streaming TV",
                "StreamingMovies": "Streaming Movies",
                "Contract": "Contract",
                "PaperlessBilling": "Paperless Billing",
                "PaymentMethod": "Payment Method",
            }

            feature = mapping.get(feature, feature)

            return f"{feature}: {value}"

    return name


def explain_prediction(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate SHAP values for a customer.

    Positive values push the prediction toward churn.
    Negative values push the prediction away from churn.
    """

    pipeline = load_pipeline()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_transformed
    )

    # Binary classification compatibility
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = pd.DataFrame(
        shap_values,
        columns=feature_names,
        index=X.index
    )

    return shap_values


def get_top_contributions(
    X: pd.DataFrame,
    top_n: int = 8
) -> pd.DataFrame:
    """
    Return the most influential SHAP features
    for a customer.
    """

    shap_values = explain_prediction(X)

    contributions = (
        shap_values
        .iloc[0]
        .sort_values(
            key=abs,
            ascending=False
        )
        .head(top_n)
    )

    result = pd.DataFrame({
        "feature": contributions.index,
        "shap_value": contributions.values
    })

    result["feature"] = (
        result["feature"]
        .apply(clean_feature_name)
    )

    result["direction"] = result["shap_value"].apply(
        lambda value:
        "Increases churn risk"
        if value > 0
        else "Decreases churn risk"
    )

    result["impact"] = (
        result["shap_value"]
        .abs()
    )

    return result