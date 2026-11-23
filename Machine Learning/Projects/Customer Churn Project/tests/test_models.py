from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/churn_pipeline.joblib"
)


def create_customer():

    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        ]
    )


def test_model_file_exists():

    assert MODEL_PATH.exists()


def test_model_can_be_loaded():

    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None


def test_model_prediction_probability():

    model = joblib.load(
        MODEL_PATH
    )

    customer = create_customer()

    probability = model.predict_proba(
        customer
    )[:, 1][0]

    assert 0 <= probability <= 1


def test_model_prediction_is_binary():

    model = joblib.load(
        MODEL_PATH
    )

    customer = create_customer()

    prediction = model.predict(
        customer
    )[0]

    assert prediction in [0, 1]