import pandas as pd

from src.feature_engineering import prepare_features


def create_sample_data():

    return pd.DataFrame(
        {
            "customerID": ["001", "002"],

            "gender": [
                "Female",
                "Male",
            ],

            "SeniorCitizen": [
                0,
                1,
            ],

            "Partner": [
                "Yes",
                "No",
            ],

            "Dependents": [
                "No",
                "Yes",
            ],

            "tenure": [
                1,
                24,
            ],

            "PhoneService": [
                "No",
                "Yes",
            ],

            "MultipleLines": [
                "No phone service",
                "No",
            ],

            "InternetService": [
                "DSL",
                "Fiber optic",
            ],

            "OnlineSecurity": [
                "No",
                "Yes",
            ],

            "OnlineBackup": [
                "Yes",
                "No",
            ],

            "DeviceProtection": [
                "No",
                "Yes",
            ],

            "TechSupport": [
                "No",
                "Yes",
            ],

            "StreamingTV": [
                "No",
                "Yes",
            ],

            "StreamingMovies": [
                "No",
                "Yes",
            ],

            "Contract": [
                "Month-to-month",
                "Two year",
            ],

            "PaperlessBilling": [
                "Yes",
                "No",
            ],

            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
            ],

            "MonthlyCharges": [
                29.85,
                90.00,
            ],

            "TotalCharges": [
                29.85,
                2160.00,
            ],

            "Churn": [
                "Yes",
                "No",
            ],
        }
    )


def test_prepare_features_shape():

    df = create_sample_data()

    X, y = prepare_features(df)

    assert X.shape == (2, 19)

    assert y.shape == (2,)


def test_customer_id_is_removed():

    df = create_sample_data()

    X, _ = prepare_features(df)

    assert "customerID" not in X.columns


def test_target_is_encoded():

    df = create_sample_data()

    _, y = prepare_features(df)

    assert y.tolist() == [1, 0]


def test_target_is_integer():

    df = create_sample_data()

    _, y = prepare_features(df)

    assert y.dtype == "int64"