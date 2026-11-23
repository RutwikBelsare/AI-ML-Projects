import pandas as pd


TARGET_COLUMN = "Churn"

DROP_COLUMNS = [
    "customerID"
]

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def prepare_features(df: pd.DataFrame):
    """
    Separate input features and target variable.
    """

    df = df.copy()

    # Convert target to binary
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # Validate target conversion
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Invalid values found in Churn column."
        )

    # Remove identifier
    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS
    )

    y = df[TARGET_COLUMN].astype(int)

    return X, y