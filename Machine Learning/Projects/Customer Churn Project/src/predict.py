from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/churn_pipeline.joblib"
)

PRODUCTION_THRESHOLD = 0.30


def load_model():
    """
    Load the trained churn prediction pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_churn(customer_data: pd.DataFrame):

    model = load_model()

    probability = model.predict_proba(
        customer_data
    )[:, 1][0]

    prediction = int(
        probability >= PRODUCTION_THRESHOLD
    )

    return {
        "churn_probability": float(
            probability
        ),
        "churn_prediction": prediction
    }