from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI

from api.schemas import CustomerData

#Configuration
MODEL_PATH = Path(
    "models/churn_pipeline.joblib"
)

PRODUCTION_THRESHOLD = 0.30

#Fast Api app
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-style API for predicting "
        "customer churn."
    ),
    version="1.0.0"
)

#Loading Model 
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(
    MODEL_PATH
)

#Health check endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


#Model information endpoint
@app.get("/model-info")
def model_info():

    return {
        "model": "XGBClassifier",
        "version": "1.0.0",
        "threshold": PRODUCTION_THRESHOLD
    }

#Prediction Endpoint
@app.post("/predict")
def predict(customer: CustomerData):

    # Convert Pydantic model to dictionary
    customer_dict = customer.model_dump()

    # Convert to DataFrame
    input_df = pd.DataFrame(
        [customer_dict]
    )

    # Get churn probability
    churn_probability = (
        model.predict_proba(
            input_df
        )[:, 1][0]
    )

    # Apply business threshold
    churn_prediction = int(
        churn_probability
        >= PRODUCTION_THRESHOLD
    )

    # Risk category
    if churn_probability >= 0.70:
        risk_level = "HIGH"

    elif churn_probability >= 0.30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "churn_probability": round(
            float(churn_probability),
            4
        ),
        "churn_prediction": churn_prediction,
        "risk_level": risk_level,
        "threshold": PRODUCTION_THRESHOLD
    }