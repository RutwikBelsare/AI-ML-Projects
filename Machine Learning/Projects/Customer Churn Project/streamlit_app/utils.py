from pathlib import Path

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

API_URL = "http://127.0.0.1:8000"


def get_api_health():
    """Check whether the FastAPI service is available."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def predict_churn(customer_data):
    """Send customer data to FastAPI."""

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:
        st.error(
            f"Unable to connect to prediction API: {exc}"
        )
        return None


def load_model_metadata():
    """Load saved model metadata."""

    import json

    if not METADATA_PATH.exists():
        return {}

    with open(METADATA_PATH, "r") as file:
        return json.load(file)