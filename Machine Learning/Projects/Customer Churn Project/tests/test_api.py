from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


CUSTOMER_DATA = {
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


def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True


def test_model_info_endpoint():

    response = client.get(
        "/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "XGBClassifier"

    assert data["threshold"] == 0.30


def test_predict_endpoint():

    response = client.post(
        "/predict",
        json=CUSTOMER_DATA
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data

    assert "churn_prediction" in data

    assert "risk_level" in data

    assert "threshold" in data


def test_prediction_probability_range():

    response = client.post(
        "/predict",
        json=CUSTOMER_DATA
    )

    data = response.json()

    probability = data[
        "churn_probability"
    ]

    assert 0 <= probability <= 1


def test_prediction_is_binary():

    response = client.post(
        "/predict",
        json=CUSTOMER_DATA
    )

    data = response.json()

    assert data[
        "churn_prediction"
    ] in [0, 1]


def test_invalid_senior_citizen_is_rejected():

    invalid_customer = CUSTOMER_DATA.copy()

    invalid_customer[
        "SeniorCitizen"
    ] = 2

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422


def test_negative_monthly_charges_are_rejected():

    invalid_customer = CUSTOMER_DATA.copy()

    invalid_customer[
        "MonthlyCharges"
    ] = -10

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422