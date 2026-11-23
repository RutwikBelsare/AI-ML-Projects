import pandas as pd

from src.data_preprocessing import clean_data
from src.explain import explain_prediction, get_top_contributions


DATA_PATH = "data/raw/Telco-Customer-Churn.csv"


def load_customer():

    df = pd.read_csv(DATA_PATH)

    df = clean_data(df)

    X = df.drop(
        columns=[
            "Churn",
            "customerID",
        ]
    )

    return X.iloc[[0]]


def test_shap_explanation_shape():

    customer = load_customer()

    shap_values = explain_prediction(
        customer
    )

    assert shap_values.shape[0] == 1

    assert shap_values.shape[1] > 0


def test_shap_explanation_contains_numeric_values():

    customer = load_customer()

    shap_values = explain_prediction(
        customer
    )

    assert shap_values.notna().all().all()

    assert all(
        pd.api.types.is_numeric_dtype(
            shap_values[column]
        )
        for column in shap_values.columns
    )


def test_top_shap_contributions():

    customer = load_customer()

    contributions = get_top_contributions(
        customer,
        top_n=8
    )

    assert len(contributions) == 8

    assert "feature" in contributions.columns

    assert "shap_value" in contributions.columns

    assert "direction" in contributions.columns


def test_shap_direction_is_valid():

    customer = load_customer()

    contributions = get_top_contributions(
        customer,
        top_n=8
    )

    valid_directions = {
        "Increases churn risk",
        "Decreases churn risk",
    }

    assert set(
        contributions["direction"]
    ).issubset(
        valid_directions
    )