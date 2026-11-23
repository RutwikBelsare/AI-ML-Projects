import pandas as pd

from src.data_preprocessing import clean_data


def test_total_charges_blank_values_are_converted_to_zero():

    df = pd.DataFrame(
        {
            "TotalCharges": ["100.50", " ", "250.00"],
            "tenure": [5, 0, 10],
        }
    )

    cleaned_df = clean_data(df)

    assert cleaned_df["TotalCharges"].tolist() == [
        100.50,
        0.0,
        250.00,
    ]


def test_total_charges_is_numeric():

    df = pd.DataFrame(
        {
            "TotalCharges": ["100", "200", "300"],
        }
    )

    cleaned_df = clean_data(df)

    assert pd.api.types.is_numeric_dtype(
        cleaned_df["TotalCharges"]
    )


def test_invalid_total_charges_raises_error():

    df = pd.DataFrame(
        {
            "TotalCharges": [
                "100",
                "INVALID",
            ]
        }
    )

    try:

        clean_data(df)

        assert False, (
            "Expected ValueError was not raised"
        )

    except ValueError:

        assert True