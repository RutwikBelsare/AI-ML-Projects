import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the raw customer churn dataset.
    """

    df = df.copy()

    # Normalize TotalCharges
    df["TotalCharges"] = (
        df["TotalCharges"]
        .str.strip()
        .replace("", "0")
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Validate conversion
    if df["TotalCharges"].isna().any():
        raise ValueError(
            "TotalCharges contains invalid numeric values."
        )

    return df