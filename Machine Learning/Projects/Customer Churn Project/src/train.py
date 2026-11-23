from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import clean_data
from src.feature_engineering import (
    prepare_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")


def build_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )


def main():

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Clean data
    df = clean_data(df)

    # Prepare X and y
    X, y = prepare_features(df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget proportion:")
    print(y.value_counts(normalize=True))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Preprocessor
    preprocessor = build_preprocessor()

    # Model
    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    random_state=42,
    n_jobs=-1
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    print("\nModel training completed.")


if __name__ == "__main__":
    main()