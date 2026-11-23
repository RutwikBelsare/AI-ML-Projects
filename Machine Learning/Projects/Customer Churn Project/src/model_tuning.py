from pathlib import Path

import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from src.data_preprocessing import clean_data
from src.feature_engineering import (
    prepare_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.train import build_preprocessor


# ==================================================
# Configuration
# ==================================================

DATA_PATH = Path(
    "data/raw/Telco-Customer-Churn.csv"
)

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR / "churn_pipeline.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "model_metadata.json"
)

RANDOM_STATE = 42

PRODUCTION_THRESHOLD = 0.30


# ==================================================
# Main
# ==================================================

def main():

    # --------------------------------------------------
    # 1. Create model directory
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------
    # 3. Clean data
    # --------------------------------------------------

    print("Cleaning data...")

    df = clean_data(df)

    # --------------------------------------------------
    # 4. Prepare X and y
    # --------------------------------------------------

    X, y = prepare_features(df)

    print(
        f"Features shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # --------------------------------------------------
    # 5. Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------
    # 6. Build preprocessing pipeline
    # --------------------------------------------------

    preprocessor = build_preprocessor()

    # --------------------------------------------------
    # 7. Base XGBoost model
    # --------------------------------------------------

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # --------------------------------------------------
    # 8. Complete pipeline
    # --------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                xgb_model
            )
        ]
    )

    # --------------------------------------------------
    # 9. Hyperparameter search space
    # --------------------------------------------------

    param_distributions = {

        "model__n_estimators": [
            100,
            200,
            300,
            500
        ],

        "model__learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1
        ],

        "model__max_depth": [
            2,
            3,
            4,
            5,
            6
        ],

        "model__min_child_weight": [
            1,
            3,
            5,
            7
        ],

        "model__subsample": [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "model__colsample_bytree": [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "model__reg_alpha": [
            0.0,
            0.01,
            0.1,
            0.5,
            1.0
        ],

        "model__reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
            10.0
        ]
    }

    # --------------------------------------------------
    # 10. Randomized Search
    # --------------------------------------------------

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )

    print("\n")
    print("=" * 60)
    print("STARTING XGBOOST HYPERPARAMETER TUNING")
    print("=" * 60)

    search.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 11. Best parameters
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BEST PARAMETERS")
    print("=" * 60)

    for parameter, value in (
        search.best_params_.items()
    ):
        print(
            f"{parameter}: {value}"
        )

    print(
        f"\nBest CV PR-AUC: "
        f"{search.best_score_:.4f}"
    )

    # --------------------------------------------------
    # 12. Best pipeline
    # --------------------------------------------------

    best_pipeline = search.best_estimator_

    # --------------------------------------------------
    # 13. Test predictions
    # --------------------------------------------------

    y_proba = (
        best_pipeline
        .predict_proba(X_test)[:, 1]
    )

    # Default threshold
    y_pred_default = (
        y_proba >= 0.50
    ).astype(int)

    # Production threshold
    y_pred_production = (
        y_proba >= PRODUCTION_THRESHOLD
    ).astype(int)

    # --------------------------------------------------
    # 14. Default threshold evaluation
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST PERFORMANCE — DEFAULT THRESHOLD 0.50")
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{accuracy_score(y_test, y_pred_default):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred_default, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, y_pred_default, zero_division=0):.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1_score(y_test, y_pred_default, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc_score(y_test, y_proba):.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{average_precision_score(y_test, y_proba):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred_default
        )
    )

    # --------------------------------------------------
    # 15. Production threshold evaluation
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        f"TEST PERFORMANCE — "
        f"PRODUCTION THRESHOLD {PRODUCTION_THRESHOLD:.2f}"
    )
    print("=" * 60)

    production_accuracy = accuracy_score(
        y_test,
        y_pred_production
    )

    production_precision = precision_score(
        y_test,
        y_pred_production,
        zero_division=0
    )

    production_recall = recall_score(
        y_test,
        y_pred_production,
        zero_division=0
    )

    production_f1 = f1_score(
        y_test,
        y_pred_production,
        zero_division=0
    )

    print(
        f"Accuracy : "
        f"{production_accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{production_precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{production_recall:.4f}"
    )

    print(
        f"F1 Score : "
        f"{production_f1:.4f}"
    )

    print("\nConfusion Matrix:")

    production_cm = confusion_matrix(
        y_test,
        y_pred_production
    )

    print(production_cm)

    # --------------------------------------------------
    # 16. Save pipeline
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    joblib.dump(
        best_pipeline,
        MODEL_PATH
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    # --------------------------------------------------
    # 17. Save metadata
    # --------------------------------------------------

    metadata = {

        "model_type": "XGBClassifier",

        "target": "Churn",

        "random_state": RANDOM_STATE,

        "production_threshold":
            PRODUCTION_THRESHOLD,

        "best_cv_pr_auc":
            float(search.best_score_),

        "test_metrics_default_threshold": {

            "accuracy":
                float(
                    accuracy_score(
                        y_test,
                        y_pred_default
                    )
                ),

            "precision":
                float(
                    precision_score(
                        y_test,
                        y_pred_default,
                        zero_division=0
                    )
                ),

            "recall":
                float(
                    recall_score(
                        y_test,
                        y_pred_default,
                        zero_division=0
                    )
                ),

            "f1":
                float(
                    f1_score(
                        y_test,
                        y_pred_default,
                        zero_division=0
                    )
                ),

            "roc_auc":
                float(
                    roc_auc_score(
                        y_test,
                        y_proba
                    )
                ),

            "pr_auc":
                float(
                    average_precision_score(
                        y_test,
                        y_proba
                    )
                )
        },

        "test_metrics_production_threshold": {

            "accuracy":
                float(
                    production_accuracy
                ),

            "precision":
                float(
                    production_precision
                ),

            "recall":
                float(
                    production_recall
                ),

            "f1":
                float(
                    production_f1
                )
        },

        "best_parameters": {
            key: value
            for key, value
            in search.best_params_.items()
        }
    }

    with open(
        METADATA_PATH,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Metadata saved to: {METADATA_PATH}"
    )

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()