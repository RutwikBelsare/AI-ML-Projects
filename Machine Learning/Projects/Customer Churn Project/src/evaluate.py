import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.data_preprocessing import clean_data
from src.feature_engineering import prepare_features
from src.train import build_preprocessor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # ==================================================
    # 1. Load Data
    # ==================================================

    df = pd.read_csv(DATA_PATH)

    print("Raw data shape:", df.shape)

    # ==================================================
    # 2. Clean Data
    # ==================================================

    df = clean_data(df)

    # ==================================================
    # 3. Prepare Features and Target
    # ==================================================

    X, y = prepare_features(df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget proportion:")
    print(y.value_counts(normalize=True))

    # ==================================================
    # 4. Train/Test Split
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ==================================================
    # 5. Build Preprocessor
    # ==================================================

    preprocessor = build_preprocessor()

    # ==================================================
    # 6. Create XGBoost Model
    # ==================================================

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    # ==================================================
    # 7. Build Complete Pipeline
    # ==================================================

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # ==================================================
    # 8. Train Model
    # ==================================================

    print("\nTraining XGBoost model...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # ==================================================
    # 9. Generate Predictions
    # ==================================================

    # Class predictions using default threshold = 0.50
    y_pred = pipeline.predict(X_test)

    # Probability of Churn
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ==================================================
    # 10. Standard Model Evaluation
    # ==================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )

    pr_auc = average_precision_score(
        y_test,
        y_proba
    )

    print("\n")
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    # ==================================================
    # 11. Confusion Matrix
    # ==================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n")
    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)

    print(cm)

    print("\nConfusion Matrix Breakdown:")

    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    # ==================================================
    # 12. Classification Report
    # ==================================================

    print("\n")
    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    # ==================================================
    # 13. Threshold Analysis
    # ==================================================

    print("\n")
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
        f"{'Predicted Churn %':<20}"
    )

    print("-" * 70)

    thresholds = np.arange(
        0.20,
        0.81,
        0.05
    )

    threshold_results = []

    for threshold in thresholds:

        # Convert probabilities into classes
        y_threshold = (
            y_proba >= threshold
        ).astype(int)

        threshold_precision = precision_score(
            y_test,
            y_threshold,
            zero_division=0
        )

        threshold_recall = recall_score(
            y_test,
            y_threshold,
            zero_division=0
        )

        threshold_f1 = f1_score(
            y_test,
            y_threshold,
            zero_division=0
        )

        predicted_churn_percentage = (
            y_threshold.mean() * 100
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": threshold_precision,
            "recall": threshold_recall,
            "f1": threshold_f1,
            "predicted_churn_percentage":
                predicted_churn_percentage
        })

        print(
            f"{threshold:<12.2f}"
            f"{threshold_precision:<12.3f}"
            f"{threshold_recall:<12.3f}"
            f"{threshold_f1:<12.3f}"
            f"{predicted_churn_percentage:<20.2f}"
        )

    # ==================================================
    # 14. Find Best Threshold by F1
    # ==================================================

    threshold_df = pd.DataFrame(
        threshold_results
    )

    best_threshold_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print("\n")
    print("=" * 50)
    print("BEST THRESHOLD BY F1")
    print("=" * 50)

    print(
        f"Threshold : "
        f"{best_threshold_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_threshold_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_threshold_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_threshold_row['f1']:.4f}"
    )

    print(
        f"Predicted Churn % : "
        f"{best_threshold_row['predicted_churn_percentage']:.2f}%"
    )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()