import pandas as pd

from data_preprocessing import clean_data


DATA_PATH = "data/raw/Telco-Customer-Churn.csv"


df = pd.read_csv(DATA_PATH)

print("Before cleaning:")
print(df["TotalCharges"].dtype)

df_clean = clean_data(df)

print("\nAfter cleaning:")
print(df_clean["TotalCharges"].dtype)

print("\nMissing TotalCharges:")
print(df_clean["TotalCharges"].isna().sum())

print("\nTotalCharges for tenure=0:")
print(
    df_clean.loc[
        df_clean["tenure"] == 0,
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ]
)

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())