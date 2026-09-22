import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/raw/cropdata_updated.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. Clean dataset
# --------------------------------------------------

# Keep only the two valid irrigation classes:
# 0 = No irrigation
# 1 = Irrigation
df = df[df["result"].isin([0, 1])]

# Remove exact duplicate rows
df = df.drop_duplicates()

# Reset index after cleaning
df = df.reset_index(drop=True)

print("Cleaned dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["result"].value_counts())


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop("result", axis=1)
y = df["result"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Define features
# --------------------------------------------------

categorical_features = [
    "crop ID",
    "soil_type",
    "Seedling Stage",
]

numerical_features = [
    "MOI",
    "temp",
    "humidity",
]


# --------------------------------------------------
# 6. Create preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
    ]
)


# --------------------------------------------------
# 7. Logistic Regression model
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)


print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


# --------------------------------------------------
# 8. Evaluate Logistic Regression
# --------------------------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred,
)

print("\n========== Logistic Regression ==========")

print("Accuracy:", round(logistic_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_pred,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_pred,
    )
)


# --------------------------------------------------
# 9. Random Forest model
# --------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


print("\nTraining Random Forest...")

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)


# --------------------------------------------------
# 10. Evaluate Random Forest
# --------------------------------------------------

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred,
)

print("\n========== Random Forest ==========")

print(
    "Accuracy:",
    round(random_forest_accuracy, 4),
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        random_forest_pred,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        random_forest_pred,
    )
)


# --------------------------------------------------
# 11. Save models
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_model,
    "models/irrigation_logistic_model.joblib",
)

joblib.dump(
    random_forest_model,
    "models/irrigation_random_forest_model.joblib",
)


# --------------------------------------------------
# 12. Save the main irrigation model
# --------------------------------------------------

joblib.dump(
    logistic_model,
    "models/irrigation_model.joblib",
)


# --------------------------------------------------
# 13. Final summary
# --------------------------------------------------

print("\n========================================")
print("IRRIGATION MODEL TRAINING COMPLETED")
print("========================================")

print(
    "Logistic Regression accuracy:",
    round(logistic_accuracy, 4),
)

print(
    "Random Forest accuracy:",
    round(random_forest_accuracy, 4),
)

print("\nSaved models:")

print("- models/irrigation_model.joblib")
print("- models/irrigation_logistic_model.joblib")
print("- models/irrigation_random_forest_model.joblib")

print("\nMain model:")
print("Logistic Regression")

print("\nTraining completed successfully.")