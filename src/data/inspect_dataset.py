import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file_path = "data/raw/cropdata_updated.csv"

df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values:")
print(df.nunique())

print("\nCrop distribution:")
print(df["crop ID"].value_counts())

print("\nSoil type distribution:")
print(df["soil_type"].value_counts())

print("\nResult distribution:")
print(df["result"].value_counts())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nResult distribution by crop:")
print(pd.crosstab(df["crop ID"], df["result"]))

print("\nResult percentages by crop:")
print(
    pd.crosstab(
        df["crop ID"],
        df["result"],
        normalize="index"
    ).round(3)
)

print("\nRemoving result = 2...")

df = df[df["result"].isin([0, 1])]

print("New shape:", df.shape)

print("\nNew result distribution:")
print(df["result"].value_counts())

print("\nNumerical feature statistics:")
print(df[["MOI", "temp", "humidity"]].describe())

print("\nUnique crop types:")
print(df["crop ID"].unique())

print("\nUnique soil types:")
print(df["soil_type"].unique())

print("\nUnique seedling stages:")
print(df["Seedling Stage"].unique())

print("\nAverage features by irrigation result:")
print(
    df.groupby("result")[["MOI", "temp", "humidity"]].mean()
)

print("\nFeature ranges by irrigation result:")
print(
    df.groupby("result")[["MOI", "temp", "humidity"]]
      .agg(["min", "max", "mean"])
)

print("\nIrrigation distribution by crop:")
print(pd.crosstab(df["crop ID"], df["result"], normalize="index").round(3))

print("\nIrrigation distribution by soil:")
print(pd.crosstab(df["soil_type"], df["result"], normalize="index").round(3))

print("\nIrrigation distribution by growth stage:")
print(pd.crosstab(df["Seedling Stage"], df["result"], normalize="index").round(3))

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nSample duplicate rows:")
print(df[df.duplicated(keep=False)].sort_values(
    by=["crop ID", "soil_type", "Seedling Stage", "MOI", "temp", "humidity"]
).head(20))

print("\nRemoving duplicate rows...")

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Rows before:", before)
print("Rows after:", after)
print("Duplicates removed:", before - after)

print("\nRemaining duplicates:", df.duplicated().sum())

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal target distribution:")
print(df["result"].value_counts())

X = df.drop("result", axis=1)
y = df["result"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\ny distribution:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

categorical_features = [
    "crop ID",
    "soil_type",
    "Seedling Stage"
]

numerical_features = [
    "MOI",
    "temp",
    "humidity"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

print("Preprocessor created successfully.")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)

print("Model trained successfully.")

y_pred = model.predict(X_test)

print("Predictions made successfully.")

print("\nFirst 20 predictions:")
print(y_pred[:20])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

results = X_test.copy()

results["actual"] = y_test
results["predicted"] = y_pred

print("\nPerformance by crop:")
print(
    results.groupby("crop ID")
    .apply(
        lambda group: accuracy_score(
            group["actual"],
            group["predicted"]
        )
    )
    .round(3)
)

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest predictions made successfully.")

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

rf_train_pred = rf_model.predict(X_train)

print("\nRandom Forest Training Accuracy:")
print(accuracy_score(y_train, rf_train_pred))

print("\nRandom Forest Test Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nTarget distribution by MOI:")
print(
    df.groupby("result")["MOI"]
      .agg(["min", "max", "mean", "median"])
)

print("\nTarget distribution by temperature:")
print(
    df.groupby("result")["temp"]
      .agg(["min", "max", "mean", "median"])
)

print("\nTarget distribution by humidity:")
print(
    df.groupby("result")["humidity"]
      .agg(["min", "max", "mean", "median"])
)

print("\nFirst 30 rows after cleaning:")
print(
    df[
        ["crop ID", "soil_type", "Seedling Stage",
         "MOI", "temp", "humidity", "result"]
    ].head(30)
)

print("\nLast 30 rows after cleaning:")
print(
    df[
        ["crop ID", "soil_type", "Seedling Stage",
         "MOI", "temp", "humidity", "result"]
    ].tail(30)
)