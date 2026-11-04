
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("dataset/Data_Train.xlsx")

print("Original rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()
print("Rows after removing duplicates:", len(df))

# Remove rows with missing important values
df = df.dropna(subset=[
    "Airline", "Source", "Destination",
    "Date_of_Journey", "Dep_Time",
    "Arrival_Time", "Duration",
    "Total_Stops", "Price"
]).copy()

# Extract date
date = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
df["Day"] = date.dt.day
df["Month"] = date.dt.month

# Extract departure and arrival hours
df["Dep_Hour"] = df["Dep_Time"].str.split(":").str[0].astype(int)

df["Arr_Hour"] = (
    df["Arrival_Time"].str.split().str[0]
    .str.split(":").str[0].astype(int)
)

# Convert duration into minutes
def get_duration(value):
    h = re.search(r"(\d+)h", str(value))
    m = re.search(r"(\d+)m", str(value))
    hours = int(h.group(1)) if h else 0
    minutes = int(m.group(1)) if m else 0
    return hours * 60 + minutes

df["Duration_Min"] = df["Duration"].apply(get_duration)

# Convert stops into numbers
df["Stops"] = (
    df["Total_Stops"]
    .str.extract(r"(\d+)", expand=False)
    .fillna("0")
    .astype(int)
)

# Select features
features = [
    "Airline", "Source", "Destination",
    "Day", "Month", "Dep_Hour",
    "Arr_Hour", "Duration_Min", "Stops"
]

X = df[features]
y = df["Price"]

cat = ["Airline", "Source", "Destination"]
num = [
    "Day", "Month", "Dep_Hour",
    "Arr_Hour", "Duration_Min", "Stops"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("text", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20
        ))
    ]), cat),

    ("number", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num)
])

# Models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# Same train/test split for all models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []
trained_models = {}

print("\nTRAINING AND COMPARING MODELS")

for name, model in models.items():

    pipeline = Pipeline([
        ("prepare", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Five-fold cross-validation on training data
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )

    cv_mae = -cv_scores.mean()

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "CV_MAE": round(cv_mae, 2)
    })

    trained_models[name] = pipeline

# Display comparison
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Select model with lowest test MAE
best_name = results_df.loc[
    results_df["MAE"].idxmin(), "Model"
]

best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)

# Save selected model
joblib.dump(best_model, "flight_price_model.pkl")

# Save results for the report
results_df.to_csv("model_comparison.csv", index=False)

print("\nModel saved successfully!")
print("Comparison saved to model_comparison.csv")
