
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_excel("dataset/Data_Train.xlsx")

# Remove rows with missing important values
df = df.dropna(subset=[
    "Airline", "Source", "Destination",
    "Date_of_Journey", "Dep_Time",
    "Arrival_Time", "Duration", "Total_Stops", "Price"
])


# Extract journey date
date = pd.to_datetime(
    df["Date_of_Journey"], dayfirst=True
)

df["Day"] = date.dt.day
df["Month"] = date.dt.month


# Extract departure hour
df["Dep_Hour"] = (
    df["Dep_Time"].str.split(":").str[0].astype(int)
)


# Extract arrival hour
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
    .str.extract(r"(\d+)")
    .fillna(0)
    .astype(int)
)


# Select features
features = [
    "Airline",
    "Source",
    "Destination",
    "Day",
    "Month",
    "Dep_Hour",
    "Arr_Hour",
    "Duration_Min",
    "Stops"
]

X = df[features]
y = df["Price"]


# Categorical and numerical columns
cat = ["Airline", "Source", "Destination"]

num = [
    "Day",
    "Month",
    "Dep_Hour",
    "Arr_Hour",
    "Duration_Min",
    "Stops"
]


# Prepare data
preprocessor = ColumnTransformer([
    (
        "text",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=20
            ))
        ]),
        cat
    ),
    (
        "number",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num
    )
])


# Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)


# Complete pipeline
pipeline = Pipeline([
    ("prepare", preprocessor),
    ("model", model)
])


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
print("Training model...")

pipeline.fit(X_train, y_train)


# Evaluate model
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


print("\nMODEL RESULTS")
print("----------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Save model
joblib.dump(pipeline, "flight_price_model.pkl")

print("\nModel saved successfully!")
