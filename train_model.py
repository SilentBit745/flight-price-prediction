
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_excel("dataset/Data_Train.xlsx")

# Remove missing values
df = df.dropna(subset=["Route", "Total_Stops"])

# Date
date = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
df["Day"] = date.dt.day
df["Month"] = date.dt.month

# Time
def get_time(x):
    return pd.Series(
        [int(x.split(":")[0]), int(x.split(":")[1])]
    )

df[["Dep_Hour", "Dep_Min"]] = df["Dep_Time"].apply(get_time)
df[["Arr_Hour", "Arr_Min"]] = (
    df["Arrival_Time"].str.split().str[0].apply(get_time)
)

# Duration
def get_duration(x):
    h = re.search(r"(\d+)h", str(x))
    m = re.search(r"(\d+)m", str(x))
    hours = int(h.group(1)) if h else 0
    minutes = int(m.group(1)) if m else 0
    return hours * 60 + minutes

df["Duration_Min"] = df["Duration"].apply(get_duration)

# Stops
df["Stops"] = df["Total_Stops"].str.extract(r"(\d+)").fillna(0).astype(int)

# Features
features = [
    "Airline", "Source", "Destination", "Additional_Info",
    "Day", "Month", "Dep_Hour", "Dep_Min",
    "Arr_Hour", "Arr_Min", "Duration_Min", "Stops"
]

X = df[features]
y = df["Price"]

# Text and number columns
cat = ["Airline", "Source", "Destination", "Additional_Info"]
num = [c for c in features if c not in cat]

# Prepare data
preprocessor = ColumnTransformer([
    ("text", OneHotEncoder(handle_unknown="ignore"), cat),
    ("number", StandardScaler(), num)
])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_model = None
best_score = float("inf")

# Train and compare
for name, model in models.items():

    pipe = Pipeline([
        ("prepare", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 3))

    if mae < best_score:
        best_score = mae
        best_model = pipe

# Save best model
joblib.dump(best_model, "flight_price_model.pkl")

print("\nBest model saved successfully!")
