"""
train_model.py
Trains the ML downscaling model: given a Block-level forecast + a
panchayat's geospatial/historical features, predict panchayat-level
rainfall, temperature, humidity and wind.

Swap RandomForest for XGBoost by installing xgboost and changing the
MODEL constructor below -- kept as RandomForest here so it runs with
zero extra native deps for your first prototype/demo.

Run:
    python data_generator.py   # first, to create training_data.csv
    python train_model.py
Produces: downscaling_model.joblib
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGETS = ["true_rainfall_mm", "true_temp_c", "true_humidity_pct", "true_wind_kmph"]
NUMERIC = ["month", "block_forecast_rain_mm", "block_forecast_temp_c",
           "lat", "lon", "elevation_m", "hist_avg_rainfall_mm", "hist_avg_temp_c"]
CATEGORICAL = ["soil_type", "land_use"]

df = pd.read_csv("training_data.csv")
X = df[NUMERIC + CATEGORICAL]
y = df[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
], remainder="passthrough")

model = Pipeline([
    ("prep", preprocess),
    ("reg", MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    )),
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Validation MAE per target:")
for i, t in enumerate(TARGETS):
    print(f"  {t:22s}: {mean_absolute_error(y_test[t], preds[:, i]):.2f}")

joblib.dump(model, "downscaling_model.joblib")
print("\nSaved model -> downscaling_model.joblib")
