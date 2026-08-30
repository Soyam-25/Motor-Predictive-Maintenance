"""
train_model.py
---------------
Loads sensor_data.csv, engineers rolling-window features per unit,
trains:
    1. A RandomForestRegressor to predict RUL (Remaining Useful Life)
    2. A RandomForestClassifier to predict Risk_Level (Low/Medium/High)

Train/test split is done BY UNIT, not by row - if you split rows randomly,
cycles from the same motor leak into both train and test and the model
looks far more accurate than it really is. Holding out entire units is
the honest way to evaluate this.

Outputs:
    processed_data.csv   - data + engineered features, used by the dashboard
    rul_model.pkl        - trained regressor
    risk_model.pkl        - trained classifier
    feature_importance.csv
    model_metrics.txt
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score

SENSORS = ["Vibration_mm_s", "Temperature_C", "Current_A", "RPM", "Bearing_Noise_dB"]
ROLL_WINDOW = 5

df = pd.read_csv("sensor_data.csv")
df = df.sort_values(["Unit_ID", "Cycle"])

# ---- rolling-window features per unit (mean + std over last 5 cycles) ----
# gives the model a sense of *trend*, not just the instantaneous reading,
# which matters a lot for degradation - a slowly rising vibration reading
# is a very different signal from a noisy but flat one.
for sensor in SENSORS:
    df[f"{sensor}_roll_mean"] = (
        df.groupby("Unit_ID")[sensor]
          .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    )
    df[f"{sensor}_roll_std"] = (
        df.groupby("Unit_ID")[sensor]
          .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).std().fillna(0))
    )

feature_cols = SENSORS + [f"{s}_roll_mean" for s in SENSORS] + [f"{s}_roll_std" for s in SENSORS] + ["Load_Factor"]

# ---- split by unit: 30 units train, 10 units held out for test ----
unit_ids = df["Unit_ID"].unique()
rng = np.random.RandomState(42)
rng.shuffle(unit_ids)
test_units = set(unit_ids[:10])

train_df = df[~df["Unit_ID"].isin(test_units)]
test_df = df[df["Unit_ID"].isin(test_units)]

X_train, y_train_rul = train_df[feature_cols], train_df["RUL_clipped"]
X_test, y_test_rul = test_df[feature_cols], test_df["RUL_clipped"]

# ---- 1. RUL regression ----
rul_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rul_model.fit(X_train, y_train_rul)
pred_rul = rul_model.predict(X_test)

mae = mean_absolute_error(y_test_rul, pred_rul)
rmse = np.sqrt(mean_squared_error(y_test_rul, pred_rul))

# ---- 2. Risk classification ----
y_train_risk = train_df["Risk_Level"]
y_test_risk = test_df["Risk_Level"]

risk_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
risk_model.fit(X_train, y_train_risk)
pred_risk = risk_model.predict(X_test)

acc = accuracy_score(y_test_risk, pred_risk)
f1 = f1_score(y_test_risk, pred_risk, average="weighted")

# ---- save everything ----
joblib.dump(rul_model, "rul_model.pkl")
joblib.dump(risk_model, "risk_model.pkl")

df["Predicted_RUL"] = np.nan
df.loc[test_df.index, "Predicted_RUL"] = pred_rul
df["Predicted_Risk"] = None
df.loc[test_df.index, "Predicted_Risk"] = pred_risk
df["Data_Split"] = np.where(df["Unit_ID"].isin(test_units), "test", "train")
df.to_csv("processed_data.csv", index=False)

importances = pd.Series(rul_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.to_csv("feature_importance.csv", header=["Importance"])

with open("model_metrics.txt", "w") as f:
    f.write("=== RUL Regression (RandomForestRegressor) ===\n")
    f.write(f"MAE:  {mae:.2f} cycles\n")
    f.write(f"RMSE: {rmse:.2f} cycles\n\n")
    f.write("=== Risk Classification (RandomForestClassifier) ===\n")
    f.write(f"Accuracy: {acc:.3f}\n")
    f.write(f"Weighted F1: {f1:.3f}\n\n")
    f.write(f"Test units (held out entirely): {sorted(test_units)}\n")
    f.write(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}\n\n")
    f.write("Top 5 features by importance:\n")
    f.write(importances.head(5).to_string())

print(open("model_metrics.txt").read())
print("Saved: rul_model.pkl, risk_model.pkl, processed_data.csv, feature_importance.csv, model_metrics.txt")
