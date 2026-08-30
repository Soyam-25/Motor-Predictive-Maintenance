# Industrial Motor Predictive Maintenance System

**Tools used:** Python (Pandas, Scikit-learn, Streamlit, Plotly)

## 1. Why this project

Reactive maintenance (fix it after it breaks) and calendar-based
preventive maintenance (service it every N months regardless of actual
condition) both waste money — one from unplanned downtime, the other
from replacing parts that still had life left. Predictive maintenance
uses sensor trends to estimate how much life a machine has left, so
maintenance happens exactly when it's needed. This project builds that
pipeline end-to-end: simulated sensor data → feature engineering →
ML model → a live dashboard a maintenance engineer could actually use.

## 2. About the data

Real vibration/temperature sensor logs from a plant are confidential
and I didn't have access to any, so `generate_sensor_data.py`
simulates a **fleet of 40 motors, each run to failure**, logging 5
sensors per cycle: vibration, temperature, current draw, RPM, and
bearing noise. Degradation follows a realistic wear-out curve — slow
at first, then accelerating near end of life — with a random load
factor per motor so not every unit degrades at the same rate.

This mirrors the structure of NASA's CMAPSS turbofan degradation
dataset (the standard academic benchmark for this exact problem),
adapted to a motor/pump context to fit an industrial automation angle.

## 3. Pipelinegenerate_sensor_data.py  →  sensor_data.csv (9,139 rows, 40 motors)
train_model.py            →  rul_model.pkl, risk_model.pkl, processed_data.csv
app.py (Streamlit)         →  live dashboard
## 4. Methodology

- **Feature engineering:** rolling mean + std (5-cycle window) per
  sensor, on top of raw readings — captures *trend*, not just the
  instantaneous value, since a steadily rising signal is a very
  different story from a noisy-but-flat one.
- **RUL clipping at 130 cycles:** a motor's early-life readings don't
  actually carry information about exactly how many hundreds of
  cycles are left — only the wear-out phase does. Capping the target
  (a standard trick used with the CMAPSS dataset) stops the model
  from learning a meaningless straight-line guess during the healthy
  phase.
- **Train/test split by unit, not by row:** if you split rows
  randomly, cycles from the same motor leak into both train and test
  and accuracy looks artificially high. All 10 test motors were held
  out completely — the model never saw a single reading from them
  during training.
- **Two models:** a `RandomForestRegressor` predicting RUL (a number),
  and a `RandomForestClassifier` predicting a Low/Medium/High risk
  label (easier for a non-technical dashboard viewer to act on than a
  raw cycle count).

## 5. Results (on the 10 held-out test motors)

| Metric | Value |
|---|---|
| RUL MAE | ~9 cycles |
| RUL RMSE | ~15 cycles |
| Risk classification accuracy | ~95% |
| Risk classification weighted F1 | ~95% |

Top predictive features: rolling-average current draw, raw vibration,
and rolling-average bearing noise — current draw dominating makes
physical sense, since a motor under mechanical strain pulls more
current to maintain torque.

*(Re-running the generator with a different seed will shift these
numbers slightly — check `model_metrics.txt` for the exact run.)*

## 6. Running it yourself

```bash
pip install -r requirements.txt
python generate_sensor_data.py
python train_model.py
streamlit run app.py
