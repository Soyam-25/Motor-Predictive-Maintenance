"""
generate_sensor_data.py
------------------------
Simulates run-to-failure sensor logs for a fleet of industrial motors.

Each motor ("unit") runs for a number of operating cycles (think:
hours of operation, logged once per cycle) until it fails. Sensor
readings start near a healthy baseline and degrade as the unit
approaches failure - slowly at first, then accelerating in the last
stretch of its life, which is how real bearing/motor wear behaves.

This mirrors the structure of the classic NASA CMAPSS turbofan
degradation dataset (same idea: multiple units, run-to-failure,
multivariate sensors), just built for a motor/pump context instead
of a jet engine, since that fits an industrial automation project.

Output: sensor_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(7)

N_UNITS = 40
MIN_LIFE, MAX_LIFE = 150, 320  # cycles until failure, varies per unit

SENSORS = ["Vibration_mm_s", "Temperature_C", "Current_A", "RPM", "Bearing_Noise_dB"]

# healthy baseline + how much each sensor drifts by end-of-life
baseline = {
    "Vibration_mm_s":   {"start": 1.5,  "end_drift": 6.5,  "noise": 0.12},
    "Temperature_C":    {"start": 45.0, "end_drift": 30.0, "noise": 0.8},
    "Current_A":        {"start": 12.0, "end_drift": 5.5,  "noise": 0.25},
    "RPM":              {"start": 1480, "end_drift": -120, "noise": 6.0},   # RPM drops as it degrades
    "Bearing_Noise_dB": {"start": 55.0, "end_drift": 18.0, "noise": 0.6},
}

rows = []
for unit_id in range(1, N_UNITS + 1):
    life = np.random.randint(MIN_LIFE, MAX_LIFE + 1)
    # each unit runs at a slightly different load level, which affects
    # how fast it wears out (higher load = faster degradation)
    load_factor = np.random.uniform(0.85, 1.25)

    for cycle in range(1, life + 1):
        wear_fraction = cycle / life  # 0 (new) -> 1 (failure point)
        # wear-out curve: slow early, accelerates near end of life
        # (power curve gives that characteristic "hockey stick" shape)
        degradation = wear_fraction ** 2.5

        row = {"Unit_ID": unit_id, "Cycle": cycle, "Life_Cycles": life}
        for sensor, p in baseline.items():
            drift = p["end_drift"] * degradation * load_factor
            value = p["start"] + drift + np.random.normal(0, p["noise"])
            row[sensor] = round(value, 2)
        row["Load_Factor"] = round(load_factor, 3)
        row["RUL"] = life - cycle  # Remaining Useful Life (cycles left)
        rows.append(row)

df = pd.DataFrame(rows)

# Clip RUL at 130: standard practice (as in CMAPSS-style work) since a
# unit's early-life sensor readings don't really carry information about
# exactly how many hundreds of cycles are left - only the wear-out phase
# does. Capping keeps the model from learning a meaningless straight-line
# guess during the healthy phase.
df["RUL_clipped"] = df["RUL"].clip(upper=130)

# risk category for the dashboard traffic-light indicator
def risk_label(rul):
    if rul <= 15:
        return "High"
    elif rul <= 40:
        return "Medium"
    return "Low"

df["Risk_Level"] = df["RUL"].apply(risk_label)

df.to_csv("sensor_data.csv", index=False)
print(f"sensor_data.csv written: {len(df)} rows across {N_UNITS} units")
print(df.groupby("Unit_ID")["Life_Cycles"].first().describe())
