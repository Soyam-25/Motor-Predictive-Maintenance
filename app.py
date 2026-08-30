"""
app.py
------
Streamlit dashboard for the Industrial Motor Predictive Maintenance
System. Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Motor Predictive Maintenance", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv")
    importance = pd.read_csv("feature_importance.csv", index_col=0)
    with open("model_metrics.txt") as f:
        metrics_text = f.read()
    return df, importance, metrics_text

df, importance, metrics_text = load_data()

RISK_COLOR = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

st.title("🔧 Industrial Motor Predictive Maintenance Dashboard")
st.caption("Fleet of 40 simulated motors · RandomForest RUL regression + risk classification")

# ---------------------------------------------------------------
# Fleet overview: a snapshot in time across the fleet
# ---------------------------------------------------------------
# NOTE: this dataset logs each motor's FULL run-to-failure history, so
# the very last row for every motor is always its failure point (RUL=0).
# Using that as "current status" would show every motor as High risk,
# which isn't a meaningful fleet snapshot. Instead we pick a single
# cycle number (140) that every motor has already reached without
# having failed yet (all motors run for a minimum of 150 cycles) - that
# gives a realistic "if you checked the fleet today" snapshot with a
# genuine mix of risk levels instead of a false all-failing picture.
SNAPSHOT_CYCLE = 140
latest = df[df["Cycle"] == SNAPSHOT_CYCLE].copy().sort_values("RUL")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Motors in fleet", latest["Unit_ID"].nunique())
col2.metric("High risk now", int((latest["Risk_Level"] == "High").sum()))
col3.metric("Medium risk now", int((latest["Risk_Level"] == "Medium").sum()))
col4.metric("Low risk now", int((latest["Risk_Level"] == "Low").sum()))

st.subheader(f"Fleet Risk Overview (snapshot at cycle {SNAPSHOT_CYCLE} for every motor)")
fig_fleet = px.bar(
    latest.sort_values("RUL"),
    x="Unit_ID", y="RUL", color="Risk_Level",
    color_discrete_map=RISK_COLOR,
    labels={"RUL": "Remaining Useful Life (cycles)", "Unit_ID": "Motor ID"},
    title=None,
)
fig_fleet.update_xaxes(type="category")
st.plotly_chart(fig_fleet, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------
# Per-unit drill-down
# ---------------------------------------------------------------
st.subheader("Per-Motor Drill-Down")
unit_choice = st.selectbox("Select a motor", sorted(df["Unit_ID"].unique()))
unit_df = df[df["Unit_ID"] == unit_choice].sort_values("Cycle")
is_test_unit = (unit_df["Data_Split"] == "test").any()

left, right = st.columns([2, 1])

with left:
    st.markdown("**Sensor trends over the motor's lifetime**")
    sensor_choice = st.multiselect(
        "Sensors to plot",
        ["Vibration_mm_s", "Temperature_C", "Current_A", "RPM", "Bearing_Noise_dB"],
        default=["Vibration_mm_s", "Current_A"],
    )
    if sensor_choice:
        fig_sensors = go.Figure()
        for s in sensor_choice:
            fig_sensors.add_trace(go.Scatter(x=unit_df["Cycle"], y=unit_df[s], name=s, mode="lines"))
        fig_sensors.update_layout(xaxis_title="Cycle", yaxis_title="Sensor value", height=380)
        st.plotly_chart(fig_sensors, use_container_width=True)

with right:
    st.markdown("**Current status**")
    latest_row = unit_df.iloc[-1]
    risk = latest_row["Risk_Level"]
    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background-color:{RISK_COLOR[risk]}22;"
        f"border:2px solid {RISK_COLOR[risk]}'>"
        f"<b>Risk level: {risk}</b><br>Actual RUL: {int(latest_row['RUL'])} cycles<br>"
        f"Life so far: {int(latest_row['Cycle'])}/{int(latest_row['Life_Cycles'])} cycles"
        f"</div>", unsafe_allow_html=True
    )
    if is_test_unit:
        st.caption("This motor was held out during training — predictions below are genuinely out-of-sample.")
    else:
        st.caption("This motor was used in training — see a held-out motor for an honest accuracy check.")

if is_test_unit:
    st.markdown("**Predicted vs Actual RUL (this motor was NOT used in training)**")
    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=unit_df["Cycle"], y=unit_df["RUL_clipped"], name="Actual RUL", mode="lines"))
    fig_rul.add_trace(go.Scatter(x=unit_df["Cycle"], y=unit_df["Predicted_RUL"], name="Predicted RUL", mode="lines", line=dict(dash="dash")))
    fig_rul.update_layout(xaxis_title="Cycle", yaxis_title="RUL (cycles)", height=350)
    st.plotly_chart(fig_rul, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------
# Model performance + feature importance
# ---------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Feature Importance")
    fig_imp = px.bar(importance.head(8).sort_values("Importance"), x="Importance", y=importance.head(8).sort_values("Importance").index, orientation="h")
    fig_imp.update_layout(yaxis_title="", height=350)
    st.plotly_chart(fig_imp, use_container_width=True)

with c2:
    st.subheader("Model Performance (held-out test motors)")
    st.code(metrics_text, language=None)

st.caption("Data is simulated — see README.md for methodology, assumptions, and how this would change with real sensor data.")
