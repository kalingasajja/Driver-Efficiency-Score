import streamlit as st
import pandas as pd
import numpy as np
import joblib

# loading models
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# user input
def user_input():
    speed_kmh = st.number_input("Speed (km/h)", 0.0, 150.0, 60.0)
    accel_mps2 = st.number_input("Acceleration (m/s²)", -6.0, 7.0, 0.0)
    brake_pct = st.number_input("Brake %", 0.0, 100.0, 5.0)
    gear = st.number_input("Gear", 1, 6, 4)
    rpm = st.number_input("RPM", 600, 2000, 700)
    throttle_pct = st.number_input("Throttle %", 0.0, 100.0, 40.0)
    road_grade = st.number_input("Road Grade %", -12.0, 12.0, 0.0)
    idling_time = st.number_input("Idling Time (s)", 0.0, 60.0, 0.0)
    fuel_rate = st.number_input("Fuel Rate (Lph)", 0.5, 10.0, 4.0)
    E_desired = st.number_input("Desired Efficiency (kmpl)", 14.0, 22.0, 18.0)
    E_actual = st.number_input("Actual Efficiency (kmpl)", 7.0, 24.0, 16.0)
    delta_control = st.number_input("Delta Control", -7.0, 11.0, 2.0)

    return pd.DataFrame({
        "speed_kmh": [speed_kmh],
        "accel_mps2": [accel_mps2],
        "brake_%": [brake_pct],
        "gear": [gear],
        "rpm": [rpm],
        "throttle_%": [throttle_pct],
        "road_grade_%": [road_grade],
        "idling_time_s": [idling_time],
        "fuel_rate_Lph": [fuel_rate],
        "E_desired_kmpl": [E_desired],
        "E_actual_kmpl": [E_actual],
        "delta_control": [delta_control]
    })


# feature engineering
def compute_features(df):
    df = df.copy()  

    df['jerk_mps3'] = df['accel_mps2'].diff().fillna(0)
    df['jerk_abs'] = df['jerk_mps3'].abs()

    window = 5
    df['speed_rolling_mean'] = df['speed_kmh'].rolling(window, min_periods=1).mean().fillna(0)
    df['speed_rolling_std'] = df['speed_kmh'].rolling(window, min_periods=1).std().fillna(0)
    df['accel_rolling_std'] = df['accel_mps2'].rolling(window, min_periods=1).std().fillna(0)
    df['brake_rolling_mean'] = df['brake_%'].rolling(window, min_periods=1).mean().fillna(0)
    df['brake_rolling_std'] = df['brake_%'].rolling(window, min_periods=1).std().fillna(0)
    df['throttle_rolling_std'] = df['throttle_%'].rolling(window, min_periods=1).std().fillna(0)

    df['harsh_brake'] = (df['brake_%'] > 30).astype(int)
    df['harsh_accel'] = (df['accel_mps2'] > 3).astype(int)
    df['harsh_decel'] = (df['accel_mps2'] < -3).astype(int)

    window_agg = 10
    df['harsh_brake_count_10s'] = df['harsh_brake'].rolling(window_agg, min_periods=1).sum()
    df['harsh_accel_count_10s'] = df['harsh_accel'].rolling(window_agg, min_periods=1).sum()
    df['aggressive_events_10s'] = (
        df['harsh_brake_count_10s'] +
        df['harsh_accel_count_10s'] +
        df['harsh_decel'].rolling(window_agg, min_periods=1).sum()
    )

    df['efficiency_ratio'] = df['E_actual_kmpl'] / df['E_desired_kmpl']
    df['efficiency_penalty'] = np.maximum(0, df['E_desired_kmpl'] - df['E_actual_kmpl'])
    df['fuel_per_distance'] = df['fuel_rate_Lph'] / (df['speed_kmh'] + 0.1)
    df['rpm_per_gear'] = df['rpm'] / (df['gear'] + 1)
    df['rpm_speed_ratio'] = df['rpm'] / (df['speed_kmh'] + 1)

    df['uphill'] = (df['road_grade_%'] > 2).astype(int)
    df['downhill'] = (df['road_grade_%'] < -2).astype(int)
    df['throttle_grade_interaction'] = df['throttle_%'] * df['road_grade_%']

    df['coasting'] = ((df['speed_kmh'] > 10) & (df['throttle_%'] < 5)).astype(int)
    df['coasting_downhill'] = df['coasting'] * df['downhill']

    df['speed_brake_interaction'] = df['speed_kmh'] * df['brake_%']
    df['rpm_throttle_interaction'] = df['rpm'] * df['throttle_%']
    df['idle_fuel_waste'] = df['idling_time_s'] * df['fuel_rate_Lph']

    df = df.fillna(0)

    required_columns = [
        'delta_control', 'accel_mps2', 'E_actual_kmpl', 'brake_%', 
        'accel_rolling_std', 'speed_rolling_std', 'throttle_rolling_std', 
        'speed_rolling_mean', 'E_desired_kmpl', 'jerk_mps3', 
        'speed_brake_interaction', 'throttle_%', 'brake_rolling_std', 
        'brake_rolling_mean', 'road_grade_%', 'fuel_per_distance', 
        'fuel_rate_Lph', 'rpm_speed_ratio', 'jerk_abs', 
        'throttle_grade_interaction', 'speed_kmh', 'rpm_per_gear', 
        'rpm', 'aggressive_events_10s', 'harsh_brake_count_10s'
    ]

    # Converting to float
    df = df[required_columns].astype(float)

    return df

# prediction function
def predict(df):
    df_full = compute_features(df)
    df_full = df_full[scaler.feature_names_in_]
    X_scaled = scaler.transform(df_full)
    score = rf_model.predict(X_scaled)[0]
    return score

st.title("Driver Efficiency Predictor 🚗")

input_df = user_input()
score = predict(input_df)

st.subheader("Predicted Driver Score:")
st.metric(label="Score", value=round(score, 2))
