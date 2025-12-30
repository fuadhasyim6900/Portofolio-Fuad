# ==============================
# IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# PATH CONFIG

MODEL_PATH = r"D:\TakeHomeTest\DS\Models"


# LOAD MODEL & ARTIFACTS

model = joblib.load(f"{MODEL_PATH}/xgb_model.pkl")
encoder = joblib.load(f"{MODEL_PATH}/encoder.pkl")

num_cols = joblib.load(f"{MODEL_PATH}/num_cols.pkl")
cat_cols = joblib.load(f"{MODEL_PATH}/cat_cols.pkl")
feature_order = joblib.load(f"{MODEL_PATH}/feature_order.pkl")

num_medians = joblib.load(f"{MODEL_PATH}/num_medians.pkl")
cat_modes = joblib.load(f"{MODEL_PATH}/cat_modes.pkl")


# PAGE CONFIG

st.set_page_config(
    page_title="Food Delivery ETA Prediction",
    page_icon="🚚",
    layout="centered"
)


# GLOBAL STYLE (DARK GRADIENT + INTER FONT)

st.markdown("""
<style>

/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #020617, #0F172A);
    color: #F9FAFB;
}

/* Headings */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #F9FAFB;
}

/* Text */
p, label {
    color: #CBD5E1;
    font-size: 0.95rem;
}

/* Card container */
div[data-testid="stVerticalBlock"] > div {
    background: rgba(15, 23, 42, 0.9);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    margin-bottom: 18px;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
    opacity: 0.95;
}

</style>
""", unsafe_allow_html=True)


# HEADER

st.title("Food Delivery ETA Prediction")
st.caption(
    "Machine learning system to estimate food delivery time based on operational conditions"
)


# INPUT SECTION

st.subheader("Order Details")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input(
        "Distance (km)", min_value=0.1, max_value=50.0, value=5.0
    )
    prep_time = st.number_input(
        "Preparation Time (minutes)", min_value=1, max_value=60, value=15
    )
    experience = st.number_input(
        "Courier Experience (years)", min_value=0.0, max_value=10.0, value=2.0
    )

with col2:
    weather = st.selectbox(
        "Weather", ["Clear", "Rainy", "Foggy", "Windy", "Snowy"]
    )
    traffic = st.selectbox(
        "Traffic Level", ["Low", "Medium", "High"]
    )
    time_of_day = st.selectbox(
        "Time of Day", ["Morning", "Afternoon", "Evening", "Night"]
    )
    vehicle = st.selectbox(
        "Vehicle Type", ["Bike", "Scooter", "Car"]
    )


# DATA WARNING (OUT OF DISTRIBUTION)

if distance > 20:
    st.warning(
        "⚠️ Distance exceeds training data range (max ≈ 20 km). "
        "Prediction may be less reliable."
    )


# CREATE INPUT DATAFRAME

input_df = pd.DataFrame([{
    "Distance_km": distance,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_of_day,
    "Vehicle_Type": vehicle,
    "Preparation_Time_min": prep_time,
    "Courier_Experience_yrs": experience
}])


# FEATURE ENGINEERING

input_df["Distance_Category"] = pd.cut(
    input_df["Distance_km"],
    bins=[0, 3, 7, 12, 20],
    labels=["Very_Near", "Near", "Medium", "Far"]
)

input_df["Is_Peak_Hour"] = input_df["Time_of_Day"].isin(
    ["Morning", "Evening"]
).astype(int)

input_df["Experience_Level"] = pd.cut(
    input_df["Courier_Experience_yrs"],
    bins=[-1, 2, 5, 10],
    labels=["Junior", "Mid", "Senior"]
)

traffic_weight = {"Low": 1.0, "Medium": 1.3, "High": 1.6}
input_df["Distance_x_Traffic"] = (
    input_df["Distance_km"] *
    input_df["Traffic_Level"].map(traffic_weight)
)

input_df["Prep_x_Peak"] = (
    input_df["Preparation_Time_min"] *
    input_df["Is_Peak_Hour"]
)


# HANDLE MISSING VALUES

for col in num_cols:
    input_df[col] = input_df[col].fillna(num_medians[col])

for col in cat_cols:
    input_df[col] = input_df[col].fillna(cat_modes[col])


# ENCODING & FEATURE ALIGNMENT

input_cat = encoder.transform(input_df[cat_cols])

input_cat_df = pd.DataFrame(
    input_cat,
    columns=encoder.get_feature_names_out(cat_cols),
    index=input_df.index
)

input_final = pd.concat(
    [input_df[num_cols], input_cat_df],
    axis=1
)

input_final = input_final.reindex(
    columns=feature_order,
    fill_value=0
)


# PREDICTION

st.markdown("---")

if st.button("Predict Delivery Time"):
    prediction = model.predict(input_final)[0]
    st.success(
        f"Estimated Delivery Time: **{prediction:.1f} minutes**"
    )
