import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "model.pkl"

# ---------------------------------------------
# Create dataset (same logic as the trainer)
# ---------------------------------------------
def create_dataset(n=1500):
    rng = np.random.RandomState(42)

    study_time = rng.randint(0, 21, n)
    g1 = np.clip(rng.normal(10, 4, n), 0, 20).round()
    g2 = np.clip((g1 + rng.normal(1, 3, n)), 0, 20).round()
    absences = rng.randint(0, 51, n)
    health = rng.randint(1, 6, n)

    base = 0.4*g1 + 0.5*g2 + 0.2*study_time - 0.1*absences + (health*0.8)
    g3 = np.clip(base + rng.normal(0, 2, n), 0, 20).round()

    return pd.DataFrame({
        "study_time": study_time,
        "g1": g1,
        "g2": g2,
        "absences": absences,
        "health": health,
        "final_grade": g3
    })

# ---------------------------------------------
# Auto-train model function
# ---------------------------------------------
def train_and_save_model():
    df = create_dataset()
    X = df.drop("final_grade", axis=1)
    y = df["final_grade"]

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)

    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_PATH)

# ---------------------------------------------
# Delete wrong models (fix mismatch issue)
# ---------------------------------------------
def ensure_correct_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        if len(data["features"]) != 5:
            st.warning("Old model detected. Regenerating with 5 features...")
            os.remove(MODEL_PATH)
            train_and_save_model()
    else:
        st.info("No model found — training a new one...")
        train_and_save_model()

# Ensure model consistency
ensure_correct_model()

# Load correct model
data = joblib.load(MODEL_PATH)
model = data["model"]
features = data["features"]

# ---------------------------------------------
# Streamlit UI (matches EXACT 5 features)
# ---------------------------------------------
st.title("🎓 Student Performance Prediction (5 Inputs)")
st.write("Predict final exam grade (0–20) using simplified inputs.")

study_time = st.number_input("📘 Weekly Study Time (hours/week)", 0, 20, 5)
g1 = st.number_input("📝 First Internal Exam Score (out of 20)", 0, 20, 10)
g2 = st.number_input("📝 Second Internal Exam Score (out of 20)", 0, 20, 12)
absences = st.number_input("🏫 Total Number of Absences (days)", 0, 50, 2)
health = st.number_input("💗 Overall Health Condition (1–5)", 1, 5, 4)

if st.button("Predict"):
    values = [study_time, g1, g2, absences, health]
    prediction = model.predict([values])[0]
    final = int(np.clip(round(prediction), 0, 20))
    st.success(f"Predicted Final Grade: **{final} / 20**")

