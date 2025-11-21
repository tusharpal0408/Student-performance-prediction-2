import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "model.pkl"

# -----------------------------------
# Auto-training code if model missing
# -----------------------------------
def create_dataset(n=1500):
    rng = np.random.RandomState(42)

    study_time = rng.randint(0, 21, size=n)            # hours/week
    g1 = np.clip(rng.normal(10, 4, n), 0, 20).round()   # 1st internal
    g2 = np.clip((g1 + rng.normal(1, 3, n)), 0, 20).round()  # 2nd internal
    absences = rng.randint(0, 51, size=n)              # days absent
    health = rng.randint(1, 6, size=n)                 # 1–5 scale

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


def train_and_save_model():
    df = create_dataset()
    X = df.drop("final_grade", axis=1)
    y = df["final_grade"]

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)

    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_PATH)


# Train model automatically if missing
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training a new model now...")
    train_and_save_model()
    st.success("Model trained and saved successfully!")

# Load trained model
data = joblib.load(MODEL_PATH)
model = data["model"]
features = data["features"]


# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("🎓 Student Performance Prediction")
st.write("Fill the details below to predict the final exam grade (0–20).")

# Input fields
study_time = st.number_input("📘 Weekly Study Time (hours/week)", 0, 20, 5)
g1 = st.number_input("📝 First Internal Exam Score (out of 20)", 0, 20, 10)
g2 = st.number_input("📝 Second Internal Exam Score (out of 20)", 0, 20, 12)
absences = st.number_input("🏫 Total Number of Absences (days)", 0, 50, 2)
health = st.number_input("💗 Overall Health Condition (1–5)", 1, 5, 4)

if st.button("Predict Final Grade"):
    input_values = [study_time, g1, g2, absences, health]
    prediction = model.predict([input_values])[0]
    final = int(np.clip(round(prediction), 0, 20))

    st.success(f"Predicted Final Grade: **{final} / 20**")
