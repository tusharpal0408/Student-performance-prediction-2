import streamlit as st
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

MODEL_PATH = "model.pkl"

# -----------------------------------
# Function to create synthetic dataset
# -----------------------------------
def create_dataset(n=1500):
    rng = np.random.RandomState(42)
    study = rng.randint(1, 5, size=n)
    fail = rng.randint(0, 4, size=n)
    absn = rng.poisson(3, size=n)
    g1 = np.clip(rng.normal(10, 3, n).round(), 0, 20)
    g2 = np.clip((g1 + rng.normal(1, 2, n)).round(), 0, 20)
    famrel = rng.randint(1, 6, size=n)
    goout = rng.randint(1, 6, size=n)
    health = rng.randint(1, 6, size=n)

    base = 0.5*g2 + 0.3*g1 - 1.5*fail - 0.1*absn + rng.normal(0, 2, n)
    g3 = np.clip(base.round(), 0, 20)

    return pd.DataFrame({
        "studytime": study,
        "failures": fail,
        "absences": absn,
        "G1": g1,
        "G2": g2,
        "famrel": famrel,
        "goout": goout,
        "health": health,
        "G3": g3
    })


# -----------------------------------
# Auto-train model if missing
# -----------------------------------
def train_and_save_model():
    df = create_dataset()
    X = df.drop("G3", axis=1)
    y = df["G3"]

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)

    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_PATH)


# Check if model file exists; if not → auto-generate
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training a new model... (only happens once)")
    train_and_save_model()
    st.success("Model trained & saved successfully!")

# Load trained model
data = joblib.load(MODEL_PATH)
model = data["model"]
features = data["features"]


# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("🎓 Student Performance Prediction App")
st.write("Fill the student details below and get predicted final grade (G3).")

inputs = {}
for feature in features:
    inputs[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict"):
    arr = [inputs[f] for f in features]
    pred = model.predict([arr])[0]
    pred = int(np.clip(round(pred), 0, 20))

    st.success(f"Predicted Final Grade: **{pred} / 20**")

