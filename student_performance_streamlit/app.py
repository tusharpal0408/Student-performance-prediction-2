
import streamlit as st
import joblib
import numpy as np

data = joblib.load("model.pkl")
model = data["model"]
features = data["features"]

st.title("🎓 Student Performance Prediction App")
st.write("Enter student details to predict the final grade (0–20).")

inputs = {}
for f in features:
    inputs[f] = st.number_input(f"{f}", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict"):
    values = [inputs[f] for f in features]
    pred = model.predict([values])[0]
    pred = int(np.clip(round(pred), 0, 20))
    st.success(f"Predicted Final Grade (G3): {pred} / 20")
