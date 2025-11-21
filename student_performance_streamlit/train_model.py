import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def create_dataset(n=1500):
    rng = np.random.RandomState(42)

    # 5 simplified features
    study_time = rng.randint(0, 21, size=n)               # hours/week (0–20)
    g1 = np.clip(rng.normal(10, 4, n), 0, 20).round()      # 1st internal exam (0–20)
    g2 = np.clip((g1 + rng.normal(1, 3, n)), 0, 20).round() # 2nd internal exam (0–20)
    absences = rng.randint(0, 51, size=n)                 # days (0–50)
    health = rng.randint(1, 6, size=n)                    # health scale (1–5)

    # Target variable (final grade out of 20)
    base = 0.4*g1 + 0.5*g2 + 0.2*study_time - 0.1*absences + (health*0.8)
    g3 = np.clip(base + rng.normal(0, 2, n), 0, 20).round()

    df = pd.DataFrame({
        "study_time": study_time,
        "g1": g1,
        "g2": g2,
        "absences": absences,
        "health": health,
        "final_grade": g3
    })
    return df


# Load dataset
df = create_dataset()
X = df.drop("final_grade", axis=1)
y = df["final_grade"]

# Train model
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)

# Save model
joblib.dump({"model": model, "features": list(X.columns)}, "model.pkl")
print("Model trained and saved as model.pkl")
