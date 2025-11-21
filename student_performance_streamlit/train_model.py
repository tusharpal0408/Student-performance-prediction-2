import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def create_dataset(n=1500):
    rng = np.random.RandomState(42)

    study_time = rng.randint(0, 21, n)                    # hours/week
    g1 = np.clip(rng.normal(10, 4, n), 0, 20).round()
    g2 = np.clip((g1 + rng.normal(1, 3, n)), 0, 20).round()
    absences = rng.randint(0, 51, n)                      # days
    health = rng.randint(1, 6, n)                         # 1–5

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

df = create_dataset()
X = df.drop("final_grade", axis=1)
y = df["final_grade"]

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)

joblib.dump({"model": model, "features": list(X.columns)}, "model.pkl")

print("Model.pkl generated successfully with EXACT 5 features:", list(X.columns))
