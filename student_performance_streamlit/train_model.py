
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

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

df = create_dataset()
X = df.drop("G3", axis=1)
y = df["G3"]

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)

joblib.dump({"model": model, "features": list(X.columns)}, "model.pkl")
print("Model saved as model.pkl")
