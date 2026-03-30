import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle
import json
import os

print("Loading dataset...")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "airfoil_dataset.csv"))

# The columns in the CSV are: airfoil,Re,AoA,thickness,thickness_loc,...
# We need: Re, alpha (AoA), thickness, thickness_loc, camber, camber_loc

X = df[["Re", "alpha", "thickness", "thickness_loc", "camber", "camber_loc"]]
y_cl = df["Cl"]
y_cd = df["Cd"]

print("Training Cl model...")
m_cl = HistGradientBoostingRegressor().fit(X, y_cl)

print("Training Cd model...")
m_cd = HistGradientBoostingRegressor().fit(X, y_cd)

print("Saving models...")
with open(os.path.join(os.path.dirname(__file__), "model_cl.pkl"), "wb") as f:
    pickle.dump(m_cl, f)

with open(os.path.join(os.path.dirname(__file__), "model_cd.pkl"), "wb") as f:
    pickle.dump(m_cd, f)

with open(os.path.join(os.path.dirname(__file__), "features.json"), "w") as f:
    json.dump({"features": ["Re", "alpha", "thickness", "thickness_loc", "camber", "camber_loc"]}, f)

print("Done retraining!")
