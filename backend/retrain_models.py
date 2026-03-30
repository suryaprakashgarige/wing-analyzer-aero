import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle
import json
import os

print("Loading dataset...")
df = pd.read_csv(r"c:\Wing Analyzer\airfoil_project\airfoil_dataset.csv")

# The columns in the CSV are: airfoil,Re,AoA,thickness,thickness_loc,...
# We need: Re, alpha (AoA), thickness, thickness_loc, camber, camber_loc

X = df[["Re", "AoA", "thickness", "thickness_loc", "camber", "camber_loc"]]
y_cl = df["Cl"]
y_cd = df["Cd"]

print("Training Cl model...")
m_cl = HistGradientBoostingRegressor().fit(X, y_cl)

print("Training Cd model...")
m_cd = HistGradientBoostingRegressor().fit(X, y_cd)

print("Saving models...")
with open(r"c:\Wing Analyzer\wing-analyzer\backend\model_cl.pkl", "wb") as f:
    pickle.dump(m_cl, f)

with open(r"c:\Wing Analyzer\wing-analyzer\backend\model_cd.pkl", "wb") as f:
    pickle.dump(m_cd, f)

with open(r"c:\Wing Analyzer\wing-analyzer\backend\features.json", "w") as f:
    json.dump({"features": ["Re", "alpha", "thickness", "thickness_loc", "camber", "camber_loc"]}, f)

print("Done retraining!")
