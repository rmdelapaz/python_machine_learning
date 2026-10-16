# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (Lesson 8.6). This handout is deliberately broken.

A colleague's fraud-detector report. One bug stops the script; three more run quietly and make the
report wrong (a split, a sign and a label convention). Fix all four. Fixed, it prints:
outliers flagged in June: 1.65% of payments; AUC 0.848; frauds in the top 165: 32; LOF novelty AUC 0.757.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
numeric = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
train = tx.sample(frac=0.8, random_state=42)          # the past ...
test = tx[tx["timestamp"] >= "2025-06-01"]             # ... and June
y = test["is_fraud"].to_numpy()

def prep():
    return ColumnTransformer([("num", StandardScaler(), numeric),
                              ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"])])

iso = make_pipeline(prep(), IsolationForest(n_estimators=300, contamination=0.02, random_state=42))
iso.fit(train[numeric + ["category"]])
pred = iso.predict(test)
print(f"outliers flagged in June: {(pred == 1).mean():.2%} of payments")

score = iso.score_samples(test)
print(f"AUC {roc_auc_score(y, score):.3f}")
queue = np.argsort(-score)[:165]
print("frauds in the top 165:", y[queue].sum())

lof = make_pipeline(prep(), LocalOutlierFactor(n_neighbors=50))
lof.fit(train[numeric + ["category"]])
print(f"LOF novelty AUC {roc_auc_score(y, -lof.score_samples(test)):.3f}")
