# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.6 A (SOLUTION): does a detector fitted in March keep working through June?

Lesson 8.6 fitted on January to May and scored June. Here the payments team asks an operations
question: fit once on the first quarter, then score April, May and June month by month. Does the
queue still catch frauds, and what happens to a fixed threshold set in March?
"""
# %% Step 1: payments, a first-quarter fit window, three months to score
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
tx["month"] = tx["timestamp"].dt.month
numeric = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
fit_rows = tx[tx["month"] <= 3]
print(f"fit on January to March: {len(fit_rows):,} payments (labels not used)")

# %% Step 2: the lesson's pipeline, fitted once, no labels
prep = ColumnTransformer([("num", StandardScaler(), numeric),
                          ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"])])
iso = make_pipeline(prep, IsolationForest(n_estimators=300, random_state=42))
iso.fit(fit_rows[numeric + ["category"]])

# %% Step 3: the sign convention, checked once on April
april = tx[tx["month"] == 4]
raw = iso.score_samples(april)
print(f"April AUC of score_samples : {roc_auc_score(april['is_fraud'], raw):.3f}  (higher = MORE normal)")
print(f"April AUC of -score_samples: {roc_auc_score(april['is_fraud'], -raw):.3f}  (the anomaly score)")

# %% Step 4: a 2% review queue every month
rows = []
for month in [4, 5, 6]:
    part = tx[tx["month"] == month]
    y = part["is_fraud"].to_numpy()
    anomaly_score = -iso.score_samples(part)
    budget = round(0.02 * len(part))
    queue = np.argsort(-anomaly_score)[:budget]
    rows.append({"month": month, "payments": len(part), "frauds": int(y.sum()), "budget": budget,
                 "caught": int(y[queue].sum()), "auc": round(roc_auc_score(y, anomaly_score), 3)})
report = pd.DataFrame(rows).set_index("month")
report["random_queue"] = (report["budget"] * report["frauds"] / report["payments"]).round(1)
print(report)

# %% Step 5: a fixed threshold set on the fit months drifts
fixed = make_pipeline(prep, IsolationForest(n_estimators=300, contamination=0.02, random_state=42))
fixed.fit(fit_rows[numeric + ["category"]])
for month in [1, 2, 3, 4, 5, 6]:
    part = tx[tx["month"] == month]
    print(f"month {month}: fixed threshold flags {(fixed.predict(part) == -1).mean():.2%}")
