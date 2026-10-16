# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.6 A: does a detector fitted in March keep working through June?

Lesson 8.6 fitted on January to May and scored June. Here the payments team asks an operations
question: fit once on the first quarter, then score April, May and June month by month. Does the
queue still catch frauds, and what happens to a fixed threshold set in March?
Work through TODO 1 to 4; run the file after each one. It runs as it is.
"""
# %% Step 1: payments, a first-quarter fit window, three months to score (done for you)
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
prep = ColumnTransformer([("num", StandardScaler(), numeric),
                          ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"])])

# %% Step 2: fit once, no labels
# TODO 1: iso = make_pipeline(prep, IsolationForest(n_estimators=300, random_state=42)), fitted on
#         fit_rows[numeric + ["category"]]. Never pass is_fraud.
iso = None
if iso is None:
    print("TODO 1: fit the Isolation Forest pipeline on January to March")

# %% Step 3: the sign convention
# TODO 2: on April, print roc_auc_score(is_fraud, iso.score_samples(april)) and the same with a minus sign.
#         Which one is the anomaly score? (An AUC far below 0.5 is almost always a flipped sign.)
print("TODO 2: AUC of score_samples and of -score_samples on April")

# %% Step 4: a 2% review queue every month
# TODO 3: for each of months 4, 5, 6: anomaly_score = -iso.score_samples(part); budget = round(0.02 * len(part));
#         count the frauds among the budget highest scores; the AUC; and what a random queue would hold
#         (budget * frauds / payments). Put the rows in a DataFrame and print it.
print("TODO 3: one row per month: payments, frauds, budget, caught, AUC, random queue")

# %% Step 5: a fixed threshold drifts
# TODO 4: refit with IsolationForest(n_estimators=300, contamination=0.02, random_state=42) and print, for
#         months 1 to 6, the share of payments predict(...) labels -1. Why is it not 2.00% every month?
print("TODO 4: share flagged by a fixed threshold, months 1 to 6")

print("\nExpected when done: April AUC 0.153 for score_samples and 0.847 for -score_samples; caught 33 of 72")
print("(April), 35 of 65 (May), 33 of 64 (June) with queues of 166, 171 and 165 (a random queue holds about")
print("1.3); AUC 0.847, 0.872, 0.848; the fixed threshold flags 1.88% to 2.16% in the fit months and 1.72%")
print("in June.")
