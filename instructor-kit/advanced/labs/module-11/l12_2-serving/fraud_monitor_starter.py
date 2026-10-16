# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 12.2 B: monitoring the fraud model month by month, on REAL months.

The lesson simulated its drift. Here nothing is simulated until Step 5: a fraud model trained on
January to March scores April, May and June as they arrive. Does anything drift? Then the labels
come in: did the model get worse? Finally a broken data feed, to see what a real alarm looks like.
Step 1 is done for you. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1: the model as it was deployed on March 31
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
tx["month"] = tx["timestamp"].dt.month
NUMERIC = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
FEATURES = NUMERIC + ["category"]
cat_type = pd.CategoricalDtype(sorted(tx["category"].unique()))
reference = tx[tx["month"] <= 3]
model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=42, learning_rate=0.05,
                                       max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
model.fit(reference[FEATURES].astype({"category": cat_type}), reference["is_fraud"])


def score(frame):
    return model.predict_proba(frame[FEATURES].astype({"category": cat_type}))[:, 1]


# In-sample scores are a little optimistic; for a holding threshold at the 1% capacity they are close
# enough (the lesson's cross_val_predict is the careful version).
THRESHOLD = float(np.quantile(score(reference), 0.99))
print(f"reference: {len(reference):,} payments, fraud rate {reference['is_fraud'].mean():.2%}, "
      f"hold threshold {THRESHOLD:.4f}")

# %% Step 2: PSI for numbers and for categories
def psi(ref, cur, bins=10):
    """Population Stability Index of one numeric column; missing values get their own bin."""
    # TODO 1: write it (the lesson's version: decile edges of `ref`, open ends, a bin for missing values,
    #         shares clipped at 0.0001, sum of (cur - ref) * ln(cur / ref)). Return a float.
    return None


def category_psi(ref, cur):
    # TODO 2: the same idea with the categories as bins (value_counts(normalize=True) on both, aligned,
    #         missing categories filled with 0 and clipped at 0.0001).
    return None


if psi(reference["amount"], reference["amount"]) is None:
    print("TODO 1-2: psi and category_psi")

# %% Step 3: the monthly report, without labels
# TODO 3: for April, May and June build one row each: the largest PSI over NUMERIC against `reference`,
#         the category PSI, the mean score and the hold rate (score >= THRESHOLD). Print it next to the
#         reference's own mean score and hold rate. Is anything drifting?
print("TODO 3: the monthly drift report")

# %% Step 4: months later, the labels arrive
# TODO 4: per month: fraud rate, payments held, frauds among them, all frauds, average precision.
#         The fraud rate fell from 0.95% to 0.77%. Did the model get worse at ranking?
print("TODO 4: the label check")

# %% Step 5: a broken feed
june = tx[tx["month"] == 6].copy()
broken = june["timestamp"] >= "2025-06-16"
june.loc[broken, "velocity_24h"] = 0          # from June 16 the counter arrives as 0 for every payment
# TODO 5: for June 1-15 and June 16-30 print the velocity PSI, the hold rate and the frauds held. Then do
#         the same for June 16-30 with the feed intact (from `tx`). What would you tell the payments team?
print("TODO 5: the broken feed")

print("\nExpected when done: every monthly PSI under 0.01 (max 0.003; category 0.0007 to 0.0013), hold rates")
print("0.0089 / 0.0089 / 0.0080 against 0.0100; average precision 0.519 / 0.521 / 0.531 (steady); broken feed:")
print("velocity PSI 2.523, hold rate 0.0025, 10 of 35 frauds held against 19 of 35 with the feed intact.")
