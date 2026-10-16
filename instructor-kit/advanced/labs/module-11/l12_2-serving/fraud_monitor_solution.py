# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 12.2 B (SOLUTION): monitoring the fraud model month by month, on REAL months.

The lesson simulated its drift. Here nothing is simulated until Step 5: a fraud model trained on
January to March scores April, May and June as they arrive. Does anything drift? Then the labels
come in: did the model get worse? Finally a broken data feed, to see what a real alarm looks like.
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
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(values):
        counts = np.append(np.histogram(values.dropna(), bins=edges)[0], values.isna().sum())
        return np.clip(counts / len(values), 1e-4, None)

    r, c = shares(ref), shares(cur)
    return float(np.sum((c - r) * np.log(c / r)))


def category_psi(ref, cur):
    s = pd.concat([ref.value_counts(normalize=True), cur.value_counts(normalize=True)], axis=1)
    s = s.fillna(0).clip(lower=1e-4)
    return float(((s.iloc[:, 1] - s.iloc[:, 0]) * np.log(s.iloc[:, 1] / s.iloc[:, 0])).sum())

# %% Step 3: the monthly report, without labels
rows = []
for m, name in [(4, "April"), (5, "May"), (6, "June")]:
    batch = tx[tx["month"] == m]
    values = {c: psi(reference[c], batch[c]) for c in NUMERIC}
    s = score(batch)
    rows.append({"month": name, "max_psi": round(max(values.values()), 3),
                 "category_psi": round(category_psi(reference["category"], batch["category"]), 4),
                 "mean_score": round(float(s.mean()), 4), "hold_rate": round(float((s >= THRESHOLD).mean()), 4)})
report = pd.DataFrame(rows).set_index("month")
ref_s = score(reference)
print(report)
print(f"reference mean_score {ref_s.mean():.4f}, hold_rate {(ref_s >= THRESHOLD).mean():.4f}")

# %% Step 4: months later, the labels arrive
rows = []
for m, name in [(4, "April"), (5, "May"), (6, "June")]:
    batch = tx[tx["month"] == m]
    s = score(batch)
    held = s >= THRESHOLD
    rows.append({"month": name, "fraud_rate": round(batch["is_fraud"].mean(), 4),
                 "held": int(held.sum()), "frauds_held": int(batch["is_fraud"].to_numpy()[held].sum()),
                 "frauds": int(batch["is_fraud"].sum()),
                 "avg_precision": round(average_precision_score(batch["is_fraud"], s), 3)})
print(pd.DataFrame(rows).set_index("month"))

# %% Step 5: a broken feed: from June 16 the velocity_24h counter arrives as 0 for every payment
june = tx[tx["month"] == 6].copy()
broken = june["timestamp"] >= "2025-06-16"
june.loc[broken, "velocity_24h"] = 0
for label, part in [("June 1-15 ", june[~broken]), ("June 16-30", june[broken])]:
    s = score(part)
    held = s >= THRESHOLD
    print(f"{label}: velocity PSI {psi(reference['velocity_24h'], part['velocity_24h']):.3f}, "
          f"hold rate {held.mean():.4f}, frauds held {int(part['is_fraud'].to_numpy()[held].sum())} "
          f"of {int(part['is_fraud'].sum())}")
intact = tx[(tx["month"] == 6) & (tx["timestamp"] >= "2025-06-16")]
held = score(intact) >= THRESHOLD
print(f"June 16-30 with the feed intact: hold rate {held.mean():.4f}, frauds held "
      f"{int(intact['is_fraud'].to_numpy()[held].sum())} of {int(intact['is_fraud'].sum())}")

# %% Step 6: the note for the payments team
print("\nApril to June: no input drift (every PSI under 0.01) and the hold rate stays near 1%. When the labels")
print("arrived, average precision was steady, so the falling fraud rate did not hurt the ranking.")
print("June 16 to 30: velocity_24h shifted (PSI far above 0.25) because the feed broke. Fix the feed and")
print("re-score those payments; do not retrain on them.")
