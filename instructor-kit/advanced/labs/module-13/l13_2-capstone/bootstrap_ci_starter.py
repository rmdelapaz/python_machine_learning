# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Capstone lab 3: how sure are you? Intervals for the test score (Milestone 6).

The rubric's Evaluation row asks for "an interval (bootstrap) for the test metric" and the model card
for the model "next to the baseline". Two problems, two lessons: on fraud the June test month holds only
64 frauds, so the interval is wide and two good models cannot be told apart; on demand the gain over the
baseline is clear, but weeks are not independent, so a block bootstrap gives the honest (wider) range. Step 1 is done for you; work through TODO 1 to 4.
"""
# %% Step 1: fraud: two final candidates, fitted on January to May, scored ONCE on June
sys.path.insert(0, str(pathlib.Path(globals().get("__file__", "lab.py")).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from capstone_start import start_demand, start_fraud

fr = start_fraud()
NUM = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
       "country_mismatch", "card_attempts", "velocity_24h"]
cat_type = pd.CategoricalDtype(sorted(fr["train"]["category"].unique()))
X_tr = fr["train"][NUM + ["category"]].astype({"category": cat_type})
X_te = fr["test"][NUM + ["category"]].astype({"category": cat_type})
y_tr, y_te = fr["train"]["is_fraud"].to_numpy(), fr["test"]["is_fraud"].to_numpy()
lr = make_pipeline(make_column_transformer((StandardScaler(), NUM), (OneHotEncoder(handle_unknown="ignore"), ["category"])),
                   LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=42, learning_rate=0.05, max_iter=300,
                                     max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                     early_stopping=True).fit(X_tr, y_tr)
s_lr, s_hgb = lr.predict_proba(X_te)[:, 1], hgb.predict_proba(X_te)[:, 1]
print(f"June: {len(y_te):,} payments, {y_te.sum()} frauds; no-model AP {fr['baseline'][1]:.3f}")

# %% Step 2: a paired bootstrap
rng = np.random.default_rng(42)
# TODO 1: 1,000 times: draw rows = rng.integers(0, len(y_te), len(y_te)); compute the average precision of
#         s_lr and of s_hgb on THOSE rows, and their difference. Print each point estimate with the 2.5th and
#         97.5th percentiles. Before you run it: can you tell the two models apart? ______
print("TODO 1: paired bootstrap of average precision")

# %% Step 3: the operating point
# TODO 2: hold the top 1% of s_hgb (np.quantile(s_hgb, 0.99)). How many June frauds are held? Bootstrap the
#         recall at that capacity the same way (recompute the 1% cut inside each resample).
print("TODO 2: frauds held at the team's capacity, with an interval")

# %% Step 4: demand: row bootstrap vs week bootstrap
# TODO 3: fit HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42) on 2024 with
#         store, product_id, category (as categoricals), month, list_price, discount_pct, price, promo and
#         holiday_week; per-row absolute errors on 2025 for the model and for the per-product mean.
# TODO 4: an interval for the units saved per row, (a) resampling rows, (b) resampling whole WEEKS (all
#         rows of a sampled week together). Which interval belongs in the model card, and why?
print("TODO 3-4: demand, row and week bootstrap")

print("\nExpected when done: AP logistic regression 0.539 (95% CI 0.408 to 0.658), boosting 0.556 (0.423 to")
print("0.668), difference +0.017 (-0.046 to +0.077): not distinguishable. Top 1%: 35 of 64 frauds held (recall")
print("0.55, 0.43 to 0.68). Demand: MAE 8.19 vs 13.42, saved 5.23 units; rows 4.90 to 5.53, weeks 4.68 to 5.81.")
