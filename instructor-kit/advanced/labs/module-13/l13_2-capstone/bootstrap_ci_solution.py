# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Capstone lab 3 (SOLUTION): how sure are you? Intervals for the test score (Milestone 6).

The rubric's Evaluation row asks for "an interval (bootstrap) for the test metric" and the model card
for the model "next to the baseline". Two problems, two lessons: on fraud the June test month holds only
64 frauds, so the interval is wide and two good models cannot be told apart; on demand the gain over the
baseline is clear, but weeks are not independent, so a block bootstrap gives the honest (wider) range.
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

# %% Step 2: a paired bootstrap: resample June's payments, score BOTH models on the same resample
rng = np.random.default_rng(42)
boot = []
for _ in range(1000):
    rows = rng.integers(0, len(y_te), len(y_te))
    a, b = average_precision_score(y_te[rows], s_lr[rows]), average_precision_score(y_te[rows], s_hgb[rows])
    boot.append((a, b, b - a))
boot = np.array(boot)
for i, name in enumerate(["logistic regression", "tuned boosting", "boosting minus LR"]):
    point = [average_precision_score(y_te, s_lr), average_precision_score(y_te, s_hgb)]
    value = point[i] if i < 2 else point[1] - point[0]
    low, high = np.percentile(boot[:, i], [2.5, 97.5])
    print(f"{name:20s} AP {value:+.3f}  95% CI {low:+.3f} to {high:+.3f}")

# %% Step 3: the operating point the team cares about: frauds among the 83 payments held (top 1%)
held = s_hgb >= np.quantile(s_hgb, 0.99)
caught = []
for _ in range(1000):
    rows = rng.integers(0, len(y_te), len(y_te))
    h = s_hgb[rows] >= np.quantile(s_hgb[rows], 0.99)
    caught.append(y_te[rows][h].sum() / max(y_te[rows].sum(), 1))
low, high = np.percentile(caught, [2.5, 97.5])
print(f"boosting, top 1%: {y_te[held].sum()} of {y_te.sum()} frauds held (recall {y_te[held].sum() / y_te.sum():.2f}, "
      f"95% CI {low:.2f} to {high:.2f})")

# %% Step 4: demand: the per-product mean against gradient boosting on 2025, row bootstrap vs WEEK bootstrap
from sklearn.ensemble import HistGradientBoostingRegressor

dm = start_demand()
tr, te = dm["train"].copy(), dm["test"].copy()
CATS = ["store", "product_id", "category"]
cols = CATS + ["month", "list_price", "discount_pct", "price", "promo", "holiday_week"]
for frame in (tr, te):
    frame["month"] = frame["week_start"].dt.month
    for c in CATS:
        frame[c] = pd.Categorical(frame[c], categories=sorted(tr[c].unique()))
reg = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42).fit(tr[cols], tr["units_sold"])
err_model = np.abs(te["units_sold"].to_numpy() - reg.predict(te[cols]))
base = te["product_id"].astype(str).map(dm["train"].groupby("product_id")["units_sold"].mean()).to_numpy()
err_base = np.abs(te["units_sold"].to_numpy() - base)
saved = err_base - err_model
print(f"\ndemand 2025: model MAE {err_model.mean():.2f}, per-product mean {err_base.mean():.2f}, saved {saved.mean():.2f} units")
row_ci = np.percentile([saved[rng.integers(0, len(saved), len(saved))].mean() for _ in range(1000)], [2.5, 97.5])
weeks = te["week_start"].to_numpy()
week_ids = np.unique(weeks)
by_week = {w: np.flatnonzero(weeks == w) for w in week_ids}
week_means = []
for _ in range(1000):
    pick = rng.choice(week_ids, len(week_ids), replace=True)
    week_means.append(saved[np.concatenate([by_week[w] for w in pick])].mean())
week_ci = np.percentile(week_means, [2.5, 97.5])
print(f"95% CI, rows resampled:  {row_ci[0]:.2f} to {row_ci[1]:.2f}  ({len(saved):,} rows treated as independent)")
print(f"95% CI, weeks resampled: {week_ci[0]:.2f} to {week_ci[1]:.2f}  ({len(week_ids)} weeks: the honest unit)")
