# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.4 B: blend two demand forecasters the time-aware way.

Lesson 5.4 warned that stacking needs a partition of the rows, so time-ordered data is blended on a
LATER period instead. Here: fit two forecasters on January to September 2024, choose the blend weight
on October to December 2024, then refit both on all of 2024 and judge the blend once on 2025.
Work through TODO 1 to 4; the file runs as it is (about 10 seconds when finished).
"""
# %% Step 1: data, the time split, and a holdout inside 2024 (done for you)
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

demand = pd.read_csv(DATA / "weekly_demand.csv")
features = ["store", "product_id", "category", "week_of_year", "list_price",
            "discount_pct", "price", "promo", "holiday_week", "avg_high_temp_f"]
cats = ["store", "product_id", "category"]
X = demand[features].copy()
X[cats] = X[cats].astype("category")
y = demand["units_sold"]
train = demand["year"] == 2024
early = train & (demand["week_start"] < "2024-10-01")
late = train & ~early
print(f"fit Jan-Sep 2024: {early.sum():,} rows | blend on Oct-Dec 2024: {late.sum():,} | test 2025: {(~train).sum():,}")

boost = HistGradientBoostingRegressor(loss="poisson", random_state=42)
forest = make_pipeline(make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                                               remainder="passthrough"),
                       RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=42))

# %% Step 2: fit both on Jan-Sep, predict Oct-Dec
# TODO 1: for each of boost and forest, clone(model).fit(X[early], y[early]).predict(X[late]); keep the
#         predictions in a dict hold and print each one's MAE on Oct-Dec. Then print the correlation of
#         the two forecasts (np.corrcoef). Why must the weight NOT be chosen on 2025?
hold = None
if hold is None:
    print("TODO 1: fit on Jan-Sep 2024, forecast Oct-Dec 2024, MAE and correlation")

# %% Step 3: choose the weight on the holdout
# TODO 2: for w in 0.0, 0.1, ..., 1.0, the MAE of w * boost + (1 - w) * forest on Oct-Dec.
#         Keep the best weight in best_w.
best_w = None
if best_w is None:
    print("TODO 2: blend weights 0.0 to 1.0 on the holdout; keep the best")

# %% Step 4: refit on all of 2024 and judge 2025 once
# TODO 3: refit both on X[train], forecast X[~train], and print the 2025 MAE of boost, forest, the blend
#         with best_w, and a plain 50/50 average.
print("TODO 3: 2025 MAE of both forecasters, your blend and a 50/50 average")

# %% Step 5: the verdict
# TODO 4: how many units per row does the blend save against the booster alone? Print a verdict from your
#         variables: is a second model worth running?
print("TODO 4: the gain and a verdict")

print("\nExpected when done: holdout MAE boost 9.78, forest 10.92, correlation 0.95; the holdout MAE falls")
print("steadily to the weight 1.0 (all booster), so best_w = 1.0; 2025 MAE boost 8.05, forest 8.47, blend 8.05,")
print("50/50 average 8.11; the blend saves 0.00 units: keep the single booster.")
