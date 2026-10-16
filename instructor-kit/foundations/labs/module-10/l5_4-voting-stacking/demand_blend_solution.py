# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.4 B (solution): blend two demand forecasters the time-aware way.

Lesson 5.4 warned that stacking needs a partition of the rows, so time-ordered data is blended on a
LATER period instead. Here: two forecasters are trained on January to September 2024, the blend weight
is chosen on October to December 2024 (the "later period"), then both are refitted on all of 2024 and
the blend is judged once on 2025.
"""
# %% Step 1: data, the time split, and a holdout inside 2024
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

# %% Step 2 (TODO 1): fit both on Jan-Sep, predict Oct-Dec
hold = {}
for name, model in [("boost", boost), ("forest", forest)]:
    hold[name] = clone(model).fit(X[early], y[early]).predict(X[late])
    print(f"{name:6s} MAE on Oct-Dec 2024: {mean_absolute_error(y[late], hold[name]):.2f}")
print("correlation of the two holdout forecasts:", round(float(np.corrcoef(hold["boost"], hold["forest"])[0, 1]), 3))

# %% Step 3 (TODO 2): choose the weight on the holdout
weights = np.round(np.arange(0, 1.01, 0.1), 1)
blend_mae = [mean_absolute_error(y[late], w * hold["boost"] + (1 - w) * hold["forest"]) for w in weights]
for w, e in zip(weights, blend_mae):
    print(f"  weight on boost {w:.1f}: holdout MAE {e:.2f}")
best_w = float(weights[int(np.argmin(blend_mae))])
print("chosen weight on boost:", best_w)

# %% Step 4 (TODO 3): refit both on all of 2024 and judge 2025 once
test_pred = {name: clone(model).fit(X[train], y[train]).predict(X[~train])
             for name, model in [("boost", boost), ("forest", forest)]}
test_pred["blend"] = best_w * test_pred["boost"] + (1 - best_w) * test_pred["forest"]
test_pred["50/50 average"] = 0.5 * test_pred["boost"] + 0.5 * test_pred["forest"]
print("\n2025 (opened once):")
for name, pred in test_pred.items():
    print(f"  {name:14s} MAE {mean_absolute_error(y[~train], pred):.2f}")

# %% Step 5 (TODO 4): the verdict
gain = mean_absolute_error(y[~train], test_pred["boost"]) - mean_absolute_error(y[~train], test_pred["blend"])
print(f"\nblend saves {gain:.2f} units per row against the booster alone "
      f"({gain / mean_absolute_error(y[~train], test_pred['boost']):.1%})")
print("Verdict:", "blend" if gain > 0.1 else "keep the single booster: the blend's gain is too small to pay for a second model")
