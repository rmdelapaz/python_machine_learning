# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 6.1 B: curves for a demand forecaster, with folds that respect time.

Lesson 6.1 drew a validation curve and learning curves for churn. The demand forecaster from Lesson
5.3 (train 2024, test 2025) raises two new questions: are shuffled folds a fair way to score a
forecaster, and would it gain from more history?
Scores are mean absolute errors in units (scikit-learn reports them negated: bigger is better, so put
a minus sign in front). Work through TODO 1 to 3; the finished file takes about a minute.
"""
# %% Step 1: the 2024 training rows (done for you; 2025 stays locked in this lab)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, learning_curve

demand = pd.read_csv(DATA / "weekly_demand.csv")
features = ["store", "product_id", "category", "week_of_year", "list_price",
            "discount_pct", "price", "promo", "holiday_week", "avg_high_temp_f"]
cats = ["store", "product_id", "category"]
X = demand[features].copy()
X[cats] = X[cats].astype("category")
train = demand["year"] == 2024
X_train, y_train = X[train], demand.loc[train, "units_sold"]
model = HistGradientBoostingRegressor(loss="poisson", random_state=42)
print(len(X_train), "rows of 2024, sorted by week")

# %% Step 2: shuffled folds against blocks of weeks
# TODO 1: score the model with cross_val_score(..., scoring="neg_mean_absolute_error") twice: with
#         KFold(5, shuffle=True, random_state=42) and with KFold(5) (no shuffle: the rows are sorted by
#         week, so each fold is a block of about ten weeks). Print each mean MAE and its std.
#         The 2025 test MAE of this model is 8.05 (Lesson 5.3). Which CV estimate is closer, and why
#         does the other one flatter the model?
print("TODO 1: CV MAE with shuffled folds and with blocks of weeks")

# %% Step 3: a learning curve on blocks of weeks
# TODO 2: learning_curve(model, X_train, y_train, train_sizes=[0.33, 0.67, 1.0], cv=KFold(5),
#         scoring="neg_mean_absolute_error") returns sizes, train scores and validation scores. Print a
#         small table (rows, training MAE, validation MAE) and save the chart to OUTPUT.
n = tr_l = va_l = None
if n is None:
    print("TODO 2: a three-point learning curve and a saved chart")

# %% Step 4: the diagnosis
# TODO 3: from your arrays: the gap between validation and training MAE at full size, and how much the
#         last third of the rows lowered the validation MAE. Print a diagnosis from those numbers: would a
#         third year of history help, or are the curves already flat?
print("TODO 3: the gap, the last improvement, and a diagnosis")

print("\nExpected when done: shuffled folds 7.60 +/- 0.19, blocks of weeks 8.49 +/- 1.34 (2025 test 8.05);")
print("learning curve 1343 / 2726 / 4070 rows: training MAE 3.42 / 4.82 / 5.32, validation MAE 10.66 / 9.96 / 8.41;")
print("gap 3.09 units at full size, the last third cut validation MAE by 1.55: more history should help.")
