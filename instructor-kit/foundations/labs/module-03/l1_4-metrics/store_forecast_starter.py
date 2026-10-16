# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.4 B: which daily sales forecast should the store managers trust?

Lesson 1.4 graded a weekly product forecast. The store managers have a simpler question: tomorrow's
sales for their store. Four forecasts that need no model at all compete on 2025, using the regression
metrics from the lesson. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1: load, sort by store and date, and build the "copy an earlier day" forecasts (done for you)
import numpy as np
import pandas as pd

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
sales = sales.sort_values(["store", "date"]).reset_index(drop=True)
sales["weekday"] = sales["date"].dt.day_name()
sales["yesterday"] = sales.groupby("store")["sales"].shift(1)        # same store, one row earlier
sales["last_week"] = sales.groupby("store")["sales"].shift(7)        # same store, same weekday last week
train = sales[sales["date"].dt.year < 2025]
test = sales[sales["date"].dt.year == 2025]
print(f"train {len(train):,} rows (2023-2024), test {len(test):,} rows (2025), closed days in 2025: "
      f"{(test['sales'] == 0).sum()}")

# %% Step 2: four forecasts, every number learned from 2023-2024 or from days already past
# TODO 1: the store-weekday mean: train.groupby(["store", "weekday"])["sales"].mean(), joined onto the
#         test rows with test.join(..., on=["store", "weekday"]). Then fill the dictionary.
forecasts = {
    "overall mean": np.full(len(test), train["sales"].mean()),
    "store-weekday mean": None,
    "yesterday": test["yesterday"].to_numpy(),
    "same day last week": test["last_week"].to_numpy(),
}

# %% Step 3: the scoreboard
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

y = test["sales"]
# TODO 2: for each forecast print MAE, RMSE, R2 and the median absolute error (truth first!).
#         Predict first: which forecast wins, and why is RMSE always at least the MAE? ______
for name, pred in forecasts.items():
    if pred is None:
        print(f"TODO 1: build the {name!r} forecast")
print("TODO 2: the scoreboard: MAE, RMSE, R2, median absolute error for each forecast")

# %% Step 4: the MAPE trap, and an honest percentage
from sklearn.metrics import mean_absolute_percentage_error

# TODO 3: for "same day last week", print mean_absolute_percentage_error on all 2025 days, then on the
#         open days only (y > 0), then MAE divided by the mean of y. Which would you put in a report?
print("TODO 3: MAPE on all days, on open days, and MAE / mean sales")

# %% Step 5: where does the best forecast miss?
# TODO 4: the absolute error of "same day last week" per row (test.assign(abs_err=...)); then per store
#         the MAE, the mean sales and their ratio. Which store is worst in dollars, and in relative terms?
# TODO 5: the three rows with the largest error (nlargest), with date, store, sales, last_week, event.
print("TODO 4-5: error by store, and the three biggest misses")

print("\nExpected when done: 2,924 / 1,460 rows, 8 closed days in 2025; MAE overall mean 1,967 (R2 -0.048),")
print("store-weekday mean 1,292 (R2 0.545), yesterday 1,350 (R2 0.493), same day last week 1,072 (RMSE 1,628,")
print("R2 0.620, median error 692); MAPE 1.33e+17 on all days but 0.168 on open days; MAE / mean 0.173;")
print("Boise has the largest MAE (1,356) but the smallest share (0.163); the three biggest misses are two")
print("Black Friday weekend days and Christmas Day (closed).")
