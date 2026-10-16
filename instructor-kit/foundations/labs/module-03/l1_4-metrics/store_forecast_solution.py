# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.4 B (SOLUTION): which daily sales forecast should the store managers trust?

Lesson 1.4 graded a weekly product forecast. The store managers have a simpler question: tomorrow's
sales for their store. Four forecasts that need no model at all compete on 2025, using the regression
metrics from the lesson: MAE, RMSE, R², the median absolute error, and the MAPE trap (the stores close
on Thanksgiving and Christmas Day, so some true values are 0).
"""
# %% Step 1: load, sort by store and date, and build the "copy an earlier day" forecasts
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
store_weekday = train.groupby(["store", "weekday"])["sales"].mean().rename("store_weekday_mean")
forecasts = {
    "overall mean": np.full(len(test), train["sales"].mean()),
    "store-weekday mean": test.join(store_weekday, on=["store", "weekday"])["store_weekday_mean"].to_numpy(),
    "yesterday": test["yesterday"].to_numpy(),
    "same day last week": test["last_week"].to_numpy(),
}

# %% Step 3: the scoreboard
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

y = test["sales"]
print()
for name, pred in forecasts.items():
    print(f"{name:19s} MAE {mean_absolute_error(y, pred):6,.0f}   RMSE {root_mean_squared_error(y, pred):6,.0f}   "
          f"R2 {r2_score(y, pred):6.3f}   median error {median_absolute_error(y, pred):5,.0f}")

# %% Step 4: the MAPE trap, and an honest percentage
from sklearn.metrics import mean_absolute_percentage_error

best = forecasts["same day last week"]
is_open = (y > 0).to_numpy()
print(f"\nMAPE, all 2025 days : {mean_absolute_percentage_error(y, best):.3g}")
print(f"MAPE, open days only: {mean_absolute_percentage_error(y[is_open], best[is_open]):.3f}")
print(f"MAE / mean sales    : {mean_absolute_error(y, best) / y.mean():.3f}")

# %% Step 5: where does the best forecast miss? Absolute and relative error by store
errors = test.assign(abs_err=(y - best).abs())
by_store = errors.groupby("store").agg(mae=("abs_err", "mean"), mean_sales=("sales", "mean"))
by_store["mae_share"] = by_store["mae"] / by_store["mean_sales"]
print()
print(by_store.sort_values("mae", ascending=False).round(3).to_string())

# %% Step 6: the three biggest misses (why RMSE is larger than MAE)
worst = errors.nlargest(3, "abs_err")[["date", "store", "sales", "last_week", "event"]]
print()
print(worst.to_string(index=False))
