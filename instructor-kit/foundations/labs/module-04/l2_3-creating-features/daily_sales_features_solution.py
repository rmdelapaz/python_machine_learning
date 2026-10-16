# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.3 B (SOLUTION): features for tomorrow's store sales.

The lesson built weekly features, where every row is a Monday. Pinecrest's store managers want a
forecast of TOMORROW's sales for each store (daily_store_sales.csv), and daily data has a rhythm weekly
data cannot show: the day of the week. Build calendar features and history features per store, use
them first as simple forecasts, then in one linear pipeline. Train on 2023-2024, test on 2025.
"""
# %% Step 1: load with real dates; calendar parts
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
sales["event"] = sales["event"].fillna("none")
sales["day_of_week"] = sales["date"].dt.dayofweek          # 0 = Monday
sales["month"] = sales["date"].dt.month
sales["year"] = sales["date"].dt.year
history = sales[sales["year"] < 2025]
print("mean sales by day of week, 2023-2024 (Mon..Sun):",
      history.groupby("day_of_week")["sales"].mean().round(0).astype(int).tolist())

# %% Step 2: history features per store, past only
sales = sales.sort_values(["store", "date"])
by_store = sales.groupby("store")["sales"]
sales["lag_1"] = by_store.shift(1)                                           # yesterday
sales["lag_7"] = by_store.shift(7)                                           # same weekday last week
sales["rolling_7"] = by_store.transform(lambda s: s.shift(1).rolling(7).mean())   # the 7 days before
sales["store_day"] = sales["store"] + "-" + sales["day_of_week"].astype(str)
print(sales.loc[sales["store"] == "Boise", ["date", "sales", "lag_1", "lag_7", "rolling_7"]].iloc[7:10].round(0))
sales = sales.sort_index()

# %% Step 3: each history feature as a forecast on its own (2025)
test = sales[sales["year"] == 2025]
for col in ["lag_1", "lag_7", "rolling_7"]:
    print(f"{col:9s} as the forecast: MAE {mean_absolute_error(test['sales'], test[col]):,.0f} dollars")

# %% Step 4: one linear pipeline, one idea at a time
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = sales[(sales["year"] < 2025)].dropna(subset=["lag_7", "rolling_7"])

def evaluate(name, num_cols, cat_cols):
    pipe = make_pipeline(
        make_column_transformer((StandardScaler(), num_cols), (OneHotEncoder(handle_unknown="ignore"), cat_cols)),
        Ridge())
    pipe.fit(train[num_cols + cat_cols], train["sales"])
    mae = mean_absolute_error(test["sales"], pipe.predict(test[num_cols + cat_cols]))
    print(f"{name:36s} MAE {mae:,.0f}")

calendar = ["store", "day_of_week", "month", "event"]
evaluate("calendar only", ["promo"], calendar)
evaluate("calendar + lag_7 + rolling_7", ["promo", "lag_7", "rolling_7"], calendar)
evaluate("+ store x day + lag_1", ["promo", "lag_1", "lag_7", "rolling_7"], ["store_day", "month", "event"])

# %% Step 5: the same model with a window that includes TODAY (a silent leak)
sales = sales.sort_values(["store", "date"])
sales["rolling_7_leaky"] = sales.groupby("store")["sales"].transform(lambda s: s.rolling(7).mean())
sales = sales.sort_index()
train = sales[(sales["year"] < 2025)].dropna(subset=["lag_7", "rolling_7_leaky"])
test = sales[sales["year"] == 2025]
evaluate("leaky window (includes today)", ["promo", "lag_1", "lag_7", "rolling_7_leaky"],
         ["store_day", "month", "event"])
