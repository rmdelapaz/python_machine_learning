# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.3 B: features for tomorrow's store sales.

The lesson built weekly features, where every row is a Monday. Pinecrest's store managers want a
forecast of TOMORROW's sales for each store (daily_store_sales.csv), and daily data has a rhythm weekly
data cannot show: the day of the week. Build calendar features and history features per store, use
them first as simple forecasts, then in one linear pipeline. Train on 2023-2024, test on 2025.
Work in pairs; swap the keyboard at TODO 4. The file runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: load with real dates; calendar parts
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
sales["event"] = sales["event"].fillna("none")
# TODO 1: add day_of_week (.dt.dayofweek, 0 = Monday), month and year from sales["date"]. Then print the
#         mean sales by day of week over the training years only (year < 2025). Which day is biggest?
if "day_of_week" not in sales:
    print("TODO 1: calendar parts and mean sales by day of week (2023-2024)")

# %% Step 2: history features per store, past only
# TODO 2: sort by store and date; by_store = sales.groupby("store")["sales"]; then
#         lag_1 = by_store.shift(1); lag_7 = by_store.shift(7);
#         rolling_7 = by_store.transform(lambda s: s.shift(1).rolling(7).mean());
#         store_day = store + "-" + day_of_week as text. Print Boise's rows 7 to 9 (iloc[7:10]) with
#         date, sales, lag_1, lag_7, rolling_7, and check one lag by hand. Restore with sort_index().
if "lag_7" not in sales:
    print("TODO 2: lag_1, lag_7, rolling_7 and store_day")

# %% Step 3: each history feature as a forecast on its own (2025)
# TODO 3: test = sales[sales["year"] == 2025]; print the MAE of lag_1, lag_7 and rolling_7 used directly
#         as the forecast. Why does "same day last week" beat both "yesterday" and "the last 7 days"?
print("TODO 3: three history features as forecasts")

# %% Step 4: one linear pipeline, one idea at a time
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO 4: train = the rows with year < 2025, minus the first days that have no lag_7 or rolling_7
#         (dropna(subset=[...])). Write evaluate(name, num_cols, cat_cols): a pipeline that scales the
#         numeric columns, one-hot encodes the categorical ones (handle_unknown="ignore"), ends with
#         Ridge(), fits on train and prints the 2025 MAE. Then run three versions:
#         "calendar only"                 ["promo"]                               + store, day_of_week, month, event
#         "calendar + lag_7 + rolling_7"  ["promo", "lag_7", "rolling_7"]         + the same four
#         "+ store x day + lag_1"         ["promo", "lag_1", "lag_7", "rolling_7"] + store_day, month, event
print("TODO 4: three linear models on a scoreboard")

# %% Step 5: the same model with a window that includes TODAY (a silent leak)
# TODO 5: rolling_7_leaky = the 7-day rolling mean WITHOUT shift(1). Rebuild train and test, and rerun
#         the third model with rolling_7_leaky in place of rolling_7. It looks better. Why can it not be
#         used on the evening before the day you forecast?
print("TODO 5: the leaky window")

print("\nExpected when done: mean sales Mon..Sun [4639, 4270, 4512, 4750, 6024, 8209, 6988]; Boise on")
print("2023-01-09: sales 3922, lag_1 7949, lag_7 4433, rolling_7 5781; as forecasts lag_1 MAE 1,350,")
print("lag_7 1,072, rolling_7 1,451; models: calendar only 996, + lag_7 + rolling_7 708, + store x day +")
print("lag_1 586; leaky window 551 (better, and useless).")
