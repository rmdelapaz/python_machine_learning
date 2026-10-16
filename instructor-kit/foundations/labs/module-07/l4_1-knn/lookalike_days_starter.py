# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.1 B: lookalike days for the Boise store.

KNeighborsRegressor forecasts a day by averaging the most similar past days. What makes two days "similar":
the weekday, the season, a sale event? And what can a model that only averages the past never do? Train on
2023 and 2024, forecast every day of 2025 for Boise. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1 (done for you): Boise, calendar features, split by year
import numpy as np
import pandas as pd

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
boise = sales[sales["store"] == "Boise"].copy()
boise["dow"] = boise["date"].dt.dayofweek                       # 0 = Monday ... 6 = Sunday
boise["doy"] = boise["date"].dt.dayofyear                       # 1 ... 366
boise["dow_sin"], boise["dow_cos"] = np.sin(2 * np.pi * boise["dow"] / 7), np.cos(2 * np.pi * boise["dow"] / 7)
boise["season_sin"] = np.sin(2 * np.pi * boise["doy"] / 365.25)
boise["season_cos"] = np.cos(2 * np.pi * boise["doy"] / 365.25)
boise["sale_event"] = boise["event"].isin(["Memorial Day sale", "Labor Day sale", "Black Friday"]).astype(int)
boise["holiday_rush"] = (boise["event"] == "Holiday rush").astype(int)
boise["closed"] = (boise["event"] == "Closed").astype(int)
train, test = boise[boise["date"].dt.year < 2025], boise[boise["date"].dt.year == 2025]
print(len(train), "training days,", len(test), "days to forecast")

# %% Step 2: the baseline
from sklearn.metrics import mean_absolute_error

# TODO 1: the 2025 MAE of "Boise's 2023-2024 average for that weekday" (groupby("dow")).
print("TODO 1: weekday-mean baseline")

# %% Step 3: raw calendar numbers
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

raw = ["dow", "doy"]
# TODO 2: KNeighborsRegressor(n_neighbors=5) on raw, unscaled and inside make_pipeline(StandardScaler(), ...).
#         Print both MAEs. Unscaled, which column decides who the neighbors are (0-6 against 1-366)?
print("TODO 2: raw features, unscaled and scaled")

# %% Step 4: cycles and events
cyc = ["dow_sin", "dow_cos", "season_sin", "season_cos", "sale_event", "holiday_rush", "closed"]
# TODO 3: scaled kNN on cyc for k = 1, 5 and 15: print the MAE and the mean error (actual - forecast).
#         Why are Sunday and Monday close in dow_sin/dow_cos but far apart as 6 and 0?
print("TODO 3: cycles and events, three values of k")

# %% Step 5: growth
# TODO 4: print Boise's mean daily sales in 2023, 2024 and 2025. Why is the mean error positive for every k?
# TODO 5: refit k=5 on 2024 only and print its MAE and mean error. What did you trade?
print("TODO 4-5: growth, and a model trained on 2024 only")

print("\nExpected when done: weekday mean $1,669; raw unscaled $1,889, raw scaled $1,327; cycles and events")
print("k=1 $1,389 (+$517), k=5 $1,244 (+$921), k=15 $1,256 (+$1,016); Boise $6,831, $7,821, $8,342 a day in")
print("2023, 2024, 2025; 2024 only, k=5: $1,202 (+$458).")
