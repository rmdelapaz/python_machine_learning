# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.1 B (SOLUTION): lookalike days for the Boise store.

KNeighborsRegressor forecasts a day by averaging the most similar past days. What makes two days "similar":
the weekday, the season, a sale event? And what can a model that only averages the past never do? Train on
2023 and 2024, forecast every day of 2025 for Boise.
"""
# %% Step 1: Boise, calendar features, split by year
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

# %% Step 2: the baseline: Boise's average for that weekday
from sklearn.metrics import mean_absolute_error

dow_mean = train.groupby("dow")["sales"].mean()
print(f"weekday mean             MAE ${mean_absolute_error(test['sales'], test['dow'].map(dow_mean)):,.0f}")

# %% Step 3: raw calendar numbers, unscaled and scaled
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

raw = ["dow", "doy"]
for name, model in [("raw, unscaled", KNeighborsRegressor(n_neighbors=5)),
                    ("raw, scaled", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)))]:
    pred = model.fit(train[raw], train["sales"]).predict(test[raw])
    print(f"{name:24s} MAE ${mean_absolute_error(test['sales'], pred):,.0f}")

# %% Step 4: cycles and events, and the dial k
cyc = ["dow_sin", "dow_cos", "season_sin", "season_cos", "sale_event", "holiday_rush", "closed"]
for k in [1, 5, 15]:
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)).fit(train[cyc], train["sales"])
    pred = model.predict(test[cyc])
    print(f"cycles + events, k={k:2d}   MAE ${mean_absolute_error(test['sales'], pred):,.0f}"
          f"   mean error (actual - forecast) ${(test['sales'] - pred).mean():+,.0f}")

# %% Step 5: what averaging the past cannot do: growth
print(f"\nBoise's mean daily sales: 2023 ${boise.loc[boise['date'].dt.year == 2023, 'sales'].mean():,.0f},"
      f" 2024 ${boise.loc[boise['date'].dt.year == 2024, 'sales'].mean():,.0f}, 2025 ${test['sales'].mean():,.0f}")
recent = train[train["date"].dt.year == 2024]
model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)).fit(recent[cyc], recent["sales"])
pred = model.predict(test[cyc])
print(f"2024 only, k= 5           MAE ${mean_absolute_error(test['sales'], pred):,.0f}"
      f"   mean error ${(test['sales'] - pred).mean():+,.0f}")
