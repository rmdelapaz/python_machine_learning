# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.4 B: attention is a lookup; the keys decide what it finds.

Lesson 10.4's hand-built attention used weekday keys over the last 28 days and beat the seasonal naive
forecast. But on Black Friday it can only average ordinary Fridays. Change WHAT the keys describe
(add the promotion and closed flags) and HOW FAR back the binder reaches (28 days or a year), and
measure each choice separately on ordinary days and on special days. No training: just dot products,
a softmax and a weighted average. Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1: the data, the calendar and the lesson's attention function
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
wide = sales.pivot(index="date", columns="store", values="sales")
is_test_day = wide.index >= "2025-01-01"
events = sales.groupby("date")[["promo", "event"]].first()
weekday = np.eye(7)[wide.index.dayofweek]
promo = events["promo"].to_numpy().astype(float)
closed = (events["event"] == "Closed").to_numpy().astype(float)
special = (promo + closed) > 0
print(f"2025: {is_test_day.sum()} days, {special[is_test_day].sum()} of them special (promotion or closed)")


def attend(query, keys, values, sharpness=5.0):
    scores = sharpness * (keys @ query)
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()
    return weights @ values, weights


def forecast_2025(keys, memory):
    """Attention forecasts for every 2025 store-day; returns predictions, actuals and the special-day mask."""
    pred, actual, spec = [], [], []
    for store in wide.columns:
        values = wide[store].to_numpy()
        for t in np.flatnonzero(is_test_day):
            f, _ = attend(keys[t], keys[t - memory:t], values[t - memory:t])
            pred.append(f)
            actual.append(values[t])
            spec.append(special[t])
    return np.array(pred), np.array(actual), np.array(spec)


def report(name, pred, actual, spec):
    print(f"{name:40s} all ${mean_absolute_error(actual, pred):6,.0f}   ordinary ${mean_absolute_error(actual[~spec], pred[~spec]):6,.0f}"
          f"   special ${mean_absolute_error(actual[spec], pred[spec]):6,.0f}")


# %% Step 2: the yardstick and the lesson's attention, split by kind of day
last_week = wide.shift(7)[is_test_day].to_numpy().T.ravel()       # store by store, like forecast_2025
p28, actual, spec = forecast_2025(weekday, 28)
print()
report("seasonal naive (same day last week)", last_week, actual, spec)
report("weekday keys, 28 days (the lesson)", p28, actual, spec)

# %% Step 3: a longer memory with the same keys
# TODO 1: p364 = forecast_2025(weekday, 364)[0] and report it. Predict first: better or worse on ordinary
#         days than 28 days? On special days?
print("TODO 1: weekday keys, 364 days")

# %% Step 4: richer keys: weekday + promotion + closed
# TODO 2: calendar_keys = np.column_stack([weekday, promo, closed]). Report forecast_2025 with these keys
#         for 28 days and for 364 days. Which change helps special days: the keys, the memory, or both?
print("TODO 2: calendar keys, 28 and 364 days")

# %% Step 5: route each query
# TODO 3: routed = np.where(spec, <calendar keys 364 predictions>, p28): ordinary days look back 28 days,
#         special days a year. Report it. Why can a query "choose" its memory here but not in the lesson?
print("TODO 3: the routed forecast")

# %% Step 6: open the binder for Black Friday 2025 in Boise
# TODO 4: for t = the index of 2025-11-28, call attend(calendar_keys[t], calendar_keys[t - 364:t],
#         boise[t - 364:t]) and print the forecast, the actual and the four past days with the highest
#         weights (date, weekday, event, sales). Which days did the query find?
print("TODO 4: the Black Friday weights")

print("\nExpected when done (MAE all / ordinary / special): seasonal naive $1,072 / $949 / $3,194; lesson")
print("attention $979 / $856 / $3,097; weekday 364 days $1,151 / $1,030 / $3,238; calendar keys 28 days")
print("$950 / $856 / $2,571; calendar keys 364 days $1,039 / $1,030 / $1,198; routed $875 / $856 / $1,198.")
print("Black Friday Boise: forecast $12,014, actual $12,702; weight 0.226 on each of four sale Fridays.")
