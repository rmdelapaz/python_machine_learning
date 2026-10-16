# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.4 B (SOLUTION): attention is a lookup; the keys decide what it finds.

Lesson 10.4's hand-built attention used weekday keys over the last 28 days and beat the seasonal naive
forecast. But on Black Friday it can only average ordinary Fridays. Change WHAT the keys describe
(add the promotion and closed flags) and HOW FAR back the binder reaches (28 days or a year), and
measure each choice separately on ordinary days and on special days. No training: just dot products,
a softmax and a weighted average.
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
p364, _, _ = forecast_2025(weekday, 364)
report("weekday keys, 364 days", p364, actual, spec)

# %% Step 4: richer keys: weekday + promotion + closed
calendar_keys = np.column_stack([weekday, promo, closed])
c28, _, _ = forecast_2025(calendar_keys, 28)
c364, _, _ = forecast_2025(calendar_keys, 364)
report("calendar keys, 28 days", c28, actual, spec)
report("calendar keys, 364 days", c364, actual, spec)

# %% Step 5: route each query: ordinary days look back 28 days, special days look back a year
routed = np.where(spec, c364, p28)
report("routed (28 ordinary / 364 special)", routed, actual, spec)

# %% Step 6: open the binder for Black Friday 2025 in Boise
t = int(np.flatnonzero(wide.index == "2025-11-28")[0])
boise = wide["Boise"].to_numpy()
f, w = attend(calendar_keys[t], calendar_keys[t - 364:t], boise[t - 364:t])
top = np.argsort(w)[::-1][:4]
print(f"\nBoise, {wide.index[t].date()} ({events['event'].iloc[t]}): forecast ${f:,.0f}, actual ${boise[t]:,.0f}")
for i in top:
    d = t - 364 + i
    print(f"  weight {w[i]:.3f} on {wide.index[d].date()} {wide.index[d].day_name():9s} ({events['event'].iloc[d]}) ${boise[d]:,.0f}")
print("Its query (Friday + promotion) matched the past year's sale Fridays, not ordinary Fridays.")
