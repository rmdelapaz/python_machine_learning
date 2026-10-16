# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.2 A: a dress rehearsal for next year's demand forecast.

The planning team wants each store's weekly units for every product NEXT year, from a model trained on
this year. Lesson 1.2 measured a time split on daily store sales; here the same question meets
weekly_demand.csv: does a random split flatter the forecast, and which free baseline must the model beat?
Work through TODO 1 to 5; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: load, and look at the calendar (done for you)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

demand = pd.read_csv(DATA / "weekly_demand.csv", parse_dates=["week_start"])
numeric = ["week_of_year", "store_sqft", "list_price", "price", "discount_pct",
           "promo", "holiday_week", "avg_high_temp_f"]
print(demand.shape, demand["week_start"].min().date(), "to", demand["week_start"].max().date())

# %% Step 2: the honest split: train on 2024, test on 2025
# TODO 1: is_2025 = demand["week_start"].dt.year == 2025; past = the other rows, future = the 2025 rows.
#         Print the row counts, the last training week and the first test week, and the number of distinct
#         weeks on each side (.nunique()). Predict first: 104 weeks, so 52 and 52?
past = future = None
if past is None:
    print("TODO 1: time split (2024 train, 2025 test)")

# %% Step 3: the careless split: the same number of test rows, drawn at random
# TODO 2: shuffled_train, shuffled_test = train_test_split(demand, test_size=len(future), random_state=42)
#         Print the share of shuffled_train rows whose week_start is in 2025.
shuffled_train = shuffled_test = None
if shuffled_train is None:
    print("TODO 2: random split with test_size=len(future)")


# %% Step 4: the same forest, graded both ways
def forest_mae(train, test):
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    forest.fit(train[numeric], train["units_sold"])
    return mean_absolute_error(test["units_sold"], forest.predict(test[numeric]))


# TODO 3: random_mae = forest_mae(shuffled_train, shuffled_test); time_mae = forest_mae(past, future).
#         Print both. Before you run it, write your prediction: how much lower will the random split be? ____
random_mae = time_mae = None
if time_mae is None:
    print("TODO 3: forest MAE on the random split and on the time split")

# %% Step 5: three free baselines on the time split
# TODO 4: compute the 2025 MAE of
#   a) DummyRegressor(strategy="mean") fitted on past
#   b) each store and product's 2024 average:
#        usual = past.groupby(["store", "product_id"])["units_sold"].mean().rename("usual_2024")
#        with_usual = future.join(usual, on=["store", "product_id"])
#   c) the same week last year: sort demand by store, product_id, week_start, then
#        groupby(["store", "product_id"])["units_sold"].shift(52), and keep the 2025 rows.
#        Check that no 2025 row is missing a value.
print("TODO 4: three baselines (training mean, each product's 2024 average, same week last year)")

# %% Step 6: verdict
# TODO 5: put the three baselines and the forest's time-split MAE in one pd.Series, sort it, print it.
#         Which is the strongest FREE baseline, and does the forest beat it?
print("TODO 5: one sorted table, and a one-line verdict")

print("\nExpected when done: 5,088 train rows (last week 2024-12-30) and 4,896 test rows (first week")
print("2025-01-06), 53 and 51 weeks; half the random split's training rows come from 2025; forest MAE 10.45")
print("on the random split vs 10.27 on the time split (no flattery here: measure, do not assume); baselines")
print("12.95 (each product's 2024 average), 13.81 (same week last year), 17.13 (training mean).")
