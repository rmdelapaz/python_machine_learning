# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.2 A (SOLUTION): a dress rehearsal for next year's demand forecast.

The planning team wants each store's weekly units for every product NEXT year, from a model trained on
this year. Lesson 1.2 measured a time split on daily store sales; here the same question meets
weekly_demand.csv: how much does a random split flatter the forecast, and which free baseline must the
model beat?
"""
# %% Step 1: load, and look at the calendar
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
is_2025 = demand["week_start"].dt.year == 2025
past, future = demand[~is_2025], demand[is_2025]
print(f"time split: {len(past):,} train rows (last week {past['week_start'].max().date()}), "
      f"{len(future):,} test rows (first week {future['week_start'].min().date()})")
print("weeks on each side:", past["week_start"].nunique(), future["week_start"].nunique())

# %% Step 3: the careless split: the same number of test rows, drawn at random
shuffled_train, shuffled_test = train_test_split(demand, test_size=len(future), random_state=42)
share_2025 = (shuffled_train["week_start"].dt.year == 2025).mean()
print(f"random split: {len(shuffled_train):,} train rows, {share_2025:.0%} of them from 2025 (the 'future')")

# %% Step 4: the same forest, graded both ways
def forest_mae(train, test):
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    forest.fit(train[numeric], train["units_sold"])
    return mean_absolute_error(test["units_sold"], forest.predict(test[numeric]))

random_mae = forest_mae(shuffled_train, shuffled_test)
time_mae = forest_mae(past, future)
print(f"\nforest MAE, random split: {random_mae:.2f} units per week")
print(f"forest MAE, time split:   {time_mae:.2f} units per week")
verdict = "flatters" if random_mae < time_mae else "does not flatter"
print(f"the random split {verdict} the forest here (difference {random_mae - time_mae:+.2f})")

# %% Step 5: three free baselines on the time split
mean_model = DummyRegressor(strategy="mean").fit(past[numeric], past["units_sold"])
mae_mean = mean_absolute_error(future["units_sold"], mean_model.predict(future[numeric]))

usual = past.groupby(["store", "product_id"])["units_sold"].mean().rename("usual_2024")
with_usual = future.join(usual, on=["store", "product_id"])
mae_usual = mean_absolute_error(with_usual["units_sold"], with_usual["usual_2024"])

ordered = demand.sort_values(["store", "product_id", "week_start"]).copy()
ordered["same_week_last_year"] = ordered.groupby(["store", "product_id"])["units_sold"].shift(52)
last_year = ordered[ordered["week_start"].dt.year == 2025]
print("2025 rows without a value 52 weeks back:", last_year["same_week_last_year"].isna().sum())
mae_last_year = mean_absolute_error(last_year["units_sold"], last_year["same_week_last_year"])

board = pd.Series({"training mean": mae_mean, "each product's 2024 average": mae_usual,
                   "same week last year": mae_last_year, "forest (time split)": time_mae})
print("\nMAE on 2025, units per week (lower is better):")
print(board.sort_values().round(2).to_string())

# %% Step 6: verdict
best_baseline = board.drop("forest (time split)").idxmin()
print(f"\nstrongest free baseline: {best_baseline} ({board[best_baseline]:.2f}); "
      f"the forest {'beats' if time_mae < board[best_baseline] else 'does NOT beat'} it")

# %% Stretch: where does the forest miss most? (MAE by category, time split)
forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
forest.fit(past[numeric], past["units_sold"])
errors = (future["units_sold"] - forest.predict(future[numeric])).abs()
print("\nforest MAE by category, 2025:")
print(errors.groupby(future["category"]).mean().sort_values(ascending=False).round(1).head(3).to_string())
