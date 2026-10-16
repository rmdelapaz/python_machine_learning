# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Review a classmate's capstone: the demand forecast that "beats the baseline by 55%".

This runs without a single error and prints a result that would look great in a model card. It breaks
FOUR of the rules in the lesson's responsible-ML checklist and rubric. Find each one, say which rubric
row it costs, fix it, and rerun. With all four fixed, the honest numbers are in the answer key; the
model still beats the template's baseline, by less.
"""
# %% The classmate's notebook
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

weeks = pd.read_csv(DATA / "weekly_demand.csv", parse_dates=["week_start"])
weeks["month"] = weeks["week_start"].dt.month
for c in ["store", "product_id", "category"]:
    weeks[c] = weeks[c].astype("category")
features = ["store", "product_id", "category", "month", "price", "promo", "holiday_week", "avg_high_temp_f"]

train, test = train_test_split(weeks, test_size=0.3, random_state=42)

best_mae, best_rate = None, None
for rate in [0.03, 0.05, 0.1, 0.2, 0.3]:
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", learning_rate=rate, random_state=42)
    model.fit(train[features], train["units_sold"])
    mae = mean_absolute_error(test["units_sold"], model.predict(test[features]))
    if best_mae is None or mae < best_mae:
        best_mae, best_rate = mae, rate

baseline = mean_absolute_error(test["units_sold"], np.full(len(test), train["units_sold"].mean()))
print(f"best learning rate {best_rate}: test MAE {best_mae:.2f}")
print(f"baseline (predict the average week): MAE {baseline:.2f}")
print(f"MODEL CARD: our forecast beats the baseline by {1 - best_mae / baseline:.0%}.")
