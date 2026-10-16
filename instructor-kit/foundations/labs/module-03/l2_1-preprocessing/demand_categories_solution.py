# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.1 B (SOLUTION): how much does it help to know WHAT is being sold?

A linear demand forecast on numbers alone (price, discount, promo, holiday, temperature, week) cannot
tell socks from tents. Add the store, the product's category, or the product itself as one-hot
columns, and grade each version on 2025 with the MAE from Lesson 1.4 (trained on 2024: a time split).
Then the catch: next year Pinecrest launches a new tent, and the most detailed encoding knows nothing
about it.
"""
# %% Step 1: load and split by time (train 2024, test 2025)
import numpy as np
import pandas as pd

demand = pd.read_csv(DATA / "weekly_demand.csv")
numbers = ["price", "discount_pct", "promo", "holiday_week", "avg_high_temp_f", "week_of_year"]
train = demand[demand["year"] == 2024]
test = demand[demand["year"] == 2025]
print(f"train {len(train):,} rows, test {len(test):,} rows")

# %% Step 2: the baseline every forecast must beat
from sklearn.metrics import mean_absolute_error

baseline_mae = mean_absolute_error(test["units_sold"], np.full(len(test), train["units_sold"].mean()))
print(f"predict the 2024 mean: MAE {baseline_mae:.2f} units per week")

# %% Step 3: one function, four versions of the preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def demand_model(categories):
    """Scale the numbers, one-hot the listed text columns, then a linear regression."""
    branches = [("nums", StandardScaler(), numbers)]
    if categories:
        branches.append(("cats", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categories))
    return make_pipeline(ColumnTransformer(branches), LinearRegression())


versions = {"numbers only": [], "+ store": ["store"], "+ store + category": ["store", "category"],
            "+ store + product_id": ["store", "product_id"]}
fitted = {}
print()
for name, cats in versions.items():
    model = demand_model(cats).fit(train[numbers + cats], train["units_sold"])
    fitted[name] = model
    n_cols = model[0].transform(train[numbers + cats].head(1)).shape[1]
    mae = mean_absolute_error(test["units_sold"], model.predict(test[numbers + cats]))
    print(f"{name:21s} {n_cols:2d} columns   MAE {mae:5.2f}")

# %% Step 4: next year's new tent: the product encoder has never seen it
row = test[(test["product_id"] == "TNT-2001") & (test["store"] == "Boise")].iloc[[25]]
new_tent = row.assign(product_id="TNT-2004")
cols = numbers + ["store", "category", "product_id"]
print(f"\nweek {int(row['week_of_year'].iloc[0])}, Boise, {row['product_name'].iloc[0]}: sold {int(row['units_sold'].iloc[0])}")
for name in ["+ store + category", "+ store + product_id"]:
    m = fitted[name]
    known, unknown = m.predict(row[cols])[0], m.predict(new_tent[cols])[0]
    print(f"{name:21s} TNT-2001 {known:5.1f}   new TNT-2004 {unknown:5.1f}")

# %% Stretch: keep BOTH the category and the product, so an unknown product still has a category
both = demand_model(["store", "category", "product_id"]).fit(train[cols], train["units_sold"])
print(f"\n+ store + category + product_id: MAE {mean_absolute_error(test['units_sold'], both.predict(test[cols])):.2f}   "
      f"TNT-2001 {both.predict(row[cols])[0]:.1f}   new TNT-2004 {both.predict(new_tent[cols])[0]:.1f}")
