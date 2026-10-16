# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.1 B: how much does it help to know WHAT is being sold?

A linear demand forecast on numbers alone (price, discount, promo, holiday, temperature, week) cannot
tell socks from tents. Add the store, the product's category, or the product itself as one-hot
columns, and grade each version on 2025 with the MAE from Lesson 1.4 (trained on 2024: a time split).
Then the catch: next year Pinecrest launches a new tent. Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1: load and split by time (done for you: train 2024, test 2025)
import numpy as np
import pandas as pd

demand = pd.read_csv(DATA / "weekly_demand.csv")
numbers = ["price", "discount_pct", "promo", "holiday_week", "avg_high_temp_f", "week_of_year"]
train = demand[demand["year"] == 2024]
test = demand[demand["year"] == 2025]
print(f"train {len(train):,} rows, test {len(test):,} rows")

# %% Step 2: the baseline every forecast must beat
from sklearn.metrics import mean_absolute_error

# TODO 1: the MAE on 2025 of predicting the 2024 mean of units_sold for every week (np.full(...)).
print("TODO 1: the baseline MAE")

# %% Step 3: one function, four versions of the preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def demand_model(categories):
    """Scale the numbers, one-hot the listed text columns, then a linear regression."""
    branches = [("nums", StandardScaler(), numbers)]
    # TODO 2: if `categories` is not empty, append a ("cats", OneHotEncoder(...), categories) branch:
    #         sparse_output=False and handle_unknown="ignore".
    return make_pipeline(ColumnTransformer(branches), LinearRegression())


versions = {"numbers only": [], "+ store": ["store"], "+ store + category": ["store", "category"],
            "+ store + product_id": ["store", "product_id"]}
fitted = {}
print("TODO 2: add the one-hot branch to demand_model (until then every version is 'numbers only')")
# TODO 3: for each version, fit demand_model(cats) on train[numbers + cats] and train["units_sold"],
#         keep it in fitted[name], and print the number of columns the preprocessor makes
#         (model[0].transform(train[numbers + cats].head(1)).shape[1]) and the 2025 MAE.
#         Predict first: which version wins, and by how much? ______
print("TODO 3: four versions: columns and MAE")

# %% Step 4: next year's new tent: the product encoder has never seen it
row = test[(test["product_id"] == "TNT-2001") & (test["store"] == "Boise")].iloc[[25]]
new_tent = row.assign(product_id="TNT-2004")
cols = numbers + ["store", "category", "product_id"]
print(f"\nweek {int(row['week_of_year'].iloc[0])}, Boise, {row['product_name'].iloc[0]}: sold {int(row['units_sold'].iloc[0])}")
# TODO 4: for "+ store + category" and "+ store + product_id", predict `row` and `new_tent` (pass
#         row[cols] and new_tent[cols]: each pipeline picks the columns it knows). Why can the product
#         model not say anything sensible about TNT-2004, and where does its number come from?
print("TODO 4: forecasts for TNT-2001 and the new TNT-2004")

# Stretch: demand_model(["store", "category", "product_id"]): its MAE, and its forecast for TNT-2004.

print("\nExpected when done: 5,088 / 4,896 rows; baseline MAE 17.13; numbers only 6 columns MAE 14.71,")
print("+ store 10 / 14.26, + store + category 19 / 13.26, + store + product_id 34 / 11.31; week 27 in Boise")
print("the tent sold 131; the category model forecasts 52.6 for both tents, the product model 45.7 for")
print("TNT-2001 and 87.4 for TNT-2004. Stretch: MAE 11.31, TNT-2001 45.7, new TNT-2004 59.6.")
