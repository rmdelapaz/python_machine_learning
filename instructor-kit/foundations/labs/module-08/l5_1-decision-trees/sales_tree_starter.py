# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.1 B: a regression tree for daily store sales.

The store managers want a simple rule of thumb for tomorrow's sales: which days are big, which are
quiet? A regression tree gives exactly that: each leaf predicts the average sales of its training days.
Train on 2023 and 2024, test on 2025 (the past predicts the future), choose the depth with time-ordered
cross-validation, and find the one thing a tree cannot do.
Work through TODO 1 to 5; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: calendar features from the date, and a split by time (done for you)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
sales["day_of_week"] = sales["date"].dt.dayofweek                  # Monday 0 ... Sunday 6
sales["month"] = sales["date"].dt.month
sales["closed"] = (sales["event"] == "Closed").astype(int)
sales["event_day"] = (sales["event"].notna() & (sales["event"] != "Closed")).astype(int)

X = pd.get_dummies(sales[["store"]], dtype=int).join(sales[["day_of_week", "month", "promo", "closed", "event_day"]])
y = sales["sales"]
is_train = sales["date"].dt.year < 2025
X_train, X_test, y_train, y_test = X[is_train], X[~is_train], y[is_train], y[~is_train]
print(X_train.shape, X_test.shape)
print(X.columns.tolist())

# %% Step 2: two baselines a manager could compute without a model
# TODO 1: the test MAE of DummyRegressor() (the overall training mean), and of each store's training
#         mean mapped onto the 2025 rows (y_train.groupby(sales.loc[is_train, "store"]).mean()).
print("TODO 1: two baselines")

# %% Step 3: a depth-3 tree, read as a rule of thumb
# TODO 2: fit DecisionTreeRegressor(max_depth=3, random_state=42) and print
#         export_text(tree, feature_names=list(X.columns), decimals=0). What is the very first question?
#         Write the biggest leaf as one sentence a manager would understand.
print("TODO 2: a depth-3 rule of thumb")

# %% Step 4: choose the depth with time-ordered cross-validation (Lesson 1.3), training days only
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cv = TimeSeriesSplit(n_splits=4)
# TODO 3: for max_depth in [2, 3, 4, 6, 8, 10, None], print the CV MAE
#         (-cross_val_score(..., cv=cv, scoring="neg_mean_absolute_error").mean()).
#         The unlimited tree barely suffers here, unlike the churn tree. Why might that be? (Hint: how many
#         training days share exactly the same store, weekday, month, promo and event values?)
print("TODO 3: depth by time-ordered CV")

# %% Step 5: fit the chosen depth once and test once
# TODO 4: fit max_depth=10 as `tree`, keep pred = tree.predict(X_test), print the test MAE and the
#         number of leaves.
tree = None
pred = None
if tree is None:
    print("TODO 4: the chosen tree on 2025")

# %% Step 6: what a tree cannot do: follow a trend
# TODO 5: per store, compare the average 2025 prediction with the average actual sales in 2025 and in
#         2024. Then compare pred.max() with y_train.max(). Which stores grew, and what did the tree do?
print("TODO 5: predictions vs the trend")

# %% Stretch: which calendar features carry the tree? (feature_importances_, top five)

print("\nExpected when done: overall mean MAE $1,967, per-store mean $1,670; the depth-3 tree asks about")
print("the weekday first (day_of_week <= 4) and its biggest leaf is Boise weekends from May on ($10,059);")
print("CV MAE $1,374 at depth 2 falling to $901 at depth 10 ($913 unlimited); depth 10: test MAE $976,")
print("298 leaves; Boise 2025 actual $8,342 vs predicted $7,339 (2024: $7,821), Flagstaff $6,561 vs")
print("$5,341: the tree cannot follow growth; highest prediction $18,918 vs highest training day $19,755.")
