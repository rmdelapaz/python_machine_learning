# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.1 B (SOLUTION): a regression tree for daily store sales.

The store managers want a simple rule of thumb for tomorrow's sales: which days are big, which are
quiet? A regression tree gives exactly that: each leaf predicts the average sales of its training days.
Train on 2023 and 2024, test on 2025 (the past predicts the future), choose the depth with time-ordered
cross-validation, and find the one thing a tree cannot do.
"""
# %% Step 1: calendar features from the date, and a split by time
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
mean_mae = mean_absolute_error(y_test, DummyRegressor().fit(X_train, y_train).predict(X_test))
store_mean = y_train.groupby(sales.loc[is_train, "store"]).mean()
store_mae = mean_absolute_error(y_test, sales.loc[~is_train, "store"].map(store_mean))
print(f"overall mean     test MAE ${mean_mae:,.0f}")
print(f"per-store mean   test MAE ${store_mae:,.0f}")

# %% Step 3: a depth-3 tree, read as a rule of thumb
rule_tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
print(export_text(rule_tree, feature_names=list(X.columns), decimals=0))

# %% Step 4: choose the depth with time-ordered cross-validation (Lesson 1.3), training days only
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cv = TimeSeriesSplit(n_splits=4)
for depth in [2, 3, 4, 6, 8, 10, None]:
    mae = -cross_val_score(DecisionTreeRegressor(max_depth=depth, random_state=42), X_train, y_train,
                           cv=cv, scoring="neg_mean_absolute_error").mean()
    print(f"max_depth={str(depth):4s}  CV MAE ${mae:,.0f}")

# %% Step 5: fit the chosen depth once and test once
tree = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train, y_train)
pred = tree.predict(X_test)
print(f"\ndepth-10 tree   test MAE ${mean_absolute_error(y_test, pred):,.0f}   leaves {tree.get_n_leaves()}")

# %% Step 6: what a tree cannot do: follow a trend
check = pd.DataFrame({"store": sales.loc[~is_train, "store"], "actual 2025": y_test, "predicted": pred})
by_store = check.groupby("store")[["actual 2025", "predicted"]].mean()
by_store["2024 actual"] = y_train[sales.loc[is_train, "date"].dt.year == 2024].groupby(
    sales.loc[is_train & (sales["date"].dt.year == 2024), "store"]).mean()
print(by_store.round(0))
print(f"highest prediction ${pred.max():,.0f} vs highest training day ${y_train.max():,.0f}")

# %% Stretch: which calendar features carry the tree?
importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n" + importance.round(3).head(5).to_string())
