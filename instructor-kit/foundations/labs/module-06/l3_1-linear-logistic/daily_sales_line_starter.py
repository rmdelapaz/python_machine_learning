# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.1 A: a line for daily store sales.

Lesson 3.1 fitted lines to weekly product demand. Here the finance team asks about whole stores, day by
day: how fast is each store growing, what is a Saturday worth, and what does a sale event add? Train on
2023 and 2024, test on 2025, exactly as a forecaster would. Work through TODO 1 to 6; the file runs as it is.
"""
# %% Step 1 (done for you): load, calendar columns, split by year
import numpy as np
import pandas as pd

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
sales["dow"] = sales["date"].dt.day_name().str[:3]
sales["month"] = sales["date"].dt.month.astype(str)
sales["event"] = sales["event"].fillna("none")
sales["trend"] = (sales["date"] - sales["date"].min()).dt.days / 365.25      # years since 2023-01-01
train, test = sales[sales["date"].dt.year < 2025], sales[sales["date"].dt.year == 2025]
print(len(train), "training days x stores,", len(test), "test")

# %% Step 2: two baselines a planner could use without a model
from sklearn.metrics import mean_absolute_error

# TODO 1: the MAE on 2025 of (a) each store's 2023-2024 mean and (b) each store's mean for that weekday
#         (groupby(["store", "dow"])). Any model has to beat the better of the two.
print("TODO 1: two baselines")

# %% Step 3: one store, one feature: the growth line
from sklearn.linear_model import LinearRegression

boise = train[(train["store"] == "Boise") & (train["event"] != "Closed")]
# TODO 2: slope = covariance of trend and sales / variance of trend; intercept = mean(y) - slope * mean(x).
#         Then fit LinearRegression on boise[["trend"]] and compare. What does the slope mean in dollars?
print("TODO 2: Boise growth by hand and by scikit-learn")

# %% Step 4: all stores, many features, in a pipeline (an Asheville Monday in January with no event is the reference)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

num_cols, cat_cols = ["promo", "trend"], ["store", "dow", "month", "event"]
reference = ["Asheville", "Mon", "1", "none"]      # each category's coefficient is measured against these


def encoder(num):
    return ColumnTransformer([("num", "passthrough", num),
                              ("cat", OneHotEncoder(drop=reference, sparse_output=False), cat_cols)]
                             ).set_output(transform="pandas")


# TODO 3: fit make_pipeline(encoder(num_cols), LinearRegression()) on the training rows, print the 2025 MAE,
#         and print the coefficients of cat__dow_Sat, cat__dow_Tue, num__trend, num__promo and
#         cat__event_Black Friday (pd.Series(model[-1].coef_, index=model[-1].feature_names_in_)).
print("TODO 3: linear regression MAE and five coefficients")

# %% Step 5: is promo a twin?
# TODO 4: print pd.crosstab(train["event"], train["promo"]). Then fit the same model without "promo"
#         (encoder(["trend"]) on train[["trend"] + cat_cols]) and check with np.allclose whether the 2025
#         predictions change. What does that say about the promo coefficient?
print("TODO 4: crosstab and the model without promo")

# %% Step 6: a log target
from sklearn.compose import TransformedTargetRegressor

# TODO 5: wrap the no-promo pipeline in TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1),
#         print its 2025 MAE, then turn its coefficients into percent changes with (np.exp(coef) - 1) * 100.
# TODO 6: in one sentence each: what is a Saturday worth, and how fast are sales growing?
print("TODO 5-6: log-target model and percent effects")

print("\nExpected when done: baselines $1,670 (store mean) and $1,292 (store and weekday); Boise grows")
print("$1,333 a year from $6,037 (by hand and sklearn agree); linear regression MAE $949 with Saturday +3,509,")
print("Tuesday -168, trend +487 a year, promo +2,625 and Black Friday +2,083; promo is 1 exactly on the sale")
print("and holiday-rush days, and dropping it changes no prediction (True); log target MAE $853, Saturday")
print("+74.4%, Tuesday -5.3%, trend +8.1% a year, Black Friday +82.1%.")
