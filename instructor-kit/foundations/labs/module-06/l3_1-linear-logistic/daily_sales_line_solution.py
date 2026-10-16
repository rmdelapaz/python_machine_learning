# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.1 A (SOLUTION): a line for daily store sales.

Lesson 3.1 fitted lines to weekly product demand. Here the finance team asks about whole stores, day by
day: how fast is each store growing, what is a Saturday worth, and what does a sale event add? Train on
2023 and 2024, test on 2025, exactly as a forecaster would.
"""
# %% Step 1: load, calendar columns, split by year
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

store_mean = train.groupby("store")["sales"].mean()
store_dow = train.groupby(["store", "dow"])["sales"].mean()
print(f"store's mean          MAE ${mean_absolute_error(test['sales'], test['store'].map(store_mean)):,.0f}")
print(f"store's weekday mean  MAE ${mean_absolute_error(test['sales'], store_dow.loc[list(zip(test['store'], test['dow']))].to_numpy()):,.0f}")

# %% Step 3: one store, one feature: the growth line, by hand and by scikit-learn
from sklearn.linear_model import LinearRegression

boise = train[(train["store"] == "Boise") & (train["event"] != "Closed")]
x, y = boise["trend"], boise["sales"]
slope = np.cov(x, y)[0, 1] / x.var()
intercept = y.mean() - slope * x.mean()
line = LinearRegression().fit(boise[["trend"]], y)
print(f"\nBoise by hand: ${slope:,.0f} per year, start ${intercept:,.0f}")
print(f"Boise sklearn: ${line.coef_[0]:,.0f} per year, start ${line.intercept_:,.0f}")

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


linear = make_pipeline(encoder(num_cols), LinearRegression()).fit(train[num_cols + cat_cols], train["sales"])
pred = linear.predict(test[num_cols + cat_cols])
coefs = pd.Series(linear[-1].coef_, index=linear[-1].feature_names_in_)
print(f"\nlinear regression     MAE ${mean_absolute_error(test['sales'], pred):,.0f}")
print(coefs[["cat__dow_Sat", "cat__dow_Tue", "num__trend", "num__promo", "cat__event_Black Friday"]].round(0).to_string())

# %% Step 5: the promo coefficient looks odd. Is promo a twin?
print()
print(pd.crosstab(train["event"], train["promo"]))
no_promo = make_pipeline(encoder(["trend"]), LinearRegression()).fit(train[["trend"] + cat_cols], train["sales"])
same = np.allclose(pred, no_promo.predict(test[["trend"] + cat_cols]))
print("same predictions without promo:", same)

# %% Step 6: a log target: effects in percent
from sklearn.compose import TransformedTargetRegressor

log_model = TransformedTargetRegressor(regressor=make_pipeline(encoder(["trend"]), LinearRegression()),
                                       func=np.log1p, inverse_func=np.expm1)
log_model.fit(train[["trend"] + cat_cols], train["sales"])
log_pred = log_model.predict(test[["trend"] + cat_cols])
log_coefs = pd.Series(log_model.regressor_[-1].coef_, index=log_model.regressor_[-1].feature_names_in_)
pct = (np.exp(log_coefs) - 1) * 100
print(f"\nlog-target model       MAE ${mean_absolute_error(test['sales'], log_pred):,.0f}")
print(pct[["cat__dow_Sat", "cat__dow_Tue", "num__trend", "cat__event_Black Friday"]].round(1).to_string())
