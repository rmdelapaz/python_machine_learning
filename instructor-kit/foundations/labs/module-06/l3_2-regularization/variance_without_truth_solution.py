# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.2 B (SOLUTION): measure variance without knowing the truth.

Lesson 3.2 said you cannot compute bias and variance on real data, because you do not know the true curve.
That is true for bias. Variance is different: it only asks how much the predictions change when the
training sample changes, and you can measure that by refitting on many random pilot samples. Do it for
daily store sales with a model full of interactions, for least squares and three Ridge penalties.
"""
# %% Step 1: daily sales, calendar columns, 2023-2024 to learn from, 2025 to predict
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
sales = sales[sales["event"] != "Closed"].copy()          # closed days are known zeros, not forecasts
sales["dow"] = sales["date"].dt.day_name().str[:3]
sales["month"] = sales["date"].dt.month.astype(str)
sales["event"] = sales["event"].fillna("none")
cols = ["store", "dow", "month", "event"]
train, test = sales[sales["date"].dt.year < 2025], sales[sales["date"].dt.year == 2025]


def model(regressor):
    """One-hot store, weekday, month and event, all pairwise products, log target."""
    design = make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                           PolynomialFeatures(2, interaction_only=True, include_bias=False))
    return TransformedTargetRegressor(make_pipeline(design, regressor), func=np.log, inverse_func=np.exp)


n_columns = model(Ridge()).regressor[0].fit_transform(train[cols]).shape[1]
print(len(train), "training rows,", len(test), "rows to predict;", n_columns, "columns")

# %% Step 2: thirty pilots of 200 rows: refit, predict the same 2025 days, measure the spread
candidates = {"least squares": LinearRegression, "ridge alpha=1": lambda: Ridge(alpha=1),
              "ridge alpha=10": lambda: Ridge(alpha=10), "ridge alpha=100": lambda: Ridge(alpha=100)}
summary = {}
for name, make in candidates.items():
    preds, maes = [], []
    for seed in range(30):
        pilot = train.sample(200, random_state=seed)
        p = model(make()).fit(pilot[cols], pilot["sales"]).predict(test[cols])
        preds.append(p)
        maes.append(mean_absolute_error(test["sales"], p))
    preds = np.array(preds)
    summary[name] = {"spread ($)": preds.std(axis=0).mean(), "mean MAE ($)": np.mean(maes),
                     "worst MAE ($)": np.max(maes)}
print()
print(pd.DataFrame(summary).T.round(0).to_string())

# %% Step 3: the same with 500-row pilots, for least squares and the middle penalty
for name in ["least squares", "ridge alpha=1"]:
    preds = np.array([model(candidates[name]()).fit(p[cols], p["sales"]).predict(test[cols])
                      for p in (train.sample(500, random_state=seed) for seed in range(30))])
    print(f"500-row pilots, {name:14s} spread ${preds.std(axis=0).mean():,.0f}")
