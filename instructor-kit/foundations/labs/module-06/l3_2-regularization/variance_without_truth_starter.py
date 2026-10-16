# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.2 B: measure variance without knowing the truth.

Lesson 3.2 said you cannot compute bias and variance on real data, because you do not know the true curve.
That is true for bias. Variance is different: it only asks how much the predictions change when the
training sample changes, and you can measure that by refitting on many random pilot samples. Do it for
daily store sales with a model full of interactions, for least squares and three Ridge penalties. Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1 (done for you): daily sales, calendar columns, 2023-2024 to learn from, 2025 to predict
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

# %% Step 2: thirty pilots of 200 rows
candidates = {"least squares": LinearRegression, "ridge alpha=1": lambda: Ridge(alpha=1),
              "ridge alpha=10": lambda: Ridge(alpha=10), "ridge alpha=100": lambda: Ridge(alpha=100)}
# TODO 1: for each candidate, for seed in range(30): take pilot = train.sample(200, random_state=seed), fit
#         model(make()) on it and predict test[cols]. Keep the 30 prediction arrays and the 30 MAEs.
# TODO 2: per candidate, print the spread (np.array(preds).std(axis=0).mean(): how much the forecast for the
#         same day changes from pilot to pilot), the mean MAE and the worst MAE.
#         Before you run it, predict: which candidate has the smallest spread? Is it also the most accurate?
print("TODO 1-2: spread and MAE over 30 pilots")

# %% Step 3: more data
# TODO 3: repeat for least squares and ridge alpha=1 with 500-row pilots; print only the spread.
# TODO 4: one sentence each: which number is variance, why is bias missing from this table, and why does the
#         most stable model (alpha=100) still forecast worst?
print("TODO 3-4: 500-row pilots")

print("\nExpected when done: 406 columns; over 30 pilots of 200 rows: least squares spread $1,217, mean MAE")
print("$1,351, worst $1,739; ridge alpha=1: $526, $1,089, $1,175; alpha=10: $275, $1,122, $1,232; alpha=100: $167,")
print("$1,527, $1,633; with 500-row pilots the spread falls to $486 (least squares) and $384 (alpha=1).")
