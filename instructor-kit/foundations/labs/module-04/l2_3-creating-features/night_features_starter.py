# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.3 A: fix the backwards hour weight.

In Session 1's fraud lab the logistic regression gave `hour` a NEGATIVE weight, as if late payments
were safe, although fraud clusters at night. A straight line in hour 0..23 cannot say "night". Look at
the training rows, write the questions the model could not ask as features, put them in the pipeline
with FunctionTransformer, and measure the lift with cross-validation before one look at the test set.
Work through TODO 1 to 6. The file runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the same split and the plain pipeline from Lab 2.2 A (done for you)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
num_cols = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(
    tx[num_cols + ["category"]], tx["is_fraud"], test_size=0.3, stratify=tx["is_fraud"], random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% Step 2: look before you build (training rows only)
train = X_train.assign(is_fraud=y_train)
# TODO 1: cut hour into bands with pd.cut(train["hour"], bins=[-1, 5, 11, 17, 23],
#         labels=["0-5", "6-11", "12-17", "18-23"]) and print the fraud rate and size of each band.
# TODO 2: a pivot table of the fraud rate with index = night (hour < 6) and columns = device_new.
#         Where is the fraud? Is it night, a new device, or the two together?
print("TODO 1-2: fraud rate by hour band; night x new device")

# %% Step 3: the questions as features (one pure function)
# TODO 3: complete add_fraud_features: copy df, then add
#         night            = 1 when hour < 6
#         night_new_device = night * device_new           (an interaction term)
#         log_amount       = np.log1p(amount)
#         and return the copy. Then print the mean of the three new columns on X_train.
def add_fraud_features(df):
    df = df.copy()
    return df

new_cols = ["night", "night_new_device", "log_amount"]
if "night" not in add_fraud_features(X_train.head()):
    print("TODO 3: add_fraud_features")

# %% Step 4: measure the lift with cross-validation (the feature step lives INSIDE the pipeline)
def fraud_pipeline(numeric, features=None):
    prep = ColumnTransformer([
        ("scale", StandardScaler(), numeric),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"]),
    ])
    steps = [("features", FunctionTransformer(features))] if features else []
    return Pipeline(steps + [("prep", prep), ("model", LogisticRegression(max_iter=1000))])

# TODO 4: pipes = {"raw columns": fraud_pipeline(num_cols),
#                  "+ night features": fraud_pipeline(num_cols + new_cols, add_fraud_features)}
#         cross_validate each with scoring=["roc_auc", "average_precision"] and print both means.
#         Which metric moves, and which one matters to a review team that checks the top of the list?
print("TODO 4: CV ROC AUC and CV average precision, raw vs + night features")

# %% Step 5: what the model learned now
# TODO 5: fit each pipeline; weights = pd.Series(pipe[-1].coef_[0],
#         index=pipe.named_steps["prep"].get_feature_names_out()). Print the weights of scale__hour,
#         scale__night, scale__night_new_device and scale__device_new (those that exist).
#         (pipe[:-1].get_feature_names_out() fails here: a FunctionTransformer without
#         feature_names_out cannot name its output, so ask the "prep" step.)
print("TODO 5: the hour weight before and after")

# %% Step 6: one look at the test set
from sklearn.metrics import average_precision_score, roc_auc_score

# TODO 6: for each pipeline print test ROC AUC, test average precision and the frauds among the
#         100 test payments with the highest probability.
print("TODO 6: test ROC AUC, average precision, frauds in the top 100")

print("\nExpected when done: night (0-5) 6.71% fraud vs about 0.65% in every other band; night with a new")
print("device 0.425, everything else 0.022 or less; new-column means night 0.037, night_new_device 0.006,")
print("log_amount 5.009; CV ROC AUC 0.867 -> 0.864 but CV average precision 0.535 -> 0.564; hour weight")
print("-0.28 -> 0.01, night_new_device 0.19; test average precision 0.463 -> 0.504, top 100: 56 -> 61 frauds.")
