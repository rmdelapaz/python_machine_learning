# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.3 A (SOLUTION): fix the backwards hour weight.

In Session 1's fraud lab the logistic regression gave `hour` a NEGATIVE weight, as if late payments
were safe, although fraud clusters at night. A straight line in hour 0..23 cannot say "night". Look at
the training rows, write the questions the model could not ask as features, put them in the pipeline
with FunctionTransformer, and measure the lift with cross-validation before one look at the test set.
"""
# %% Step 1: the same split and the plain pipeline from Lab 2.2 A
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
bands = pd.cut(train["hour"], bins=[-1, 5, 11, 17, 23], labels=["0-5", "6-11", "12-17", "18-23"])
print(train.groupby(bands, observed=True)["is_fraud"].agg(["mean", "size"]).round(4))
night = (train["hour"] < 6).rename("night")
print(train.pivot_table(index=night, columns="device_new", values="is_fraud", aggfunc="mean").round(3))

# %% Step 3: the questions as features (one pure function)
def add_fraud_features(df):
    df = df.copy()
    df["night"] = (df["hour"] < 6).astype(int)
    df["night_new_device"] = df["night"] * df["device_new"]
    df["log_amount"] = np.log1p(df["amount"])
    return df

new_cols = ["night", "night_new_device", "log_amount"]
print(add_fraud_features(X_train)[new_cols].mean().round(3).to_dict())

# %% Step 4: measure the lift with cross-validation (the feature step lives INSIDE the pipeline)
def fraud_pipeline(numeric, features=None):
    prep = ColumnTransformer([
        ("scale", StandardScaler(), numeric),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"]),
    ])
    steps = [("features", FunctionTransformer(features))] if features else []
    return Pipeline(steps + [("prep", prep), ("model", LogisticRegression(max_iter=1000))])

pipes = {"raw columns": fraud_pipeline(num_cols),
         "+ night features": fraud_pipeline(num_cols + new_cols, add_fraud_features)}
for name, pipe in pipes.items():
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
    print(f"{name:18s} CV ROC AUC {res['test_roc_auc'].mean():.3f}   "
          f"CV average precision {res['test_average_precision'].mean():.3f}")

# %% Step 5: what the model learned now
for name, pipe in pipes.items():
    pipe.fit(X_train, y_train)
    # pipe[:-1].get_feature_names_out() fails when the first step is a FunctionTransformer without
    # feature_names_out, so ask the "prep" step (the last one before the model) for the names.
    w = pd.Series(pipe[-1].coef_[0], index=pipe.named_steps["prep"].get_feature_names_out())
    show = [c for c in ["scale__hour", "scale__night", "scale__night_new_device", "scale__device_new"] if c in w]
    print(f"{name:18s}", w[show].round(2).to_dict())

# %% Step 6: one look at the test set
from sklearn.metrics import average_precision_score, roc_auc_score

for name, pipe in pipes.items():
    p = pipe.predict_proba(X_test)[:, 1]
    top100 = pd.Series(p, index=X_test.index).nlargest(100).index
    print(f"{name:18s} test ROC AUC {roc_auc_score(y_test, p):.3f}   average precision "
          f"{average_precision_score(y_test, p):.3f}   frauds in the top 100: {y_test.loc[top100].sum()}")
