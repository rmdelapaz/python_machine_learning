# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 2.2 A: a deployable fraud pipeline.

Lesson 2.2 sealed the churn workflow into one Pipeline. The payments team wants the same for card
fraud: one object that takes raw payments (text category and all), is compared with a baseline by
cross-validation on the training rows only, is tested once, and is saved to a file that scores new
payments, including a category the model has never seen. Work through TODO 1 to 7; run the file
after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: raw columns and a stratified 70/30 split (done for you: the same rows as Session 1's fraud lab)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
num_cols = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
cat_cols = ["category"]
X_train, X_test, y_train, y_test = train_test_split(
    tx[num_cols + cat_cols], tx["is_fraud"], test_size=0.3, stratify=tx["is_fraud"], random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(X_train.shape, X_test.shape, f"frauds in train: {y_train.sum()}")

# %% Step 2: one preprocessor, one pipeline with named steps
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO 1: a ColumnTransformer called preprocess with two branches: ("scale", StandardScaler(), num_cols)
#         and ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols).
#         Then fraud_pipe = Pipeline([("prep", preprocess), ("model", LogisticRegression())]).
preprocess = None
fraud_pipe = None
if fraud_pipe is None:
    print("TODO 1: build preprocess and fraud_pipe")
else:
    print(list(fraud_pipe.named_steps))

# %% Step 3: compare with a baseline pipeline by cross-validation (training rows only)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

# TODO 2: baseline_pipe = the same "prep" step with DummyClassifier(strategy="prior") as "model".
#         For both pipelines, cross_validate(pipe, X_train, y_train, cv=cv,
#         scoring=["roc_auc", "average_precision"]) and print the mean of res["test_roc_auc"] and of
#         res["test_average_precision"]. Before you run it: what will the baseline's average precision be?
print("TODO 2: baseline vs logistic regression, CV ROC AUC and CV average precision")

# %% Step 4: reach inside the fitted pipeline
# TODO 3: fit fraud_pipe on the training rows. From fraud_pipe.named_steps["prep"].named_transformers_["onehot"]
#         print how many categories it learned (categories_[0]). Then build
#         pd.Series(fraud_pipe[-1].coef_[0], index=fraud_pipe[:-1].get_feature_names_out())
#         and print the three largest weights. Which category pushes hardest toward fraud?
print("TODO 3: categories learned and the three largest weights")

# %% Step 5: turn a knob through the pipeline (step__parameter)
from sklearn.model_selection import cross_val_score

# TODO 4: for C in [0.01, 1.0]: fraud_pipe.set_params(model__C=C), then print the CV average precision.
#         Put C back to 1.0 afterwards.
print("TODO 4: model__C sweep")

# %% Step 6: open the test set once
from sklearn.metrics import average_precision_score, roc_auc_score

# TODO 5: fit on all training rows, then print test ROC AUC, test average precision, and how many real
#         frauds are among the 100 test payments with the highest probability.
print("TODO 5: one look at the test set")

# %% Step 7: save, reload, and score raw payments (one category the model has never seen)
import joblib

incoming = pd.DataFrame([
    {"amount": 450.0, "account_age_days": 0, "hour": 3, "items": 1, "device_new": 1,
     "billing_shipping_mismatch": 1, "country_mismatch": 0, "card_attempts": 3, "velocity_24h": 4,
     "category": "Gift Cards"},
    {"amount": 89.5, "account_age_days": 1200, "hour": 14, "items": 2, "device_new": 0,
     "billing_shipping_mismatch": 0, "country_mismatch": 0, "card_attempts": 1, "velocity_24h": 0,
     "category": "Footwear"},
    {"amount": 129.0, "account_age_days": 300, "hour": 11, "items": 1, "device_new": 0,
     "billing_shipping_mismatch": 0, "country_mismatch": 0, "card_attempts": 1, "velocity_24h": 0,
     "category": "Bikes"},
], index=["P1", "P2", "P3"])
# TODO 6: joblib.dump(fraud_pipe, OUTPUT / "fraud_pipeline.joblib"), then joblib.load it into `reloaded`.
# TODO 7: print reloaded's step names and len(reloaded.feature_names_in_), then the fraud probability of
#         the three incoming payments. Pinecrest sells no bikes: why does P3 still get a score?
print("TODO 6-7: save, reload, score the three incoming payments")

print("\nExpected when done: baseline CV ROC AUC 0.500 and average precision 0.009; logistic regression")
print("0.867 and 0.535; 10 categories, largest weight onehot__category_Gift Cards 0.95; model__C=0.01")
print("gives 0.518, C=1.0 0.535; test ROC AUC 0.843, average precision 0.463, 56 frauds in the top 100;")
print("reloaded steps ['prep', 'model'], 10 columns; probabilities [0.999, 0.0, 0.002].")
