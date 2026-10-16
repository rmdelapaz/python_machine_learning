# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 2.2 A (SOLUTION): a deployable fraud pipeline.

Lesson 2.2 sealed the churn workflow into one Pipeline. The payments team wants the same for card
fraud: one object that takes raw payments (text category and all), is compared with a baseline by
cross-validation on the training rows only, is tested once, and is saved to a file that scores new
payments, including a category the model has never seen.
"""
# %% Step 1: raw columns and a stratified 70/30 split (the same rows as Session 1's fraud lab)
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

preprocess = ColumnTransformer([
    ("scale", StandardScaler(), num_cols),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
])
fraud_pipe = Pipeline([("prep", preprocess), ("model", LogisticRegression())])
print(list(fraud_pipe.named_steps))

# %% Step 3: compare with a baseline pipeline by cross-validation (training rows only)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

baseline_pipe = Pipeline([("prep", preprocess), ("model", DummyClassifier(strategy="prior"))])
for name, pipe in [("baseline", baseline_pipe), ("logistic regression", fraud_pipe)]:
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
    print(f"{name:20s} CV ROC AUC {res['test_roc_auc'].mean():.3f}   "
          f"CV average precision {res['test_average_precision'].mean():.3f}")

# %% Step 4: reach inside the fitted pipeline
fraud_pipe.fit(X_train, y_train)
onehot = fraud_pipe.named_steps["prep"].named_transformers_["onehot"]
print("categories learned:", len(onehot.categories_[0]), list(onehot.categories_[0][:3]), "...")
names = fraud_pipe[:-1].get_feature_names_out()
weights = pd.Series(fraud_pipe[-1].coef_[0], index=names)
print("largest weights:", weights.sort_values().round(2).tail(3).to_dict())

# %% Step 5: turn a knob through the pipeline (step__parameter)
from sklearn.model_selection import cross_val_score

for c_value in [0.01, 1.0]:
    fraud_pipe.set_params(model__C=c_value)
    ap = cross_val_score(fraud_pipe, X_train, y_train, cv=cv, scoring="average_precision").mean()
    print(f"model__C={c_value:<5} CV average precision {ap:.3f}")
fraud_pipe.set_params(model__C=1.0)

# %% Step 6: open the test set once
from sklearn.metrics import average_precision_score, roc_auc_score

fraud_pipe.fit(X_train, y_train)
p_test = fraud_pipe.predict_proba(X_test)[:, 1]
top100 = pd.Series(p_test, index=X_test.index).nlargest(100).index
print(f"test ROC AUC {roc_auc_score(y_test, p_test):.3f}   test average precision "
      f"{average_precision_score(y_test, p_test):.3f}   frauds in the top 100: {y_test.loc[top100].sum()}")

# %% Step 7: save, reload, and score raw payments (one category the model has never seen)
import joblib

path = OUTPUT / "fraud_pipeline.joblib"
joblib.dump(fraud_pipe, path)
reloaded = joblib.load(path)
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
print("reloaded steps:", [name for name, _ in reloaded.steps], "| columns expected:", len(reloaded.feature_names_in_))
print("fraud probability:", reloaded.predict_proba(incoming)[:, 1].round(3).tolist())
