# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.3 A (SOLUTION): cross-validating a fraud model that will meet next month's payments.

The fraud model will be retrained at the end of every month and used on the month after. Lesson 1.3
cross-validated churn (shuffled folds) and weekly demand (time-ordered folds). Here both kinds meet one
question: on January to May, which folds tell the truth about June? The June payments stay locked
until the very end.
"""
# %% Step 1: lock June away; everything else happens on January to May
import numpy as np
import pandas as pd

tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
month = tx["timestamp"].dt.month
train, june = tx[month < 6], tx[month == 6]
X_train, y_train = train[features], train["is_fraud"]
print(f"train (Jan-May): {len(train):,} payments, {y_train.sum()} frauds; June locked: {len(june):,}")

# %% Step 2: what is inside each validation fold?
from sklearn.model_selection import KFold, StratifiedKFold

plain = KFold(n_splits=5)                                   # no shuffle: consecutive blocks of rows
strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, splitter in [("KFold, no shuffle", plain), ("StratifiedKFold", strat)]:
    counts = [int(y_train.iloc[val].sum()) for _, val in splitter.split(X_train, y_train)]
    print(f"{name:18s} frauds per validation fold {counts}")

# %% Step 3: three candidates on the same shuffled, stratified folds
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

candidates = {
    "baseline": DummyClassifier(strategy="prior"),
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=42),
}
fold_auc = {}
print()
for name, model in candidates.items():
    res = cross_validate(model, X_train, y_train, cv=strat, scoring="roc_auc", return_train_score=True)
    fold_auc[name] = res["test_score"]
    print(f"{name:20s} validation AUC {res['test_score'].mean():.3f} +/- {res['test_score'].std():.3f}"
          f"   train AUC {res['train_score'].mean():.3f}")

# %% Step 4: is the lead real? Compare fold by fold
diff = fold_auc["tree (depth 5)"] - fold_auc["logistic regression"]
print(f"\ntree ahead of logistic regression in {(diff > 0).sum()} of {len(diff)} folds, "
      f"mean difference {diff.mean():+.3f}")

# %% Step 5: time-ordered folds: train on the months before, validate on the next month
time_folds = []
train_month = month[month < 6].to_numpy()
for m in [2, 3, 4, 5]:
    time_folds.append((np.flatnonzero(train_month < m), np.flatnonzero(train_month == m)))
from sklearn.model_selection import cross_val_score

logreg = candidates["logistic regression"]
by_month = cross_val_score(logreg, X_train, y_train, cv=time_folds, scoring="roc_auc")
print("\nlogistic regression, next-month AUC (Feb, Mar, Apr, May):", by_month.round(3).tolist(),
      f"mean {by_month.mean():.3f}")

# %% Step 6: open June, once
final = logreg.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score

june_auc = roc_auc_score(june["is_fraud"], final.predict_proba(june[features])[:, 1])
print(f"June AUC {june_auc:.3f}   (shuffled CV said {fold_auc['logistic regression'].mean():.3f}, "
      f"time-ordered CV said {by_month.mean():.3f})")

# %% Stretch: out-of-fold scores give an honest review queue on the training months
from sklearn.model_selection import cross_val_predict

oof = cross_val_predict(logreg, X_train, y_train, cv=strat, method="predict_proba")[:, 1]
top = pd.Series(oof).nlargest(400).index
print(f"\nout-of-fold top 400 of {len(train):,}: {int(y_train.iloc[top].sum())} frauds "
      f"(a random 400 would hold {400 * y_train.mean():.1f})")
