# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.3 A: cross-validating a fraud model that will meet next month's payments.

The fraud model will be retrained at the end of every month and used on the month after. Lesson 1.3
cross-validated churn (shuffled folds) and weekly demand (time-ordered folds). Here both kinds meet one
question: on January to May, which folds tell the truth about June? The June payments stay locked
until the very end. Work through TODO 1 to 5; it runs as it is.
"""
# %% Step 1: lock June away (done for you)
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
# TODO 1: for each splitter, count the frauds in every validation fold:
#         [int(y_train.iloc[val].sum()) for _, val in splitter.split(X_train, y_train)]
#         The rows are in time order. Predict first: which splitter gives equal counts, and why does the
#         other one's count fall?
print("TODO 1: frauds per validation fold, KFold vs StratifiedKFold")

# %% Step 3: three candidates on the same shuffled, stratified folds
from sklearn.model_selection import cross_validate, cross_val_score
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
# TODO 2: cross_validate each candidate with cv=strat, scoring="roc_auc", return_train_score=True.
#         Keep res["test_score"] in a dict fold_auc; print mean +/- std of the validation AUC and the
#         mean train AUC.
fold_auc = {}
if not fold_auc:
    print("TODO 2: validation and train AUC for the three candidates")

# %% Step 4: is the lead real?
# TODO 3: diff = fold_auc["tree (depth 5)"] - fold_auc["logistic regression"]: in how many of the 5 folds
#         is the tree ahead, and by how much on average?
print("TODO 3: paired comparison, fold by fold")

# %% Step 5: time-ordered folds: train on the months before, validate on the next month
# TODO 4: build time_folds = [(train positions, validation positions), ...] for validation months
#         2, 3, 4 and 5, where train = every earlier month. Hint:
#             train_month = month[month < 6].to_numpy()
#             (np.flatnonzero(train_month < m), np.flatnonzero(train_month == m))
#         Then cross_val_score(candidates["logistic regression"], X_train, y_train, cv=time_folds,
#         scoring="roc_auc"). Print the four AUCs and their mean.
print("TODO 4: next-month AUC for February to May")

# %% Step 6: open June, once
# TODO 5: fit the logistic regression pipeline on all of X_train, compute roc_auc_score on June
#         (from sklearn.metrics import roc_auc_score), and print it next to both CV estimates.
print("TODO 5: June AUC, once")

print("\nExpected when done: frauds per fold KFold [83, 84, 71, 75, 61] vs stratified [75, 75, 75, 74, 75];")
print("validation AUC baseline 0.500, logistic regression 0.854 +/- 0.023, tree 0.776 +/- 0.020 (ahead in")
print("0 of 5 folds); next-month AUC [0.864, 0.844, 0.86, 0.881], mean 0.862; June AUC 0.874.")
