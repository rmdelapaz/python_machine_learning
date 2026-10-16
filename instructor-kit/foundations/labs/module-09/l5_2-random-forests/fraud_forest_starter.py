# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.2 A: a random forest for the payments team's review queue.

Session 1's fraud lab ranked payments with logistic regression; Session 8 gave the reviewers an
eight-rule tree. Now the team wants the best possible list of 100 payments to review each batch.
Build a forest, tune it out-of-bag (no cross-validation loop needed), test it once against a single
tree and logistic regression, and check which columns it really relies on.
Work through TODO 1 to 4; run the file after each one. It runs as it is: unfinished steps print a reminder.
The finished file takes about 25 seconds (the forests and the permutation importance).
"""
# %% Step 1: the same stratified 70/30 split as the earlier fraud labs (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(
    tx[features], tx["is_fraud"], test_size=0.3, stratify=tx["is_fraud"], random_state=42
)


def top100(scores):
    """Real frauds among the 100 highest-scored test payments."""
    return int(y_test.to_numpy()[np.argsort(-scores, kind="stable")[:100]].sum())


print(X_train.shape, f"frauds in training: {y_train.sum()}; a random 100 test payments hold {100 * y_test.mean():.1f}")

# %% Step 2: tune the leaf size out-of-bag: one fit per setting, training rows only
# TODO 1: for min_samples_leaf in [1, 3, 5, 10], fit RandomForestClassifier(n_estimators=300,
#         min_samples_leaf=leaf, oob_score=True, random_state=42, n_jobs=-1) and print the OOB average
#         precision and OOB ROC AUC computed from forest.oob_decision_function_[:, 1] against y_train.
#         Which leaf size would you pick, judging by average precision?
print("TODO 1: OOB tuning of the leaf size")

# %% Step 3: one test, three models
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# TODO 2: fit make_pipeline(StandardScaler(), LogisticRegression()), DecisionTreeClassifier(random_state=42)
#         and RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1).
#         Keep each model's test probabilities in a dict `scores`; print test AUC, test AP and top100(...).
#         Keep the fitted forest as `forest`.
scores = {}
forest = None
if forest is None:
    print("TODO 2: one test, three models")

# %% Step 4: why the single tree's list is poor
# TODO 3: print how many DISTINCT scores the single tree and the forest give the 15,000 test payments
#         (len(np.unique(...))). What does that mean for a top-100 list?
print("TODO 3: distinct scores")

# %% Step 5: impurity importance (MDI) against permutation importance on the test set
from sklearn.inspection import permutation_importance

# TODO 4: perm = permutation_importance(forest, X_test, y_test, scoring="average_precision", n_repeats=3,
#         random_state=42, n_jobs=-1). Put forest.feature_importances_ and perm.importances_mean side by
#         side in a DataFrame indexed by features. Which column does MDI underrate, which two does it
#         overrate, and how many distinct values does each have (tx[col].nunique())?
print("TODO 4: MDI against permutation importance")

print("\nExpected when done: OOB AP 0.489 / 0.511 / 0.513 / 0.503 for leaf 1 / 3 / 5 / 10 (AUC 0.829 / 0.848 /")
print("0.863 / 0.861); test: logistic regression AUC 0.848, AP 0.444, 53 frauds in the top 100; one tree")
print("AUC 0.684, AP 0.129, 41; forest AUC 0.842, AP 0.479, 57; distinct scores 2 vs 10,197; device_new")
print("MDI 0.081 but permutation 0.122 (2 distinct values), amount MDI 0.176 but permutation 0.058 (27,674).")
