# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.1 A: a fraud rule book for Pinecrest's payment reviewers.

Lesson 5.1 built a churn rule book for the care team. The payments team wants the same for card
fraud: at most eight rules a reviewer can follow, each with its fraud rate, honest about how well
they work on payments the tree never saw. Fraud is rare (0.88%), so average precision (Lesson 1.4)
is the yardstick here, with ROC AUC beside it.
Work through TODO 1 to 6; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: features, target and the stratified 70/30 split from Session 1's fraud lab (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, average_precision_score

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(
    tx[features], tx["is_fraud"], test_size=0.3, stratify=tx["is_fraud"], random_state=42
)
print(X_train.shape, X_test.shape, f"frauds: train {y_train.sum()}, test {y_test.sum()}")

# %% Step 2: a depth-2 tree, read as rules
# TODO 1: fit DecisionTreeClassifier(max_depth=2, random_state=42) as `small` and print
#         export_text(small, feature_names=features, show_weights=True).
#         Which leaf would you hand a reviewer first, and how many of its training payments were fraud?
small = None
if small is None:
    print("TODO 1: a depth-2 tree as rules")


# %% Step 3: the root split's Gini decrease, by hand, checked against the fitted tree
def gini(counts):
    p = np.asarray(counts) / np.sum(counts)
    return 1 - np.sum(p ** 2)


# TODO 2: root = np.bincount(y_train); left = the counts for velocity_24h <= 3.5 (minlength=2);
#         right = root - left. Print the three Gini values and the weighted decrease, then compare with
#         small.tree_.impurity for node 0 and its two children (tree_.children_left[0], children_right[0]).
#         Why is the root's Gini so small?
print("TODO 2: Gini by hand")

# %% Step 4: an unlimited tree, judged honestly
# TODO 3: fit DecisionTreeClassifier(random_state=42). Print its number of leaves, its TRAINING AUC,
#         and its test AUC and test average precision.
print("TODO 3: the unlimited tree")

# %% Step 5: choose the rule book's size by cross-validation (training rows only)
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO 4: for max_leaf_nodes in [4, 8, 12, 16] (always with min_samples_leaf=50, random_state=42),
#         print the CV average precision (scoring="average_precision") and the CV ROC AUC.
print("TODO 4: CV for four rule-book sizes")

# %% Step 6: the eight-rule book, tested once, and its segments ranked by fraud rate
# TODO 5: the team allows eight rules: fit max_leaf_nodes=8, min_samples_leaf=50 as `rules`, print its
#         test AUC and AP and its export_text. Then put rules.apply(X_test), the predicted probability
#         and y_test in a DataFrame and group by leaf: payments, predicted, actual fraud rate, frauds,
#         sorted by predicted (the lesson's segment table, for payments).
rules = None
if rules is None:
    print("TODO 5: the rule book and its segment table")

# %% Step 7: which features did the rule book use?
# TODO 6: print rules.feature_importances_ as a sorted Series. `items` gets 0.000: compare the mean
#         items of fraud and legitimate payments in tx. What is the right sentence about items?
print("TODO 6: importances, and what a zero means")

print("\nExpected when done: depth-2 leaf 'velocity_24h > 3.5 and account_age_days <= 366' holds 40")
print("frauds of 42; Gini root 0.0174, left 0.0151, right 0.3457, decrease 0.0018 (matches tree_); the")
print("unlimited tree has 433 leaves, training AUC 1.000, test AUC 0.684, AP 0.129; CV AP 0.178 / 0.259 /")
print("0.287 / 0.295 for 4 / 8 / 12 / 16 leaves; the 8-rule book scores test AUC 0.756, AP 0.253, and its")
print("riskiest segment (velocity > 3.5) caught 15 of 19 test payments (78.9%); items importance 0.000")
print("although frauds average 2.39 items against 1.89.")
