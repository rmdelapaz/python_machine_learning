# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.1 A (SOLUTION): a fraud rule book for Pinecrest's payment reviewers.

Lesson 5.1 built a churn rule book for the care team. The payments team wants the same for card
fraud: at most eight rules a reviewer can follow, each with its fraud rate, honest about how well
they work on payments the tree never saw. Fraud is rare (0.88%), so average precision (Lesson 1.4)
is the yardstick here, with ROC AUC beside it.
"""
# %% Step 1: features, target and the stratified 70/30 split from Session 1's fraud lab
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
small = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_train, y_train)
print(export_text(small, feature_names=features, show_weights=True))

# %% Step 3: the root split's Gini decrease, by hand, checked against the fitted tree


def gini(counts):
    p = np.asarray(counts) / np.sum(counts)
    return 1 - np.sum(p ** 2)


root = np.bincount(y_train)
left = np.bincount(y_train[X_train["velocity_24h"] <= 3.5], minlength=2)
right = root - left
weighted = (left.sum() * gini(left) + right.sum() * gini(right)) / root.sum()
print("root", root, " left", left, " right", right)
print(f"Gini root {gini(root):.4f}  left {gini(left):.4f}  right {gini(right):.4f}  decrease {gini(root) - weighted:.4f}")
t = small.tree_
print("scikit-learn's values:", t.impurity[[0, t.children_left[0], t.children_right[0]]].round(4))

# %% Step 4: an unlimited tree, judged honestly
full = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
p_full = full.predict_proba(X_test)[:, 1]
print(f"\nunlimited tree: {full.get_n_leaves()} leaves, training AUC {roc_auc_score(y_train, full.predict_proba(X_train)[:, 1]):.3f}, "
      f"test AUC {roc_auc_score(y_test, p_full):.3f}, test AP {average_precision_score(y_test, p_full):.3f}")

# %% Step 5: choose the rule book's size by cross-validation (training rows only)
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for leaves in [4, 8, 12, 16]:
    model = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=50, random_state=42)
    ap = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision").mean()
    auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    print(f"max_leaf_nodes={leaves:2d}  CV average precision {ap:.3f}   CV ROC AUC {auc:.3f}")

# %% Step 6: the eight-rule book, tested once, and its segments ranked by fraud rate
rules = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=50, random_state=42).fit(X_train, y_train)
p_rules = rules.predict_proba(X_test)[:, 1]
print(f"\nrule book: test AUC {roc_auc_score(y_test, p_rules):.3f}, test AP {average_precision_score(y_test, p_rules):.3f}")
print(export_text(rules, feature_names=features, show_weights=True))

segments = pd.DataFrame({"leaf": rules.apply(X_test), "predicted": p_rules, "fraud": y_test.to_numpy()})
table = (segments.groupby("leaf")
         .agg(payments=("fraud", "size"), predicted=("predicted", "first"), actual=("fraud", "mean"),
              frauds=("fraud", "sum"))
         .sort_values("predicted", ascending=False))
print(table.round(3))

# %% Step 7: which features did the rule book use?
importance = pd.Series(rules.feature_importances_, index=features).sort_values(ascending=False)
print("\n" + importance.round(3).to_string())
print("\nmean items: fraud", round(tx.loc[tx["is_fraud"] == 1, "items"].mean(), 2),
      " legit", round(tx.loc[tx["is_fraud"] == 0, "items"].mean(), 2))
