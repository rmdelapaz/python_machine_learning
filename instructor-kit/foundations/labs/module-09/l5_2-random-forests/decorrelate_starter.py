# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.2 B: the variance formula, measured, as max_features moves.

Lesson 5.2's formula says the average of n models has variance rho * sigma² + (1 - rho) * sigma² / n:
better individual trees help, and so do LESS ALIKE trees. max_features pulls those two in opposite
directions. Sweep it on the churn data and measure both forces: the average single tree's AUC, the
average correlation between trees, and the forest that results.
Work through TODO 1 to 4; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the churn split from Lesson 1.1, and a helper that takes a forest apart (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)
X_test_array = X_test.to_numpy()          # the inner trees were fitted without column names


def per_tree(forest):
    """Churn probabilities of every tree in the forest on the test customers, one row per tree."""
    return np.array([tree.predict_proba(X_test_array)[:, 1] for tree in forest.estimators_])


# %% Step 2: one forest, taken apart
# TODO 1: fit RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
#         get probas = per_tree(forest), print probas.shape, and check that averaging the rows yourself
#         gives the same test AUC as forest.predict_proba(X_test)[:, 1].
print("TODO 1: a forest is an average")

# %% Step 3: the two forces, as max_features moves from 1 to all 10
# TODO 2: for max_features in [1, 2, 3, 5, 10]: fit the same forest with max_features=... and
#         oob_score=True; p = per_tree(model). Record the average single-tree test AUC, the average
#         correlation between trees (np.corrcoef(p)[np.triu_indices(len(p), k=1)].mean()), the OOB AUC
#         and the test AUC. Print the table. Before you run it, predict which max_features wins: ______
print("TODO 2: the max_features sweep")

# %% Step 4: how many trees?
# TODO 3: fit max_features=3 (the default "sqrt" of 10 features), p3 = per_tree(...), and print the test
#         AUC of the average of the first k trees for k in [1, 5, 10, 25, 50, 100, 200].
p3 = None
if p3 is None:
    print("TODO 3: the first k trees")

# %% Step 5: where the panel disagrees
# TODO 4: spread = p3.std(axis=0) is how much the trees disagree about each customer. Among the 50
#         customers with the highest average vote, count the real churners and compare their average
#         spread with the whole test set's. Print the five most divided of the top 50.
print("TODO 4: the panel's disagreement")

print("\nExpected when done: probas (200, 1500), forest AUC 0.865 both ways; single-tree AUC rises")
print("0.709 -> 0.747 and tree correlation 0.357 -> 0.535 as max_features goes 1 -> 10, while the forest")
print("peaks in the middle (test AUC 0.854 / 0.864 / 0.865 / 0.861 / 0.856); first-k AUC 0.755, 0.843,")
print("0.850, 0.861, 0.864, 0.866, 0.865; top 50 holds 41 churners with spread 0.284 vs 0.178 overall.")
