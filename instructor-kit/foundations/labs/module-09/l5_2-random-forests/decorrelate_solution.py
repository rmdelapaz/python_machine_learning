# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.2 B (SOLUTION): the variance formula, measured, as max_features moves.

Lesson 5.2's formula says the average of n models has variance rho * sigma² + (1 - rho) * sigma² / n:
better individual trees help, and so do LESS ALIKE trees. max_features pulls those two in opposite
directions. Sweep it on the churn data and measure both forces: the average single tree's AUC, the
average correlation between trees, and the forest that results.
"""
# %% Step 1: the churn split from Lesson 1.1 (ten numeric features)
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
forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(X_train, y_train)
probas = per_tree(forest)
print("per-tree predictions:", probas.shape)
print(f"forest test AUC {roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1]):.3f}; "
      f"same as averaging the trees yourself: {roc_auc_score(y_test, probas.mean(axis=0)):.3f}")

# %% Step 3: the two forces, as max_features moves from 1 to all 10
rows = []
for max_features in [1, 2, 3, 5, 10]:
    model = RandomForestClassifier(n_estimators=200, max_features=max_features, min_samples_leaf=5,
                                   oob_score=True, random_state=42, n_jobs=-1).fit(X_train, y_train)
    p = per_tree(model)
    rows.append({
        "max_features": max_features,
        "one_tree_auc": np.mean([roc_auc_score(y_test, tree_p) for tree_p in p]),
        "tree_correlation": np.corrcoef(p)[np.triu_indices(len(p), k=1)].mean(),
        "oob_auc": roc_auc_score(y_train, model.oob_decision_function_[:, 1]),
        "test_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %% Step 4: how many trees? Average the first k trees of the max_features=3 forest
three = RandomForestClassifier(n_estimators=200, max_features=3, min_samples_leaf=5, random_state=42, n_jobs=-1)
p3 = per_tree(three.fit(X_train, y_train))
print("\ntest AUC with the first k trees:",
      {k: round(roc_auc_score(y_test, p3[:k].mean(axis=0)), 3) for k in [1, 5, 10, 25, 50, 100, 200]})

# %% Step 5: where the panel disagrees: the spread of the trees' votes per customer
spread = p3.std(axis=0)
panel = pd.DataFrame({"p_churn": p3.mean(axis=0), "spread": spread, "churned": y_test.to_numpy()})
top50 = panel.nlargest(50, "p_churn")
print(f"\ntop 50: {top50['churned'].sum()} real churners; average spread of the trees' votes {top50['spread'].mean():.3f}")
print(f"all test customers: average spread {panel['spread'].mean():.3f}")
print("the five most divided of the top 50:")
print(top50.nlargest(5, "spread").round(3).to_string())
