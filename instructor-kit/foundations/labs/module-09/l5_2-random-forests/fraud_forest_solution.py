# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.2 A (SOLUTION): a random forest for the payments team's review queue.

Session 1's fraud lab ranked payments with logistic regression; Session 8 gave the reviewers an
eight-rule tree. Now the team wants the best possible list of 100 payments to review each batch.
Build a forest, tune it out-of-bag (no cross-validation loop needed), test it once against a single
tree and logistic regression, and check which columns it really relies on.
"""
# %% Step 1: the same stratified 70/30 split as the earlier fraud labs
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
for leaf in [1, 3, 5, 10]:
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, oob_score=True,
                                    random_state=42, n_jobs=-1).fit(X_train, y_train)
    oob = forest.oob_decision_function_[:, 1]
    print(f"min_samples_leaf={leaf:2d}  OOB average precision {average_precision_score(y_train, oob):.3f}"
          f"   OOB ROC AUC {roc_auc_score(y_train, oob):.3f}")

# %% Step 3: one test, three models
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "one unlimited tree": DecisionTreeClassifier(random_state=42),
    "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
}
scores = {}
for name, model in models.items():
    scores[name] = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print(f"{name:20s} test AUC {roc_auc_score(y_test, scores[name]):.3f}   AP {average_precision_score(y_test, scores[name]):.3f}"
          f"   frauds in top 100: {top100(scores[name])}")
forest = models["random forest"]

# %% Step 4: why the single tree's list is poor
print("\ndistinct scores: one tree", len(np.unique(scores["one unlimited tree"])),
      "| forest", len(np.unique(scores["random forest"])))

# %% Step 5: impurity importance (MDI) against permutation importance on the test set
from sklearn.inspection import permutation_importance

perm = permutation_importance(forest, X_test, y_test, scoring="average_precision", n_repeats=3,
                              random_state=42, n_jobs=-1)
importance = pd.DataFrame({"impurity (MDI)": forest.feature_importances_,
                           "permutation (AP drop)": perm.importances_mean}, index=features)
print(importance.sort_values("impurity (MDI)", ascending=False).round(3))
print("\ndistinct values: amount", tx["amount"].nunique(), "| account_age_days", tx["account_age_days"].nunique(),
      "| device_new", tx["device_new"].nunique())
