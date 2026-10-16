# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 6.2 B (solution): see the winner's curse with your own eyes.

Lesson 6.2 said best_score_ is the best of many noisy scores, so it is optimistic. Here is an
experiment that makes the optimism visible: shuffle the churn labels so that NOTHING can be learned,
tune a decision tree anyway, and compare three numbers: the search's best_score_, a nested
cross-validation estimate, and the score on the untouched test customers. Then repeat with the
real labels.
"""
# %% Step 1: the churn split and a target with no signal in it
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(customers[features], customers["churned"], test_size=0.25,
                                                    stratify=customers["churned"], random_state=42)
rng = np.random.default_rng(42)
y_noise_train = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
y_noise_test = pd.Series(rng.permutation(y_test.to_numpy()), index=y_test.index)
print(f"shuffled labels keep the churn rate ({y_noise_train.mean():.3f}) but break every link to the features")

# %% Step 2 (TODO 1): tune a tree on the shuffled labels
space = {"max_depth": randint(1, 15), "min_samples_leaf": randint(1, 100), "max_features": uniform(0.1, 0.9)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def make_search(n_iter):
    return RandomizedSearchCV(DecisionTreeClassifier(random_state=42), space, n_iter=n_iter, cv=cv,
                              scoring="roc_auc", random_state=42)

for n_iter in [1, 10, 40]:
    s = make_search(n_iter).fit(X_train, y_noise_train)
    print(f"{n_iter:3d} settings tried: best_score_ {s.best_score_:.3f}")

# %% Step 3 (TODO 2): the honest numbers for the 40-setting search
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
nested = cross_val_score(make_search(40), X_train, y_noise_train, cv=outer, scoring="roc_auc")
search = make_search(40).fit(X_train, y_noise_train)
test_auc = roc_auc_score(y_noise_test, search.predict_proba(X_test)[:, 1])
print(f"\nshuffled labels: best_score_ {search.best_score_:.3f} | nested {nested.mean():.3f} | test {test_auc:.3f}")

# %% Step 4 (TODO 3): the same three numbers with the real labels
search_real = make_search(40).fit(X_train, y_train)
nested_real = cross_val_score(make_search(40), X_train, y_train, cv=outer, scoring="roc_auc")
test_real = roc_auc_score(y_test, search_real.predict_proba(X_test)[:, 1])
print(f"real labels:     best_score_ {search_real.best_score_:.3f} | nested {nested_real.mean():.3f} | test {test_real:.3f}")
print("\nThe number to promise is the nested estimate or the one-time test score, never best_score_.")
