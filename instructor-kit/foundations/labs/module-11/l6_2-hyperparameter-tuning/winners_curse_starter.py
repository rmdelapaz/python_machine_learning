# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 6.2 B: see the winner's curse with your own eyes.

Lesson 6.2 said best_score_ is the best of many noisy scores, so it is optimistic. Here is an
experiment that makes the optimism visible: shuffle the churn labels so that NOTHING can be learned,
tune a decision tree anyway, and compare three numbers: the search's best_score_, a nested
cross-validation estimate, and the score on the untouched test customers. Then repeat with the
real labels. Work through TODO 1 to 3; the finished file takes about 25 seconds.
"""
# %% Step 1: the churn split and a target with no signal in it (done for you)
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

space = {"max_depth": randint(1, 15), "min_samples_leaf": randint(1, 100), "max_features": uniform(0.1, 0.9)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def make_search(n_iter):
    return RandomizedSearchCV(DecisionTreeClassifier(random_state=42), space, n_iter=n_iter, cv=cv,
                              scoring="roc_auc", random_state=42)

# %% Step 2: tune a tree on the shuffled labels
# TODO 1: for n_iter in 1, 10 and 40, fit make_search(n_iter) on X_train, y_noise_train and print
#         best_score_. Predict first: what should an honest AUC be when the labels are pure noise?
print("TODO 1: best_score_ after 1, 10 and 40 settings on shuffled labels")

# %% Step 3: the honest numbers for the 40-setting search
# TODO 2: nested = cross_val_score(make_search(40), X_train, y_noise_train,
#         cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc"); then fit one
#         make_search(40) and score it on X_test against y_noise_test. Print best_score_, nested mean, test.
print("TODO 2: best_score_, nested and test AUC on shuffled labels")

# %% Step 4: the same three numbers with the real labels
# TODO 3: repeat TODO 2 with y_train and y_test. How big is the gap now, and why is it smaller?
print("TODO 3: the three numbers with the real labels")

print("\nExpected when done: best_score_ 0.504 / 0.514 / 0.537 after 1 / 10 / 40 settings on noise;")
print("shuffled labels: best_score_ 0.537, nested 0.501, test 0.524; real labels: best_score_ 0.848,")
print("nested 0.840, test 0.859.")
