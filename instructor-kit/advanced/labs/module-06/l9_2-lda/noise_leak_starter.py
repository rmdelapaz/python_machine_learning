# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.2 B: the leak, on real customers.

Lesson 9.2 showed LDA "separating" pure noise when it was fitted before cross-validation. Here the
data are real: 400 customers, their ten churn features, plus 290 columns of random noise, and churn
labels that have been SHUFFLED, so no honest model can beat chance. Then the same check with the true
labels. Work through TODO 1 to 3.
"""
# %% Step 1: 400 customers, 10 real + 290 noise columns (done for you)
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

customers = pd.read_csv(DATA / "customers_churn.csv").sample(400, random_state=42)
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
rng = np.random.default_rng(42)
X = np.hstack([StandardScaler().fit_transform(customers[features]), rng.normal(size=(400, 290))])
y_true = customers["churned"].to_numpy()
y_shuffled = rng.permutation(y_true)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(X.shape, f"churn rate {y_true.mean():.3f}")

# %% Step 2: shuffled labels
# TODO 1: write compare(y, name) that prints two cross-validated ROC AUCs (cv=cv, scoring="roc_auc"):
#   leaky : LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y) on ALL rows, then
#           cross_val_score(KNeighborsClassifier(n_neighbors=15), reduced, y, ...)
#   honest: cross_val_score(make_pipeline(LinearDiscriminantAnalysis(n_components=1),
#                                         KNeighborsClassifier(n_neighbors=15)), X, y, ...)
#   Call it with y_shuffled. Predict first: what should an honest score be on shuffled labels?
print("TODO 1: leaky vs honest on shuffled labels")

# %% Step 3: true labels
# TODO 2: call compare(y_true, "true labels"). Is the honest score impressive?
print("TODO 2: leaky vs honest on the true labels")

# %% Step 4: the real columns alone
# TODO 3: the honest pipeline on X[:, :10] with y_true. What did 290 noise columns cost?
print("TODO 3: honest pipeline on the 10 real columns")

print("\nExpected when done: shuffled labels: leaky AUC 0.998, honest 0.481 (chance); true labels: leaky 1.000,")
print("honest 0.541; the 10 real columns alone: honest AUC 0.776.")
