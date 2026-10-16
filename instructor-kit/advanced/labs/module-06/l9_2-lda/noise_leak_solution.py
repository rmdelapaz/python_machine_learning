# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.2 B (SOLUTION): the leak, on real customers.

Lesson 9.2 showed LDA "separating" pure noise when it was fitted before cross-validation. Here the
data are real: 400 customers, their ten churn features, plus 290 columns of random noise, and churn
labels that have been SHUFFLED, so no honest model can beat chance. Then the same check with the true
labels, where there is real signal to find.
"""
# %% Step 1: 400 customers, 10 real + 290 noise columns
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
print(X.shape, f"churn rate {y_true.mean():.3f}; chance accuracy of 'always stays' {1 - y_true.mean():.3f}")

# %% Step 2: shuffled labels, LDA fitted on ALL rows first (the leak)
def compare(y, name):
    reduced = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)
    leaky = cross_val_score(KNeighborsClassifier(n_neighbors=15), reduced, y, cv=cv, scoring="roc_auc").mean()
    honest_pipe = make_pipeline(LinearDiscriminantAnalysis(n_components=1), KNeighborsClassifier(n_neighbors=15))
    honest = cross_val_score(honest_pipe, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"{name:16s} AUC  LDA outside CV {leaky:.3f}   LDA inside the pipeline {honest:.3f}")

compare(y_shuffled, "shuffled labels")

# %% Step 3: the true labels
compare(y_true, "true labels")

# %% Step 4: the same honest check on the 10 real columns only
real_only = make_pipeline(LinearDiscriminantAnalysis(n_components=1), KNeighborsClassifier(n_neighbors=15))
auc = cross_val_score(real_only, X[:, :10], y_true, cv=cv, scoring="roc_auc").mean()
print(f"true labels, 10 real columns only, inside the pipeline: AUC {auc:.3f}")
