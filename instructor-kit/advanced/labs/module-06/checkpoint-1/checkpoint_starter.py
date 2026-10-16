# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Checkpoint 1 (Modules 7 to 9). 30 minutes, open notebook: your lesson notebooks and the lesson pages
are allowed; another person is not.

Pinecrest's retention lead asks three questions about the same 6,000 customers. Answer each with code
in the cell under it, then write Part D. Everything runs as shipped; fill in the TODOs in order.
  A. Which features does our churn model rely on? (Module 7)                         about 6 minutes
  B. What customer groups are there, and how sure are we about each customer? (8)   about 8 minutes
  C. Which single direction separates churners, and is it the direction of most spread? (9)  8 minutes
  D. Three sentences for the retention lead.                                          about 5 minutes
"""
# %% Setup (given)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pd.set_option("display.width", 120)
customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X = customers[features]
y = customers["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
print("model fitted; test rows:", len(X_test))

# %% Part A (Module 7)
from sklearn.inspection import permutation_importance
# TODO A: permutation importance of `model` on the TEST rows, scoring="roc_auc", n_repeats=5, random_state=42.
#         Print the three columns whose shuffling costs the most AUC.
print("TODO A: top three columns by permutation importance")

# %% Part B (Module 8)
from sklearn.mixture import GaussianMixture
behavior = ["orders_12m", "avg_order_value", "discount_share", "return_rate"]
# TODO B: scale the four behavior columns for all customers (churn is NOT a feature), fit
#         GaussianMixture(n_components=5, n_init=3, random_state=42), and report per component: size,
#         mean discount_share, churn rate. How many customers have a top probability below 0.8?
print("TODO B: five soft segments, their churn, and the borderline count")

# %% Part C (Module 9)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
# TODO C: on the scaled TRAINING rows (scaler fitted on X_train only): how many PCA components keep 90%?
#         Fit LinearDiscriminantAnalysis on (scaled X_train, y_train): which feature has the largest |LD1| weight?
#         Which feature has the largest |PC1| loading? Then the test ROC AUC of PC1 alone (use max(auc, 1 - auc),
#         a PCA sign is arbitrary) and of LD1 alone.
print("TODO C: PCA count for 90%, LD1 vs PC1, and their AUCs")

# %% Part D: write three sentences here, as comments, for the retention lead
# D1 (what the model relies on, and what that does NOT prove):
# D2 (which group to act on first, with one number):
# D3 (why a churn map should use LDA rather than PCA, with one number):
