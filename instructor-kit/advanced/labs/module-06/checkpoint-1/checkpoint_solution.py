# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Checkpoint 1 (Modules 7 to 9), MODEL ANSWER. 30 minutes, open notebook.

Pinecrest's retention lead asks three questions about the same 6,000 customers:
  A. Which features does our churn model rely on? (Module 7)
  B. What customer groups are there, and how sure are we about each customer? (Module 8)
  C. Which single direction in the data separates churners, and is it the direction of most spread? (Module 9)
  D. Three sentences for the retention lead.
"""
# %% Setup (given in the starter)
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

# %% Part A: permutation importance on the TEST rows (Module 7)
from sklearn.inspection import permutation_importance

imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False)
print("A. drop in test AUC when each column is shuffled (top 3):")
print(importance.head(3).round(3).to_string())

# %% Part B: soft segments on the four behavior columns, churn NOT used (Module 8)
from sklearn.mixture import GaussianMixture

behavior = ["orders_12m", "avg_order_value", "discount_share", "return_rate"]
Z = StandardScaler().fit_transform(customers[behavior])
gm = GaussianMixture(n_components=5, n_init=3, random_state=42).fit(Z)
proba = gm.predict_proba(Z)
customers["component"] = proba.argmax(axis=1)
customers["confidence"] = proba.max(axis=1)
profile = customers.groupby("component").agg(customers=("customer_id", "size"),
                                             discount=("discount_share", "mean"), churn=("churned", "mean"))
print("\nB. components (churn used only to profile):")
print(profile.round(3).to_string())
print(f"customers below 0.8 confidence: {(customers['confidence'] < 0.8).sum()}")

# %% Part C: PCA vs LDA on the scaled training rows (Module 9)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

scaler = StandardScaler().set_output(transform="pandas").fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
pca = PCA().fit(Xs_train)
print(f"\nC. components for 90% of the variance: {PCA(n_components=0.90).fit(Xs_train).n_components_}")
lda = LinearDiscriminantAnalysis().fit(Xs_train, y_train)
weights = pd.Series(lda.scalings_[:, 0], index=features)
print("largest LD1 weight:", weights.abs().idxmax(), round(weights[weights.abs().idxmax()], 2))
print("largest PC1 loading:", pd.Series(np.abs(pca.components_[0]), index=features).idxmax())

from sklearn.metrics import roc_auc_score
pc1_auc = roc_auc_score(y_test, pca.transform(Xs_test)[:, 0])
ld1_auc = roc_auc_score(y_test, lda.transform(Xs_test)[:, 0])
print(f"test AUC of one number: PC1 {max(pc1_auc, 1 - pc1_auc):.3f} (sign-free)   LD1 {ld1_auc:.3f}")
