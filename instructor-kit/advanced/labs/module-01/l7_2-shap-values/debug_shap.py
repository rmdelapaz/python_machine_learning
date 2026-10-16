# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 7.2: a "why this customer?" script with three bugs (deliberately broken handout).

Two bugs stop the script with an error; one does not, and prints a "probability" that is wrong.
Fix them one at a time, reading each error from the last line up. When all three are fixed it prints
  C104051: rebuilt probability 0.972 (model says 0.972)
  forest's top reason for C104051: days_since_last_order
"""
# %%
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
            "email_opens_90d"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, X_test = train[features].astype(float), test[features].astype(float)

logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression()).fit(X_train, train["churned"])
hgb = HistGradientBoostingClassifier(random_state=42).fit(X_train, train["churned"])
proba = hgb.predict_proba(X_test)[:, 1]
i = int(proba.argmax())

# %% Rebuild the riskiest customer's probability from the SHAP values
sv = shap.TreeExplainer(logreg)(X_test)
rebuilt = proba.mean() + sv.values[i].sum()
print(f"{test['customer_id'].iloc[i]}: rebuilt probability {rebuilt:.3f} (model says {proba[i]:.3f})")

# %% A second opinion from a random forest
forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
forest.fit(X_train, train["churned"])
forest_sv = shap.TreeExplainer(forest)(X_test.iloc[[i]])
top = pd.Series(forest_sv.values[0], index=features).idxmax()
print(f"forest's top reason for {test['customer_id'].iloc[i]}: {top}")
