# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 7.1: a churn-drivers script with four bugs (deliberately broken handout).

Two bugs stop the script with an error. Two do not: the script runs and prints numbers that are wrong.
Fix them one at a time, reading each error from the last line up. When all four are fixed it prints
  odds ratio per 30 more days without an order: 1.36
  average predicted churn at 60 / 150 / 240 days: [0.077, 0.214, 0.443]
  top driver on held-out customers: days_since_last_order (0.140)
"""
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance, partial_dependence

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
            "email_opens_90d"]
X = customers[features]
y = customers["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# %% Odds ratio per 30 days without an order
logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression()).fit(X_train, y_train)
coefs = pd.Series(logreg.coef_[0], index=features)
per_unit = coefs / pd.Series(logreg[1].scale_, index=features)
print(f"odds ratio per 30 more days without an order: {30 * per_unit['days_since_last_order']:.2f}")

# %% Risk at three points on the recency dial
hgb = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
pd_days = partial_dependence(hgb, X_test, ["days_since_last_order"], method="brute",
                             custom_values={"days_since_last_order": [60, 150, 240]})
print("average predicted churn at 60 / 150 / 240 days:", pd_days["average"][0].round(3).tolist())

# %% The top driver
result = permutation_importance(hgb, X_train, y_train, scoring="roc_auc", n_repeats=10, random_state=42)
imp = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
print(f"top driver on held-out customers: {imp.index[0]} ({imp.iloc[0]:.3f})")
