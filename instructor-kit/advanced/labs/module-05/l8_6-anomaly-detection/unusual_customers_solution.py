# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.6 B (SOLUTION): the most unusual customers, by two detectors.

Lesson 8.6 hunted fraud in payments. Here marketing asks a softer question: which 60 customers (1%)
behave least like everyone else, do Isolation Forest and Local Outlier Factor agree on them, and is
"unusual" the same as "about to churn"? There is no fraud label here: the profile is the evidence.
"""
# %% Step 1: ten scaled behavior columns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

pd.set_option("display.width", 130)
customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
Z = StandardScaler().fit_transform(customers[features])

# %% Step 2: two anomaly scores, both "higher = more unusual"
iso = IsolationForest(n_estimators=300, random_state=42).fit(Z)
customers["iso_score"] = -iso.score_samples(Z)
lof = LocalOutlierFactor(n_neighbors=30).fit(Z)                  # default novelty=False: scores the training rows
customers["lof_score"] = -lof.negative_outlier_factor_
print(customers[["iso_score", "lof_score"]].describe().loc[["min", "50%", "max"]].round(3))

# %% Step 3: do they agree?
from scipy.stats import spearmanr

top_iso = set(customers.nlargest(60, "iso_score").index)
top_lof = set(customers.nlargest(60, "lof_score").index)
rho = spearmanr(customers["iso_score"], customers["lof_score"]).statistic
print(f"\nrank correlation of the two scores: {rho:.3f}")
print(f"customers in both top-60 lists: {len(top_iso & top_lof)}")

# %% Step 4: what makes the Isolation Forest's top 60 unusual?
customers["iso_top60"] = customers.index.isin(list(top_iso))
cols = ["orders_12m", "spend_12m", "avg_order_value", "returns_12m", "support_tickets_12m", "discount_share", "churned"]
print(customers.groupby("iso_top60")[cols].mean().round(2).to_string())

# %% Step 5: and the LOF's top 60?
customers["lof_top60"] = customers.index.isin(list(top_lof))
print(customers.groupby("lof_top60")[cols].mean().round(2).to_string())
print(f"\nchurn: overall {customers['churned'].mean():.3f}, IF top 60 "
      f"{customers.loc[customers['iso_top60'], 'churned'].mean():.3f}, LOF top 60 "
      f"{customers.loc[customers['lof_top60'], 'churned'].mean():.3f}")
