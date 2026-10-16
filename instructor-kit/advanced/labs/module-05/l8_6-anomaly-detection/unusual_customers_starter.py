# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.6 B: the most unusual customers, by two detectors.

Lesson 8.6 hunted fraud in payments. Here marketing asks a softer question: which 60 customers (1%)
behave least like everyone else, do Isolation Forest and Local Outlier Factor agree on them, and is
"unusual" the same as "about to churn"? There is no fraud label here: the profile is the evidence.
Work through TODO 1 to 4.
"""
# %% Step 1: ten scaled behavior columns (done for you)
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

# %% Step 2: two anomaly scores
# TODO 1: customers["iso_score"] = -IsolationForest(n_estimators=300, random_state=42).fit(Z).score_samples(Z)
#         customers["lof_score"] = -LocalOutlierFactor(n_neighbors=30).fit(Z).negative_outlier_factor_
#         (both now "higher = more unusual"). Print min, median and max of each.
print("TODO 1: two anomaly scores")

# %% Step 3: agreement
from scipy.stats import spearmanr

# TODO 2: the Spearman rank correlation of the two scores, and how many customers are in BOTH top-60 lists.
print("TODO 2: rank correlation and overlap of the two top-60 lists")

# %% Step 4: profiles
cols = ["orders_12m", "spend_12m", "avg_order_value", "returns_12m", "support_tickets_12m", "discount_share", "churned"]
# TODO 3: flag the Isolation Forest's top 60 and compare the means of cols for flagged vs the rest.
# TODO 4: the same for the LOF's top 60. Compare both churn rates with the overall 0.165. Is "unusual" the
#         same as "about to leave"?
print("TODO 3-4: profile both top-60 lists")

print("\nExpected when done: rank correlation 0.515; only 9 customers in both lists; the Isolation Forest's")
print("top 60 are heavy buyers (9.67 orders, $1,810 spend, 4.78 returns) who churn at 0.067; the LOF's top 60")
print("are much closer to average but churn at 0.333 (overall 0.165).")
