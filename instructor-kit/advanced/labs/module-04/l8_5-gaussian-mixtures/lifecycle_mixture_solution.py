# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.5 B (SOLUTION): lifecycle stages as a soft clustering.

Lesson 8.5 segmented customers by HOW they shop. Here the question is WHERE they are in their life
with Pinecrest: how long they have been members, how recently they ordered, how much they spend.
Choose the number of components with BIC, name the components by profile, test whether a different
seed gives the same answer, and score a brand-new customer.
"""
# %% Step 1: three lifecycle features, logged where skewed, scaled
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

pd.set_option("display.width", 120)
customers = pd.read_csv(DATA / "customers_churn.csv")
life = pd.DataFrame({
    "tenure_months": customers["tenure_months"],
    "log_recency": np.log1p(customers["days_since_last_order"]),
    "log_spend": np.log(customers["spend_12m"]),
})
scaler = StandardScaler().set_output(transform="pandas")
Z = scaler.fit_transform(life)

# %% Step 2: BIC for 1 to 6 full-covariance components
rows = []
for k in range(1, 7):
    gm = GaussianMixture(n_components=k, n_init=3, random_state=42).fit(Z)
    rows.append({"k": k, "bic": gm.bic(Z)})
crit = pd.DataFrame(rows).set_index("k")
crit["drop"] = -crit["bic"].diff()
print(crit.round(0))

# %% Step 3: four components, profiled and named
gm4 = GaussianMixture(n_components=4, n_init=3, random_state=42).fit(Z)
proba = gm4.predict_proba(Z)
customers["component"] = proba.argmax(axis=1)
customers["confidence"] = proba.max(axis=1)
profile = customers.groupby("component").agg(
    customers=("customer_id", "size"), tenure=("tenure_months", "mean"),
    days_since=("days_since_last_order", "median"), spend=("spend_12m", "median"), churn=("churned", "mean"))
profile["weight"] = gm4.weights_
print(profile.round(3))

# %% Step 4: same answer with another seed?
from sklearn.metrics import adjusted_rand_score

other = GaussianMixture(n_components=4, n_init=3, random_state=7).fit(Z).predict(Z)
print(f"\nARI between seeds 42 and 7: {adjusted_rand_score(customers['component'], other):.3f}")
print(pd.crosstab(customers["component"], other, rownames=["seed 42"], colnames=["seed 7"]))

# %% Step 5: how sure is the model?
print(f"\nconfidence below 0.8: {(customers['confidence'] < 0.8).sum()} customers "
      f"({(customers['confidence'] < 0.8).mean():.1%})")
print(customers.groupby("component")["confidence"].median().round(3).to_dict())

# %% Step 6: a brand-new customer, through the SAME scaler
newcomer = pd.DataFrame({"tenure_months": [2], "log_recency": [np.log1p(20)], "log_spend": [np.log(150.0)]})
print("\nnew customer:", gm4.predict_proba(scaler.transform(newcomer)).round(3))
