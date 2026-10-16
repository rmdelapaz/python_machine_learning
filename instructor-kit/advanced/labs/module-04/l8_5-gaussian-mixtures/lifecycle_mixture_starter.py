# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.5 B: lifecycle stages as a soft clustering.

Lesson 8.5 segmented customers by HOW they shop. Here the question is WHERE they are in their life
with Pinecrest: how long they have been members, how recently they ordered, how much they spend.
Choose the number of components with BIC, name the components by profile, test whether a different
seed gives the same answer, and score a brand-new customer. Work through TODO 1 to 5.
"""
# %% Step 1: three lifecycle features, logged where skewed, scaled (done for you)
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

# %% Step 2: BIC
# TODO 1: BIC for 1 to 6 full-covariance components (n_init=3, random_state=42), with the drop per step.
print("TODO 1: BIC for 1 to 6 components")

# %% Step 3: four components, profiled
# TODO 2: fit 4 components (n_init=3, random_state=42); store argmax and max of predict_proba as
#         customers["component"] and customers["confidence"]; profile each component: size, mean tenure,
#         MEDIAN days since last order and spend, churn rate, and weights_. Give each a name.
gm4 = None
if gm4 is None:
    print("TODO 2: four components and their profile")

# %% Step 4: another seed
from sklearn.metrics import adjusted_rand_score

# TODO 3: fit the same model with random_state=7, then print the ARI between the two labelings and a
#         pd.crosstab of the two. Are the segments different, or only their numbers?
print("TODO 3: seed 42 vs seed 7")

# %% Step 5: confidence
# TODO 4: how many customers have confidence below 0.8? Median confidence per component?
print("TODO 4: borderline customers")

# %% Step 6: a new customer
newcomer = pd.DataFrame({"tenure_months": [2], "log_recency": [np.log1p(20)], "log_spend": [np.log(150.0)]})
# TODO 5: predict_proba for newcomer, through the SAME fitted scaler (scaler.transform, not fit_transform).
print("TODO 5: score the newcomer")

print("\nExpected when done: BIC drops 3,408 from one to two components, then 493, 385, 243 and 40 (no clean")
print("minimum); four components: 2,169 long-tenure regulars (68 months, churn 0.109), 1,272 established (22")
print("months, 0.071), 1,293 fading (median 101 days since the last order, $194 spend, 0.205), 1,266 newcomers")
print("(3 months, 0.317); seeds 42 and 7: ARI 0.981, the same groups under other numbers; 1,340 customers (22.3%)")
print("below 0.8; the newcomer is 0.977 in the newcomer component.")
