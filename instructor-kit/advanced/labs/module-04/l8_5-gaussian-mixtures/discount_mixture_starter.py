# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.5 A: are there two kinds of discount shoppers?

Lesson 8.5 opened with a one-feature mixture on made-up data. Here is the same move on a real column:
the share of each customer's spend that was discounted. Does BIC see one population or two, where does
the model switch its vote, and do the soft memberships line up with churn (which the model never sees)?
Work through TODO 1 to 5; run the file after each one. It runs as it is.
"""
# %% Step 1: one column, as a 2-D table (done for you)
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

customers = pd.read_csv(DATA / "customers_churn.csv")
x = customers[["discount_share"]].to_numpy()            # double brackets: scikit-learn wants 2-D
print(x.shape, f"mean {x.mean():.3f}, median {np.median(x):.3f}, max {x.max():.3f}")

# %% Step 2: how many bell curves?
# TODO 1: for k in 1..5 fit GaussianMixture(n_components=k, n_init=3, random_state=42) on x and record
#         gm.bic(x). Print the BIC and the drop from the previous k. Where do the big drops stop?
print("TODO 1: BIC for 1 to 5 components")

# %% Step 3: two components, named by their means
# TODO 2: fit two components (n_init=3, random_state=42). Print means_, the standard deviations
#         (np.sqrt of covariances_) and weights_. Set deal = the index of the LARGER mean
#         (gm.means_.ravel().argmax()): never assume component 1 is the one you want.
gm = None
deal = None
if gm is None:
    print("TODO 2: two components, means, stds, weights, which one is deal-driven")

# %% Step 4: where does the vote switch?
probe = np.array([[0.10], [0.20], [0.25], [0.30], [0.40]])
# TODO 3: print gm.predict_proba(probe)[:, deal] for the five probe values. Between which two does it pass 0.5?
print("TODO 3: P(deal-driven) for 0.10, 0.20, 0.25, 0.30, 0.40")

# %% Step 5: soft membership against churn
# TODO 4: customers["p_deal"] = gm.predict_proba(x)[:, deal]; cut it into the bands [0, 0.1, 0.5, 0.9, 1.0]
#         (pd.cut(..., include_lowest=True)) and print the number of customers and the churn rate per band.
print("TODO 4: churn by band of p_deal")

# %% Step 6: borderline customers and a hard rule
# TODO 5: count customers with p_deal between 0.2 and 0.8 and the range of their discount_share. Then compare
#         the hard rule discount_share > 0.3 with the mixture's majority vote (p_deal > 0.5).
print("TODO 5: borderline customers; hard rule vs mixture")

print("\nExpected when done: BIC drops 3,596 from one to two components, then only 330, 65 and 54; means 0.11")
print("and 0.44, stds 0.063 and 0.146, weights 0.705 and 0.295; P(deal-driven) passes 0.5 between 0.25 (0.471)")
print("and 0.30 (0.911); churn 0.119 for p_deal up to 0.1 and 0.305 above 0.9; 430 borderline customers with")
print("discount shares 0.219 to 0.281; the rule flags 1,431, the mixture's majority vote 1,682.")
