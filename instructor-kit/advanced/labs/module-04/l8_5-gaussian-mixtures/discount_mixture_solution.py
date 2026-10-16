# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.5 A (SOLUTION): are there two kinds of discount shoppers?

Lesson 8.5 opened with a one-feature mixture on made-up data. Here is the same move on a real column:
the share of each customer's spend that was discounted. Does BIC see one population or two, where does
the model switch its vote, and do the soft memberships line up with churn (which the model never sees)?
"""
# %% Step 1: one column, as a 2-D table
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

customers = pd.read_csv(DATA / "customers_churn.csv")
x = customers[["discount_share"]].to_numpy()            # double brackets: scikit-learn wants 2-D
print(x.shape, f"mean {x.mean():.3f}, median {np.median(x):.3f}, max {x.max():.3f}")

# %% Step 2: how many bell curves? BIC for 1 to 5 components
bics = {k: GaussianMixture(n_components=k, n_init=3, random_state=42).fit(x).bic(x) for k in range(1, 6)}
bic = pd.Series(bics, name="bic")
print(pd.DataFrame({"bic": bic.round(0), "drop": (-bic.diff()).round(0)}))

# %% Step 3: fit two components and name them by their means, never by their numbers
gm = GaussianMixture(n_components=2, n_init=3, random_state=42).fit(x)
deal = int(gm.means_.ravel().argmax())                   # the component with the higher mean
print("means  :", gm.means_.ravel().round(3))
print("stds   :", np.sqrt(gm.covariances_.ravel()).round(3))
print("weights:", gm.weights_.round(3), " deal-driven component:", deal)

# %% Step 4: where does the vote switch?
probe = np.array([[0.10], [0.20], [0.25], [0.30], [0.40]])
print(pd.Series(gm.predict_proba(probe)[:, deal].round(3), index=probe.ravel(), name="P(deal-driven)"))

# %% Step 5: soft membership against churn (churn was never used to fit)
customers["p_deal"] = gm.predict_proba(x)[:, deal]
bands = pd.cut(customers["p_deal"], [0, 0.1, 0.5, 0.9, 1.0], include_lowest=True)
print(customers.groupby(bands, observed=True)["churned"].agg(customers="size", churn="mean").round(3))

# %% Step 6: borderline customers, and a hard rule for comparison
borderline = customers["p_deal"].between(0.2, 0.8)
rule = customers["discount_share"] > 0.3
print(f"borderline (0.2 to 0.8): {borderline.sum()} customers, discount share "
      f"{customers.loc[borderline, 'discount_share'].min():.3f} to {customers.loc[borderline, 'discount_share'].max():.3f}")
print(f"hard rule 'discount_share > 0.3' flags {rule.sum()}; the mixture's majority vote flags "
      f"{(customers['p_deal'] > 0.5).sum()}")
