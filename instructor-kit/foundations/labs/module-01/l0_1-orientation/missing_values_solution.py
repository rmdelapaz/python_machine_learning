# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 0.1 A (SOLUTION): why is it missing?

Lesson 0.1 counted the gaps in every file. A count is only the start: a data scientist asks WHY a
value is missing, because the reason decides what you may do with it. This lab checks three gaps in
customers_churn.csv and one in interactions.csv.
"""
# %% Step 1: load the customers and count the gaps
import pandas as pd

customers = pd.read_csv(DATA / "customers_churn.csv")
gaps = customers.isna().sum()
print(gaps[gaps > 0])

# %% Step 2: is age_group missing more often for online shoppers?
# A True/False column groups the rows; the mean of isna() is the share missing.
online = customers["home_store"] == "Online"
age_missing = customers["age_group"].isna().groupby(online).mean()
print(f"\nage_group missing: online {age_missing[True]:.1%}, store shoppers {age_missing[False]:.1%}")

# %% Step 3: does a missing email count go with churn?
email_missing = customers["email_opens_90d"].isna()
churn_by_gap = customers["churned"].groupby(email_missing).mean()
print(f"churn rate: email count missing {churn_by_gap[True]:.1%}, email count present {churn_by_gap[False]:.1%}")

# %% Step 4: who has a cancellation_reason?
print()
print(pd.crosstab(customers["cancellation_reason"].notna(), customers["churned"],
                  rownames=["has reason"], colnames=["churned"]))

# %% Step 5: interactions: a gap that means something
interactions = pd.read_csv(DATA / "interactions.csv")
share = interactions["rating"].isna().mean()
print(f"\ninteractions without a star rating: {interactions['rating'].isna().sum():,} of {len(interactions):,} ({share:.1%})")

# %% Verdicts (the answers to the questions in the starter)
print("""
Verdicts
  age_group           missing far more often online: depends on WHERE the customer shops, not on churn.
  email_opens_90d     missing customers churn more (26.1% vs 15.6%): the gap itself carries a little signal.
  cancellation_reason filled only for churners (625 of 625): written AFTER the outcome. A leak, never a feature.
  rating              most buyers never rate: an empty rating means "no review", not "bad product".""")
