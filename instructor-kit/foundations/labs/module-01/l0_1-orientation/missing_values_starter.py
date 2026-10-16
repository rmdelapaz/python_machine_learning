# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 0.1 A: why is it missing?

Lesson 0.1 counted the gaps in every file. A count is only the start: ask WHY a value is missing,
because the reason decides what you may do with it. Work through TODO 1 to 5 in order; run the file
after each one. The file runs as it is: the lines you have not written yet print a reminder.
"""
# %% Step 1: load the customers and count the gaps (done for you)
import pandas as pd

customers = pd.read_csv(DATA / "customers_churn.csv")
gaps = customers.isna().sum()
print(gaps[gaps > 0])

# %% Step 2: is age_group missing more often for online shoppers?
online = customers["home_store"] == "Online"
# TODO 1: the share of customers whose age_group is missing, separately for online (True) and
#         store (False) shoppers. Hint: customers["age_group"].isna().groupby(online).mean()
age_missing = None
if age_missing is None:
    print("\nTODO 1: share of missing age_group, online vs store shoppers")
else:
    print(f"\nage_group missing: online {age_missing[True]:.1%}, store shoppers {age_missing[False]:.1%}")

# %% Step 3: does a missing email count go with churn?
email_missing = customers["email_opens_90d"].isna()
# TODO 2: the churn rate for customers whose email count is missing (True) and present (False).
#         Hint: group the churned column by email_missing and take the mean.
churn_by_gap = None
if churn_by_gap is None:
    print("TODO 2: churn rate when email_opens_90d is missing vs present")
else:
    print(f"churn rate: email count missing {churn_by_gap[True]:.1%}, email count present {churn_by_gap[False]:.1%}")

# %% Step 4: who has a cancellation_reason?
# TODO 3: a crosstab of "has a cancellation_reason" (notna()) against churned.
#         pd.crosstab(row_values, column_values) counts every combination.
print("\nTODO 3: crosstab of cancellation_reason.notna() against churned")

# %% Step 5: interactions: a gap that means something
# TODO 4: load interactions.csv from DATA and print how many rows have no rating, and what share.
print("\nTODO 4: interactions without a star rating (count and share)")

# %% Step 6: your verdicts
# TODO 5: for each of the four columns, write ONE line in a comment here: why is it missing, and may a
#         model use it? (The solution prints its verdicts; compare after you write yours.)

print("\nExpected when done: age_group missing 12.7% online vs 4.4% in stores; churn 26.1% when the email")
print("count is missing vs 15.6% when present; every one of the 625 cancellation reasons belongs to a churner;")
print("19,970 of 32,178 interactions (62.1%) have no rating.")
