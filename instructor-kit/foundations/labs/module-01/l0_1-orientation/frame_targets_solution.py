# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 0.1 B (SOLUTION): frame three questions the business asks.

Framing is step 1 of the workflow: which table, which target column, which kind of problem, and what
does "knowing nothing" already score? Three requests from Pinecrest teams, none of them in the lesson.
"""
# %% Request 1, the product team: "Will a reviewer recommend the product?"
import pandas as pd

reviews = pd.read_csv(DATA / "reviews.csv")
target = reviews["recommended"]
print("1. reviews.csv  target: recommended  kind: classification")
print(f"   distinct values: {target.nunique()}, share recommended: {target.mean():.1%}")
print(f"   always guessing 'recommends' would be right {target.mean():.1%} of the time")

# %% Request 2, the Boise store manager: "How much will we sell tomorrow?"
sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
boise = sales[sales["store"] == "Boise"]
print("\n2. daily_store_sales.csv (Boise rows)  target: sales  kind: regression (forecasting)")
print(f"   days: {len(boise):,}, distinct sales values: {boise['sales'].nunique():,}")
print(f"   mean ${boise['sales'].mean():,.2f}, median ${boise['sales'].median():,.2f}, closed days (sales 0): {(boise['sales'] == 0).sum()}")

# %% Request 3, the payments team: "Is this payment fraud?" (and does the answer stay the same?)
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
by_month = tx.groupby(tx["timestamp"].dt.month.rename("month"))["is_fraud"].agg(["size", "sum", "mean"])
by_month.columns = ["payments", "fraud", "fraud_rate"]
by_month["always_legit_accuracy"] = 1 - by_month["fraud_rate"]
print("\n3. transactions.csv  target: is_fraud  kind: classification (a very rare class)")
print(by_month.round(4).to_string())

# %% What the numbers say
print("""
Reading the numbers
  1. Two values, about two in three reviews recommend: a classification target, not too unbalanced.
  2. More than a thousand distinct amounts and 6 closed days: a quantity, so regression. Closed days are
     real zeros the model must expect (holidays), not missing data.
  3. Fraud falls from 1.00% of payments in January to 0.77% in June, so "always legitimate" gets MORE
     accurate every month while catching nobody. A score that rises for a useless model is a warning sign.""")
