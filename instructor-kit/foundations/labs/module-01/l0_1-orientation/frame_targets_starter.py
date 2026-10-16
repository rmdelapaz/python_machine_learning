# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 0.1 B: frame three questions the business asks.

Framing is step 1 of the workflow: which table, which target column, which kind of problem, and what
does "knowing nothing" already score? Three Pinecrest teams have sent you a request. For each one,
FIRST write your answer in the comment, THEN run the check. The file runs as it is.
"""
# %% Request 1, the product team: "Will a reviewer recommend the product?"
import pandas as pd

reviews = pd.read_csv(DATA / "reviews.csv")
# Your framing (write it before you run anything): target column = ______  kind = ______
# TODO 1: put the target column in `target`, then print how many distinct values it has, the share of 1s,
#         and how often "always guess 1" would be right.
target = None
if target is None:
    print("TODO 1: reviews.csv: the target, its distinct values and the share of 1s")

# %% Request 2, the Boise store manager: "How much will we sell tomorrow?"
sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
# Your framing: target column = ______  kind = ______
# TODO 2: keep only the Boise rows, then print the number of days, the number of distinct sales values,
#         the mean and median sales, and how many days had sales of exactly 0.
print("TODO 2: Boise: days, distinct sales values, mean, median, days with sales 0")

# %% Request 3, the payments team: "Is this payment fraud?" (and does the answer stay the same?)
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
# Your framing: target column = ______  kind = ______
# TODO 3: group by month (tx["timestamp"].dt.month) and compute, per month, the number of payments,
#         the number of frauds and the fraud rate. Then add the accuracy of always guessing "legitimate".
print("TODO 3: transactions: fraud rate by month and the accuracy of always guessing 'legitimate'")

# %% Step 4: one sentence each
# TODO 4: under each request above, add one comment line: what would make this problem hard?

print("\nExpected when done: recommended has 2 values and 65.5% are 1; Boise has 1,096 days, 1,091 distinct")
print("sales values, mean $7,664.57, median $7,140.00 and 6 closed days; fraud falls from 1.00% in January")
print("to 0.77% in June, so 'always legitimate' scores 0.9900 in January and 0.9923 in June.")
