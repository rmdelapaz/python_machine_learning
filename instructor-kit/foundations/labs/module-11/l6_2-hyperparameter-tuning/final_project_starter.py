# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Final project scaffold: milestone 1 for your chosen problem.

Set PROBLEM to the problem you chose in Session 10: "churn", "demand" or "fraud". This file makes the
course's leak-safe split for it (done for you, so everyone's numbers can be compared), then asks you
for the two things every project needs before any clever model: the bar to beat, and a first model
scored by cross-validation on the TRAINING rows only. Copy the finished cells into your project
notebook. The file runs as it is; unfinished steps print a reminder.
"""
# %% Step 1: choose the problem and make the leak-safe split (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROBLEM = "churn"          # "churn", "demand" or "fraud"

if PROBLEM == "churn":
    data = pd.read_csv(DATA / "customers_churn.csv")
    train, test = train_test_split(data, test_size=0.25, stratify=data["churned"], random_state=42)
    target = "churned"
    print("never a feature: cancellation_reason (filled in only after a customer has left)")
elif PROBLEM == "demand":
    data = pd.read_csv(DATA / "weekly_demand.csv")
    train, test = data[data["year"] == 2024], data[data["year"] == 2025]
    target = "units_sold"
    print("the forecast is made at the start of 2025: no feature may use a 2025 sales figure")
elif PROBLEM == "fraud":
    data = pd.read_csv(DATA / "transactions.csv")
    is_june = data["timestamp"] >= "2025-06-01"
    train, test = data[~is_june], data[is_june]
    target = "is_fraud"
    print("train on January to May, test on June: never shuffle months together")
else:
    raise ValueError(f"PROBLEM must be 'churn', 'demand' or 'fraud', not {PROBLEM!r}")
print(f"{PROBLEM}: {len(train):,} training rows, {len(test):,} test rows (the test rows stay locked)")

# %% Step 2: the bar to beat
# TODO 1: compute the do-nothing numbers for your problem, the ones every later model is compared with:
#   churn : the random-order AP (the churn rate of the test rows) and the profit of contacting everyone,
#           60 * churners - 15 * customers (dollars); contacting nobody earns 0
#   demand: the MAE on 2025 of the 2024 overall mean and of each product's 2024 mean
#   fraud : the June cost of reviewing nothing (375 per fraud) and of reviewing everything
#           (10 per payment, nothing missed)
print("TODO 1: the bar to beat for", PROBLEM)

# %% Step 3: a first model, cross-validated on the training rows only
# TODO 2: churn and fraud: a logistic-regression pipeline (impute, scale, one-hot as in Lesson 6.1),
#         5-fold StratifiedKFold(shuffle=True, random_state=42), scoring="average_precision".
#         demand: HistGradientBoostingRegressor(loss="poisson", random_state=42) on the Lesson 5.3 features
#         with store, product_id and category as category dtype, cv=KFold(5) (blocks of weeks, no shuffle),
#         scoring="neg_mean_absolute_error". Print the mean and the std.
print("TODO 2: a first cross-validated model")

print("\nExpected when done (your problem only):")
print("  churn : 4,500 / 1,500; random-order AP 0.165; contact everyone -7,620 dollars; logistic CV AP 0.528 +/- 0.042")
print("  demand: 5,088 / 4,896; overall mean MAE 17.13, per-product mean 13.42; boosting CV MAE 8.49 +/- 1.34")
print("  fraud : 41,738 / 8,262 (374 / 64 frauds); nothing $24,000, everything $82,620; logistic CV AP 0.506 +/- 0.054")
