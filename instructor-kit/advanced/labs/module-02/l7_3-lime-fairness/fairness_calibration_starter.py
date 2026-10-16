# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.3 B: is the churn model equally honest for every age group?

Lesson 7.3 compared recall and flag rates by age group. Two more views matter to Pinecrest: calibration
(when the model says 15%, do 15% churn, in every group?) and money (the $15 / $60 campaign from
Foundations: what does the campaign earn per customer in each group?). Age is used only to audit. Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1: the churn model and the campaign rule (done for you)
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
            "email_opens_90d"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
model = HistGradientBoostingClassifier(random_state=42).fit(train[features].astype(float), train["churned"])

audit = test[["customer_id", "age_group", "churned"]].copy()
audit["age"] = audit["age_group"].fillna("unknown")
audit["p"] = model.predict_proba(test[features].astype(float))[:, 1]
audit["offer"] = audit["p"] >= 0.25                     # the $15 / $60 break-even
print(f"offers {audit['offer'].sum()} of {len(audit)}; churners among them {audit.loc[audit['offer'], 'churned'].mean():.1%}")

# %% Step 2: calibration by group
# TODO 1: by_age = audit.groupby("age").agg(customers=("churned", "size"), actual=("churned", "mean"),
#         predicted=("p", "mean")); add gap = predicted - actual and print it rounded to 3.
#         Which groups does the model under-predict, which over-predict?
by_age = None
if by_age is None:
    print("TODO 1: actual and predicted churn by age group")

# %% Step 3: precision and money by group
# TODO 2: add precision (churn rate among the offered customers of each group) and profit_per_customer:
#         (60 x offered churners - 15 x offers) / customers, per group. Which group earns least, and why?
print("TODO 2: precision and campaign profit per customer by age group")

# %% Step 4: is the largest calibration gap more than chance?
# TODO 3: write worst_gap(frame): the largest |mean p - churn rate| over age groups. Compute it for audit,
#         then for 500 copies with the age labels shuffled (rng = np.random.default_rng(42)); print the
#         observed gap, the median shuffled gap and the share of shuffles at least as large.
# TODO 4 (think): write two sentences for the analytics lead: what was checked, what was found, what was not.
print("TODO 3: permutation test on the largest calibration gap")

print("\nExpected when done: 300 offers, 52.3% churners (the same 300 as the lesson's top 20%); gaps from")
print("-0.042 (35-44) to +0.025 (45-54); 45-54 has the lowest precision 0.41 and profit $2.00 per customer,")
print("unknown the highest $4.80; largest gap 0.042 against 0.046 by chance, 64% of shuffles at least as large.")
