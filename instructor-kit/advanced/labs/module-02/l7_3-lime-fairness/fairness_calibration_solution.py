# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.3 B (SOLUTION): is the churn model equally honest for every age group?

Lesson 7.3 compared recall and flag rates by age group. Two more views matter to Pinecrest: calibration
(when the model says 15%, do 15% churn, in every group?) and money (the $15 / $60 campaign from
Foundations: what does the campaign earn per customer in each group?). Age is used only to audit.
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
by_age = audit.groupby("age").agg(customers=("churned", "size"), actual=("churned", "mean"),
                                  predicted=("p", "mean"))
by_age["gap"] = by_age["predicted"] - by_age["actual"]
print(by_age.round(3))

# %% Step 3: precision and money by group
by_age["precision"] = audit[audit["offer"]].groupby("age")["churned"].mean()
tp = (audit["offer"] & (audit["churned"] == 1)).groupby(audit["age"]).sum()
offers = audit.groupby("age")["offer"].sum()
by_age["profit_per_customer"] = (60 * tp - 15 * offers) / by_age["customers"]
print(by_age[["precision", "profit_per_customer"]].round(2))

# %% Step 4: is the largest calibration gap more than chance?


def worst_gap(frame):
    g = frame.groupby("age").agg(p=("p", "mean"), y=("churned", "mean"))
    return (g["p"] - g["y"]).abs().max()


observed = worst_gap(audit)
rng = np.random.default_rng(42)
shuffled = np.array([worst_gap(audit.assign(age=rng.permutation(audit["age"].to_numpy()))) for _ in range(500)])
print(f"largest calibration gap {observed:.3f}; typical by chance {np.median(shuffled):.3f}; "
      f"share of shuffles at least as large {(shuffled >= observed).mean():.3f}")
