# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 13.1 B: when the costs change, does the plan change?

The case study set the offer threshold from the manager's figures ($15 an offer, $60 a churner
reached). Next quarter marketing wants a bigger coupon, and finance wants to know what each extra
offer earns. Here you answer both from out-of-fold predictions on the TRAINING customers, then open
the test set once to price the decision. Step 1 is done for you; work through TODO 1 to 4.
"""
# %% Step 1: the case study's model (tuned settings, no age), and honest training probabilities
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
NUMERIC = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
           "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
CATEGORICAL = ["home_store", "region", "loyalty_tier", "preferred_category"]
FEATURES = NUMERIC + CATEGORICAL
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
])
model = Pipeline([("prep", prep), ("model", HistGradientBoostingClassifier(
    random_state=42, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=0.1, max_iter=100))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(model, train[FEATURES], train["churned"], cv=cv, method="predict_proba")[:, 1]
y = train["churned"].to_numpy()
print(f"{len(train)} training customers, out-of-fold probabilities from 5 folds, mean {oof.mean():.3f}")


def profit(y_true, offered, cost, value=60):
    y_true, offered = np.asarray(y_true), np.asarray(offered, dtype=bool)
    return int(value * (offered & (y_true == 1)).sum() - cost * offered.sum())

# %% Step 2: four offer costs: break-even theory against the best threshold on the training data
grid = np.round(np.arange(0.05, 0.71, 0.01), 2)
# TODO 1: for offer costs 10, 15, 20 and 25 (value stays $60): the break-even probability cost / 60, the
#         threshold in `grid` with the highest profit(y, oof >= t, cost), the offers it sends, that profit,
#         and the profit of keeping the old 0.25. One row per cost in a DataFrame `scenarios`.
#         Before you run it, predict: does the best threshold follow cost / 60? ______
scenarios = None
if scenarios is None:
    print("TODO 1: the cost scenarios")

# %% Step 3: what does each extra block of 100 offers earn? (the $15 offer)
order = np.argsort(-oof)
# TODO 2: walk down the ranking in blocks of 100 (offers 1-100, 101-200, ... up to 1,600). For each block
#         print the lowest probability in it, the churners in it and its profit (60 x churners - 15 x 100).
# TODO 3: which block is the first to lose money, and what probability does it end at? Compare with 0.25.
print("TODO 2-3: profit of each block of 100 offers")

# %% Step 4: open the test set once: the $25 offer, with the old and the new threshold
# TODO 4: fit `model` on all training customers, score the test set once, and print offers and profit for
#         a $25 offer at the old threshold 0.25, at your new best threshold, and at break-even 25 / 60.
print("TODO 4: price the decision on the test set")

print("\nExpected when done: best thresholds 0.17 / 0.24 / 0.31 / 0.43 for offer costs $10 / $15 / $20 / $25")
print("(break-even 0.167 / 0.250 / 0.333 / 0.417); keeping 0.25 at $25 earns $6,485 instead of $8,490 on the")
print("training data. Blocks of 100: the block ending at offer 1,200 (lowest p 0.213) is the first to lose")
print("money. Test, $25 offer: 0.25 gives 360 offers and $1,620; 0.43 gives 183 offers and $2,745.")
