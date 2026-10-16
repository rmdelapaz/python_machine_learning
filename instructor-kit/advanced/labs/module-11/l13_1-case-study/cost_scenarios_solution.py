# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 13.1 B (SOLUTION): when the costs change, does the plan change?

The case study set the offer threshold from the manager's figures ($15 an offer, $60 a churner
reached). Next quarter marketing wants a bigger coupon, and finance wants to know what each extra
offer earns. Here you answer both from out-of-fold predictions on the TRAINING customers, then open
the test set once to price the decision.
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
rows = []
for cost in [10, 15, 20, 25]:
    profits = [profit(y, oof >= t, cost) for t in grid]
    best = grid[int(np.argmax(profits))]
    rows.append({"offer_cost": cost, "break_even": round(cost / 60, 3), "best_threshold": best,
                 "offers": int((oof >= best).sum()), "profit_best": max(profits),
                 "profit_at_0.25": profit(y, oof >= 0.25, cost)})
scenarios = pd.DataFrame(rows).set_index("offer_cost")
print(scenarios)

# %% Step 3: what does each extra block of 100 offers earn? (the $15 offer)
order = np.argsort(-oof)
blocks = []
for k in range(100, 1601, 100):
    chosen = order[k - 100:k]
    blocks.append({"offers": k, "lowest_p": round(float(oof[order[k - 1]]), 3),
                   "churners": int(y[chosen].sum()), "block_profit": int(60 * y[chosen].sum() - 15 * 100)})
blocks = pd.DataFrame(blocks).set_index("offers")
print(blocks.to_string())
first_loss = blocks[blocks["block_profit"] <= 0].index.min()
print(f"the block ending at offer {first_loss} is the first that loses money (its lowest probability is "
      f"{blocks.loc[first_loss, 'lowest_p']})")

# %% Step 4: open the test set once: the $25 offer, with the old and the new threshold
model.fit(train[FEATURES], train["churned"])
p_test = model.predict_proba(test[FEATURES])[:, 1]
y_test = test["churned"].to_numpy()
new_t = scenarios.loc[25, "best_threshold"]
for label, t in [("old threshold 0.25", 0.25), (f"new threshold {new_t}", new_t), ("break-even 0.417", 25 / 60)]:
    print(f"$25 offer, {label:20s}: {int((p_test >= t).sum()):4d} offers, profit ${profit(y_test, p_test >= t, 25):,}")
