# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.2 A (SOLUTION): reasons for the fraud review queue.

Lesson 7.2 gave churn agents three reasons per customer. The payments team reviews the 25 riskiest
payments each morning and wants the same: why is THIS payment flagged? Same explainer (TreeExplainer on
a boosting classifier, so the units are log-odds), a different model, and an interaction to find.
"""
# %% Step 1: the fraud model from Lab 7.1 A (done for you)
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
X_train, X_test = train[features].astype(float), test[features].astype(float)
model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       early_stopping=True, random_state=42).fit(X_train, train["is_fraud"])
test = test.assign(p_fraud=model.predict_proba(X_test)[:, 1])
print(len(train), len(test), f"fraud rate {tx['is_fraud'].mean():.4f}")

# %% Step 2: explain every test payment and check the bookkeeping
sv = shap.TreeExplainer(model)(X_test)
gap = np.abs(sv.base_values + sv.values.sum(axis=1) - model.decision_function(X_test)).max()
print(sv.values.shape, f"base value {sv.base_values[0]:.3f} log-odds = probability {expit(sv.base_values[0]):.4f}")
print(f"largest additivity gap: {gap:.1e}")

# %% Step 3: the global picture from 15,000 local explanations
mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=features).sort_values(ascending=False)
print("mean |SHAP|:", mean_abs.round(3).to_dict())

# %% Step 4: the morning review queue, with reasons in words
phrases = {
    "velocity_24h": "{:.0f} payments on this card in 24 hours",
    "amount": "amount ${:,.0f}",
    "account_age_days": "account only {:.0f} days old",
    "device_new": "first payment from this device",
    "hour": "placed at {:.0f}:00",
    "card_attempts": "{:.0f} card attempts",
    "billing_shipping_mismatch": "billing and shipping addresses differ",
    "country_mismatch": "card country differs from shipping",
    "items": "{:.0f} items",
}
queue = test.nlargest(25, "p_fraud")
rows = [test.index.get_loc(label) for label in queue.index]          # positions in X_test / sv
shares = pd.DataFrame(sv.values[rows], columns=features, index=queue["transaction_id"])


def reasons(tid, n=3):
    row = queue.set_index("transaction_id").loc[tid]
    top = shares.loc[tid][shares.loc[tid] > 0].sort_values(ascending=False).index[:n]
    return [phrases[f].format(row[f]) for f in top]


for tid in queue["transaction_id"].iloc[:3]:
    print(tid, round(queue.set_index("transaction_id").loc[tid, "p_fraud"], 3), reasons(tid))
print("top reason across the 25:", shares.idxmax(axis=1).value_counts().to_dict())
print(f"real frauds in the queue: {queue['is_fraud'].sum()} of 25")

# %% Step 5: the night x new-device interaction, through SHAP
when = np.where(X_test["hour"] < 6, "night 0-5", "day 6-23")
table = pd.DataFrame({"when": when, "new_device": X_test["device_new"].astype(int),
                      "hour_shap": sv[:, "hour"].values})
print(table.pivot_table(index="when", columns="new_device", values="hour_shap", aggfunc="mean").round(2))
