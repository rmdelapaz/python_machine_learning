# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.2 A: reasons for the fraud review queue.

Lesson 7.2 gave churn agents three reasons per customer. The payments team reviews the 25 riskiest
payments each morning and wants the same: why is THIS payment flagged? Same explainer (TreeExplainer on
a boosting classifier, so the units are log-odds), a different model, and an interaction to find. Work through TODO 1 to 5; the file runs as it is.
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
# TODO 1: sv = shap.TreeExplainer(model)(X_test). Print sv.values.shape, the base value in log-odds and as a
#         probability (expit), and the largest gap between base value + row sum and model.decision_function(X_test).
#         Why is the base value as a probability (about 0.003) lower than the fraud rate (0.0088)?
sv = None
if sv is None:
    print("TODO 1: SHAP values for the 15,000 test payments, and the additivity check")

# %% Step 3: the global picture from 15,000 local explanations
# TODO 2: the mean absolute SHAP value per feature, largest first. Compare with Lab 7.1 A's permutation
#         ranking (velocity_24h first there): which feature moves predictions most here?
print("TODO 2: mean |SHAP| per feature")

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
# TODO 3: queue = test.nlargest(25, "p_fraud"). Find each queued payment's POSITION in X_test
#         (test.index.get_loc(label)), take those rows of sv.values as a DataFrame indexed by transaction_id,
#         and write reasons(tid): the phrases of the (up to) three features with the largest POSITIVE shares.
#         Print the first three payments with their probability and reasons.
# TODO 4: the most common top reason across the 25 (idxmax(axis=1).value_counts()) and how many of the
#         25 are real frauds.
print("TODO 3-4: the review queue with three reasons per payment")

# %% Step 5: the night x new-device interaction, through SHAP
# TODO 5: a table with when = "night 0-5" / "day 6-23", new_device (0/1) and the SHAP value of hour;
#         pivot_table of the mean hour SHAP value. What does the hour add at night with a new device?
print("TODO 5: mean SHAP value of hour by time of day and device")

print("\nExpected when done: (15000, 9), base value -5.840 log-odds = 0.0029, additivity gap about 1e-14;")
print("mean |SHAP| device_new 0.247, velocity_24h 0.235, account_age_days 0.211; T031633 0.99 with")
print("'5 payments on this card in 24 hours', 'amount $836', 'account only 10 days old'; top reasons")
print("velocity_24h 19, device_new 5, card_attempts 1; 25 of 25 real frauds; hour adds +1.24 log-odds at")
print("night with a new device, +0.33 at night with a known one, about 0 by day.")
