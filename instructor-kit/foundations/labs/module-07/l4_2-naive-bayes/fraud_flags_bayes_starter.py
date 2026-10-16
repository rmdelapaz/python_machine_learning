# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.2 A: Bayes' rule for fraud flags.

Lesson 4.2 counted words in reviews. A fraud analyst counts warning signs instead: a night-time payment, a
new device, a mismatched address. Use Bayes' rule on one flag, combine two flags the naive way and check
the answer against direct counting, then let BernoulliNB use nine flags at once. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1 (done for you): nine 0/1 warning signs and the stratified split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
tx["night"] = (tx["hour"] < 6).astype(int)
tx["new_account"] = (tx["account_age_days"] < 30).astype(int)
tx["gift_card"] = (tx["category"] == "Gift Cards").astype(int)
tx["big_amount"] = (tx["amount"] > 400).astype(int)
tx["multi_attempt"] = (tx["card_attempts"] > 1).astype(int)
tx["busy_24h"] = (tx["velocity_24h"] >= 2).astype(int)
flags = ["night", "device_new", "new_account", "gift_card", "big_amount", "billing_shipping_mismatch",
         "country_mismatch", "multi_attempt", "busy_24h"]
train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
y = train["is_fraud"]
prior = y.mean()
print(f"prior P(fraud) = {prior:.4f}")

# %% Step 2: Bayes' rule for one flag
# TODO 1: for country_mismatch compute P(flag | fraud), P(flag | legit) and P(flag) on the training rows, then
#         the posterior P(fraud | flag) = P(flag | fraud) * prior / P(flag). Check it by direct counting:
#         y[train["country_mismatch"] == 1].mean().
print("TODO 1: one flag by Bayes' rule and by counting")

# %% Step 3: every flag's likelihood ratio
# TODO 2: for each flag, P(flag | fraud) / P(flag | legit). Keep them in a Series `ratios`, sorted, and print it.
ratios = None
print("TODO 2: likelihood ratios")

# %% Step 4: two flags, the naive way and the real way
# TODO 3: the naive posterior for two flags multiplies the prior odds by both ratios:
#         odds = prior / (1 - prior) * ratios[f1] * ratios[f2]; posterior = odds / (1 + odds).
#         Compare it with direct counting for night + device_new and for big_amount + gift_card.
#         Which way is the naive answer wrong, and what does that say about the two flags among frauds?
print("TODO 3: naive against counted, for two pairs")

# %% Step 5: nine flags at once
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB

# TODO 4: fit BernoulliNB() and LogisticRegression() on train[flags]; print each one's test ROC AUC and AP.
# TODO 5: how many test payments does each put above 0.9, and how many of those are fraud?
print("TODO 4-5: BernoulliNB against logistic regression")

print("\nExpected when done: prior 0.0088; country_mismatch 0.137 against 0.0121, posterior 0.091 both ways;")
print("ratios from country_mismatch 11.3 down to new_account 1.9; naive against counted: night + device_new")
print("0.280 against 0.425 (193 payments), big_amount + gift_card 0.185 against 0.508 (59); BernoulliNB AUC 0.810,")
print("AP 0.439 (21 above 0.9, all fraud); logistic regression AUC 0.810, AP 0.435 (8 above 0.9, all fraud).")
