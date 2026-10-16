# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.2 A (SOLUTION): Bayes' rule for fraud flags.

Lesson 4.2 counted words in reviews. A fraud analyst counts warning signs instead: a night-time payment, a
new device, a mismatched address. Use Bayes' rule on one flag, combine two flags the naive way and check
the answer against direct counting, then let BernoulliNB use nine flags at once.
"""
# %% Step 1: nine 0/1 warning signs and the stratified split
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

# %% Step 2: Bayes' rule for one flag, checked by counting
has = train["country_mismatch"] == 1
p_flag_fraud, p_flag_legit, p_flag = has[y == 1].mean(), has[y == 0].mean(), has.mean()
print(f"\nP(country_mismatch | fraud) {p_flag_fraud:.3f}   P(country_mismatch | legit) {p_flag_legit:.4f}")
print(f"posterior by Bayes' rule {p_flag_fraud * prior / p_flag:.3f}   by counting {y[has].mean():.3f}")

# %% Step 3: every flag's likelihood ratio
ratios = pd.Series({f: (train[f][y == 1].mean()) / (train[f][y == 0].mean()) for f in flags}).sort_values(ascending=False)
print("\nlikelihood ratio P(flag | fraud) / P(flag | legit):")
print(ratios.round(1).to_string())

# %% Step 4: two flags, the naive way and the real way
def naive_posterior(f1, f2):
    odds = prior / (1 - prior) * ratios[f1] * ratios[f2]
    return odds / (1 + odds)

print()
for f1, f2 in [("night", "device_new"), ("big_amount", "gift_card")]:
    both = (train[f1] == 1) & (train[f2] == 1)
    print(f"{f1} + {f2}: naive {naive_posterior(f1, f2):.3f}   counted {y[both].mean():.3f}   ({both.sum()} payments)")

# %% Step 5: BernoulliNB on all nine flags, against logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB

print()
for name, model in [("BernoulliNB", BernoulliNB()), ("logistic regression", LogisticRegression())]:
    p = model.fit(train[flags], y).predict_proba(test[flags])[:, 1]
    sure = p > 0.9
    print(f"{name:20s} ROC AUC {roc_auc_score(test['is_fraud'], p):.3f}   AP {average_precision_score(test['is_fraud'], p):.3f}"
          f"   above 0.9: {sure.sum()} payments, {test['is_fraud'][sure].mean():.0%} fraud")
