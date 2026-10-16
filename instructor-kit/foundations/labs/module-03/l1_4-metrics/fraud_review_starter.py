# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.4 A: a review queue for card fraud, chosen from costs.

Lesson 1.4 chose a retention threshold from a profit. Here the payments team asks the mirror-image
question about card fraud, framed as a COST to keep low: a reviewer checks each flagged payment
($10 of staff time), and every fraud that slips through costs about $375 (chargeback, fees, goods).
Work through TODO 1 to 6; run the file after each one. It runs as it is: unfinished steps print a
reminder.
"""
# %% Step 1: split and fit (done for you: the model from Lesson 1.1's fraud lab, now in a pipeline)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(
    tx[features], tx["is_fraud"], test_size=0.3, stratify=tx["is_fraud"], random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print(f"test payments {len(y_test):,}, frauds {y_test.sum()}")

# %% Step 2: the confusion matrix at the default threshold, in business words
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

y_pred = model.predict(X_test)
# TODO 1: tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel(); print the four counts, then
#         precision, recall and F1 with 3 decimals (truth first, prediction second!).
#         Then one sentence in words: how many payments go to review, how many of them are fraud,
#         and how many frauds slip through?
print("TODO 1: confusion matrix, precision, recall, F1, and one sentence in words")

# %% Step 3: ranking quality: ROC AUC against average precision (and its baseline)
from sklearn.metrics import average_precision_score, roc_auc_score

# TODO 2: print ROC AUC and average precision from `proba` (scores, not labels), and the fraud rate
#         y_test.mean(), which is what a RANDOM ranking scores on average precision.
#         Before you run it: which of the two will look more comfortable? ______
print("TODO 2: ROC AUC, average precision, and the fraud rate")

# %% Step 4: the cost of a review policy, and a threshold chosen on TRAINING data only
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REVIEW_COST = 10    # dollars of staff time per payment sent to review
FRAUD_LOSS = 375    # dollars lost per fraud that is NOT reviewed


def review_cost(y_true, flagged):
    """Total cost of a policy: every review costs REVIEW_COST, every missed fraud FRAUD_LOSS."""
    y_true, flagged = np.asarray(y_true), np.asarray(flagged)
    # TODO 3: count the missed frauds (flagged 0, really 1) and return
    #         int(REVIEW_COST * number_flagged + FRAUD_LOSS * missed)
    return None


cv = StratifiedKFold(5, shuffle=True, random_state=42)
# TODO 4: oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
#         Compute review_cost(y_train, (oof >= t).astype(int)) for t in np.round(np.arange(0.01, 0.51, 0.01), 2)
#         and keep the CHEAPEST threshold as best_t. Print it with its cost, the cost at 0.5, and the
#         break-even probability REVIEW_COST / FRAUD_LOSS. Never use the test set here.
best_t = None
if review_cost([1, 0], [0, 1]) is None:
    print("TODO 3: finish review_cost (check: review_cost([1, 0], [0, 1]) should be 385)")
if best_t is None:
    print("TODO 4: out-of-fold probabilities, the cost per threshold, and the cheapest threshold")

# %% Step 5: open the test set once and compare four policies
# TODO 5: for "threshold best_t", "threshold 0.50", "review everything" (all ones) and "review nothing"
#         (all zeros), print the number of reviews, precision (zero_division=0), recall and the cost.
print("TODO 5: four policies on the test set: reviews, precision, recall, cost")

# %% Step 6: make the threshold part of the model
from sklearn.model_selection import FixedThresholdClassifier

# TODO 6: wrap the model in FixedThresholdClassifier(model, threshold=best_t), fit it on the training
#         rows, and print how many test payments its predict() sends to review.
print("TODO 6: FixedThresholdClassifier")

# Stretch: TunedThresholdClassifierCV(model, scoring=make_scorer(review_cost, greater_is_better=False),
#          cv=cv). Why greater_is_better=False? Compare its threshold and test cost with yours.

print("\nExpected when done: TN 14867 FP 2 FN 93 TP 38; precision 0.950, recall 0.290, F1 0.444; ROC AUC 0.848,")
print("average precision 0.444 against a fraud rate of 0.0087; break-even 0.027; cheapest training threshold")
print("0.04 (cost 52,320 vs 75,380 at 0.5); on the test set it reviews 390 payments, recall 0.580, cost 24,525,")
print("against 35,275 at 0.5, 49,125 reviewing nothing and 150,000 reviewing everything; the wrapped model")
print("sends 390 to review. Stretch: tuned threshold 0.040, test cost 24,485.")
