# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.4 A (SOLUTION): a review queue for card fraud, chosen from costs.

Lesson 1.4 chose a retention threshold from a profit. Here the payments team asks the mirror-image
question about card fraud, framed as a COST to keep low: a reviewer checks each flagged payment
($10 of staff time), and every fraud that slips through costs about $375 (chargeback, fees, goods).
Same tools as the lesson: confusion matrix, precision and recall, the threshold, ROC AUC against
average precision, and a threshold chosen on out-of-fold training probabilities, tested once.
"""
# %% Step 1: split and fit (the model from Lesson 1.1's fraud lab, now in a pipeline)
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
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nTN {tn}  FP {fp}  FN {fn}  TP {tp}")
print(f"precision {precision_score(y_test, y_pred):.3f}  recall {recall_score(y_test, y_pred):.3f}  "
      f"F1 {f1_score(y_test, y_pred):.3f}")
print(f"In words: of {tp + fp} payments sent to review, {tp} are fraud; {fn} of {tp + fn} frauds slip through.")

# %% Step 3: ranking quality: ROC AUC against average precision (and its baseline)
from sklearn.metrics import average_precision_score, roc_auc_score

print(f"\nROC AUC {roc_auc_score(y_test, proba):.3f}   average precision {average_precision_score(y_test, proba):.3f}   "
      f"fraud rate (AP of a random ranking) {y_test.mean():.4f}")

# %% Step 4: the cost of a review policy, and a threshold chosen on TRAINING data only
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REVIEW_COST = 10    # dollars of staff time per payment sent to review
FRAUD_LOSS = 375    # dollars lost per fraud that is NOT reviewed


def review_cost(y_true, flagged):
    """Total cost of a policy: every review costs REVIEW_COST, every missed fraud FRAUD_LOSS."""
    y_true, flagged = np.asarray(y_true), np.asarray(flagged)
    missed = ((flagged == 0) & (y_true == 1)).sum()
    return int(REVIEW_COST * flagged.sum() + FRAUD_LOSS * missed)


cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2)
costs = np.array([review_cost(y_train, (oof >= t).astype(int)) for t in thresholds])
best_t = thresholds[costs.argmin()]
print(f"\nbreak-even probability {REVIEW_COST / FRAUD_LOSS:.3f}")
print(f"cheapest threshold on the training payments (out-of-fold): {best_t:.2f}, cost {costs.min():,} "
      f"(threshold 0.5: {costs[thresholds == 0.5][0]:,})")

# %% Step 5: open the test set once and compare four policies
policies = {
    f"threshold {best_t:.2f}": (proba >= best_t).astype(int),
    "threshold 0.50": y_pred,
    "review everything": np.ones(len(y_test), dtype=int),
    "review nothing": np.zeros(len(y_test), dtype=int),
}
print()
for name, flagged in policies.items():
    print(f"{name:17s} reviews {flagged.sum():6,}   precision {precision_score(y_test, flagged, zero_division=0):.3f}   "
          f"recall {recall_score(y_test, flagged):.3f}   cost {review_cost(y_test, flagged):7,}")

# %% Step 6: make the threshold part of the model
from sklearn.model_selection import FixedThresholdClassifier

queue_model = FixedThresholdClassifier(model, threshold=best_t).fit(X_train, y_train)
print(f"\nFixedThresholdClassifier sends {queue_model.predict(X_test).sum()} test payments to review")

# %% Stretch: let TunedThresholdClassifierCV search (a cost is minimized: greater_is_better=False)
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

tuned = TunedThresholdClassifierCV(model, scoring=make_scorer(review_cost, greater_is_better=False),
                                   cv=cv).fit(X_train, y_train)
print(f"TunedThresholdClassifierCV threshold {tuned.best_threshold_:.3f}, "
      f"test cost {review_cost(y_test, tuned.predict(X_test)):,}")
