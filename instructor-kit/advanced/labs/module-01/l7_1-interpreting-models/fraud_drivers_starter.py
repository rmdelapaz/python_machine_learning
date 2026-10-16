# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.1 A: what drives the fraud model?

Lesson 7.1 explained the churn model. The payments team has a fraud model and the same three questions:
which signals does it rely on, does the answer depend on the metric, and how does risk move with the
hour of the day? Same tools (permutation importance, partial dependence), a different model and a much
rarer class. Work through TODO 1 to 5; the file runs as it is, and unfinished steps print a reminder.
"""
# %% Step 1: the fraud model (stratified 70/30 split, a regularized booster)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance, partial_dependence

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[features].astype(float)              # float columns: partial dependence refuses integers
y = tx["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       early_stopping=True, random_state=42).fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
print(f"test ROC AUC {roc_auc_score(y_test, p_test):.3f}   average precision {average_precision_score(y_test, p_test):.3f}"
      f"   (a random ranking: AP {y_test.mean():.3f})")

# %% Step 2: permutation importance with the metric the review team cares about
# The review team checks the riskiest payments first, so use scoring="average_precision".
# TODO 1: permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=5,
#         random_state=42). Put importances_mean and importances_std in a DataFrame called drivers
#         (columns ap_drop, ap_std; index = features), sorted from largest drop down, and print it.
drivers = None
if drivers is None:
    print("TODO 1: permutation importance on the test set, scored by average precision")

# %% Step 3: the same model, another yardstick
# TODO 2: repeat with scoring="roc_auc" and add the means as a column auc_drop. Print both columns for
#         velocity_24h, device_new, hour and country_mismatch. Which feature changes most, and why might
#         a metric that focuses on the top of the list see it differently?
print("TODO 2: the same importances scored by ROC AUC")

# %% Step 4: turn the hour dial (partial dependence on all 24 hours)
# TODO 3: partial_dependence(model, X_test, ["hour"], method="brute", custom_values={"hour": list(range(24))}).
#         Print the 24 averages rounded to 3, then the mean for night (hours 0-5) and day (6-23).
print("TODO 3: partial dependence of the predicted fraud rate on the hour")

# %% Step 5: does the hour matter the same for every payment?
# TODO 4: compute the same partial dependence separately for payments from a known device
#         (device_new == 0) and a new device (== 1), at hours 2, 10, 15 and 21. Is this an interaction?
print("TODO 4: the hour curve for known and new devices")

# %% Step 6: check the curve against the data itself
# TODO 5: cut hour into night (0-5), morning (6-11), afternoon (12-17), evening (18-23) with
#         pd.cut(X_test["hour"], [-1, 5, 11, 17, 23], labels=[...]) and print, per band, the number of
#         payments, the real fraud rate and the average predicted probability (p_test).
print("TODO 5: real and predicted fraud rate by time of day")

print("\nExpected when done: AP drops led by velocity_24h 0.242, then account_age_days 0.120, device_new 0.116,")
print("amount 0.102, hour 0.074; by ROC AUC hour drops to -0.005 (about zero). Partial dependence: night 0.0174")
print("vs day 0.0072; known devices 0.006 at 2 a.m., new devices 0.122 at 2 a.m. vs 0.028 at 10 a.m.;")
print("real fraud at night 6.40% (predicted 5.89%) against under 1% in every daytime band.")
