# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.1 A (SOLUTION): what drives the fraud model?

Lesson 7.1 explained the churn model. The payments team has a fraud model and the same three questions:
which signals does it rely on, does the answer depend on the metric, and how does risk move with the
hour of the day? Same tools (permutation importance, partial dependence), a different model and a much
rarer class.
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
ap_imp = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=42)
drivers = pd.DataFrame({"ap_drop": ap_imp.importances_mean, "ap_std": ap_imp.importances_std},
                       index=features).sort_values("ap_drop", ascending=False)
print(drivers.round(3))

# %% Step 3: the same model, another yardstick
auc_imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42)
drivers["auc_drop"] = pd.Series(auc_imp.importances_mean, index=features)
print(drivers[["ap_drop", "auc_drop"]].round(3).loc[["velocity_24h", "device_new", "hour", "country_mismatch"]])
# hour costs 0.07 of AP but nothing of AUC: it lifts a few night frauds to the very TOP of the list,
# which average precision rewards and the whole-ranking AUC hardly notices.

# %% Step 4: turn the hour dial (partial dependence on all 24 hours)
pd_hour = partial_dependence(model, X_test, ["hour"], method="brute", custom_values={"hour": list(range(24))})
risk = pd.Series(pd_hour["average"][0], index=range(24))
print("average predicted fraud by hour:", risk.round(3).tolist())
print(f"night (0-5) {risk.loc[0:5].mean():.4f}   day (6-23) {risk.loc[6:23].mean():.4f}")

# %% Step 5: does the hour matter the same for every payment? (new device or not)
for flag, label in [(0, "known device"), (1, "new device")]:
    group = X_test[X_test["device_new"] == flag]
    pd_g = partial_dependence(model, group, ["hour"], method="brute", custom_values={"hour": [2, 10, 15, 21]})
    print(f"{label:12s} ({len(group):,} payments)  hour 2, 10, 15, 21 ->", pd_g["average"][0].round(3).tolist())

# %% Step 6: check the curve against the data itself
band = pd.cut(X_test["hour"], [-1, 5, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"])
check = pd.DataFrame({"band": band, "actual": y_test, "predicted": p_test})
print(check.groupby("band", observed=True).agg(payments=("actual", "size"), actual=("actual", "mean"),
                                               predicted=("predicted", "mean")).round(4))
