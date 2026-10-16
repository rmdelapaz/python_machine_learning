# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.4 A (SOLUTION): a lean fraud model.

Lesson 2.4 trimmed the churn model from 16 columns to 6. Here the payments team asks the same question
about fraud: nine raw columns plus five engineered ones (night, a night-on-a-new-device interaction,
gift cards, the log of the amount, brand-new accounts). Which of the fourteen earn their place? Same
tools as the lesson, a rare class, so the score is average precision (AP) instead of ROC AUC.
"""
# %% Step 1: features, engineered features and a stratified split
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
tx["night"] = (tx["hour"] < 6).astype(int)
tx["night_new_device"] = tx["night"] * tx["device_new"]
tx["gift_card"] = (tx["category"] == "Gift Cards").astype(int)
tx["log_amount"] = np.log(tx["amount"])
tx["new_account"] = (tx["account_age_days"] < 30).astype(int)
raw = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
       "country_mismatch", "card_attempts", "velocity_24h"]
engineered = ["night", "night_new_device", "gift_card", "log_amount", "new_account"]
candidates = raw + engineered

train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
y_train, y_test = train["is_fraud"], test["is_fraud"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(len(candidates), "candidates;", len(train), "training payments,", y_train.sum(), "frauds")

# %% Step 2: free wins first: which pairs move together?
corr = train[candidates].corr()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print("\npairs with |correlation| above 0.5:")
print(upper[upper.abs() > 0.5].sort_values(key=abs, ascending=False).round(2).to_string())

# %% Step 3: a filter ranks the columns one at a time
from sklearn.feature_selection import f_classif

f_scores, _ = f_classif(train[candidates], y_train)
print("\nF-test, top 6:", pd.Series(f_scores, index=candidates).sort_values(ascending=False).head(6).round(0).to_dict())

# %% Step 4: forward selection with the real model, judged by cross-validated AP
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_ap(columns):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_score(model, train[columns], y_train, cv=cv, scoring="average_precision")


full = cv_ap(candidates)
print(f"\nall 14 columns: CV AP {full.mean():.4f} +/- {full.std():.4f}")
print(f"raw 9 columns : CV AP {cv_ap(raw).mean():.4f}")

chosen, remaining, history = [], list(candidates), []
while remaining and len(chosen) < 10:
    step_scores = {col: cv_ap(chosen + [col]).mean() for col in remaining}
    best = max(step_scores, key=step_scores.get)
    chosen.append(best)
    remaining.remove(best)
    history.append(step_scores[best])
    print(f"{len(chosen):2d}  + {best:26s} CV AP {step_scores[best]:.4f}")

# %% Step 5: the tolerance rule (the folds disagree by about 0.06, so 0.01 is a modest tolerance)
path = pd.Series(history, index=range(1, len(history) + 1))
line = path.max() - 0.01
n_lean = int(path.index[path >= line].min())
lean = chosen[:n_lean]
print(f"\nbest {path.max():.4f}, tolerance line {line:.4f}: {n_lean} columns")
print(lean)

# %% Step 6: one look at the test set
from sklearn.metrics import average_precision_score

print()
for name, cols in [("raw 9", raw), ("all 14", candidates), (f"lean {n_lean}", lean)]:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[cols], y_train)
    ap = average_precision_score(y_test, model.predict_proba(test[cols])[:, 1])
    print(f"{name:7s} test AP {ap:.3f}")
print(f"random ranking AP {y_test.mean():.4f}")
