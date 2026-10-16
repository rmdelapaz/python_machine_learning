# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.1 B (SOLUTION): odds ratios a fraud analyst can read.

Lesson 3.1 read churn coefficients as odds ratios per standard deviation. A fraud analyst wants sentences
in their own units: "a payment from a new device has N times the odds of fraud", "each extra $100
multiplies the odds by M". Build the model so the coefficients come out in those units.
"""
# %% Step 1: the stratified split from Lesson 2.5, plus a night flag and the amount in hundreds
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
tx["night"] = (tx["hour"] < 6).astype(int)
tx["amount_100"] = tx["amount"] / 100
train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
y_train, y_test = train["is_fraud"], test["is_fraud"]
print(len(train), "training payments,", y_train.sum(), "frauds")

# %% Step 2: one feature, the sigmoid by hand
one = LogisticRegression().fit(train[["velocity_24h"]], y_train)
b0, b1 = one.intercept_[0], one.coef_[0, 0]
print(f"\nz = {b0:.3f} + {b1:.3f} * velocity_24h")
grid = pd.DataFrame({"velocity_24h": [0, 1, 2, 3, 5]})
grid["p_fraud"] = one.predict_proba(grid)[:, 1].round(4)
grid["by_hand"] = (1 / (1 + np.exp(-(b0 + b1 * grid["velocity_24h"])))).round(4)
print(grid.to_string(index=False))
print(f"odds ratio per extra payment in 24 hours: {np.exp(b1):.2f}")

# %% Step 3: flags and amounts in their own units (no scaler, so each coefficient is "per 1 unit")
cols = ["device_new", "night", "billing_shipping_mismatch", "country_mismatch", "amount_100", "card_attempts"]
model = LogisticRegression(C=np.inf, max_iter=1000).fit(train[cols], y_train)
odds = pd.DataFrame({"coef": model.coef_[0], "odds_ratio": np.exp(model.coef_[0])}, index=cols)
print()
print(odds.round(2).to_string())

# %% Step 4: hour as a straight line against the night flag
for name, feats in [("hour as a number", ["hour"]), ("night flag", ["night"])]:
    m = LogisticRegression().fit(train[feats], y_train)
    print(f"{name:17s} coef {m.coef_[0, 0]:+.3f}   odds ratio {np.exp(m.coef_[0, 0]):.2f}")

# %% Step 5: the 0.5 rule is the line z = 0
from sklearn.metrics import average_precision_score

z = model.decision_function(test[cols])
pred = model.predict(test[cols])
print(f"\npredict == (z > 0): {(pred == (z > 0).astype(int)).all()}")
print(f"flagged at 0.5: {pred.sum()}, of which fraud: {int(y_test[pred == 1].sum())}")
print(f"test AP {average_precision_score(y_test, model.predict_proba(test[cols])[:, 1]):.3f}")
