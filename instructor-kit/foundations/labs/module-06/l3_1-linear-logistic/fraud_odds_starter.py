# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.1 B: odds ratios a fraud analyst can read.

Lesson 3.1 read churn coefficients as odds ratios per standard deviation. A fraud analyst wants sentences
in their own units: "a payment from a new device has N times the odds of fraud", "each extra $100
multiplies the odds by M". Build the model so the coefficients come out in those units. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1 (done for you): the stratified split from Lesson 2.5, plus a night flag and the amount in hundreds
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
# TODO 1: fit LogisticRegression() on train[["velocity_24h"]]. Print z = b0 + b1 * velocity_24h, then a table
#         for velocity 0, 1, 2, 3 and 5 with predict_proba(...)[:, 1] next to 1 / (1 + np.exp(-(b0 + b1 * v))).
#         Finish with np.exp(b1): the odds ratio per extra payment in 24 hours.
print("TODO 1: one-feature model and the sigmoid by hand")

# %% Step 3: flags and amounts in their own units
cols = ["device_new", "night", "billing_shipping_mismatch", "country_mismatch", "amount_100", "card_attempts"]
# TODO 2: fit LogisticRegression(C=np.inf, max_iter=1000) on train[cols] WITHOUT a scaler, so every coefficient
#         is "per one unit": per flag, per $100, per card attempt. Print coef_ and np.exp(coef_) side by side.
#         Write one sentence an analyst could say about device_new.
print("TODO 2: odds ratios in the data's own units")

# %% Step 4: hour as a straight line against the night flag
# TODO 3: fit one model on train[["hour"]] and one on train[["night"]]; print each coefficient and odds ratio.
#         Why is the hour coefficient negative when fraud clusters at night?
print("TODO 3: hour against night")

# %% Step 5: the 0.5 rule is the line z = 0
from sklearn.metrics import average_precision_score

# TODO 4: check that model.predict(test[cols]) equals (model.decision_function(test[cols]) > 0).
# TODO 5: how many test payments does it flag at 0.5, how many are fraud, and what is the test AP?
print("TODO 4-5: decision_function, flags and AP")

print("\nExpected when done: z = -5.984 + 1.515 * velocity_24h; p from 0.0025 (0) to 0.8308 (5), by hand the")
print("same; odds ratio 4.55 per extra payment; odds ratios device_new 7.11, night 6.93, billing mismatch 3.91,")
print("country mismatch 8.94, per $100 1.31, per card attempt 6.31; hour -0.111 (0.90) against night +2.354")
print("(10.53); predict == (z > 0) True; 21 flagged, 18 fraud; test AP 0.332.")
