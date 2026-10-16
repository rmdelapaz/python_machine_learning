# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.1 A: the same estimator API on a new problem, card fraud.

Lesson 1.1 predicted churn. Here the payments team asks the same kind of question about online card
payments: which ones are fraud? Same moves as the lesson, a much rarer class. Work through TODO 1 to 6;
run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: features and target (done for you)
import pandas as pd
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[features]
y = tx["is_fraud"]
print(X.shape, f"fraud rate {y.mean():.3%}")

# %% Step 2: hold out 30% for testing
# TODO 1: train_test_split with test_size=0.3, stratify=y, random_state=42. Then print both shapes and
#         the number of frauds on each side (y_train.sum(), y_test.sum()).
X_train = X_test = y_train = y_test = None
if X_train is None:
    print("TODO 1: split 70/30, stratified, random_state=42")

# %% Step 3: the baseline that knows nothing
from sklearn.dummy import DummyClassifier

# TODO 2: fit DummyClassifier(strategy="most_frequent") on the training rows and print its test accuracy
#         with 4 decimals. Before you run it, write your prediction here: ______
print("TODO 2: baseline accuracy")

# %% Step 4: scale (fit on train only), then a logistic regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# TODO 3: StandardScaler().set_output(transform="pandas"): fit_transform on X_train, transform on X_test.
# TODO 4: fit LogisticRegression() on the scaled training rows. Print its test accuracy, then use
#         predict() to count how many test payments it flags and how many of those are really fraud.
print("TODO 3-4: scale, fit, accuracy, and how many frauds the 0.5 cut-off catches")

# %% Step 5: rank instead of yes/no
# TODO 5: add predict_proba(...)[:, 1] as a column p_fraud on a copy of X_test (keep y_test beside it),
#         take the 100 highest with nlargest, and count the real frauds among them. Compare with
#         100 * y_test.mean(), the number a random 100 would hold.
print("TODO 5: frauds among the 100 highest-risk payments vs a random 100")

# %% Step 6: a first impression of the weights
# TODO 6: print pd.Series(model.coef_[0], index=features).sort_values().round(2).
#         One weight looks backwards: which, and why? (Hint: when do night payments happen, as a number?)
print("TODO 6: the weights, sorted")

print("\nExpected when done: 35,000 / 15,000 rows with 307 and 131 frauds; baseline 0.9913; logistic")
print("regression 0.9937, flagging 40 payments of which 38 are fraud (93 frauds missed); 53 frauds among")
print("the 100 riskiest against 0.9 for a random 100; hour has a NEGATIVE weight.")
