# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.2 B: train, validation, test for a fraud review queue.

The payments team can review 100 payments out of every 10,000 by hand. Which model should fill that
queue? Lesson 1.2 chose a tree depth for churn with a validation set; here you choose between four
fraud models the same way, judging them by how many frauds land in the 100 riskiest payments, and you
open the test set exactly once. Work through TODO 1 to 4; it runs as it is.
"""
# %% Step 1: three parts, 60 / 20 / 20, stratified at every step
import pandas as pd
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
# TODO 1: carve off test first (test_size=0.20, stratify on is_fraud, random_state=42), then split the
#         rest into train and val (test_size=0.25, stratify on the REST's is_fraud, random_state=42).
#         Print the rows and the number of frauds in each part.
rest = test = train = val = None
if train is None:
    print("TODO 1: 60/20/20 split, stratified twice")


# %% Step 2: the yardstick (done for you): how many frauds would 100 RANDOM payments hold?
def frauds_in_top(scores, outcome, n=100):
    """Real frauds among the n payments with the highest scores."""
    return int(pd.Series(outcome.to_numpy()).iloc[pd.Series(scores).nlargest(n).index].sum())


if val is not None:
    print(f"random 100 validation payments hold {100 * val['is_fraud'].mean():.1f} frauds on average")

# %% Step 3: four candidates, fitted on train, compared on validation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

candidates = {
    "logistic regression": LogisticRegression(),
    "tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "tree (depth 8)": DecisionTreeClassifier(max_depth=8, random_state=42),
    "random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
}
# TODO 2: fit a StandardScaler on train[features] only. For each candidate: fit on the scaled training
#         rows, take predict_proba(...)[:, 1] on the scaled VALIDATION rows, and store
#         frauds_in_top(p, val["is_fraud"]) in a dict val_hits. Print each, then the winner (max).
#         Do not look at the test set.
val_hits = {}
if not val_hits:
    print("TODO 2: frauds in the validation top 100 for each candidate; choose one")

# %% Step 4: open the test set, once
# TODO 3: refit a NEW scaler and the chosen model on rest (train + validation), score the test rows,
#         and print the frauds in the test top 100 next to the validation number and a random 100.
# TODO 4: why is the test number lower than the validation number, and why must you NOT now go back and
#         pick the runner-up? Write two sentences in a comment.
print("TODO 3-4: one look at the test set, and two sentences")

print("\nExpected when done: 30,000 / 10,000 / 10,000 rows with 263, 87 and 88 frauds; a random 100 holds 0.9;")
print("validation top 100: logistic regression 45, tree depth 3 30, depth 8 34, random forest 47 (chosen);")
print("test top 100: 41 frauds for the forest, 47x a random 100.")
