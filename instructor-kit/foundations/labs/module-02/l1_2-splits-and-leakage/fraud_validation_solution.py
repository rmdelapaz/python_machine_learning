# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.2 B (SOLUTION): train, validation, test for a fraud review queue.

The payments team can review 100 payments out of every 10,000 by hand. Which model should fill that
queue? Lesson 1.2 chose a tree depth for churn with a validation set; here you choose between four
fraud models the same way, judging them by how many frauds land in the 100 riskiest payments, and you
open the test set exactly once.
"""
# %% Step 1: three parts, 60 / 20 / 20, stratified at every step
import pandas as pd
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
rest, test = train_test_split(tx, test_size=0.20, stratify=tx["is_fraud"], random_state=42)
train, val = train_test_split(rest, test_size=0.25, stratify=rest["is_fraud"], random_state=42)
for name, part in [("train", train), ("validation", val), ("test", test)]:
    print(f"{name:10s} {len(part):6,} rows  {part['is_fraud'].sum():3d} frauds")


# %% Step 2: the yardstick: how many frauds would 100 RANDOM payments hold?
def frauds_in_top(scores, outcome, n=100):
    """Real frauds among the n payments with the highest scores."""
    return int(pd.Series(outcome.to_numpy()).iloc[pd.Series(scores).nlargest(n).index].sum())


random_100 = 100 * val["is_fraud"].mean()
print(f"\nrandom 100 validation payments hold {random_100:.1f} frauds on average")

# %% Step 3: four candidates, fitted on train, compared on validation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler().fit(train[features])
candidates = {
    "logistic regression": LogisticRegression(),
    "tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "tree (depth 8)": DecisionTreeClassifier(max_depth=8, random_state=42),
    "random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
}
val_hits = {}
for name, model in candidates.items():
    model.fit(scaler.transform(train[features]), train["is_fraud"])
    p = model.predict_proba(scaler.transform(val[features]))[:, 1]
    val_hits[name] = frauds_in_top(p, val["is_fraud"])
    print(f"{name:20s} frauds in the validation top 100: {val_hits[name]}")
chosen = max(val_hits, key=val_hits.get)
print("chosen on validation:", chosen)

# %% Step 4: open the test set, once: refit the chosen recipe on train + validation
final_scaler = StandardScaler().fit(rest[features])
final = candidates[chosen].fit(final_scaler.transform(rest[features]), rest["is_fraud"])
p_test = final.predict_proba(final_scaler.transform(test[features]))[:, 1]
test_hits = frauds_in_top(p_test, test["is_fraud"])
print(f"\n{chosen}: {test_hits} frauds in the test top 100 "
      f"(validation said {val_hits[chosen]}; a random 100 holds {100 * test['is_fraud'].mean():.1f})")
accuracy_nothing = 1 - test["is_fraud"].mean()
print(f"for comparison, 'never fraud' is {accuracy_nothing:.4f} accurate and fills the queue with nobody")

# %% Step 5: the report, built from the variables
print(f"\nFraud review queue: {chosen}, chosen on {len(val):,} validation payments; "
      f"test top 100 holds {test_hits} frauds, {test_hits / (100 * test['is_fraud'].mean()):.0f}x a random 100.")
