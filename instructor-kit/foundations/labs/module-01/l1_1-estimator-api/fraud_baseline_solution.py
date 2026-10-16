# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.1 A (SOLUTION): the same estimator API on a new problem, card fraud.

Lesson 1.1 predicted churn. Here the payments team asks the same kind of question about online card
payments: which ones are fraud? Same five moves (X and y, split, baseline, scale, fit and rank), a
much rarer class, and a sharper lesson about accuracy.
"""
# %% Step 1: features and target
import pandas as pd
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[features]
y = tx["is_fraud"]
print(X.shape, f"fraud rate {y.mean():.3%}")

# %% Step 2: hold out 30% for testing, keeping the fraud share equal on both sides
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(X_train.shape, X_test.shape, f"frauds: train {y_train.sum()}, test {y_test.sum()}")

# %% Step 3: the baseline that knows nothing
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"baseline accuracy: {baseline.score(X_test, y_test):.4f}  (frauds caught: 0 of {y_test.sum()})")

# %% Step 4: scale (fit on train only), then a logistic regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler().set_output(transform="pandas")
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = LogisticRegression().fit(X_train_s, y_train)
flagged = model.predict(X_test_s)
caught = int(((flagged == 1) & (y_test.to_numpy() == 1)).sum())
print(f"logistic regression accuracy: {model.score(X_test_s, y_test):.4f}")
print(f"flagged at the 0.5 cut-off: {flagged.sum()} payments, {caught} of them fraud, "
      f"{y_test.sum() - caught} frauds missed")

# %% Step 5: rank instead of yes/no: review the 100 riskiest payments
scored = X_test.assign(is_fraud=y_test, p_fraud=model.predict_proba(X_test_s)[:, 1])
top100 = scored.nlargest(100, "p_fraud")
expected = 100 * y_test.mean()
print(f"\nfrauds among the 100 highest-risk payments: {top100['is_fraud'].sum()} "
      f"(a random 100 would hold {expected:.1f})")

# %% Step 6: a first impression of the weights (scaled features, so roughly comparable)
weights = pd.Series(model.coef_[0], index=features).sort_values()
print()
print(weights.round(2).to_string())

# %% Stretch: swap in a random forest without changing anything else
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
forest.fit(X_train_s, y_train)
top_forest = scored.assign(p_forest=forest.predict_proba(X_test_s)[:, 1]).nlargest(100, "p_forest")
print(f"\nrandom forest accuracy: {forest.score(X_test_s, y_test):.4f}; "
      f"frauds among its 100 riskiest: {top_forest['is_fraud'].sum()}")
