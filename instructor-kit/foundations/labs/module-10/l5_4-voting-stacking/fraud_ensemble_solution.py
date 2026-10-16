# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.4 A (solution): does a team of models catch more fraud than its best member?

Lesson 5.4 built a churn ensemble. Here the payments team asks the same question about card fraud,
where positives are rare (about 0.9%) and the model is trained on the past and used on the future:
train on January to May, judge on June. Every choice (which members, which team) is made on
out-of-fold predictions for January to May; June is opened once, at the end.
"""
# %% Step 1: the time split and three different members
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
num = ["account_age_days", "amount", "hour", "items", "device_new",
       "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[num + ["category"]].copy()
X["category"] = X["category"].astype("category")
y = tx["is_fraud"]
is_june = tx["timestamp"] >= "2025-06-01"
X_train, X_test, y_train, y_test = X[~is_june], X[is_june], y[~is_june], y[is_june]
print(f"Jan-May: {len(X_train):,} payments, {y_train.sum()} frauds | June: {len(X_test):,} payments, {y_test.sum()} frauds")

encode = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["category"]), remainder="passthrough")
members = [
    ("logreg", make_pipeline(make_column_transformer((StandardScaler(), num),
                                                     (OneHotEncoder(handle_unknown="ignore"), ["category"])),
                             LogisticRegression(max_iter=1000))),
    ("forest", make_pipeline(encode, RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                                            n_jobs=-1, random_state=42))),
    ("boost", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             early_stopping=True, random_state=42)),
]

# %% Step 2 (TODO 1): out-of-fold probabilities and how different the members are
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = pd.DataFrame(index=X_train.index)
for name, model in members:
    oof[name] = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
print("\nout-of-fold average precision (Jan-May):")
for name in oof:
    print(f"  {name:7s} {average_precision_score(y_train, oof[name]):.3f}")
print("correlation of out-of-fold probabilities:")
print(oof.corr().round(2))

# %% Step 3 (TODO 2): choose a soft vote on the out-of-fold predictions, never on June
teams = {"all three": ["logreg", "forest", "boost"], "forest + boost": ["forest", "boost"],
         "logreg + boost": ["logreg", "boost"]}
print("\nsoft votes, out-of-fold average precision:")
oof_ap = {}
for team, cols in teams.items():
    oof_ap[team] = average_precision_score(y_train, oof[cols].mean(axis=1))
    print(f"  {team:15s} {oof_ap[team]:.3f}")
best_team = max(oof_ap, key=oof_ap.get)
print("chosen on Jan-May:", best_team)

# %% Step 4 (TODO 3): a stack learns the weights from the same kind of out-of-fold predictions
stack = StackingClassifier(estimators=members, final_estimator=LogisticRegression(), cv=cv)
stack.fit(X_train, y_train)
print("\nmeta-model weights (logreg, forest, boost):", stack.final_estimator_.coef_[0].round(2))

# %% Step 5 (TODO 4): open June once: every contender, AP and frauds in a 100-payment review queue
scores = pd.DataFrame(index=X_test.index)
for name, model in members:
    scores[name] = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
scores["vote: " + best_team] = scores[teams[best_team]].mean(axis=1)
scores["stack"] = stack.predict_proba(X_test)[:, 1]
print(f"\nJune (opened once), {y_test.sum()} frauds; a random 100 would hold {100 * y_test.mean():.1f}:")
for c in scores:
    caught = int(y_test.loc[scores[c].nlargest(100).index].sum())
    print(f"  {c:22s} AP {average_precision_score(y_test, scores[c]):.3f}   frauds in top 100: {caught}")

# %% Step 6 (TODO 5): is the stack's gain over the best single member bigger than June's noise?
best_single = max(["logreg", "forest", "boost"], key=lambda c: average_precision_score(y_test, scores[c]))
rng = np.random.default_rng(42)
y_arr = y_test.to_numpy()
a, b = scores["stack"].to_numpy(), scores[best_single].to_numpy()
diffs = []
for _ in range(500):
    idx = rng.integers(0, len(y_arr), len(y_arr))
    diffs.append(average_precision_score(y_arr[idx], a[idx]) - average_precision_score(y_arr[idx], b[idx]))
low, high = np.percentile(diffs, [2.5, 97.5])
print(f"\nstack minus {best_single}: {np.mean(diffs):+.3f} AP, 95% interval [{low:+.3f}, {high:+.3f}]")
print(f"Verdict: {'ship the stack' if low > 0 else 'keep the single model (' + best_single + ')'}: "
      f"{'the gain is clear of zero' if low > 0 else 'June cannot tell them apart, and the stack costs 19 fits instead of 1'}")
