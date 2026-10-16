# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.4 A: does a team of models catch more fraud than its best member?

Lesson 5.4 built a churn ensemble. Here the payments team asks the same question about card fraud,
where positives are rare (about 0.9%) and the model is trained on the past and used on the future:
train on January to May, judge on June. Make every choice (which team, which weights) on
out-of-fold predictions for January to May; open June once, at the end.
Work through TODO 1 to 5 and run the file after each one. It runs as it is: unfinished steps print
a reminder. The full run takes about 40 seconds.
"""
# %% Step 1: the time split and three different members (done for you)
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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% Step 2: out-of-fold probabilities and how different the members are
# TODO 1: for each (name, model) in members, put
#         cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
#         into a column of the DataFrame oof. Print each column's average precision against y_train,
#         then oof.corr().round(2). Which pair of members agrees most?
oof = None
if oof is None:
    print("TODO 1: out-of-fold probabilities for the three members, their AP and their correlations")

# %% Step 3: choose a soft vote on the out-of-fold predictions, never on June
teams = {"all three": ["logreg", "forest", "boost"], "forest + boost": ["forest", "boost"],
         "logreg + boost": ["logreg", "boost"]}
# TODO 2: for each team, the average precision of oof[cols].mean(axis=1) against y_train.
#         Keep the best team's name in best_team.
best_team = None
if best_team is None:
    print("TODO 2: out-of-fold AP of three soft votes; keep the best in best_team")

# %% Step 4: a stack learns the weights
# TODO 3: StackingClassifier(estimators=members, final_estimator=LogisticRegression(), cv=cv), fitted on
#         the January-to-May rows. Print stack.final_estimator_.coef_[0].round(2).
stack = None
if stack is None:
    print("TODO 3: fit the stack and print the meta-model weights")

# %% Step 5: open June once
# TODO 4: fit each member on all of January to May and put its June probabilities in a DataFrame
#         scores; add the chosen soft vote and the stack's probabilities as two more columns. For every
#         column print the June AP and the number of real frauds among its 100 highest scores
#         (y_test.loc[scores[c].nlargest(100).index].sum()). A random 100 would hold 100 * y_test.mean().
print("TODO 4: June AP and frauds in a 100-payment queue for all five contenders")

# %% Step 6: is the gain bigger than June's noise?
# TODO 5: a paired bootstrap, as in the lesson project: 500 resamples of the June rows
#         (rng = np.random.default_rng(42); idx = rng.integers(0, n, n)), each time the stack's AP minus
#         the best single member's AP. Print the mean and the 2.5th and 97.5th percentiles, and a verdict
#         printed from your variables.
print("TODO 5: paired bootstrap of the stack's gain, and a ship-or-skip verdict")

print("\nExpected when done: out-of-fold AP logreg 0.506, forest 0.513, boost 0.524, correlations 0.89 to 0.92;")
print("soft votes 0.541 (all three), 0.531, 0.539, so all three is chosen; meta weights about [5.04 4.45 5.5];")
print("June AP logreg 0.539, forest 0.509, boost 0.556, vote 0.570, stack 0.573, and every contender has 35 or 36")
print("frauds in its top 100 (0.8 at random); stack minus boost +0.016, 95% interval [-0.016, +0.057]: keep one model.")
