# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.3 A: can LIME explain the fraud model?

Lesson 7.3 explained a churn customer with LIME and checked fidelity and stability. Here the same checks
meet a much harder case: fraud is rare (0.88%), so most random variants of any payment look legitimate.
Explain three payments at different risk levels, test fidelity and stability, and compare with SHAP. Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1: the fraud model from Session 1's labs (done for you)
import warnings

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)      # lime's own feature-name chatter
tx = pd.read_csv(DATA / "transactions.csv")
features = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
X_train, X_test = train[features].astype(float), test[features].astype(float)
model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       early_stopping=True, random_state=42).fit(X_train, train["is_fraud"])
test = test.assign(p_fraud=model.predict_proba(X_test)[:, 1])
ranked = test.sort_values("p_fraud", ascending=False)
picks = ranked.iloc[[0, 49, 199]]                  # the riskiest, the 50th and the 200th payment
print(picks[["transaction_id", "p_fraud", "is_fraud"]].round(3).to_string(index=False))


def predict_fn(rows):
    return model.predict_proba(pd.DataFrame(rows, columns=features))


# %% Step 2: one explanation, with its fidelity
# TODO 1: explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=features,
#         class_names=["legit", "fraud"], random_state=42). (No missing values here, so no filling.)
#         Explain the riskiest payment (picks.index[0]) with num_features=4, num_samples=5000. Print each rule
#         and weight, then exp.score (fidelity), exp.local_pred[0] (the surrogate) and the model's probability.
#         Before you run it: will the surrogate land near the model's 0.990?
print("TODO 1: LIME for the riskiest payment, with fidelity and the surrogate's prediction")

# %% Step 3: stability, with few and with many variants
# TODO 2: for num_samples 500 and 5000, and for each of the three picks, explain with random_state 1, 2, 3
#         (num_features=3). Print how many of the 3 seeds agree on the first rule, and the three fidelities.
print("TODO 2: stability across seeds at 500 and 5,000 variants")

# %% Step 4: a second opinion from SHAP
# TODO 3: shap.TreeExplainer(model)(X_test.loc[picks.index]); the top feature per payment (idxmax(axis=1)).
# TODO 4 (think): which of the two explanations would you hand to the review team, and why? What makes
#         LIME struggle here that did not hurt it as much on churn (16.5% positives)?
print("TODO 3: SHAP's top feature for the same three payments")

print("\nExpected when done: T031633 (0.990), T033385 (0.567), T028586 (0.057); for T031633 the rules are")
print("velocity_24h > 1.00 +0.032, country_mismatch > 0.00 +0.031 ..., fidelity 0.12 and a surrogate of 0.077")
print("against the model's 0.990; at 500 variants the top rule agrees in 3, 2 and 1 of 3 seeds, at 5,000 in")
print("2, 3 and 3, with fidelities between 0.05 and 0.15; SHAP: velocity_24h, velocity_24h, device_new.")
