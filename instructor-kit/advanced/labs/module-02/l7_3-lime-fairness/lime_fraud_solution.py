# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.3 A (SOLUTION): can LIME explain the fraud model?

Lesson 7.3 explained a churn customer with LIME and checked fidelity and stability. Here the same checks
meet a much harder case: fraud is rare (0.88%), so most random variants of any payment look legitimate.
Explain three payments at different risk levels, test fidelity and stability, and compare with SHAP.
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
explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=features,
                                 class_names=["legit", "fraud"], random_state=42)
top = picks.index[0]
e = explainer.explain_instance(X_test.loc[top].to_numpy(), predict_fn, num_features=4, num_samples=5000)
for rule, w in e.as_list():
    print(f"  {rule:35s} {w:+.3f}")
print(f"fidelity {e.score:.2f}   surrogate {e.local_pred[0]:.3f}   model {test.loc[top, 'p_fraud']:.3f}")

# %% Step 3: stability, with few and with many variants
for n_samples in [500, 5000]:
    for label, row in picks.iterrows():
        tops, fids = [], []
        for seed in [1, 2, 3]:
            ex = LimeTabularExplainer(X_train.to_numpy(), feature_names=features,
                                      class_names=["legit", "fraud"], random_state=seed)
            ee = ex.explain_instance(X_test.loc[label].to_numpy(), predict_fn, num_features=3, num_samples=n_samples)
            tops.append(ee.as_list()[0][0])
            fids.append(round(ee.score, 2))
        same = max(tops.count(t) for t in tops)
        print(f"num_samples={n_samples:4d} {row['transaction_id']}: top rule agrees in {same} of 3 seeds, "
              f"fidelity {fids}, first rule '{tops[0]}'")

# %% Step 4: a second opinion from SHAP
sv = shap.TreeExplainer(model)(X_test.loc[picks.index])
shap_top = pd.DataFrame(sv.values, columns=features, index=picks["transaction_id"]).idxmax(axis=1)
print("SHAP's top feature:", shap_top.to_dict())
