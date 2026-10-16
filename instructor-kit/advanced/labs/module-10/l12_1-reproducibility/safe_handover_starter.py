# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 12.1 B: hand a fraud model to another team, safely and provably.

The payments team will run your fraud model on their own server. They will not accept a pickle, and
they want proof of what the model is. Save it with skops, review its untrusted types (and see how the
list depends on the model), write a model card, then play the receiving team: rebuild the score from
the card and the files alone, and catch a data file that is no longer the one you trained on.
Step 1 is done for you. Work through TODO 1 to 6; run the file after each one. It runs as it is:
unfinished steps print a reminder. Everything you save goes to OUTPUT.
"""
# %% Step 1: the model (time split: January to May trains, June tests)
import hashlib
import json
import shutil
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG = {"data_file": "transactions.csv", "target": "is_fraud", "test_month": 6, "seed": 42,
          "numeric": ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
                      "country_mismatch", "card_attempts", "velocity_24h"],
          "categorical": ["category"]}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1_000_000), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path, cfg):
    tx = pd.read_csv(path, parse_dates=["timestamp"])
    is_test = tx["timestamp"].dt.month == cfg["test_month"]
    X = tx[cfg["numeric"] + cfg["categorical"]]
    return X[~is_test], X[is_test], tx.loc[~is_test, cfg["target"]], tx.loc[is_test, cfg["target"]]


def build_pipeline(model):
    prep = ColumnTransformer([("num", StandardScaler(), CONFIG["numeric"]),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CONFIG["categorical"])])
    return Pipeline([("prep", prep), ("model", model)])


X_train, X_test, y_train, y_test = load_split(DATA / CONFIG["data_file"], CONFIG)
model = build_pipeline(HistGradientBoostingClassifier(random_state=CONFIG["seed"])).fit(X_train, y_train)
test_ap = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"train {len(X_train):,}, test (June) {len(X_test):,}; test average precision {test_ap:.4f}")

# %% Step 2: save with skops; a load without a reviewed list is refused
import skops.io as sio
from skops.io.exceptions import UntrustedTypesFoundException

model_path = OUTPUT / "fraud_hgb.skops"
# TODO 1: sio.dump(model, model_path) and print sio.get_untrusted_types(file=model_path). Then try
#         sio.load(model_path) with NO trusted list inside try/except UntrustedTypesFoundException and print
#         the exception's class name. Finally write the reviewed list out BY HAND as `trusted`, load with
#         trusted=trusted, and check the reloaded model gives identical predict_proba on X_test.
trusted = None
if trusted is None:
    print("TODO 1: dump, review, refuse, load")

# %% Step 3: the list depends on the model
# TODO 2: fit build_pipeline(LogisticRegression(max_iter=1000)) and build_pipeline(RandomForestClassifier(
#         n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=42)), save each to OUTPUT and print
#         their untrusted types. Predict first: which one needs no review at all? ______
print("TODO 2: two more trust lists")

# %% Step 4: the model card
# TODO 3: a dict with model_file (just the file name), estimator, config, data_sha256, test_average_precision
#         (rounded to 4), environment (versions of scikit-learn, pandas, numpy, skops), skops_trusted_types
#         and intended_use. Write it to OUTPUT / "fraud_hgb.json" with json.dumps(card, indent=2).
card_path = OUTPUT / "fraud_hgb.json"
print("TODO 3: the model card")

# %% Step 5: the receiving team: rebuild the score from the card and the files alone
# TODO 4: write verify(card_file, data_folder) that reads the card, compares the data file's fingerprint
#         with the card, loads the model with the card's trusted list, re-splits with the card's config
#         (load_split) and returns (same_data, reproduced, average precision rounded to 4).
#         Run it on DATA.
print("TODO 4: verify from the card")

# %% Step 6: a copy, a re-save, and a one-cent edit of the data
# TODO 5: in OUTPUT / "received", verify against (a) a shutil.copy of the CSV, (b) the CSV read and re-saved
#         with pandas (index=False), (c) the same with 0.01 added to the first row's amount.
# TODO 6: in a comment: case (c) still reproduces the score. What does the fingerprint tell you, and what not?
print("TODO 5-6: three versions of the data")

# %% Stretch: compare the file sizes of the skops file and a joblib.dump of the same model.

print("\nExpected when done: untrusted types ['sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor'];")
print("UntrustedTypesFoundException without a list; same predictions True; logistic regression []; random")
print("forest ['sklearn.tree._tree.Tree']; verify on DATA (True, True, 0.4743); byte copy (True, True, 0.4743);")
print("pandas re-save (True, True, 0.4743); one cent changed (False, True, 0.4743). Stretch: skops 462 kB, joblib 142 kB.")
