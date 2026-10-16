# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 12.1 B (SOLUTION): hand a fraud model to another team, safely and provably.

The payments team will run your fraud model on their own server. They will not accept a pickle, and
they want proof of what the model is. Save it with skops, review its untrusted types (and see how the
list depends on the model), write a model card, then play the receiving team: rebuild the score from
the card and the files alone, and catch a data file that is no longer the one you trained on.
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
sio.dump(model, model_path)
print("untrusted types:", sio.get_untrusted_types(file=model_path))
try:
    sio.load(model_path)
except UntrustedTypesFoundException as e:
    print("load without trusted=:", type(e).__name__)

trusted = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]   # reviewed: HGB trees
reloaded = sio.load(model_path, trusted=trusted)
print("same predictions after reload:", np.array_equal(reloaded.predict_proba(X_test), model.predict_proba(X_test)))

# %% Step 3: the list depends on the model
others = {"logistic regression": LogisticRegression(max_iter=1000),
          "random forest": RandomForestClassifier(n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=CONFIG["seed"])}
for name, est in others.items():
    path = OUTPUT / f"fraud_{name.split()[0]}.skops"
    sio.dump(build_pipeline(est).fit(X_train, y_train), path)
    print(f"{name:20s} untrusted types: {sio.get_untrusted_types(file=path)}")

# %% Step 4: the model card
card = {
    "model_file": model_path.name,
    "estimator": type(model[-1]).__name__,
    "config": CONFIG,
    "data_sha256": file_sha256(DATA / CONFIG["data_file"]),
    "test_average_precision": round(test_ap, 4),
    "environment": {pkg: version(pkg) for pkg in ["scikit-learn", "pandas", "numpy", "skops"]},
    "skops_trusted_types": trusted,
    "intended_use": "Rank June-style payments for manual review; not for automatic account blocking.",
}
card_path = OUTPUT / "fraud_hgb.json"
card_path.write_text(json.dumps(card, indent=2))
print("card keys:", list(card))


# %% Step 5: the receiving team: rebuild the score from the card and the files alone
def verify(card_file, data_folder):
    c = json.loads(card_file.read_text())
    data_path = data_folder / c["config"]["data_file"]
    same_data = file_sha256(data_path) == c["data_sha256"]
    m = sio.load(card_file.parent / c["model_file"], trusted=c["skops_trusted_types"])
    _, X_te, _, y_te = load_split(data_path, c["config"])
    ap = average_precision_score(y_te, m.predict_proba(X_te)[:, 1])
    return same_data, round(ap, 4) == c["test_average_precision"], round(ap, 4)


print("original data  -> data unchanged, score reproduced, AP:", verify(card_path, DATA))

# %% Step 6: a copy, a re-save, and a one-cent edit of the data
copy_dir = OUTPUT / "received"
copy_dir.mkdir(exist_ok=True)
shutil.copy(DATA / CONFIG["data_file"], copy_dir / CONFIG["data_file"])
print("byte-for-byte copy -> ", verify(card_path, copy_dir))
pd.read_csv(DATA / CONFIG["data_file"]).to_csv(copy_dir / CONFIG["data_file"], index=False)
print("re-saved by pandas -> ", verify(card_path, copy_dir))
edited = pd.read_csv(DATA / CONFIG["data_file"])
edited.loc[0, "amount"] += 0.01                      # one cent on one January payment
edited.to_csv(copy_dir / CONFIG["data_file"], index=False)
print("one cent changed   -> ", verify(card_path, copy_dir))

# %% Stretch: file sizes
import joblib

joblib.dump(model, OUTPUT / "fraud_hgb.joblib")
print(f"skops {model_path.stat().st_size / 1e3:.0f} kB, joblib {(OUTPUT / 'fraud_hgb.joblib').stat().st_size / 1e3:.0f} kB")
