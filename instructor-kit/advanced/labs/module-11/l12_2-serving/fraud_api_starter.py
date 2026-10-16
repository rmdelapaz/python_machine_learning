# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 12.2 A: a fraud-review API with an order window.

Lesson 12.2 served the churn model. The payments team wants the same for card fraud: send a payment,
get a fraud score back, and a yes/no "hold for review" that matches the review team's capacity (about
1% of payments). Work through TODO 1 to 6; the file runs as it is, and unfinished steps print a reminder.
"""
# %% Step 1: train on January to April, keep May to set the threshold (done for you)
import json
import warnings

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", message=".*Starlette.*")       # fastapi's test client import note
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
FEATURES = ["account_age_days", "amount", "hour", "items", "category", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
CATEGORIES = sorted(tx["category"].unique())
cat_type = pd.CategoricalDtype(CATEGORIES)
train = tx[tx["timestamp"] < "2025-05-01"]
may = tx[(tx["timestamp"] >= "2025-05-01") & (tx["timestamp"] < "2025-06-01")]
june = tx[tx["timestamp"] >= "2025-06-01"]
X_train = train[FEATURES].astype({"category": cat_type})
model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=42,
                                       learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0)
model.fit(X_train, train["is_fraud"])
may_scores = model.predict_proba(may[FEATURES].astype({"category": cat_type}))[:, 1]
print(f"train {len(train):,} payments ({train['is_fraud'].sum()} fraud), May {len(may):,}, June {len(june):,}")

# %% Step 2: package the model and the serving metadata
# TODO 1: sio.dump(model, OUTPUT / "fraud_model.skops"), then print sio.get_untrusted_types(file=...).
#         Review the list and WRITE IT OUT as TRUSTED = [...] (never pass whatever the file reports).
# TODO 2: THRESHOLD = the 99th percentile of may_scores (the riskiest 1% = the team's capacity). Build a
#         dict `serving` with version, features, dtypes (every feature but category), categories,
#         threshold (rounded to 4), trusted_types, reference_mean_score and reference_hold_rate (on May),
#         and write it to OUTPUT / "fraud_serving.json".
serving = None
if serving is None:
    print("TODO 1-2: save the model with skops, review its types, write fraud_serving.json")

# %% Step 3: the schema
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Category = Literal["Accessories", "Apparel", "Camp Kitchen", "Climbing", "Footwear",
                   "Gift Cards", "Packs", "Sleeping", "Snow", "Tents"]


class Payment(BaseModel):
    """One online card payment, in the units the model was trained on."""
    model_config = ConfigDict(extra="forbid")
    account_age_days: int = Field(ge=0, le=20_000)
    amount: float = Field(gt=0, le=10_000)
    # TODO 3: add the other eight fields with sensible limits: hour 0..23, items 1..50, category,
    #         the three 0/1 flags, card_attempts 1..20, velocity_24h 0..100.


class Verdict(BaseModel):
    fraud_score: float
    hold: bool
    model_version: str


first = june[FEATURES].iloc[0].to_dict()
bad = {**first, "hour": 24, "category": "Groceries", "amount": -5.0}
del bad["items"]
# TODO 4: validate `bad` with Payment(**bad) inside try/except ValidationError and print each problem's
#         field and message. Before you run it: how many problems will pydantic report? ______
print("TODO 3-4: finish the schema and read the problems with `bad`")

# %% Step 4: the app
from fastapi import FastAPI
from fastapi.testclient import TestClient

# TODO 5: load the model with sio.load(..., trusted=serving["trusted_types"]); write to_frame(payments)
#         that builds a DataFrame with columns=serving["features"], applies serving["dtypes"] and turns
#         category into pd.CategoricalDtype(serving["categories"]); add /health, /score and
#         /score_batch (list[Payment] -> list[Verdict]); hold = score >= serving["threshold"].
app = None
print("TODO 5: the FastAPI app")

# %% Step 5 and 6: test it without a server, then send June through the batch endpoint
# TODO 6: client = TestClient(app). Print /health, one /score for `first`, the status and fields for
#         `bad`. Then post all June payments to /score_batch, compare with model.predict_proba on the
#         same rows (np.allclose, atol=1e-4), and count how many are held and how many frauds that holds.
print("TODO 6: the match test and June's hold rate")

print("\nExpected when done: threshold 0.0775 (reference hold rate 0.0101, mean score 0.0084); `bad` has")
print("4 problems (amount, hour, items, category) and the API answers 422; the first June payment scores")
print("0.0021 (no hold); the API matches the notebook (True); June holds 83 of 8,262 payments (1.00%),")
print("with 35 of the 64 frauds among them.")
