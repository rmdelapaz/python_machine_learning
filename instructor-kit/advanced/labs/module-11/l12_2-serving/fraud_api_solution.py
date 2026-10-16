# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 12.2 A (SOLUTION): a fraud-review API with an order window.

Lesson 12.2 served the churn model. The payments team wants the same for card fraud: send a payment,
get a fraud score back, and a yes/no "hold for review" that matches the review team's capacity (about
1% of payments). Same three habits as the lesson: a frozen artifact with its metadata, a strict
pydantic schema, and a test that the API matches the notebook.
"""
# %% Step 1: train on January to April, keep May to set the threshold and the reference numbers
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

# %% Step 2: package the model and the serving metadata (skops, not pickle)
model_path = OUTPUT / "fraud_model.skops"
sio.dump(model, model_path)
untrusted = sio.get_untrusted_types(file=model_path)
print("types to review:", untrusted)
# Reviewed by a person and written out, never copied from the file: if a later file asks for more, the
# load stops. (functools.partial and check_array come from categorical_features="from_dtype".)
TRUSTED = ["functools.partial", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
           "sklearn.utils.validation.check_array"]
print("matches the reviewed list:", sorted(untrusted) == sorted(TRUSTED))
THRESHOLD = float(np.quantile(may_scores, 0.99))          # hold the riskiest 1%, the team's capacity
serving = {
    "version": "fraud-hgb-2025-04-30",
    "features": FEATURES,
    "dtypes": {c: str(t) for c, t in X_train.dtypes.items() if c != "category"},
    "categories": CATEGORIES,
    "threshold": round(THRESHOLD, 4),
    "trusted_types": TRUSTED,
    "reference_mean_score": round(float(may_scores.mean()), 4),
    "reference_hold_rate": round(float((may_scores >= round(THRESHOLD, 4)).mean()), 4),
}
(OUTPUT / "fraud_serving.json").write_text(json.dumps(serving, indent=2))
print("threshold", serving["threshold"], " reference hold rate", serving["reference_hold_rate"],
      " reference mean score", serving["reference_mean_score"])

# %% Step 3: the schema: what a valid payment looks like
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Category = Literal["Accessories", "Apparel", "Camp Kitchen", "Climbing", "Footwear",
                   "Gift Cards", "Packs", "Sleeping", "Snow", "Tents"]


class Payment(BaseModel):
    """One online card payment, in the units the model was trained on."""
    model_config = ConfigDict(extra="forbid")
    account_age_days: int = Field(ge=0, le=20_000)
    amount: float = Field(gt=0, le=10_000)
    hour: int = Field(ge=0, le=23)
    items: int = Field(ge=1, le=50)
    category: Category
    device_new: int = Field(ge=0, le=1)
    billing_shipping_mismatch: int = Field(ge=0, le=1)
    country_mismatch: int = Field(ge=0, le=1)
    card_attempts: int = Field(ge=1, le=20)
    velocity_24h: int = Field(ge=0, le=100)


class Verdict(BaseModel):
    fraud_score: float
    hold: bool
    model_version: str


first = june[FEATURES].iloc[0].to_dict()
bad = {**first, "hour": 24, "category": "Groceries", "amount": -5.0}
del bad["items"]
try:
    Payment(**bad)
except ValidationError as err:
    print(f"\n{err.error_count()} problems")
    for problem in err.errors():
        print(" ", problem["loc"][0], "->", problem["msg"])

# %% Step 4: the app (load the reviewed file once, rebuild the training table in ONE function)
from fastapi import FastAPI
from fastapi.testclient import TestClient

MODEL = sio.load(model_path, trusted=serving["trusted_types"])
app = FastAPI(title="Pinecrest fraud review", version=serving["version"])


def to_frame(payments):
    rows = [p.model_dump() for p in payments]
    frame = pd.DataFrame(rows, columns=serving["features"]).astype(serving["dtypes"])
    return frame.astype({"category": pd.CategoricalDtype(serving["categories"])})


def to_verdict(p):
    return Verdict(fraud_score=round(float(p), 4), hold=bool(p >= serving["threshold"]),
                   model_version=serving["version"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": serving["version"]}


@app.post("/score")
def score(payment: Payment) -> Verdict:
    return to_verdict(MODEL.predict_proba(to_frame([payment]))[0, 1])


@app.post("/score_batch")
def score_batch(payments: list[Payment]) -> list[Verdict]:
    return [to_verdict(p) for p in MODEL.predict_proba(to_frame(payments))[:, 1]]


# %% Step 5: test it without a server
client = TestClient(app)
print("\n", client.get("/health").json())
r = client.post("/score", json=first)
print(r.status_code, r.json())
r = client.post("/score", json=bad)
print(r.status_code, [p["loc"][-1] for p in r.json()["detail"]])

# %% Step 6: the match test, then June through the batch endpoint
june_rows = june[FEATURES].to_dict(orient="records")       # no gaps in this table, so no NaN to convert
verdicts = pd.DataFrame(client.post("/score_batch", json=june_rows).json())
direct = model.predict_proba(june[FEATURES].astype({"category": cat_type}))[:, 1]
print("\nAPI matches the notebook:", np.allclose(verdicts["fraud_score"], direct, atol=1e-4))
held = verdicts["hold"].to_numpy()
caught = int(june["is_fraud"].to_numpy()[held].sum())
print(f"June: {held.sum()} of {len(june):,} held ({held.mean():.2%}; reference {serving['reference_hold_rate']:.2%}), "
      f"{caught} of {june['is_fraud'].sum()} frauds among them")
