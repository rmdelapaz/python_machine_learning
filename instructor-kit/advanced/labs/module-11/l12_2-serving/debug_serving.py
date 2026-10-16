# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs: a fraud API that a colleague says "works".

This service has FOUR bugs between the JSON and the model. Two stop it with an error; two do not, and
those are the dangerous ones. Fix them one at a time, rerunning after each (read the LAST line of each
error first). When all four are fixed the last lines print `API matches the notebook: True` and
`June: 83 held, 35 of them fraud`.
"""
# %% The model and its metadata (no bugs in this cell)
import json
import warnings

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", message=".*Starlette.*")
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
FEATURES = ["account_age_days", "amount", "hour", "items", "category", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
CATEGORIES = sorted(tx["category"].unique())
train = tx[tx["timestamp"] < "2025-05-01"]
may = tx[(tx["timestamp"] >= "2025-05-01") & (tx["timestamp"] < "2025-06-01")]
june = tx[tx["timestamp"] >= "2025-06-01"]
model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=42, learning_rate=0.05,
                                       max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
model.fit(train[FEATURES].astype({"category": pd.CategoricalDtype(CATEGORIES)}), train["is_fraud"])
may_scores = model.predict_proba(may[FEATURES].astype({"category": pd.CategoricalDtype(CATEGORIES)}))[:, 1]
SERVING = json.loads(json.dumps({"features": FEATURES, "categories": CATEGORIES,
                                 "threshold": f"{np.quantile(may_scores, 0.99):.4f}"}))


# %% The service
class Payment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_age_days: int
    amount: int
    hour: int
    items: int
    category: str
    device_new: int
    billing_shipping_mismatch: int
    country_mismatch: int
    card_attempts: int
    velocity_24h: int


app = FastAPI()


def to_frame(payments):
    frame = pd.DataFrame([p.model_dump() for p in payments], columns=SERVING["features"])
    frame["category"] = pd.Categorical(frame["category"], categories=SERVING["categories"])
    return frame


@app.post("/score_batch")
def score_batch(payments: list[Payment]) -> list[dict]:
    frame = to_frame(payments)
    scores = model.predict_proba(frame)[:, 1]
    holds = model.predict(frame)
    return [{"fraud_score": round(float(p), 2), "hold": bool(h), "above_threshold": bool(p >= SERVING["threshold"])}
            for p, h in zip(scores, holds)]


# %% The tests: the checkout sends ONE payment per request; the nightly job sends all of June at once
client = TestClient(app)
rows = june[FEATURES].to_dict(orient="records")
direct = model.predict_proba(june[FEATURES].astype({"category": pd.CategoricalDtype(CATEGORIES)}))[:, 1]
one_by_one = pd.DataFrame([client.post("/score_batch", json=[row]).json()[0] for row in rows[:200]])
print("API matches the notebook:", np.allclose(one_by_one["fraud_score"], direct[:200], atol=1e-4))
answer = pd.DataFrame(client.post("/score_batch", json=rows).json())
held = answer["hold"].to_numpy()
print(f"June: {held.sum()} held, {int(june['is_fraud'].to_numpy()[held].sum())} of them fraud")
