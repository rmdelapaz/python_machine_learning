# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Capstone lab 2 (SOLUTION): an evaluation harness and a first pipeline (Milestones 3 and 4).

Milestone 3 asks for ONE function that scores any candidate on the TRAINING data only, the way your
problem's split works; Milestone 4 for the simplest sensible pipeline that beats the baseline in that
harness. This solution does it for all five problems so you have reference numbers; a learner does it
for the problem they chose. The test set is never touched here.
"""
# %% Step 0: the lesson's template (capstone_start.py next to this file)
sys.path.insert(0, str(pathlib.Path(globals().get("__file__", "lab.py")).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, average_precision_score, mean_absolute_error, recall_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import csr_matrix

from capstone_start import start_demand, start_fraud, start_recs, start_reviews, start_segments

results = []

# %% Step 1: demand: rolling-origin CV on the 2024 weeks (train on the past, validate on the next block)
dm = start_demand()
d = dm["train"].sort_values("week_start").reset_index(drop=True)
d["month"] = d["week_start"].dt.month
CATS = ["store", "product_id", "category", "month"]
NUMS = ["list_price", "discount_pct", "price", "promo", "holiday_week"]


def evaluate_demand(make_model):
    maes = []
    for fit_rows, val_rows in TimeSeriesSplit(n_splits=4).split(d):
        model = make_model().fit(d.iloc[fit_rows], d["units_sold"].iloc[fit_rows])
        maes.append(mean_absolute_error(d["units_sold"].iloc[val_rows], model.predict(d.iloc[val_rows])))
    return np.mean(maes), np.std(maes)


class PerProductMean:
    """The template's baseline as an estimator, so the harness can score it the same way."""
    def fit(self, X, y):
        self.means_ = y.groupby(X["product_id"]).mean()
        self.overall_ = y.mean()
        return self

    def predict(self, X):
        return X["product_id"].map(self.means_).fillna(self.overall_).to_numpy()


def ridge():
    prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), CATS), (StandardScaler(), NUMS))
    return make_pipeline(prep, Ridge(alpha=1.0))


for name, maker in [("per-product mean", PerProductMean), ("ridge, one-hot", ridge)]:
    mean, sd = evaluate_demand(maker)
    results.append(("demand", "MAE, 4 rolling folds (lower is better)", name, mean, sd))

# %% Step 2: fraud: train January to April, validate on May (the test month stays June)
fr = start_fraud()
f = fr["train"]
is_may = (f["timestamp"] >= "2025-05-01").to_numpy()
NUM_F = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
         "country_mismatch", "card_attempts", "velocity_24h"]


def evaluate_fraud(model):
    model.fit(f[~is_may], f["is_fraud"][~is_may])
    scores = model.predict_proba(f[is_may])[:, 1]
    return average_precision_score(f["is_fraud"][is_may], scores), np.nan


class NoModel:
    def fit(self, X, y):
        self.rate_ = y.mean()
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate_), np.full(len(X), self.rate_)])


lr_fraud = make_pipeline(make_column_transformer((StandardScaler(), NUM_F), (OneHotEncoder(handle_unknown="ignore"), ["category"])),
                         LogisticRegression(max_iter=1000))
for name, model in [("no model", NoModel()), ("logistic regression", lr_fraud)]:
    mean, sd = evaluate_fraud(model)
    results.append(("fraud", "average precision on May", name, mean, sd))

# %% Step 3: reviews: stratified 5-fold CV, accuracy AND the recall of negative reviews
rv = start_reviews()
r = rv["train"]
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"accuracy": "accuracy", "neg_recall": lambda est, X, y: recall_score(y, est.predict(X), pos_label=0)}


def evaluate_reviews(model):
    cvr = cross_validate(model, r["text"], r["positive"], cv=CV5, scoring=SCORING)
    return cvr["test_accuracy"].mean(), cvr["test_neg_recall"].mean()


for name, model in [("always positive", DummyClassifier(strategy="constant", constant=1)),
                    ("tf-idf + logistic regression", make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)))]:
    acc, neg = evaluate_reviews(model)
    results.append(("reviews", "CV accuracy (second value: negative recall)", name, acc, neg))

# %% Step 4: recommendations: hide each customer's second-to-last purchase from the training rows
rc = start_recs()
train_rows = rc["train"]
val = train_rows.groupby("customer_id").tail(1)
inner = train_rows.drop(val.index)
customers = np.sort(train_rows["customer_id"].unique())
products = np.sort(pd.concat([rc["train"], rc["test"]])["product_id"].unique())
row_of = pd.Series(np.arange(len(customers)), index=customers)
col_of = pd.Series(np.arange(len(products)), index=products)
R = csr_matrix((np.ones(len(inner)), (row_of[inner["customer_id"]].to_numpy(), col_of[inner["product_id"]].to_numpy())),
               shape=(len(customers), len(products)))
owned = R.toarray() > 0
val_col = col_of[val.set_index("customer_id").loc[customers, "product_id"]].to_numpy()


def evaluate_recs(scores, k=10):
    s = np.array(scores, dtype=float)
    s[owned] = -np.inf
    return (np.argsort(-s, axis=1)[:, :k] == val_col[:, None]).any(axis=1).mean(), np.nan


item_sim = cosine_similarity(R.T)
np.fill_diagonal(item_sim, 0)
for name, scores in [("popularity", np.tile(np.asarray(R.sum(axis=0)).ravel(), (R.shape[0], 1))),
                     ("item-kNN", np.asarray(R @ item_sim))]:
    hr, _ = evaluate_recs(scores)
    results.append(("recs", "hit rate at 10, validation item", name, hr, np.nan))
for k in (3, 5, 10):
    svd = TruncatedSVD(n_components=k, random_state=42)
    hr, _ = evaluate_recs(svd.fit_transform(R) @ svd.components_)
    results.append(("recs", "hit rate at 10, validation item", f"SVD k={k}", hr, np.nan))

# %% Step 5: segments: silhouette AND stability (adjusted Rand index between two seeds, and two halves)
sg = start_segments()
X = StandardScaler().fit_transform(sg["train"][sg["features"]])
results.append(("segments", "silhouette (second value: ARI seeds 42 vs 7)", "existing loyalty tiers", sg["baseline"][1], np.nan))
rng = np.random.default_rng(42)
half = rng.permutation(len(X))[: len(X) // 2]
for k in (4, 5, 6):
    a = KMeans(k, n_init=10, random_state=42).fit(X)
    b = KMeans(k, n_init=10, random_state=7).fit(X)
    sil = silhouette_score(X, a.labels_, sample_size=3000, random_state=42)
    results.append(("segments", "silhouette (second value: ARI seeds 42 vs 7)", f"k-means k={k}", sil,
                    adjusted_rand_score(a.labels_, b.labels_)))
    if k == 5:
        on_half = KMeans(k, n_init=10, random_state=42).fit(X[half])
        half_ari = adjusted_rand_score(a.labels_, on_half.predict(X))

# %% Step 6: the table
table = pd.DataFrame(results, columns=["problem", "harness", "candidate", "score", "second"]).round(3)
print(table.set_index(["problem", "candidate"]).drop(columns="harness").to_string())
print("\nharness per problem:", dict(table.groupby("problem", sort=False)["harness"].first()))
print(f"segments k=5: model fitted on half the customers agrees with the full model, ARI {half_ari:.3f}")
