# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.2 B (SOLUTION): customers like you, and "because you just bought".

The lesson built item-based filtering and mentioned the flipped idea: find customers similar to this
one and recommend what they bought (user-based filtering). Build it, tune its one setting (how many
neighbors to listen to) on a VALIDATION item, score it once on the test item, and then try the
simplest personal list of all: neighbors of the customer's most recent purchase only.
"""
# %% Step 1: three-way split by time and the helpers
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

interactions = pd.read_csv(DATA / "interactions.csv").sort_values(["customer_id", "last_purchase_date", "product_id"])
products = pd.read_csv(DATA / "products_ml.csv")
test = interactions.groupby("customer_id").tail(1)
rest = interactions.drop(test.index)
valid = rest.groupby("customer_id").tail(1)
fit_rows = rest.drop(valid.index)

customer_ids = np.sort(interactions["customer_id"].unique())
product_ids = products["product_id"].to_numpy()
row_of = pd.Series(np.arange(len(customer_ids)), index=customer_ids)
col_of = pd.Series(np.arange(len(product_ids)), index=product_ids)


def build_matrix(rows):
    return csr_matrix((np.ones(len(rows)), (row_of[rows["customer_id"]].to_numpy(), col_of[rows["product_id"]].to_numpy())),
                      shape=(len(customer_ids), len(product_ids)))


def hit_rate_on(scores, R, target_rows, k=10):
    scores = np.array(scores, dtype=float)
    scores[R.toarray() > 0] = -np.inf
    target = col_of[target_rows.set_index("customer_id").loc[customer_ids, "product_id"]].to_numpy()
    return (np.argsort(-scores, axis=1)[:, :k] == target[:, None]).any(axis=1).mean()


def item_knn(R):
    sim = cosine_similarity(R.T)
    np.fill_diagonal(sim, 0)
    return R @ sim


R_fit = build_matrix(fit_rows)
print(f"fit {len(fit_rows):,}  validation {len(valid):,}  test {len(test):,}")

# %% Step 2: user-based filtering: listen to the n most similar customers
def user_knn(R, n_neighbors):
    sim = cosine_similarity(R)                                  # customers x customers
    np.fill_diagonal(sim, 0)                                    # you are not your own neighbor
    weakest = np.argsort(sim, axis=1)[:, :-n_neighbors]         # everyone except the n most similar
    np.put_along_axis(sim, weakest, 0, axis=1)
    return sim @ R.toarray()                                    # neighbors' purchases, weighted by similarity


print("customer similarity matrix:", cosine_similarity(R_fit).shape)

# %% Step 3: choose the number of neighbors on the validation item
val = {n: hit_rate_on(user_knn(R_fit, n), R_fit, valid) for n in [10, 50, 200, 1000]}
for n, hr in val.items():
    print(f"user-kNN, {n:4d} neighbors: validation HR@10 {hr:.3f}")
print(f"item-kNN               : validation HR@10 {hit_rate_on(item_knn(R_fit), R_fit, valid):.3f}")
best_n = max(val, key=val.get)
print("chosen:", best_n, "neighbors")

# %% Step 4: score once on the test item
R_train = build_matrix(rest)
print(f"test HR@10: user-kNN ({best_n}) {hit_rate_on(user_knn(R_train, best_n), R_train, test):.3f}, "
      f"item-kNN {hit_rate_on(item_knn(R_train), R_train, test):.3f}")

# %% Step 5: "because you just bought": neighbors of the most recent purchase only
last = rest.groupby("customer_id").tail(1)
R_last = build_matrix(last)                                     # one product per customer
sim = cosine_similarity(R_train.T)
np.fill_diagonal(sim, 0)
print(f"last purchase only: test HR@10 {hit_rate_on(R_last @ sim, R_train, test):.3f}")

# %% Stretch: blend the whole history with extra weight on the last purchase
for w in [1, 3]:
    print(f"history + {w} x last purchase: test HR@10 {hit_rate_on((R_train + w * R_last) @ sim, R_train, test):.3f}")
