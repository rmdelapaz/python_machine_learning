# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.2 B: customers like you, and "because you just bought".

The lesson built item-based filtering and mentioned the flipped idea: find customers similar to this
one and recommend what they bought (user-based filtering). Build it, tune its one setting (how many
neighbors to listen to) on a VALIDATION item, score it once on the test item, then try a list built
from the most recent purchase only. Work through TODO 1 to 5; run the file after each one.
It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: three-way split by time and the helpers (done for you)
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
# TODO 1: write user_knn(R, n_neighbors): sim = cosine_similarity(R) (customers x customers; which
#         shape do you expect? ______), zero diagonal, then keep only each row's n largest values:
#         weakest = np.argsort(sim, axis=1)[:, :-n_neighbors]; np.put_along_axis(sim, weakest, 0, axis=1).
#         Return sim @ R.toarray(): every product scored by the similar customers who bought it.
user_knn = None
if user_knn is None:
    print("TODO 1: user_knn")

# %% Step 3: choose the number of neighbors on the validation item
# TODO 2: validation HR@10 of user_knn(R_fit, n) for n in [10, 50, 200, 1000], and of item_knn(R_fit).
#         best_n = the n with the highest validation hit rate. Why might 1,000 neighbors be worse than 200?
print("TODO 2: choose n on the validation item")

# %% Step 4: score once on the test item
# TODO 3: refit on `rest` (R_train = build_matrix(rest)) and print the test HR@10 of user-kNN with best_n
#         and of item-kNN. Only now look at the test item.
print("TODO 3: one test score each")

# %% Step 5: "because you just bought": neighbors of the most recent purchase only
# TODO 4: last = rest.groupby("customer_id").tail(1); R_last = build_matrix(last) holds ONE product per
#         customer. Score with R_last @ sim, where sim is the item similarity of R_train (zero diagonal).
#         Mask with R_train (the whole history). Predict first: better or worse than the whole history?
# TODO 5: in a comment: what does the result suggest about the dates in interactions.csv?
print("TODO 4-5: last purchase only")

# %% Stretch: score (R_train + w * R_last) @ sim for w = 1 and 3.

print("\nExpected when done: user-kNN validation HR@10 0.202 / 0.282 / 0.292 / 0.266 for 10 / 50 / 200 / 1000")
print("neighbors, item-kNN 0.286; chosen 200; test HR@10 user-kNN 0.321, item-kNN 0.308; last purchase only")
print("0.212. Stretch: 0.305 (w = 1) and 0.298 (w = 3).")
